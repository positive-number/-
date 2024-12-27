#!/usr/bin/env python
# -*- coding:utf-8 -*-
# limo_application/scripts/lane_detection/control.py

import rospy
import time
import threading
from std_msgs.msg import Int32, String, Bool, Float32
from geometry_msgs.msg import Twist
from object_msgs.msg import ObjectArray
from limo_base.msg import LimoStatus
from dynamic_reconfigure.server import Server
from limo_application.cfg import controlConfig
import math

class LimoController:
    def __init__(self):
        rospy.init_node('limo_control', anonymous=True)
        self.LIMO_WHEELBASE = 0.25
        self.distance_to_ref = 0
        self.prev_distance_to_ref = 0  # 직전 차선 거리로 코너 감지에 사용
        self.crosswalk_detected = True
        self.yolo_object = "green"
        self.e_stop = "Safe"
        self.is_pedestrian_stop_available = True
        self.pedestrian_stop_time = 5.0
        self.pedestrian_stop_last_time = rospy.Time.now().to_sec()
        self.yolo_object_last_time = rospy.Time.now().to_sec()
        self.bbox_size = [0, 0]
        self.limo_mode = "ackermann"
        self.crosswalk_detected = False
        self.is_force_disabled = False
        self.redline_detected = False
        self.wall_speed = 0
        self.wall_angle = 0

        # 상태 변수 추가
        self.state = 0  # 기본 state는 0

        srv = Server(controlConfig, self.reconfigure_callback)
        rospy.Subscriber("limo_status", LimoStatus, self.limo_status_callback)
        rospy.Subscriber("/limo/lane_x", Int32, self.lane_x_callback)
        rospy.Subscriber("/limo/wall_speed", Float32, self.wall_speed_callback)
        rospy.Subscriber("/limo/wall_angle", Float32, self.wall_angle_callback)
        rospy.Subscriber("/limo/crosswalk_y", Int32, self.crosswalk_y_callback)
        rospy.Subscriber("/limo/yolo_object", ObjectArray, self.yolo_object_callback)
        rospy.Subscriber("/limo/lidar_warning", String, self.lidar_warning_callback)
        rospy.Subscriber("/limo/redline_detect", Int32, self.redline_callback)  # 빨간선 감지 구독
        self.drive_pub = rospy.Publisher(rospy.get_param("~control_topic_name", "/cmd_vel"), Twist, queue_size=1)
        rospy.Timer(rospy.Duration(0.03), self.drive_callback)

    def redline_callback(self, msg):
        '''
        빨간선 감지 콜백 함수
        '''
        if msg.data == 1:
            rospy.logwarn("Red line detected! Switching to state 1.")
            self.state = 1 # 빨간선 감지 시 state=1로 변경
            rospy.logwarn("Red line detected! Switching to state 2.")
            self.state = 2
            time.sleep(10000)
        else:
            self.redline_detected = False

    def calcTimeFromDetection(self, _last_detected_time):
        return rospy.Time.now().to_sec() - _last_detected_time

    # ==============================================
    #               Callback Functions
    # ==============================================

    def force_crosswalk_disable(self):
        self.is_force_disabled = True
        threading.Timer(5, self.restore_crosswalk_detection).start()

    def restore_crosswalk_detection(self):
        self.is_force_disabled = False
        rospy.loginfo("Crosswalk detection restored.")

    def limo_status_callback(self, _data):
        if _data.motion_mode == 1:
            if self.limo_mode == "ackermann":
                pass
            else:
                self.limo_mode = "ackermann"
                rospy.loginfo("Mode Changed --> Ackermann")
        else:
            if self.limo_mode == "diff":
                pass
            else:
                self.limo_mode = "diff"
                rospy.loginfo("Mode Changed --> Differential Drive")

    def lidar_warning_callback(self, _data):
        self.e_stop = _data.data

    def yolo_object_callback(self, _data):
        if len(_data.Objects) == 0:
            pass
        else:
            self.yolo_object = _data.Objects[0].class_name
            self.yolo_object_last_time = rospy.Time.now().to_sec()
            self.bbox_size = [
                _data.Objects[0].xmin_ymin_xmax_ymax[2] - _data.Objects[0].xmin_ymin_xmax_ymax[0],
                _data.Objects[0].xmin_ymin_xmax_ymax[3] - _data.Objects[0].xmin_ymin_xmax_ymax[1]
            ]

    def lane_x_callback(self, _data):
        if _data.data == -1:
            self.distance_to_ref = 0
        else:
            self.distance_to_ref = self.REF_X - _data.data

    def crosswalk_y_callback(self, _data):
        if _data.data == -1:
            self.crosswalk_detected = False
            self.crosswalk_distance = _data.data
        else:
            self.crosswalk_detected = True
            self.crosswalk_distance = _data.data

    def wall_speed_callback(self, _data):
        self.wall_speed = _data.data

    def wall_angle_callback(self, _data):
        self.wall_angle = _data.data

    def reconfigure_callback(self, _config, _level):
        self.BASE_SPEED = _config.base_speed
        self.LATERAL_GAIN = float(_config.lateral_gain * 0.001)
        self.REF_X = _config.reference_lane_x
        self.PEDE_STOP_WIDTH = _config.pedestrian_width_min
        self.CORNER_GAIN_MULTIPLIER = _config.get('corner_gain_multiplier', 1.0)  # 코너 감지 시 Gain 조정
        return _config

    def drive_callback(self, _event):
   
        if self.yolo_object != "green" and self.calcTimeFromDetection(self.yolo_object_last_time) > 3.0:
            self.yolo_object = "green"
            self.bbox_size = [0, 0]
           
        if self.calcTimeFromDetection(self.pedestrian_stop_last_time) >20.0:
            self.is_pedestrian_stop_available = True
           
        drive_data = Twist()
        drive_data.angular.z = self.distance_to_ref * self.LATERAL_GAIN
        rospy.loginfo("OFF_CENTER, Lateral_Gain = {}, {}".format(self.distance_to_ref, self.LATERAL_GAIN))
        rospy.loginfo("Bbox Size = {}, Bbox_width_min = {}".format(self.bbox_size, self.PEDE_STOP_WIDTH))

        # 상태에 따른 동작
        if self.state == 0:
            if self.e_stop == "Warning":
                drive_data.linear.x = 0.0
                drive_data.angular.z = 0.0
                rospy.logwarn("Obstacle Detected, Stop!")
                self.drive_pub.publish(drive_data)

            elif self.yolo_object == "yellow" or self.yolo_object == "red":
                drive_data.linear.x = 0.0
                drive_data.angular.z = 0.0
                rospy.logwarn("Traffic light is Red or Yellow, Stop!")
           
            elif self.yolo_object == "slow":
                drive_data.linear.x = self.BASE_SPEED / 2
                rospy.logwarn("Slow Traffic Sign Detected, Slow Down!")

            elif self.yolo_object == "pedestrian" and self.is_pedestrian_stop_available and self.bbox_size[0] > self.PEDE_STOP_WIDTH:
                drive_data.linear.x = 0.0
                drive_data.angular.z = 0.0
                self.is_pedestrian_stop_available = False
                self.pedestrian_stop_last_time = rospy.Time.now().to_sec()
                rospy.logwarn("Pedestrian Traffic Sign Detected, Stop {} Seconds!".format(self.pedestrian_stop_time))
                rospy.sleep(rospy.Duration(self.pedestrian_stop_time))

            elif self.crosswalk_detected and not self.is_force_disabled:
                drive_data.linear.x = 0.0
                rospy.logwarn("Crosswalk Detected, Stop!")
                self.drive_pub.publish(drive_data)
               
                time.sleep(3)
                self.crosswalk_detected = False
                rospy.logwarn("Crosswalk detection disabled for 5 seconds.")
                self.force_crosswalk_disable()

            else:
                drive_data.linear.x = self.BASE_SPEED
                rospy.loginfo("All Clear, Just Drive!")

            if self.limo_mode == "diff":
                self.drive_pub.publish(drive_data)
            elif self.limo_mode == "ackermann":
                if drive_data.linear.x == 0:
                    drive_data.angular.z = 0
                else:
                    drive_data.angular.z = \
                        math.tan(drive_data.angular.z / 2) * drive_data.linear.x / self.LIMO_WHEELBASE
                    self.drive_pub.publish(drive_data)

        elif self.state == 1 or 2:
            rospy.loginfo("State 1: Wall following mode")
            drive_data.linear.x = self.wall_speed
            drive_data.angular.z = self.wall_angle
            self.drive_pub.publish(drive_data)
            self.state = self.state+1

        elif self.state >= 3:
            rospy.loginfo("State 2: Vehicle stopped")
            drive_data.linear.x = 0.0
            drive_data.angular.z = 0.0
            self.drive_pub.publish(drive_data)


def run():
    new_class = LimoController()
    rospy.spin()

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("program down")
