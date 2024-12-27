#! /usr/bin/env python
# -*- coding: utf-8 -*-
# limo_application/scripts/lane_detection/red_line_stop.py

import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from limo_application.cfg import crosswalkConfig

import cv2
import numpy as np

class RedLineStopDetector:
    '''
        ROS 기반 빨간색 선 감지 및 정지 객체
        Private Params --> image_topic_name, visualization
        Image Topic Subscriber (CompressedImage Type)
        Distance to Stop Line Topic Publisher (Int32 Type)
    '''
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node("red_line_stop")
        # Dynamic Reconfigure 서버 초기화
        srv = Server(crosswalkConfig, self.reconfigure_callback)
        # CvBridge 객체 생성 (ROS 이미지와 OpenCV 이미지 간 변환)
        self.cvbridge = CvBridge()
        # ROS 이미지 토픽 구독
        rospy.Subscriber(rospy.get_param("~image_topic_name", "/camera/rgb/image_raw/compressed"), CompressedImage, self.Image_CB)
        # 빨간색 선 감지 결과 퍼블리싱 설정
        self.distance_pub = rospy.Publisher("/limo/redline_detect", Int32, queue_size=5)
        # 시각화 여부를 ROS 파라미터에서 가져옴
        self.viz = rospy.get_param("~visualization", False)

    def detectRedLine(self, _img):
        '''
            빨간색 영역만 검출
        '''
        # 이미지를 HSV 색 공간으로 변환
        hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
        # 빨간색 HSV 범위 정의 (2개의 범위로 분리된 빨간색)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 두 범위에 해당하는 마스크 생성
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # 두 마스크를 합쳐 최종 빨간색 마스크 생성
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        return red_mask

    def calcRedLineDistance(self, _img):
        '''
            빨간색 선이 있는지 확인하고, 거리를 계산하여 반환
        '''
        _, binary_red_mask = cv2.threshold(_img, 127, 255, cv2.THRESH_BINARY)
        M = cv2.moments(binary_red_mask)
        if M['m00'] > 0:  # 빨간색 픽셀이 존재하는 경우
            # 무게중심 x, y 좌표 계산
            self.x = int(M['m10'] / M['m00'])
            self.y = int(M['m01'] / M['m00'])
            return self.y  # y 좌표(거리) 반환
        else:
            return -1  # 빨간색 선 없음

    def visResult(self):
        '''
            결과 시각화
        '''
        # 빨간색 선의 y 좌표를 시각적으로 표시
        if not self.x <= 0 and not self.y <= 0:
            cv2.line(self.cropped_image, (0, self.y), (self.crop_size_x, self.y), (0, 0, 255), 20)
        # 원본 이미지, ROI, 빨간색 마스크를 화면에 표시
        cv2.imshow("original", self.frame)
        cv2.imshow("cropped", self.cropped_image)
        cv2.imshow("red_line", self.red_mask)
        cv2.waitKey(1)

    def imageCrop(self, _img=np.ndarray(shape=(480, 640))):
        '''
            원하는 이미지 영역 검출
        '''
        # ROI 설정 (이미지 하단의 특정 영역 선택)
        self.crop_size_x = 360
        self.crop_size_y = 60
        return _img[360:480, 200:480]

    def reconfigure_callback(self, config, level):
        '''
            Dynamic_Reconfigure를 활용하여, 설정값 갱신
        '''
        return config

    def Image_CB(self, img):
        '''
            실제 이미지를 받아 동작하는 부분
        '''
        # ROS 압축 이미지를 OpenCV 이미지로 변환
        self.frame = self.cvbridge.compressed_imgmsg_to_cv2(img, "bgr8")
        # ROI 영역 추출
        self.cropped_image = self.imageCrop(self.frame)
        # 빨간색 마스크 생성
        self.red_mask = self.detectRedLine(self.cropped_image)
        # 빨간색 선의 거리 계산
        self.red_line_distance = self.calcRedLineDistance(self.red_mask)

        # 빨간색 선이 감지되었는지 여부를 퍼블리싱
        if self.red_line_distance > 0:
            rospy.loginfo("Red line detected! Distance: {}".format(self.red_line_distance))
            self.distance_pub.publish(1)  # 빨간색 선 감지됨
        else:
            self.distance_pub.publish(0)  # 빨간색 선 없음

        # 시각화 옵션이 켜져 있으면 결과 표시
        if self.viz:
            self.visResult()

def run():
    # RedLineStopDetector 객체 생성 및 ROS 이벤트 루프 실행
    red_line_detector = RedLineStopDetector()
    rospy.spin()

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("program down")
