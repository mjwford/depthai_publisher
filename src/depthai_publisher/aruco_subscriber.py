#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import numpy as np


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'
    # camera_matrix
    mtx = np.array([[623.680552, 0, (720/2)], [0, 623.680552, (480/2)], [0, 0, 1]], dtype=np.float)
    # distortion_coefficients
    dist = np.array([[0, 0, 0, 0]], dtype=np.float)
        
    def __init__(self):
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)

        self.br = CvBridge()

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)

        aruco = self.find_aruco(frame)
        self.publish_to_ros(aruco)

        # cv2.imshow('aruco', aruco)
        # cv2.waitKey(1)

    def find_aruco(self, frame):
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))

                cv2.putText(frame, str(
                    marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                # draw frames on arucos
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(marker_corner, 0.02, self.mtx, self.dist)
                cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, 0.02)

        return frame

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

        self.aruco_pub.publish(msg_out)


def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()
