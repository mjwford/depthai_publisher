#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import math

from geometry_msgs.msg import Point, PoseStamped

from visualization_msgs.msg import Marker


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()
    
    


    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        self.time_finished_processing = rospy.Time(0)
    
        #Get input for ArUco landing ID.
        while True:
            try:
                self.landing_id = int(input("Enter ArUco Landing ID: "))
                break  # Exit the loop if the input is a valid integer
            except ValueError:
                print("Invalid input! Please enter an integer.")
        rospy.loginfo("Searching for ID " + str(self.landing_id))

        self.detected_marker = False

        #ID publisher.
        self.id_coordinate_publisher = rospy.Publisher("landing_id_coordinate", Point, queue_size=10)

        #Current position subscriber.
        self.current_uav_position = None
        self.sub_uav_position = rospy.Subscriber("current_position", Point, self.callback_uav_position)


        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)

        

        self.br = CvBridge()

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    def img_callback(self, msg_in):
        if msg_in.header.stamp > self.time_finished_processing:        
            try:
                frame = self.br.compressed_imgmsg_to_cv2(msg_in)
            except CvBridgeError as e:
                rospy.logerr(e)

            aruco = self.find_aruco(frame)
            self.publish_to_ros(aruco)

            # cv2.imshow('aruco', aruco)
            # cv2.waitKey(1)
            
            self.time_finished_processing = rospy.Time.now()
            
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

                x_pixel = (top_right[0] + top_left[0]) / 2
                y_pixel = (top_left[1] + bottom_left[1]) / 2

            


                #rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
                rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
                #Determine if its the landing id. 
                #USE FIRST LINE TO LAND ON SPECIFIC MARKER AND USE OTHER LINE TO LAND ON ANY.
                if(marker_ID == self.landing_id) and not self.detected_marker and (self.current_uav_position is not None):
                    #if not self.detected_marker and (self.current_uav_position is not None):
                
                    #if not self.detected_marker and (self.current_uav_position is not None):
                    rospy.loginfo("Correct Marker Detected")

                if(marker_ID == self.landing_id) and (self.current_uav_position is not None):
   
                    real_world_coordinates = self.convert_to_real_world(x_pixel, y_pixel, self.current_uav_position.x, self.current_uav_position.y, self.current_uav_position.z)
                    self.send_marker_aruco(real_world_coordinates[0], real_world_coordinates[1])
                    self.send_coordinate_roi(real_world_coordinates[0], real_world_coordinates[1], self.current_uav_position.z)
                    
                    #self.id_coordinate_publisher.publish(self.current_uav_position)
                    self.detected_marker = True

                cv2.putText(frame, str(
                    marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

        self.aruco_pub.publish(msg_out)

    def callback_uav_position(self, msg_in):
        self.current_uav_position = msg_in




    #Pixel coordinates to real world coordinates
    def convert_to_real_world(self, x_pixel, y_pixel, x_data, y_data, z_data):
        # Calculate Fx and Fy
        #focal_length_x = (415 * 0.5) / (math.tan(69 * 0.5 * (math.pi/180)))
        #focal_length_y = (415 * 0.5) / (math.tan(69 * 0.5 * (math.pi/180)))
        #principal_point_x = 415 * 0.5
        #principal_point_y = 415 * 0.5
        
        
        # Assuming linear mapping between depth and real-world coordinates
        #depth_value = depth_data[y_pixel, x_pixel]  # Get depth value at pixel coordinates
        # Convert the raw depth value to real-world coordinates (in meters)
        #depth_in_meters = (depth_value * 0.001)  # Convert mm to meters
        # Perform conversion based on your camera's calibration parameters
        # Replace the following line with your actual conversion formula
        # x_real = depth_value * 0.001  # Convert mm to meters
        # x_real = 
        
        #Field of view = 39.18 degrees.
        image_length = math.tan(19.59 * (math.pi/180)) * z_data * 2

        #centered_x_pixel = x_pixel - principal_point_x
        #centered_y_pixel = y_pixel - principal_point_y

        x_pixel_percentage = x_pixel / 415
        y_pixel_percentage = y_pixel / 415

        right_x_pixel = 1 - y_pixel_percentage 
        right_y_pixel = x_pixel_percentage

        ###WRONG:x_image_real = (image_length * right_x_pixel) - (image_length / 2)
        ###WRONG:y_image_real = (image_length * right_y_pixel) - (image_length / 2)

        x_image_real = (image_length - (image_length * right_y_pixel)) - (image_length/2)
        y_image_real = (image_length - (image_length * right_x_pixel)) - (image_length/2)


        x_real = x_image_real + x_data
        y_real = y_image_real + y_data
   
        #return x_real, y_real, z_real
        # rospy.loginfo("TESTING STUFF")
        # rospy.loginfo(x_data)
        # rospy.loginfo(y_data)
        # rospy.loginfo("     ")
        # rospy.loginfo(x_real)
        # rospy.loginfo(y_real)
        # rospy.loginfo("     ")
        # rospy.loginfo(x_pixel)
        # rospy.loginfo(y_pixel)
        # rospy.loginfo("     ")
        # rospy.loginfo(x_pixel_percentage)
        # rospy.loginfo(y_pixel_percentage)
        # rospy.loginfo("     ")
        # rospy.loginfo(image_length)
        # rospy.loginfo("     ")
        # rospy.loginfo(x_image_real)


       

        rospy.loginfo("Pixel Percentages")
        rospy.loginfo(right_x_pixel)
        rospy.loginfo(right_y_pixel)
        rospy.loginfo(" ")

        

        #rospy.loginfo("UAV Position")
        #rospy.loginfo(x_data)
        #rospy.loginfo(y_data)
        #rospy.loginfo("Transformed Coordinate")
        #rospy.loginfo(x_data_)



        rospy.loginfo(y_image_real)
        return x_real, y_real


    #TESTING: Send coordinate to ROI.
    def send_coordinate_roi(self, x, y, z):            
        rospy.loginfo("Aruco sending coordinate to ROI...")
        #pose = PoseStamped()
        #pose.pose.position.x = x
        #pose.pose.position.y = y
        #pose.pose.position.z = z
        #pose.pose.orientation.w = 1.0
        #pose.pose.orientation.x = 0.0
        #pose.pose.orientation.y = 0.0
        #pose.pose.orientation.z = 0.0
        #self.id_coordinate_publisher.publish(pose)      
        #rospy.loginfo("Aruco cooridinate sent to ROI.")
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        self.id_coordinate_publisher.publish(point)
        rospy.loginfo("Aruco coordinate sent to ROI.")    













    #################TESTING
    def send_marker_aruco(self, x, y):
        rospy.loginfo("SENDING ARUCO...")
        pub = rospy.Publisher("aruco_tile_marker", Marker, queue_size=10, latch=True)


        

        # Set up the message header
        msg_out = Marker()
        msg_out.header.frame_id = "map"
        msg_out.header.stamp = rospy.Time.now()

        # Namespace allows you to manage
        # adding and modifying multiple markers
        msg_out.ns = "my_aruco_marker"
        # ID is the ID of this specific marker
        msg_out.id = 0
        # Type can be most primitive shapes
        # and some custom ones (see rviz guide
        # for more information)
        msg_out.type = Marker.CUBE
        msg_out.color.r = 255
        msg_out.color.g = 0
        msg_out.color.b = 0
        msg_out.color.a = 1
        self.bag_detected = True
        # Action is to add / create a new marker
        msg_out.action = Marker.ADD
        # Lifetime set to Time(0) will make it
        # last forever
        msg_out.lifetime = rospy.Time(0)
        # Frame Locked will ensure our marker
        # moves around relative to the frame_id
        # if this is applicable
        msg_out.frame_locked = True

        # Place the marker at [1.0,1.0,0.0]
        # with no rotation
        msg_out.pose.position.x = x
        msg_out.pose.position.y = y
        msg_out.pose.position.z = 0.0
        msg_out.pose.orientation.w = 1.0
        msg_out.pose.orientation.x = 0.0
        msg_out.pose.orientation.y = 0.0
        msg_out.pose.orientation.z = 0.0

        # Make a square tile marker with
        # size 0.1x0.1m square and 0.02m high
        msg_out.scale.x = 0.1
        msg_out.scale.y = 0.1
        msg_out.scale.z = 0.02


        # Publish the marker
        pub.publish(msg_out)
        rospy.loginfo("ARUCO MARKER SENT SUCCESSFULLY.")



def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()
