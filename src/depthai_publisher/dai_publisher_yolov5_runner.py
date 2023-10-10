#!/usr/bin/env python3

'''
Run as:
# check model path line ~30is
# rosrun depthai_publisher dai_publisher_yolov5_runner
'''
############################### ############################### Libraries ###############################
from pathlib import Path
import threading
import csv
import argparse
import time
import sys
import json     # Yolo conf use json files
import cv2
import numpy as np
import depthai as dai
import rospy

import math

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String


# Library to send PoseStamped to roi
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker




############################### ############################### Parameters ###############################
# Global variables to deal with pipeline creation
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True
# model path
modelsPath = "/home/uavteam6/egh450/ros_ws/src/depthai_publisher/src/depthai_publisher/models"
# modelName = 'exp31Yolov5_ov21.4_6sh'

#modelName = 'mattModel'
#modelName = 'best_openvino_2022.1_6shave'
# confJson = 'exp31Yolov5.json'
#confJson = 'best.json'
#modelName = 'ryanModel'
#confJson = 'best.json'

modelName = 'jamesModel'
confJson = 'epoch24.json'


################################  Yolo Config File
# parse config
configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# Extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
# Parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

class DepthaiCamera():
    # res = [416, 416]
    fps = 20.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'

   

    def __init__(self):
        self.pipeline = dai.Pipeline()

         # Input image size
        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

         #Detection stuff.
        self.bag_detected = False
        self.person_detected = False

        # Pulbish ros image data
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
        # Create a publisher for the CameraInfo topic
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=10)
        # Create a timer for the callback
        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

        self.publish_label = rospy.Publisher('detection_of_object', String, queue_size=10)

        #JAMES CODE ADDED:
        #Create the target location publisher.
        self.publish_roi_location = rospy.Publisher('target_detection/roi', PoseStamped, queue_size=10)
        #Create the current position subsriber.
        #The current UAV position is a Point variable [float64 x, float64 y, float64 z].
        self.current_uav_position = None
        self.sub_uav_position = rospy.Subscriber("current_position", Point, self.callback_uav_position)
    

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

        self.br = CvBridge()

        rospy.on_shutdown(lambda: self.shutdown())

    def publish_camera_info(self, timer=None):
        # Create a publisher for the CameraInfo topic

        # Create a CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h # Set the height of the camera image
        camera_info_msg.width = self.nn_shape_w  # Set the width of the camera image

        # Set the camera intrinsic matrix (fx, fy, cx, cy)
        camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
        # Set the distortion parameters (k1, k2, p1, p2, k3)
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
        # Set the rectification matrix (identity matrix)
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Set the projection matrix (P)
        camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        # Set the distortion model
        camera_info_msg.distortion_model = "plumb_bob"
        # Set the timestamp
        camera_info_msg.header.stamp = rospy.Time.now()

        self.pub_cam_inf.publish(camera_info_msg)  # Publish the camera info message

    def rgb_camera(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(self.res[0], self.res[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self.fps)

        # Def xout / xin
        ctrl_in = self.pipeline.createXLinkIn()
        ctrl_in.setStreamName("cam_ctrl")
        ctrl_in.out.link(cam_rgb.inputControl)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("video")

        cam_rgb.preview.link(xout_rgb.input)

    def run(self):
        #self.rgb_camera()
        ############################### Run Model ###############################
        # Pipeline defined, now the device is assigned and pipeline is started
        pipeline = None
        # Get argument first
        # Model parameters
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        print(metadata)
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        print(nnPath)

        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if cam_source != "rgb" and not depth_enabled:
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline)

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            start_time = time.time()
            counter = 0
            fps = 0
            
            olor2 = (255, 255, 255)
            layer_info_printed = False
            dims = None

            while True:
                found_classes = []
                # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
                inRgb = q_nn_input.get()
                inDet = q_nn.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                else:
                    print("Cam Image empty, trying again...")
                    continue
                
                if inDet is not None:
                    detections = inDet.detections
                    # print(detections)
                    for detection in detections:
                        # print(detection)
                        # print("{},{},{},{},{},{},{}".format(detection.label,labels[detection.label],detection.confidence,detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        found_classes.append(detection.label)
                        # print(dai.ImgDetection.getData(detection))
                    found_classes = np.unique(found_classes)
                    # print(found_classes)
                    overlay = self.show_yolo(frame, detections)
                else:
                    print("Detection empty, trying again...")
                    continue

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    # cv2.imshow("nn_output_yolo", overlay)
                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    self.publish_camera_info()

                ## Function to compute FPS
                counter+=1
                if (time.time() - start_time) > 1 :
                    fps = counter / (time.time() - start_time)

                    counter = 0
                    start_time = time.time()


            # with dai.Device(self.pipeline) as device:
            #     video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

            #     while True:
            #         frame = video.get().getCvFrame()

            #         self.publish_to_ros(frame)
            #         self.publish_camera_info()

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        # Publish image raw
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image_detect.publish(msg_out)
        
    ############################### ############################### Functions ###############################
    ######### Functions for Yolo Decoding
    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        color = (255, 0, 0)
        # Both YoloDetectionNetwork and MobileNetDetectionNetwork output this message. This message contains a list of detections, which contains label, confidence, and the bounding box information (xmin, ymin, xmax, ymax).
        overlay =  frame.copy()
        for detection in detections:
            label = detection.label
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(overlay, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            if self.current_uav_position is not None:

                # Convert pixel coordinates to real-world coordinates (Added by Ryan)
                x_pixel, y_pixel = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Assuming the center of the bounding box and the bottom pixel

                real_world_coordinates = self.convert_to_real_world(x_pixel, y_pixel, self.current_uav_position.x, self.current_uav_position.y, self.current_uav_position.z) # comment out to ignore depth

                self.publish_label.publish(str(label))
                
                #TESTTING: If any target detected, coordinate to ROI.
                if (label == 1 or label == 0) and (self.current_uav_position is not None):


                    rospy.loginfo("MAX STUFF")
                    height, width, _ = frame.shape
                    max_width = width-1
                    max_height = height-1
                    rospy.loginfo(max_width)
                    rospy.loginfo(max_height)
                

                    rospy.loginfo("THIS IS A TEST")
                    ##CHANGE THESE COORDINATES TO BE THE TRANSFORMED COORDINATES.
                    #RIGHT NOW IT JUST SENDS THE CURRENT COORDINATE.
                    #self.send_coordinate_roi(self.current_uav_position.x, self.current_uav_position.y, self.current_uav_position.z, label)
                    # Sends out adjusted coordinates
                    self.send_coordinate_roi(real_world_coordinates[0], real_world_coordinates[1], self.current_uav_position.z, label)
            #rospy.loginfo(type(label))
            rospy.loginfo(label)
            rospy.sleep(2)

        return overlay

    # Start defining a pipeline
    def createPipeline(self, nnPath):

        pipeline = dai.Pipeline()

        # pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)
        # pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        # Network specific settings
        detection_nn.setConfidenceThreshold(confidenceThreshold)
        detection_nn.setNumClasses(classes)
        detection_nn.setCoordinateSize(coordinates)
        detection_nn.setAnchors(anchors)
        detection_nn.setAnchorMasks(anchorMasks)
        detection_nn.setIouThreshold(iouThreshold)
        # generic nn configs
        detection_nn.setBlobPath(nnPath)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        # Define a source - color camera
        if cam_source == 'rgb':
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(self.nn_shape_w,self.nn_shape_h)
            cam.setInterleaved(False)
            cam.preview.link(detection_nn.input)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(10)
            print("Using RGB camera...")
        elif cam_source == 'left':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            print("Using BW Left cam")
        elif cam_source == 'right':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            print("Using BW Rigth cam")

        if cam_source != 'rgb':
            manip = pipeline.create(dai.node.ImageManip)
            manip.setResize(self.nn_shape_w,self.nn_shape_h)
            manip.setKeepAspectRatio(True)
            # manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
            manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
            cam.out.link(manip.inputImage)
            manip.out.link(detection_nn.input)

        # Create outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("nn_input")
        xout_rgb.input.setBlocking(False)

        detection_nn.passthrough.link(xout_rgb.input)

        xinDet = pipeline.create(dai.node.XLinkOut)
        xinDet.setStreamName("nn")
        xinDet.input.setBlocking(False)

        detection_nn.out.link(xinDet.input)

        return pipeline


    def shutdown(self):
        cv2.destroyAllWindows()

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
    def send_coordinate_roi(self, x, y, z, label):
        if((label == 0) and not self.bag_detected):
            self.send_bag_marker_roi(x, y, z, label)
        elif((label == 1) and not self.person_detected):    
            self.send_person_marker_roi(x, y, z, label)
            
        rospy.loginfo("Imaging sending coordinate to ROI...")
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = label
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        self.publish_roi_location.publish(pose)        
        rospy.loginfo("Imaging sent coordinate to ROI.")
    
    def callback_uav_position(self, msg_in):
        self.current_uav_position = msg_in


    def send_bag_marker_roi(self, x, y, z, label):
        rospy.loginfo("SENDING BAG MARKER...")
        pub = rospy.Publisher("bag_tile_marker", Marker, queue_size=10, latch=True)


        if((label == 0) and not self.bag_detected):

            # Set up the message header
            msg_out = Marker()
            msg_out.header.frame_id = "map"
            msg_out.header.stamp = rospy.Time.now()

            # Namespace allows you to manage
            # adding and modifying multiple markers
            msg_out.ns = "my_bag_marker"
            # ID is the ID of this specific marker
            msg_out.id = 0
            # Type can be most primitive shapes
            # and some custom ones (see rviz guide
            # for more information)
            msg_out.type = Marker.CUBE
            msg_out.color.r = 0
            msg_out.color.g = 0
            msg_out.color.b = 255
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
            rospy.loginfo("BAG MARKER SENT SUCCESSFULLY.")
            
    def send_person_marker_roi(self, x, y, z, label):
        rospy.loginfo("SENDING PERSON MARKER...")
        pub = rospy.Publisher("person_tile_marker", Marker, queue_size=10, latch=True)


        if((label == 1) and not self.person_detected):

            # Set up the message header
            msg_out = Marker()
            msg_out.header.frame_id = "map"
            msg_out.header.stamp = rospy.Time.now()

            # Namespace allows you to manage
            # adding and modifying multiple markers
            msg_out.ns = "my_person_marker"
            # ID is the ID of this specific marker
            msg_out.id = 0
            # Type can be most primitive shapes
            # and some custom ones (see rviz guide
            # for more information)
            msg_out.type = Marker.CUBE
            msg_out.color.r = 0
            msg_out.color.g = 255
            msg_out.color.b = 0
            msg_out.color.a = 1
            self.person_detected = True
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
            rospy.loginfo("PERSON MARKER SENT SUCCESSFULLY.")       
            



#### Main code that creates a depthaiCamera class and run it.
def main():
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()

    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()
