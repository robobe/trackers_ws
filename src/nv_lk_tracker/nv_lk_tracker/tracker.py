#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, Detection2DArray
import pathlib
import numpy as np


TOPIC_CAMERA = "video"
TOPIC_CAMERA_RESULT = "video_out"


class MyNode(Node):
    def __init__(self):
        node_name="nv_lk_tracker"
        super().__init__(node_name)
        self.gpu_prev = None
        self.tracker = self.create_tracker()
        self.cv_bridge = CvBridge()
        
        self.init_publishers()
        self.init_subscribers()

        self.get_logger().info("Hello nv_lk_tracker")

    def init_publishers(self):
        """
        init publishers
        - tracker result
        """
        self.img_pub = self.create_publisher(
            Image,
            TOPIC_CAMERA_RESULT,  # Topic name to publish to
            qos_profile=qos_profile_system_default)
        
    def init_subscribers(self):
        """
        init subscribers
        - images
        - track request
        """
        self.img_sub = self.create_subscription(
            Image,
            TOPIC_CAMERA,  # Topic name to subscribe to
            self.image_callback,
            qos_profile=qos_profile_system_default)
        
        # self.track_sub = self.create_subscription(
        #     Detection2D,
        #     TOPIC_TRACK_REQUEST,
        #     self.track_callback,
        #     qos_profile=qos_profile_system_default)
        
    def create_tracker(self):
        tracker = cv2.cuda_NvidiaOpticalFlow_1_0.create(imageSize=(640, 480))

        return tracker

    def image_callback(self, img_msg: Image):
        """
        handler images message
        if tracker work , update result
        """
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        # put image in cache with timestamp as key
        key = Time.from_msg(img_msg.header.stamp).nanoseconds

        if self.gpu_prev is None:
            self.prev_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.gpu_prev = cv2.cuda_GpuMat()
            self.gpu_prev.upload(self.prev_gray)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gpu_curr = cv2.cuda_GpuMat()
        gpu_curr.upload(gray)

        flow_gpu = self.tracker.calc(self.gpu_prev, gpu_curr, None)
        flow = flow_gpu.download()

        # === Visualize motion (magnitude only) ===
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Normalize and display
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_norm = np.uint8(mag_norm)
        heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        self.img_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(heatmap, encoding="bgr8")
        )


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()