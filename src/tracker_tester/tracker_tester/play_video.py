#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

TOPIC_CAMERA = "video"

class VideoPlayer(Node):
    def __init__(self):
        node_name="video_player"
        super().__init__(node_name)
        self.get_logger().info("Hello ROS2")
        self.init_node_parameters()

         # Get parameters
        self.video_path = self.get_parameter('video_path').value
        publish_rate = self.get_parameter('publish_rate').value
        
        # Create publisher
        self.publisher = self.create_publisher(Image, TOPIC_CAMERA, 10)
        
        # Create timer for publishing frames
        self.timer = self.create_timer(1.0/publish_rate, self.timer_callback)
        
        # Initialize video capture and CV bridge
        self.cap = cv2.VideoCapture(self.video_path)
        self.cv_bridge = CvBridge()
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video file: {self.video_path}")
            return
            
        self.get_logger().info(f"Started video player with file: {self.video_path}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().info("End of video file reached, restarting...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
            
        # Convert the image to ROS format
        ros_image = self.cv_bridge.cv2_to_imgmsg(frame, "bgr8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = "camera_frame"
        
        # Publish the image
        self.publisher.publish(ros_image)

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

    def init_node_parameters(self):
         # Declare parameters
        self.declare_parameter(
            'video_path',
            '/workspace/src/tracker_tester/data/vtest.avi',
            ParameterDescriptor(description='Path to the video file to play')
        )
        self.declare_parameter(
            'publish_rate',
            20.0,
            ParameterDescriptor(description='Rate at which to publish frames (Hz)')
        )

def main(args=None):
    rclpy.init(args=args)
    node = VideoPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()