#!/usr/bin/env python3


import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge
import cv2
from functools import partial

WINDOW_NAME = "tracker"

class ImageClickDetector(Node):

    def __init__(self):
        super().__init__('image_click_detector')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/video', self.image_callback, 10)
        self.tracker_sub = self.create_subscription(Detection2D, '/track_result', self.tracker_callback, 10)
        self.request_tracking_pub = self.create_publisher(Detection2D, '/track_request', 10)


        self.last_image = None
        self.tracker_result = None
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse_click)
        cv2.createTrackbar("Width", WINDOW_NAME, 50, 100, partial(self.change_gate_size, "W"))
        cv2.createTrackbar("Height", WINDOW_NAME, 50, 100, partial(self.change_gate_size, "H"))
        self.timer = self.create_timer(0.05, self.display_image)

    def change_gate_size(self, t, x):
        self.get_logger().info(f"{t}: {x}")

    def tracker_callback(self, msg: Detection2D):
        self.tracker_result = msg

    def image_callback(self, msg):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def display_image(self):
        if self.tracker_result is not None:
            bbox = (
                int(self.tracker_result.bbox.center.position.x - self.tracker_result.bbox.size_x/2),
                int(self.tracker_result.bbox.center.position.y - self.tracker_result.bbox.size_y/2),
                int(self.tracker_result.bbox.size_x),
                int(self.tracker_result.bbox.size_y)
            )
            score = self.tracker_result.results[0].hypothesis.score
            score = int(score*100)
            
            p1 = (int(bbox[0]), int(bbox[1]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_location = (50, 50)
            font_scale = 0.4  # Approximate scale for 10px font size
            font_color = (0, 255, 0)  # Green color
            thickness = 1
            cv2.putText(self.last_image, str(score), text_location, font, font_scale, font_color, thickness, cv2.LINE_AA)
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(self.last_image, p1, p2, (255,0,0), 2, 1)

        if self.last_image is not None:
            cv2.imshow(WINDOW_NAME, self.last_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit on pressing 'q'
                self.get_logger().info("Exiting...")
                rclpy.shutdown()
                cv2.destroyAllWindows()
            elif key == ord('s'):  # Example: Save image on pressing 's'
                if self.last_image is not None:
                    cv2.imwrite('saved_image.jpg', self.last_image)
                    self.get_logger().info("Image saved as 'saved_image.jpg'")
            elif key == 82:  # Up arrow key
                self.get_logger().info("Up arrow key pressed")
            elif key == 81:  # Left arrow key
                self.get_logger().info("Left arrow key pressed")
            elif key == 84:  # Down arrow key
                self.get_logger().info("Down arrow key pressed")
            elif key == 83:  # Right arrow key
                self.get_logger().info("Right arrow key pressed")

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info(f"Clicked at: {x}, {y}")
            self.publish_detection(x, y)

    def publish_detection(self, x, y):
        msg = Detection2D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"

        # Bounding box centered at click
        
        msg.bbox.center.position.x = x + 100/2
        msg.bbox.center.position.y = y + 100/2
        msg.bbox.size_x = float(100)
        msg.bbox.size_y = float(100)



        self.request_tracking_pub.publish(msg)
        self.get_logger().info("Published Detection2D")

def main(args=None):
    rclpy.init(args=args)
    node = ImageClickDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
