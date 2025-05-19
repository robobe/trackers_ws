#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge
import cv2
from functools import partial
from typing import NamedTuple
from threading import Event

# Example usage of NamedTuple for a bounding box
class BBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int

# You can create a bounding box like this:
# bbox = BBox(x=10, y=20, w=100, h=50)

WINDOW_NAME = "tracker"
TRACK_WIDTH_NAME = "Width"
TRACK_HEIGHT_NAME = "Height"

class ImageClickDetector(Node):

    def __init__(self):
        super().__init__('image_click_detector')
        self.event = Event()
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/video', self.image_callback, 10)
        self.tracker_sub = self.create_subscription(Detection2D, '/track_result', self.tracker_callback, 10)
        self.request_tracking_pub = self.create_publisher(Detection2D, '/track_request', 10)
        self.allow_zoom = False
        self.zoom_factor = 2
        self.last_image = None
        self.last_image_stamp = None

        self.working_stamp = None # TODO: use this to publish the image, think about save the image message
        self.original_frame = None
        self.tracker_result = None
        self.freeze_image = None
        self.freeze_image_stamp = None
        self.frame = None
        self.drawing = False
        self.box_ready = False
        self.sx, self.sy = 0, 0
        self.ex, self.ey = 0, 0
        self.box = None

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse_click)
        cv2.createTrackbar(TRACK_WIDTH_NAME, WINDOW_NAME, 50, 100, partial(self.change_gate_size, "W"))
        cv2.createTrackbar(TRACK_HEIGHT_NAME, WINDOW_NAME, 50, 100, partial(self.change_gate_size, "H"))
        # self.timer = self.create_timer(0.05, self.display_image)

    def change_gate_size(self, t, x):
        self.get_logger().info(f"{t}: {x}")

    def tracker_callback(self, msg: Detection2D):
        self.tracker_result = msg
        self.event.set()
        self.event.clear()

    def image_callback(self, msg: Image):
        try:
            self.event.wait(1/20)
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_image_stamp = msg.header.stamp
            self.display_image()
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def display_image(self):
        if self.tracker_result is not None:
            t_key = Time.from_msg(self.tracker_result.header.stamp).nanoseconds
            i_key = Time.from_msg(self.last_image_stamp).nanoseconds
            delta = i_key - t_key
            self.get_logger().info(f"{t_key} -- {i_key} ")
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
            self.tracker_result = None

        
        self.original_frame = self.last_image.copy()
        self.frame = self.last_image.copy()

        if self.freeze_image is not None:
            self.frame = self.freeze_image
            self.working_stamp = self.freeze_image_stamp
        else:
            self.working_stamp = self.last_image_stamp

        if self.allow_zoom:
            
            h, w = self.frame.shape[:2]
            cx, cy = w // 2, h // 2  # zoom center

            # Calculate cropped region
            crop_w = w // self.zoom_factor
            crop_h = h // self.zoom_factor
            x1 = max(cx - crop_w // 2, 0)
            y1 = max(cy - crop_h // 2, 0)
            x2 = min(cx + crop_w // 2, w)
            y2 = min(cy + crop_h // 2, h)

            cropped = self.frame[y1:y2, x1:x2]
            self.frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


        if self.drawing:
            if self.freeze_image is not None:
                if self.allow_zoom:
                    self.frame = self.frame.copy()
                else:
                    self.frame = self.freeze_image.copy()
            cv2.rectangle(self.frame, (self.sx, self.sy), (self.ex, self.ey), (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, self.frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on pressing 'q'
            self.get_logger().info("Exiting...")
            rclpy.shutdown()
            cv2.destroyAllWindows()
        elif key == ord("z"):  # zoom
            self.allow_zoom = not self.allow_zoom
            self.get_logger().info("zoom:------------------")

        elif key == ord("f"):  # freeze
            self.get_logger().info("Freeze")
            if self.freeze_image is None:
                self.freeze_image = self.original_frame.copy()
                self.freeze_image_stamp = self.last_image_stamp
            else:
                self.freeze_image = None
                self.get_logger().info("Unfreeze")

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
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     self.get_logger().info(f"Clicked at: {x}, {y}")
        #     self.publish_detection(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.sx, self.sy = x, y
            self.box_ready = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.ex, self.ey = x, y
                

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.box = (min(self.sx, x), min(self.sy, y), abs(x - self.sx), abs(y - self.sy))
            self.get_logger().info(f"self.box: {self.box}")
            self.box_ready = True
            if self.allow_zoom:
                h, w = self.frame.shape[:2]
                cx, cy = w // 2, h // 2  # zoom center

                # Calculate cropped region
                scale_x = 2 #(x2 - x1) / w
                scale_y = 2 #(y2 - y1) / h
                crop_w = w // scale_x
                crop_h = h // scale_y
                x1 = max(cx - crop_w // 2, 0)
                y1 = max(cy - crop_h // 2, 0)
                
                x = int(x1 + x // scale_x)
                y = int(y1 + y // scale_y)
                w = self.box[2] // scale_x
                h = self.box[3] // scale_y
                self.get_logger().info(f"Zoomed box: {x}, {y}, {w}, {h}")
            else:
                x = min(self.sx, x)
                y = min(self.sy, y)
                w = self.box[2]
                h = self.box[3]
                self.get_logger().info(f"unZoomed box: {x}, {y}, {w}, {h}")

            self.publish_detection(x, y, w, h)
                
                
                
                
            


    def publish_detection(self, x, y ,w ,h):
        msg = Detection2D()
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.stamp = self.working_stamp
        msg.header.frame_id = "camera_frame"

        # Bounding box centered at click
        # width = cv2.getTrackbarPos(TRACK_WIDTH_NAME, WINDOW_NAME)
        # height = cv2.getTrackbarPos(TRACK_HEIGHT_NAME, WINDOW_NAME)

        msg.bbox.center.position.x = float(x)
        msg.bbox.center.position.y = float(y)
        msg.bbox.size_x = float(w)
        msg.bbox.size_y = float(h)
        



        self.request_tracking_pub.publish(msg)
        self.freeze_image = None
        self.allow_zoom = False
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
