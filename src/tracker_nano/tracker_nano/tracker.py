#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose



TOPIC_CAMERA = "video"
TOPIC_TRACK_REQUEST = "track_request"
TOPIC_TRACK_RESULT = "track_result"

class Tracker(Node):
    def __init__(self):
        super().__init__('image_viewer')
        
       
        #TODO: create on init or for each tracking request
        self.tracker = self.create_tracker()
        self.tracking_active = False
        self.tracking_request = False
        self.detection = None
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        self.init_publishers()
        self.init_subscribers()
        self.get_logger().info('Image viewer node started')

    #region private
    def init_publishers(self):
        self.track_pub = self.create_publisher(
            Detection2D,
            TOPIC_TRACK_RESULT,
            10)
        
    def init_subscribers(self):
         # Create subscriber
        self.img_sub = self.create_subscription(
            Image,
            TOPIC_CAMERA,  # Topic name to subscribe to
            self.image_callback,
            10)
        
        self.track_sub = self.create_subscription(
            Detection2D,
            TOPIC_TRACK_REQUEST,
            self.track_callback,
            10)
        
    def create_tracker(self):
        # Create tracker object
        params = cv2.TrackerNano.Params()
        # v2
        params.backbone = "/workspace/src/tracker_nano/tracker_nano/models/nanotrack_backbone_sim.onnx"
        params.neckhead = "/workspace/src/tracker_nano/tracker_nano/models/nanotrack_head_sim.onnx"

        
        params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        params.target  = cv2.dnn.DNN_TARGET_CPU

        tracker = cv2.TrackerNano.create(params)  # CSRT tracker offers good accuracy
        return tracker
    #endregion

    #region handlers
    def track_callback(self, msg: Detection2D):
        """Callback for receiving tracking requests"""
        if not self.tracking_active:
            self.detection = msg
            self.tracking_active = True
            self.tracking_request = True
            self.get_logger().info('Received tracking request')
        else:
            self.detection = None
            self.tracking_active = False
            self.get_logger().info('Received stop tracking request')

    def image_callback(self, msg: Image):
        if not self.tracking_active:
            return
        
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        if self.tracking_request:
            # Initialize tracker with first bbox
            bbox = (
                int(self.detection.bbox.center.position.x - self.detection.bbox.size_x/2),
                int(self.detection.bbox.center.position.y - self.detection.bbox.size_y/2),
                int(self.detection.bbox.size_x),
                int(self.detection.bbox.size_y)
            )
            
            self.get_logger().info(f"{cv_image.shape}")
            self.get_logger().info(f"{bbox}")

            try:
                self.tracker.init(cv_image, bbox)
            except:
                self.get_logger().error('Failed to initialize tracker')
                self.tracking_active = False
                return
            self.tracking_request = False
                    

        success, bbox = self.tracker.update(cv_image)
        self.get_logger().info("---")
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("debug", cv_image)
            cv2.waitKey(1)
            # Publish tracking result
            result = Detection2D()
            result.header = msg.header  # Use image header
            result.bbox.center.position.x = x + w/2
            result.bbox.center.position.y = y + h/2
            result.bbox.size_x = float(w)
            result.bbox.size_y = float(h)

            # Add tracking score
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "tracked_object"
            hypothesis.hypothesis.score = float(self.tracker.getTrackingScore())  # Get score from NanoTracker
            result.results.append(hypothesis)
            
            self.track_pub.publish(result)
        else:
            self.get_logger().warn('Lost tracking target')
            self.tracker = None



    #endregion
    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = Tracker()
    
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