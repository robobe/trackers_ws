#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, Detection2DArray
import pathlib

from image_cache import ThreadSafeFixedCache

NANO_TRACKER_ID = "1"
NANO_TRACKER_NAME = "nano"

TOPIC_CAMERA = "video"
TOPIC_TRACK_REQUEST = "track_request"
TOPIC_TRACK_RESULT = "track_result"

PARAM_NANOTRACK_BACKBONE_PATH = "nanotrack_backbone_path"
PARAM_NANOTRACK_HEAD_PATH = "nanotrack_head_path"


NODE_NAME = "nano_tracker_node"

class Tracker(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.cache = ThreadSafeFixedCache(capacity=1000)
        self.get_logger().info(f'{self.get_name()} started')
        self.init_parameters()
        self.tracker = self.create_tracker()
        self.tracking_active = False
        self.tracking_first_time_request = False
        self.tracking_request_msg = None
        self.last_bbox_width = None
        self.last_bbox_height = None
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        self.init_publishers()
        self.init_subscribers()
        self.get_logger().info(f'{self.get_name()} started')

    #region private
    def init_parameters(self):
        """
        init node parameters
        """
        self.declare_parameter(PARAM_NANOTRACK_BACKBONE_PATH, "/workspace/src/tracker_nano/tracker_nano/models/nanotrack_backbone_sim.onnx")
        self.declare_parameter(PARAM_NANOTRACK_HEAD_PATH, "/workspace/src/tracker_nano/tracker_nano/models/nanotrack_head_sim.onnx")

    def init_publishers(self):
        """
        init publishers
        - tracker result
        """
        self.track_pub = self.create_publisher(
            Detection2D,
            TOPIC_TRACK_RESULT,
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
        
        self.track_sub = self.create_subscription(
            Detection2D,
            TOPIC_TRACK_REQUEST,
            self.track_callback,
            qos_profile=qos_profile_system_default)
        
    def create_tracker(self):
        # Create tracker object
        params = cv2.TrackerNano.Params()
        # v2
        params.backbone = self.get_parameter(PARAM_NANOTRACK_BACKBONE_PATH).value
        params.neckhead = self.get_parameter(PARAM_NANOTRACK_HEAD_PATH).value #"/workspace/src/tracker_nano/tracker_nano/models/nanotrack_head_sim.onnx"


        if not pathlib.Path((params.backbone)).exists() or not pathlib.Path((params.neckhead)).exists():
            self.get_logger().error("Tracker models not found, check parameters or param file")
            raise Exception("Failed to init tracker")
        
        self.get_logger().info(f"------- Create tracker with models: {params.neckhead}\n{params.neckhead}")
        params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        params.target  = cv2.dnn.DNN_TARGET_CPU

        tracker = cv2.TrackerNano.create(params)  # CSRT tracker offers good accuracy
        return tracker
    #endregion

    #region handlers
    def track_callback(self, msg: Detection2D):
        if msg.bbox.center.position.x == msg.bbox.center.position.y == -1.0:
            self.tracking_request_msg = None
            self.tracking_active = False
            self.get_logger().info('Received stop tracking request')
        
        else:
            self.tracking_request_msg = msg
            self.tracking_active = True
            self.tracking_first_time_request = True
            self.get_logger().info('Received tracking request')
        
            

    def image_callback(self, img_msg: Image):
        """
        handler images message
        if tracker work , update result
        """
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        # put image in cache with timestamp as key
        key = Time.from_msg(img_msg.header.stamp).nanoseconds
        self.cache.put(key, cv_image)
        
        if not self.tracking_active:
            return
        
        if self.tracking_first_time_request:
            # get tracker request timestamp
            self.last_bbox_width = None
            self.last_bbox_height = None
            key = Time.from_msg(self.tracking_request_msg.header.stamp).nanoseconds
            # get history image from cache
            #TODO: handler if not found
            image_from_cache = self.cache.get(key)
            
            bbox = (
                int(self.tracking_request_msg.bbox.center.position.x - self.tracking_request_msg.bbox.size_x/2),
                int(self.tracking_request_msg.bbox.center.position.y - self.tracking_request_msg.bbox.size_y/2),
                int(self.tracking_request_msg.bbox.size_x),
                int(self.tracking_request_msg.bbox.size_y)
            )
            

            try:
                # init tracker with image from cache and request bbox
                self.tracker.init(image_from_cache, bbox)
            except:
                self.get_logger().error('Failed to initialize tracker')
                self.tracking_active = False
            finally:
                self.tracking_first_time_request = False
                    
            if not self.tracking_active:
                self.get_logger().error("Tracker fail to initialize exit tracking")
                return
            # iterate over cache to fast forward to current time
            # skip  last item and the first found item
            for k, image in self.cache.iterate_from_key(key, skip_first=True, skip_last=True):
                success, bbox = self.tracker.update(image)

        success, bbox = self.tracker.update(cv_image)
        # TODO: when tracker return success False ?
        if success:
            w_change = h_change = False
            x, y, w, h = [int(v) for v in bbox]
            if self.last_bbox_width is not None:
                w_change =  abs(self.last_bbox_width-w) > 5

            if self.last_bbox_height is not None:
                h_change =  abs(self.last_bbox_height-h) > 5

            if h_change or w_change:
                self.get_logger().warning("tracker gate size change, request auto tracking")
                w = self.last_bbox_width
                h = self.last_bbox_height
                self.tracking_first_time_request = True
                
                self.tracking_request_msg = Detection2D()
                self.tracking_request_msg.header.stamp = img_msg.header.stamp
                self.tracking_request_msg.bbox.center.position.x = x + w/2
                self.tracking_request_msg.bbox.center.position.y = y + h/2
                self.tracking_request_msg.bbox.size_x = float(w)
                self.tracking_request_msg.bbox.size_y = float(h)
            self.last_bbox_width = w
            self.last_bbox_height = h
            result = Detection2D()
            result.header = img_msg.header  # Use image header
            result.header.stamp = img_msg.header.stamp
            result.bbox.center.position.x = x + w/2
            result.bbox.center.position.y = y + h/2
            result.bbox.size_x = float(w)
            result.bbox.size_y = float(h)

            result.id = NANO_TRACKER_ID
            # Add tracking score
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = NANO_TRACKER_NAME
            hypothesis.hypothesis.score = float(self.tracker.getTrackingScore())  # Get score from NanoTracker
            result.results.append(hypothesis)
            
            self.track_pub.publish(result)

            

        else:
            self.get_logger().warn('Lost tracking target')
            self.tracking_active = False



    #endregion


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