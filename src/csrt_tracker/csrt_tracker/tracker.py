#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from builtin_interfaces.msg import Time as MsgTime

# from image_cache import ThreadSafeFixedCache
MAX_GATE_SIZE_CHANGE_BETWEEN_TRACKING_RESULT = 5
TRACKER_ID = "2"
TRACKER_NAME = "csrt"

TOPIC_CAMERA = "video"
TOPIC_TRACK_REQUEST = "track_request"
TOPIC_TRACK_RESULT = "track_result"
MINIMAL_WIN_SIZE_W = 50.0
MINIMAL_WIN_SIZE_H = 50.0


NODE_NAME = "csrt_tracker_node"

class Tracker(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        # self.cache = ThreadSafeFixedCache(capacity=1000)
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
        """
        create tracker instance
        using CSRT tracker with custom parameters"""
        #TODO: add csrt parameters to node parameters
        params = cv2.TrackerCSRT.Params()
        params.scale_lr = 0.5
        tracker = cv2.TrackerCSRT.create(params)
        return tracker
    #endregion

    #region handlers
    def track_callback(self, msg: Detection2D):
        """
        handler track request message
        for now because using Detection2D msg if bbox center is (-1,-1) then stop tracking
        """
        DISABLE_TRACKING_REQUEST = -1.0
        if msg.bbox.center.position.x == msg.bbox.center.position.y == DISABLE_TRACKING_REQUEST:
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
        # self.cache.put(key, cv_image)
        
        if not self.tracking_active:
            return
        
        if self.tracking_first_time_request:
            # get tracker request timestamp
            self.last_bbox_width = None
            self.last_bbox_height = None
            key = Time.from_msg(self.tracking_request_msg.header.stamp).nanoseconds
            # get history image from cache
            #TODO: handler if not found
            # image_from_cache = self.cache.get(key)
            self.tracking_request_msg.bbox.size_x = max(self.tracking_request_msg.bbox.size_x, MINIMAL_WIN_SIZE_W)
            self.tracking_request_msg.bbox.size_y = max(self.tracking_request_msg.bbox.size_y, MINIMAL_WIN_SIZE_H)

            bbox = (
                int(self.tracking_request_msg.bbox.center.position.x - self.tracking_request_msg.bbox.size_x/2),
                int(self.tracking_request_msg.bbox.center.position.y - self.tracking_request_msg.bbox.size_y/2),
                int(self.tracking_request_msg.bbox.size_x),
                int(self.tracking_request_msg.bbox.size_y)
            )
            

            try:
                # init tracker with image from cache and request bbox
                self.tracker.init(cv_image, bbox)
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
            # for k, image in self.cache.iterate_from_key(key, skip_first=True, skip_last=True):
            #     success, bbox = self.tracker.update(image)

        success, bbox = self.tracker.update(cv_image)
        # TODO: when tracker return success False ?
        if success:
            x, y, w, h = [int(v) for v in bbox]
            keep_last, w, h = self.tracker_gate_size_keeper( w, h)
            if keep_last:
                self.self_tracking_request(img_msg.header.stamp, x, y, w, h)

            self.last_bbox_width = w
            self.last_bbox_height = h
            result = Detection2D()
            result.header = img_msg.header  # Use image header
            result.header.stamp = img_msg.header.stamp
            result.bbox.center.position.x = x + w/2
            result.bbox.center.position.y = y + h/2
            result.bbox.size_x = float(w)
            result.bbox.size_y = float(h)

            result.id = TRACKER_ID
            # Add tracking score
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = TRACKER_NAME
            hypothesis.hypothesis.score = 100.0#float(self.tracker.getTrackingScore())  # Get score from NanoTracker
            result.results.append(hypothesis)
            
            self.track_pub.publish(result)

        else:
            self.get_logger().warn('Lost tracking target')
            self.tracking_active = False

    def tracker_gate_size_keeper(self, w: float, h: float) -> tuple:
        """
        check if tracker gate size changed too much
        if so, keep last size and return request for auto tracking
        """
        keep_last = False
        if self.last_bbox_width is not None:
            
            w_change =  abs(self.last_bbox_width-w) > MAX_GATE_SIZE_CHANGE_BETWEEN_TRACKING_RESULT

        if self.last_bbox_height is not None:
            h_change =  abs(self.last_bbox_height-h) > MAX_GATE_SIZE_CHANGE_BETWEEN_TRACKING_RESULT

        if h_change or w_change:
            self.get_logger().warning("tracker gate size change, request auto tracking")
            w = self.last_bbox_width
            h = self.last_bbox_height
            
            keep_last = True
            
        return keep_last,w,h

    def self_tracking_request(self, stamp: MsgTime, x: float, y: float, w: float, h: float):
        """
        create self tracking request message with given parameters
        """
        self.tracking_request_msg = Detection2D()
        self.tracking_request_msg.header.stamp = stamp
        self.tracking_request_msg.bbox.center.position.x = x + w/2
        self.tracking_request_msg.bbox.center.position.y = y + h/2
        self.tracking_request_msg.bbox.size_x = float(w)
        self.tracking_request_msg.bbox.size_y = float(h)
        self.tracking_first_time_request = True



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
