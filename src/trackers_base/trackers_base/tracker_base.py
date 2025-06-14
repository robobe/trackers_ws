#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_system_default
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image
from trackers_base.image_cache import ThreadSafeFixedCache

TOPIC_CAMERA = "video"
TOPIC_TRACK_REQUEST = "track_request"
TOPIC_TRACK_RESULT = "track_result"

class TrackerBase(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self._init_parameters()
        self.cache = ThreadSafeFixedCache(capacity=1000)
        self.tracker = self._create_tracker()
        self.tracking_active = False
        self.tracking_first_time_request = False
        self.tracking_request_msg = None
        self.last_bbox_width = None
        self.last_bbox_height = None
        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        self._init_publishers()
        self._init_subscribers()

    def _init_parameters(self):
        """
        Initialize node parameters.
        This method should be overridden in subclasses to declare specific parameters.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_publishers(self):
        """
        Initialize publishers.
        This method should be overridden in subclasses to create specific publishers.
        """
        self.track_pub = self.create_publisher(
            Detection2D,
            TOPIC_TRACK_RESULT,
            qos_profile=qos_profile_system_default)

    def _init_subscribers(self):
        """
        init subscribers
        - images
        - track request
        """
        self.img_sub = self.create_subscription(
            Image,
            TOPIC_CAMERA,  # Topic name to subscribe to
            self._image_callback,
            qos_profile=qos_profile_system_default)
        
        self.track_sub = self.create_subscription(
            Detection2D,
            TOPIC_TRACK_REQUEST,
            self.track_callback,
            qos_profile=qos_profile_system_default)
    
    def _create_tracker(self):
        """
        Create and return a tracker instance.
        This method should be overridden in subclasses to create specific tracker instances.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _image_callback(self, img_msg: Image):
        """
        # TODO: split to tracker first time request and update
        """
        raise NotImplementedError("Subclasses must implement this method.")