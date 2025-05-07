import os
import rclpy
import cv2
from cv_bridge import CvBridge
from rqt_gui_py.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from python_qt_binding.QtCore import Qt
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Pose2D
from .image_widget import TrackingImageWidget
from ament_index_python import get_resource
from std_msgs.msg import String
from python_qt_binding.QtCore import QTimer

PKG = "rqt_tracker"

class TrackingPlugin(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('TrackingPlugin')
        
        # Create QWidget
        self._widget = QWidget()
        
        # Load UI file
        _, package_path = get_resource('packages', PKG)
        ui_file = os.path.join(package_path, 'share', PKG, 'resource', 'Tracker.ui')
        loadUi(ui_file, self._widget)
        context.add_widget(self._widget)
        
        self.node = rclpy.create_node('tracking_plugin')
        
        # # Create publisher for Detection2D messages
        # self.detection_pub = self.node.create_publisher(
        #     Detection2D,
        #     'track_request',
        #     10
        # )
        
        # # Create custom image widget
        # self.image_widget = TrackingImageWidget(self.detection_pub, self.node)
        # self._widget.image_layout.addWidget(self.image_widget)
        
        # # Subscribe to image topic
        # self.image_sub = self.node.create_subscription(
        #     Image,
        #     'video',
        #     self.image_widget.image_callback,
        #     10
        # )
        self.sub = self.node.create_subscription(String, '/chatter', self.callback, 10)
        self._spin_timer = QTimer(self._widget)
        self._spin_timer.timeout.connect(lambda: rclpy.spin_once(self.node, timeout_sec=0))
        self._spin_timer.start(33)  # every 100 ms
        # Add widget to the user interface
        
        self.node.get_logger().info(f"=======start")

    def callback(self, msg):
        self.node.get_logger().info(f"======= {msg.data}")