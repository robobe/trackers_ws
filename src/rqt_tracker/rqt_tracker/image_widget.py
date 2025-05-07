from python_qt_binding.QtWidgets import QLabel
from python_qt_binding.QtCore import Qt, Signal
from python_qt_binding.QtGui import QImage, QPixmap
from cv_bridge import CvBridge
import cv2
from vision_msgs.msg import Detection2D, Pose2D
from std_msgs.msg import Header

class TrackingImageWidget(QLabel):
    def __init__(self, detection_pub, node):
        super(TrackingImageWidget, self).__init__()
        self.detection_pub = detection_pub
        self.node = node
        self.bridge = CvBridge()
        self.last_image = None
        self.start_point = None
        self.box_size = 100  # Default box size
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
    def image_callback(self, msg):
        try:
            self.node.get_logger().info("ddddd")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_image = msg
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(q_image))
        except Exception as e:
            self.node.get_logger().error(f'Error converting image: {str(e)}')
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.last_image is not None:
            # Get click position
            x = event.x()
            y = event.y()
            
            # Create Detection2D message
            detection = Detection2D()
            detection.header = self.last_image.header
            
            # Set bounding box
            detection.bbox.center.position.x = float(x)
            detection.bbox.center.position.y = float(y)
            detection.bbox.size_x = float(self.box_size)
            detection.bbox.size_y = float(self.box_size)
            
            # Publish detection
            self.detection_pub.publish(detection)
            self.node.get_logger().info(f'Published detection at ({x}, {y})')