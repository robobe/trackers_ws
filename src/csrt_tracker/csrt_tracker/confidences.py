import cv2
import numpy as np


class Fake_Confidence:
    def __init__(self):
        pass

    def init(self, img, bbox):
        """
        Initialize the confidence points (no-op for fake confidence).
        """
        pass

    def get_confidence(self, img, bbox):
        """
        Return a fixed confidence value (no-op for fake confidence).
        """
        return 1.0  # Always return maximum confidence
    

class LK_Confidence:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_points = None
        self.prev_points = None
        self.prev_gray = None
        self.lk_params = dict(winSize=(15, 15), 
                              maxLevel=2) 
                              #criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.good_features_params = dict(maxCorners=50, qualityLevel=0.01, minDistance=5, blockSize=3)

    def _get_keypoints(self, gray, bbox):
        x, y, w, h = map(int, bbox)
        roi = gray[y:y+h, x:x+w]
        points = cv2.goodFeaturesToTrack(roi, **self.good_features_params)
        
        if points is not None:
            # Offset to absolute image coordinates
            points += np.array([[x, y]], dtype=np.float32)
        return points
    
    def init(self, img, bbox):
        self.prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.prev_points = self._get_keypoints(self.prev_gray, bbox)

    def get_confidence(self, img, bbox):
        confidence = 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params)
            
            good_old = self.prev_points[status.flatten() == 1]
            good_new = next_points[status.flatten() == 1]
            confidence = len(good_new) / len(self.prev_points)
            # rclpy.logging.get_logger("csrt_tracker").info(
            #     f"{len(good_new)}/{len(self.prev_points)}")

        self.prev_gray = gray
        self.prev_points = self._get_keypoints(gray, bbox)

        return confidence