# ==============================================================================
#                           SECTION: OPENCV CAMERA CLIENT
# ==============================================================================
"""
OpenCV client for receiving frames from local camera.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from . import config


class OpenCVCamera:
    """
    Camera provider implementation using local OpenCV VideoCapture.
    """
    
    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        
    def _get_capture_device(self) -> cv2.VideoCapture:
        """
        Get or create the VideoCapture instance.
        """
        if self._cap is None or not self._cap.isOpened():
            logging.info(f"Opening local camera index {config.CAMERA_INDEX}...")
            self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
            
            # Suggest a lower resolution to reduce processing lag if possible
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Reduce buffer size to minimum to always grab the latest frame
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self._cap.isOpened():
                logging.error(f"Failed to open camera index {config.CAMERA_INDEX}")
        return self._cap

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from local camera using OpenCV.
        To avoid lag, we grab multiple times to clear the internal buffer.
        """
        cap = self._get_capture_device()
        if not cap.isOpened():
            return None
            
        # Due to openCV's internal buffer, if the processing loop is slower than 
        # the camera FPS, the displayed frames will lag behind real-time. 
        # Reading extra frames clears the buffer. On macOS we might need to grab more.
        for _ in range(5):
            if not cap.grab():
                break
            
        # Actually decode the final newest frame
        ret, frame = cap.retrieve()
        
        if not ret:
            logging.warning("Failed to retrieve frame from camera")
            return None
            
        return frame

    def send_result(self, message: str) -> None:
        """
        Log result message.
        """
        logging.info(f"[CAMERA_RESULT] {message}")
        
    def release(self) -> None:
        """
        Release the camera resource and clear windows.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        # Clean up any lingering windows just in case
        cv2.destroyAllWindows()
        import sys
        if sys.platform == "darwin":
            for _ in range(50):
                cv2.waitKey(1)
