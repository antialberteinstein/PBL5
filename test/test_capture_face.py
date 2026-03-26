"""
Test script to capture a face with optional bounding box padding.

[UTILITIES USED FROM src/]:
- recog.face_recognition.InsightFaceDetector: Core detection component
- camera.opencv_client.OpenCVCamera: Camera abstraction for uniform capture
- ui.colors: UI color constants to avoid hardcoding RGB values
"""

import os
import sys
import time
import cv2
import logging
import warnings
import argparse

# Suppress warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recog.face_recognition import InsightFaceDetector
from camera.opencv_client import OpenCVCamera
import ui.colors as colors

# ==============================================================================
#                           SECTION: CONFIGURATION
# ==============================================================================

OUTPUT_DIR = "logs/test_capture_face"
WINDOW_NAME = "Test Face Capture"

# UI Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_INFO = colors.CYAN
COLOR_SUCCESS = colors.GREEN
COLOR_ERROR = colors.RED
COLOR_TEXT = colors.WHITE

TEXT_POS_MAIN = (50, 50)
TEXT_POS_SUB = (50, 100)


def setup_test_directories(output_dir):
    """Ensure output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")


def test_capture_face(padding_ratio: float):
    """
    Function-based test to capture a face, crop it with optional padding, and save.
    
    Args:
        padding_ratio: Ratio of bounding box size to add as padding (0.0 = no padding).
    """
    # 1. Setup
    logging.basicConfig(level=logging.INFO)
    
    logging.info(
        "\n======================================================\n"
        "[UTILITIES USED FROM src/]:\n"
        "- recog.face_recognition.InsightFaceDetector\n"
        "- camera.opencv_client.OpenCVCamera\n"
        "- ui.colors\n"
        "======================================================\n"
    )
    
    setup_test_directories(OUTPUT_DIR)
    
    detector = InsightFaceDetector()
    detector.prepare()
    
    camera = OpenCVCamera()
    
    logging.info(f"Starting test with padding: {padding_ratio}")
    logging.info("Press 'S' to capture, 'Q' to quit.")
    
    started = False
    success = False
    
    while True:
        frame = camera.capture_frame()
        if frame is None:
            continue
            
        display_frame = frame.copy()
        
        if not started:
            cv2.putText(display_frame, "Press 'S' to start capture", TEXT_POS_MAIN, 
                        FONT, 1, COLOR_INFO, 2)
        elif not success:
            # Triggered capture
            detections = detector.detect(frame)
            if detections:
                face = detections[0]
                box = face.bbox.astype(int)
                
                ts = int(time.time())
                
                # Crop with padding
                x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                
                # Apply padding based on ratio
                if padding_ratio > 0:
                    padding_w = int(w_box * padding_ratio)
                    padding_h = int(h_box * padding_ratio)
                    
                    h, w = frame.shape[:2]
                    x1 = max(0, x1 - padding_w)
                    y1 = max(0, y1 - padding_h)
                    x2 = min(w, x2 + padding_w)
                    y2 = min(h, y2 + padding_h)
                
                face_crop = frame[y1:y2, x1:x2]
                crop_path = os.path.join(OUTPUT_DIR, f"face_crop_{ts}.jpg")
                cv2.imwrite(crop_path, face_crop)
                
                success = True
                logging.info(f"Saved: {crop_path} (padding ratio: {padding_ratio})")
            else:
                cv2.putText(display_frame, "No face detected, try again...", TEXT_POS_MAIN, 
                            FONT, 1, COLOR_ERROR, 2)
                started = False # Reset to allow retry
        
        if success:
            cv2.putText(display_frame, "SUCCESS! Face captured.", TEXT_POS_MAIN, 
                        FONT, 1, COLOR_SUCCESS, 2)
            cv2.putText(display_frame, "Press 'Q' to exit", TEXT_POS_SUB, 
                        FONT, 0.8, COLOR_TEXT, 2)

        cv2.imshow(WINDOW_NAME, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not success:
            started = True
        elif key == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face capture with padding.")
    parser.add_argument("--padding", type=float, default=0.2, 
                        help="Padding ratio (e.g. 0.2 for 20%%, 0.0 for no padding)")
    args = parser.parse_args()
    
    test_capture_face(args.padding)
