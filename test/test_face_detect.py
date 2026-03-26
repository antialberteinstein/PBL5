# ==============================================================================
#                           SECTION: FACE DETECTION TEST
# ==============================================================================
"""
Functional test script for the Face Recognition module specifically.
Opens a camera feed and runs the InsightFace detector, drawing bounding
boxes and landmarks on the screen.

[UTILITIES USED FROM src/]:
- recog.face_recognition.InsightFaceDetector: Core detection component
- camera.opencv_client.OpenCVCamera: Camera abstraction for uniform capture
- ui.colors: UI color constants to avoid hardcoding RGB values
- utils.pose_utils.get_pose_name: Reused pose classification logic
"""

import os
import sys
import logging
import warnings
import cv2
import numpy as np
import time

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recog.face_recognition import InsightFaceDetector
from camera.opencv_client import OpenCVCamera
import ui.colors as colors
from utils.pose_utils import get_pose_name

# ==============================================================================
#                           SECTION: CONFIGURATION
# ==============================================================================
CAMERA_ID = 0
WINDOW_NAME = "Face Detection Test"
DET_THRESHOLD = 0.5
DRAW_BBOX = True
DRAW_LANDMARKS = True
DRAW_POSE = True

OUTPUT_DIR = "logs/test_face_detect"
PADDING_RATIO = 0.4

COLOR_BBOX = colors.GREEN
COLOR_TEXT = colors.CYAN
COLOR_LANDMARK = colors.GREEN
COLOR_LANDMARK_TEXT = colors.RED

# ==============================================================================
#                                   SECTION: MAIN
# ==============================================================================

def main():
    logging.basicConfig(level=logging.INFO)
    
    logging.info(
        "\n======================================================\n"
        "[UTILITIES USED FROM src/]:\n"
        "- recog.face_recognition.InsightFaceDetector\n"
        "- camera.opencv_client.OpenCVCamera\n"
        "- ui.colors\n"
        "- utils.pose_utils.get_pose_name\n"
        "======================================================\n"
    )
    
    logging.info("Initializing InsightFaceDetector...")
    detector = InsightFaceDetector(det_threshold=DET_THRESHOLD)
    detector.prepare()
    logging.info("Detector initialized.")
    
    camera = OpenCVCamera()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
        
    logging.info(f"Press 's' to capture cropped face, 'q' to quit.")
    
    try:
        while True:
            frame = camera.capture_frame()
            if frame is None:
                logging.error("Failed to read frame from camera.")
                continue
                
            display_frame = frame.copy()
            
            # Detect faces
            detections = detector.detect(frame)
            
            for face in detections:
                box = face.bbox.astype(int)
                
                # Draw bounding box
                if DRAW_BBOX:
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), COLOR_BBOX, 2)
                    text = f"Conf: {face.confidence:.2f}"
                    cv2.putText(display_frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
                
                # Draw pose
                if DRAW_POSE and face.pose is not None:
                    # pose is roughly (pitch, yaw, roll)
                    pitch, yaw, roll = face.pose
                    pose_label = get_pose_name(face.pose)
                    pose_text = f"{pose_label} (P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f})"
                    cv2.putText(display_frame, pose_text, (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
                
                # Draw landmarks
                if DRAW_LANDMARKS and face.landmarks is not None:
                    for i, lmk in enumerate(face.landmarks.astype(int)):
                        cv2.circle(display_frame, tuple(lmk), 2, COLOR_LANDMARK, -1)
                        cv2.putText(display_frame, str(i), tuple(lmk), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_LANDMARK_TEXT, 1)
                        
            cv2.imshow(WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if detections:
                    ts = int(time.time())
                    for idx, face in enumerate(detections):
                        box = face.bbox.astype(int)
                        x1, y1, x2, y2 = box
                        w_box = x2 - x1
                        h_box = y2 - y1
                        
                        padding_w = int(w_box * PADDING_RATIO)
                        padding_h = int(h_box * PADDING_RATIO)
                        
                        h, w = display_frame.shape[:2]
                        x1 = max(0, x1 - padding_w)
                        y1 = max(0, y1 - padding_h)
                        x2 = min(w, x2 + padding_w)
                        y2 = min(h, y2 + padding_h)
                        
                        face_crop = display_frame[y1:y2, x1:x2]
                        if len(detections) > 1:
                            crop_path = os.path.join(OUTPUT_DIR, f"face_crop_{ts}_{idx}.jpg")
                        else:
                            crop_path = os.path.join(OUTPUT_DIR, f"face_crop_{ts}.jpg")
                        
                        cv2.imwrite(crop_path, face_crop)
                        logging.info(f"Saved: {crop_path}")
                else:
                    logging.warning("No face detected to capture.")
                
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logging.info("Resources released.")

if __name__ == "__main__":
    main()
