"""
Functional test script for the Face Recognition module specifically on static images.
Loads an image from a path, runs the InsightFace detector, draws bounding
boxes and landmarks, and optionally displays the result and saves face crops.

[UTILITIES USED FROM src/]:
- recog.face_recognition.InsightFaceDetector: Core detection component
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
import ui.colors as colors
from utils.pose_utils import get_pose_name

# ==============================================================================
#                           SECTION: CONFIGURATION
# ==============================================================================
INPUT_IMAGE_PATH = "logs/test_mask/face_crop_1773242068_masked.jpg" # Replace with your image path

# What to show/do
SHOW_IMAGE = True
SAVE_CROPPED_FACE = True

# Detection thresholds
DET_THRESHOLD = 0.5

# Directories and Padding for Face Crop saving
OUTPUT_DIR = "logs/test_face_detect_on_available_images"
PADDING_RATIO = 0.4

# Overlay configurations
DRAW_BBOX = True
DRAW_LANDMARKS = True
DRAW_POSE = True

COLOR_BBOX = colors.GREEN
COLOR_TEXT = colors.CYAN
COLOR_LANDMARK = colors.GREEN
COLOR_LANDMARK_TEXT = colors.RED

WINDOW_NAME = "Static Image Face Detection Test"

# ==============================================================================
#                                   SECTION: MAIN
# ==============================================================================

def main():
    logging.basicConfig(level=logging.INFO)
    
    logging.info(
        "\n======================================================\n"
        "[UTILITIES USED FROM src/]:\n"
        "- recog.face_recognition.InsightFaceDetector\n"
        "- ui.colors\n"
        "- utils.pose_utils.get_pose_name\n"
        "======================================================\n"
    )
    
    logging.info("Initializing InsightFaceDetector...")
    detector = InsightFaceDetector(det_threshold=DET_THRESHOLD)
    detector.prepare()
    logging.info("Detector initialized.")
    
    if SAVE_CROPPED_FACE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
        
    logging.info(f"Loading image from {INPUT_IMAGE_PATH}")
    frame = cv2.imread(INPUT_IMAGE_PATH)
    
    if frame is None:
        logging.error(f"Image not found at path: {INPUT_IMAGE_PATH}")
        return
        
    display_frame = frame.copy()
    
    # Detect faces
    detections = detector.detect(frame)
    
    if not detections:
        logging.warning("No face detected in the given image.")
    else:
        logging.info(f"Detected {len(detections)} face(s).")
        ts = int(time.time())
        
        # --- First Pass: Overlay Drawing ---
        for face in detections:
            box = face.bbox.astype(int)
            
            # Draw bounding box
            if DRAW_BBOX:
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), COLOR_BBOX, 2)
                text = f"Conf: {face.confidence:.2f}"
                cv2.putText(display_frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            
            # Draw pose
            if DRAW_POSE and face.pose is not None:
                pitch, yaw, roll = face.pose
                pose_label = get_pose_name(face.pose)
                pose_text = f"{pose_label} (P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f})"
                cv2.putText(display_frame, pose_text, (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            
            # Draw landmarks
            if DRAW_LANDMARKS and face.landmarks is not None:
                for i, lmk in enumerate(face.landmarks.astype(int)):
                    cv2.circle(display_frame, tuple(lmk), 2, COLOR_LANDMARK, -1)
                    cv2.putText(display_frame, str(i), tuple(lmk), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_LANDMARK_TEXT, 1)

        # --- Second Pass: Saving Cropped Face (if enabled) ---
        if SAVE_CROPPED_FACE:
            for idx, face in enumerate(detections):
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box
                
                w_box = x2 - x1
                h_box = y2 - y1
                
                padding_w = int(w_box * PADDING_RATIO)
                padding_h = int(h_box * PADDING_RATIO)
                
                h, w = display_frame.shape[:2]
                cx1 = max(0, x1 - padding_w)
                cy1 = max(0, y1 - padding_h)
                cx2 = min(w, x2 + padding_w)
                cy2 = min(h, y2 + padding_h)
                
                face_crop = display_frame[cy1:cy2, cx1:cx2]
                
                if len(detections) > 1:
                    crop_path = os.path.join(OUTPUT_DIR, f"face_crop_{ts}_{idx}.jpg")
                else:
                    crop_path = os.path.join(OUTPUT_DIR, f"face_crop_{ts}.jpg")
                    
                cv2.imwrite(crop_path, face_crop)
                logging.info(f"Saved padded face crop to: {crop_path}")

    # --- Display Result Image (if enabled) ---
    if SHOW_IMAGE:
        cv2.imshow(WINDOW_NAME, display_frame)
        logging.info("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    logging.info("Script completed.")

if __name__ == "__main__":
    main()
