"""
Test script to apply virtual masks to currently saved cropped faces.

[UTILITIES USED FROM src/]:
- recog.face_recognition.InsightFaceDetector
- utils.mask_utils.add_virtual_mask
"""

import os
import sys
import cv2
import logging
import warnings
import numpy as np

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recog.face_recognition import InsightFaceDetector
from utils.mask_utils import add_virtual_mask

# Suppress warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
#                           SECTION: CONFIGURATION
# ==============================================================================

INPUT_DIR = "logs/test_capture_face"
OUTPUT_DIR = "logs/test_mask"


def setup_directories():
    """Ensure input exists and output directory is created."""
    if not os.path.exists(INPUT_DIR):
        logging.error(f"Input directory not found: {INPUT_DIR}. Please run test_capture_face.py first.")
        return False
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
    return True


def robust_detect(detector, img):
    """
    Try to detect face with different scales if the default fails.
    Useful for cropped images where the detector context is limited.
    """
    # Try 1: Default
    detections = detector.detect(img)
    if detections:
        return detections
        
    # Try 2: Scale up (context might be too small)
    h, w = img.shape[:2]
    if h < 400 or w < 400:
        larger_img = cv2.resize(img, (w*2, h*2))
        detections = detector.detect(larger_img)
        if detections:
            # Rescale landmarks back
            for d in detections:
                d.bbox = (d.bbox / 2).astype(int)
                if d.landmarks is not None:
                    d.landmarks = (d.landmarks / 2).astype(int)
            return detections
            
    # Try 3: Pad with black border (mimic full frame context)
    pad_h, pad_w = h // 2, w // 2
    padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    detections = detector.detect(padded_img)
    if detections:
        # Shift landmarks back
        for d in detections:
            d.bbox[0] -= pad_w
            d.bbox[2] -= pad_w
            d.bbox[1] -= pad_h
            d.bbox[3] -= pad_h
            if d.landmarks is not None:
                d.landmarks[:, 0] -= pad_w
                d.landmarks[:, 1] -= pad_h
        return detections
        
    return []


def run_mask_test():
    """
    Function-based test to apply virtual masks to all images in a source folder.
    """
    # 1. Setup
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logging.info(
        "\n======================================================\n"
        "[UTILITIES USED FROM src/]:\n"
        "- recog.face_recognition.InsightFaceDetector\n"
        "- utils.mask_utils.add_virtual_mask\n"
        "======================================================\n"
    )
    
    if not setup_directories():
        return

    # 2. Initialize Detector
    detector = InsightFaceDetector()
    detector.prepare()
    
    # 3. Process Images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    
    # Filter to process all images (ignoring if it's original or crop)
    if not files:
        logging.warning(f"No images found in {INPUT_DIR}")
        return

    logging.info(f"Found {len(files)} images to process.")
    
    success_count = 0
    
    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(input_path)
        
        if img is None:
            logging.warning(f"Failed to read: {filename}")
            continue
            
        # Detect faces with robust fallback
        detections = robust_detect(detector, img)
        
        if not detections:
            logging.warning(f"Still no face detected in: {filename}. Using fallback rectangle mask.")
            # Create a mock detection for add_virtual_mask fallback (rectangle)
            h, w = img.shape[:2]
            # Use most of the image as the "face" since it's already a crop
            mock_bbox = np.array([int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)])
            mock_face = FaceDetection(bbox=mock_bbox, embedding=None, confidence=1.0, landmarks=None)
            masked_img = add_virtual_mask(img, mock_face)
        else:
            # Apply mask to the first face detected
            main_face = detections[0]
            masked_img = add_virtual_mask(img, main_face)
        
        # Save result
        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{basename}_masked.jpg")
        
        cv2.imwrite(output_path, masked_img)
        logging.info(f"Processed {filename} -> {output_path}")
        success_count += 1
        
    logging.info(f"Test completed. Successfully processed {success_count}/{len(files)} images.")


if __name__ == "__main__":
    run_mask_test()
