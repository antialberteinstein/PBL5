import cv2
import numpy as np
import logging
from typing import List, Optional


from recog.face_recognition import FaceDetection

# Global variables
# Tập lồi chứa các điểm landmarks bao quanh vùng đeo khẩu trang của người dùng.
MASK_LANDMARKS_ARRAY = \
      { 13, 14, 15, 16 } \
    | set(range(2, 9)) \
    | { 0 } \
    | set(range(23, 17, -1)) \
    | { 32, 31, 30, 29 } \
    | { 74 }


def add_virtual_mask(img: np.ndarray, face: FaceDetection) -> np.ndarray:
    """
    Create a synthetic mask over the lower half of the face using landmarks.
    
    Args:
        img: Input image as numpy array (BGR format)
        face: FaceDetection object containing bbox and landmarks
        
    Returns:
        Image with a virtual mask applied
    """
    masked_img = img.copy()
    
    if getattr(face, 'landmarks', None) is None:
        logging.debug('No landmarks detected')
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        mask_y = y1 + int(h * 0.5)
        # Use white rectangle if no landmarks
        cv2.rectangle(masked_img, (x1, mask_y), (x2, y2), (255, 255, 255), -1)
        return masked_img
    
    logging.debug("Landmarks detected")
        
    lmks = face.landmarks.astype(int)
    
    # Extract points belonging to the mask
    mask_points = [lmks[i] for i in MASK_LANDMARKS_ARRAY]
    pts = np.array(mask_points, np.int32)
    
    # Calculate the convex hull to form a proper "bao lồi" (convex polygon)
    hull = cv2.convexHull(pts)
    
    # Silver/Gray color (BGR format: 192, 192, 192)
    cv2.fillConvexPoly(masked_img, hull, (192, 192, 192))
    
    return masked_img
