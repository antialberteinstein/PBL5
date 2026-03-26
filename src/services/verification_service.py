# ==============================================================================
#                           SECTION: VERIFICATION SERVICE
# ==============================================================================
"""
Business logic for face verification.
"""

import logging
from typing import List, Dict, Any
import numpy as np
from recog.face_recognition import FaceRecognizer, FaceDetection
from pipeline.recog import RecognitionPipeline
from utils.pose_utils import get_pose_name

class VerificationService:
    """Service for verifying faces against registered embeddings."""
    
    def __init__(
        self,
        recog_pipeline: Any,
        classify_pipeline: Any,
    ):
        """
        Initialize verification service.
        """
        self.recog_pipeline = recog_pipeline
        self.classify_pipeline = classify_pipeline
    
    def verify(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Verify faces in a frame.
        
        Returns:
            List of dicts containing bbox, class_id, score, pose_text, and is_known.
        """
        faces = self.recog_pipeline.process_frame(frame)
        results = []
        for face in faces:
            class_id, score = self.classify_pipeline.predict_with_score(face.embedding)
            is_known = class_id is not None and "UNKNOWN" not in str(class_id)
            
            results.append({
                "bbox": face.bbox,
                "class_id": class_id,
                "score": score,
                "pose": face.pose,
                "pose_name": face.pose_name,
                "is_known": is_known,
                "landmarks": face.landmarks
            })
            
        return results
    
