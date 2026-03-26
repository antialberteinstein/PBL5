# ==============================================================================
#                           SECTION: RECOGNITION PIPELINE
# ==============================================================================
"""
Unified Recognition Pipeline.

Combines Face Detection (InsightFace) and Classification (PCA + Scaler + Cosine)
into a single, easy-to-use interface for all modules.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from recog.face_recognition import FaceRecognizer

@dataclass
class ProcessedFace:
    """Combines detection data with pose information."""
    bbox: np.ndarray
    embedding: np.ndarray
    confidence: float
    pose: Optional[np.ndarray]
    pose_name: str
    landmarks: Optional[np.ndarray] = None

class RecognitionPipeline:
    """
    High-level orchestrator for the entire face recognition process.
    """
    
    def __init__(self, recognizer: FaceRecognizer):
        """
        Initialize the recognition pipeline.
        
        Args:
            recognizer: The detector/embedder (e.g., InsightFace)
        """
        self.recognizer = recognizer

    def process_frame(self, frame: np.ndarray) -> List[ProcessedFace]:
        """
        Processes a full frame: detects faces and extracts pose information.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of ProcessedFace objects
        """
        # 1. Detect faces
        detections = self.recognizer.detect(frame)
        
        # 2. Extract pose for each face
        from utils.pose_utils import get_pose_name
        
        results = []
        for face in detections:
            pose_name = get_pose_name(face.pose)
            
            results.append(ProcessedFace(
                bbox=face.bbox,
                embedding=face.embedding,
                confidence=face.confidence,
                pose=face.pose,
                pose_name=pose_name,
                landmarks=getattr(face, 'landmarks', None)
            ))
            
        return results
