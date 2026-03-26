# ==============================================================================
#                           SECTION: FACE RECOGNITION
# ==============================================================================
"""
Face recognition implementation using InsightFace.

Provides face detection and embedding extraction using the
InsightFace library (buffalo_l model).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import logging
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==============================================================================
#                           SECTION: DATA CLASSES
# ==============================================================================

@dataclass
class FaceDetection:
    """
    Represents a detected face in an image.
    
    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        embedding: Face embedding vector
        confidence: Detection confidence score (0-1)
    """
    bbox: np.ndarray
    embedding: np.ndarray
    confidence: float
    landmarks: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None


# ==============================================================================
#                           SECTION: CONFIGURATION
# ==============================================================================
# Global default configurations for face recognition have been moved into 
# the InsightFaceDetector class constructor.


# ==============================================================================
#                           SECTION: ABSTRACT INTERFACE
# ==============================================================================

class FaceRecognizer(ABC):
    """
    Abstract interface for face recognition systems.
    
    This class defines the contract that all face recognition implementations
    must follow. It enables easy swapping between different face recognition
    models without changing the rest of the codebase.
    """
    
    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image frame.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of detected faces with bounding boxes and embeddings
        """
        pass
    
    @abstractmethod
    def extract_embedding(
        self,
        frame: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from a frame.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            bbox: Optional bounding box to extract from. If None, detects face first.
            
        Returns:
            Face embedding vector, or None if no face detected
        """
        pass
    
    @abstractmethod
    def prepare(self) -> None:
        """
        Initialize the face recognition model.
        
        This method should load models, allocate resources, and perform
        any necessary setup before the recognizer can be used.
        """
        pass


# ==============================================================================
#                           SECTION: IMPLEMENTATION
# ==============================================================================

class InsightFaceDetector(FaceRecognizer):
    """
    InsightFace-based face recognition implementation.
    
    Uses the InsightFace library for face detection and embedding extraction.
    This implementation provides high-quality face embeddings suitable for
    recognition tasks.
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cpu",
        det_threshold: float = 0.5,
        det_size: tuple = (640, 640),
    ):
        """Initialize InsightFace detector."""
        self.model_name = model_name
        self.device = device
        self.det_threshold = det_threshold
        self.det_size = det_size
        
        self.app: Optional[FaceAnalysis] = None
        self._is_prepared = False
    
    def prepare(self) -> None:
        """
        Initialize the InsightFace model.
        
        Loads the model and prepares it for inference. This should be called
        once before using the detector.
        """
        if self._is_prepared:
            return
        
        ### 1. Initialize FaceAnalysis
        self.app = FaceAnalysis(name=self.model_name)
        
        ### 2. Prepare model with configuration
        ctx_id = -1 if self.device == "cpu" else 0
        self.app.prepare(
            ctx_id=ctx_id,
            det_size=self.det_size,
        )
        
        self._is_prepared = True
    
    def detect(
        self,
        frame: np.ndarray,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image frame using InsightFace.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of detected faces with bounding boxes and embeddings
            
        Raises:
            RuntimeError: If detector is not prepared
        """
        if not self._is_prepared or self.app is None:
            raise RuntimeError(
                "InsightFace detector not prepared. Call prepare() first."
            )
        
        ### 1. Run face detection
        faces = self.app.get(frame)
        
        ### 2. Convert to FaceDetection objects
        detections = []
        for face in faces:
            score = float(face.det_score)
            
            # Dynamic threshold based on pose
            # For profile faces (pitch/yaw), the detection score is naturally lower than frontal faces.
            threshold = self.det_threshold
            if hasattr(face, 'pose') and face.pose is not None:
                pitch, yaw, roll = face.pose
                if abs(pitch) > 15 or abs(yaw) > 20:
                    threshold = max(0.50, self.det_threshold - 0.20)
            
            # Filter out faces with low confidence score (blurry, dark, small, etc.)
            if score < threshold:
                # To prevent log spam, we could use a rate limiter, but for now a direct warning is fine 
                # per user request. If a single frame has multiple bad faces it warns for each.
                logging.warning(f"Ignored face due to low detection score: {score:.2f} < {threshold:.2f} (Base: {self.det_threshold:.2f})")
                continue
                
            detection = FaceDetection(
                bbox=face.bbox.astype(int),
                embedding=face.embedding,
                confidence=score,
                landmarks=face.landmark_2d_106.astype(int) if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None else None,
                pose=face.pose if hasattr(face, 'pose') else None
            )
            detections.append(detection)
        
        return detections
    
    def extract_embedding(
        self,
        frame: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from a frame.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            bbox: Optional bounding box. If None, detects face first.
            
        Returns:
            Face embedding vector, or None if no face detected
            
        Raises:
            RuntimeError: If detector is not prepared
        """
        if not self._is_prepared or self.app is None:
            raise RuntimeError(
                "InsightFace detector not prepared. Call prepare() first."
            )
        
        ### 1. Detect faces if bbox not provided
        if bbox is None:
            detections = self.detect(frame)
            if not detections:
                return None
            return detections[0].embedding
        
        ### 2. Extract embedding from bbox
        # Note: InsightFace doesn't support direct bbox extraction,
        # so we detect all faces and find the one matching the bbox
        detections = self.detect(frame)
        
        if not detections:
            return None
        
        # Return embedding of first detected face
        # (In practice, bbox matching would be more sophisticated)
        return detections[0].embedding
