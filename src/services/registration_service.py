# ==============================================================================
#                           SECTION: REGISTRATION SERVICE
# ==============================================================================
"""
Business logic for face registration.
"""

import logging
from typing import Optional, Dict, List, Any
import numpy as np
import os
import time
import cv2
from recog.face_recognition import FaceRecognizer
from utils.pose_utils import POSES, get_pose_name
from utils.mask_utils import add_virtual_mask
import services.debug_config as debug_config

class RegistrationService:
    """Service for registering new faces and managing embeddings."""
    
    def __init__(
        self,
        recog_pipeline,
        classify_pipeline,
        images_per_pose: int = 3,
        same_pose_threshold: float = 0.45,
        cross_pose_threshold: float = 0.05,
        similar_embedding_threshold: float = 0.85,
    ):
        """
        Initialize registration service.
        """
        ### 1. Store dependencies
        self.recog_pipeline = recog_pipeline
        self.classify_pipeline = classify_pipeline
        
        ### 2. Store configurations
        self.images_per_pose = images_per_pose
        self.same_pose_threshold = same_pose_threshold
        self.cross_pose_threshold = cross_pose_threshold
        self.similar_embedding_threshold = similar_embedding_threshold
        
        ### 3. Initialize state
        self.poses = POSES
        self.max_registration_images = len(self.poses) * self.images_per_pose
        self.embeddings_by_pose = {p: [] for p in self.poses}
        self.current_pose_idx = 0
        self.total_collected = 0
        self.pose_counts = {p: 0 for p in self.poses}
        
    def get_pose_count(self, pose: str) -> int:
        return self.pose_counts.get(pose, 0)
        
    def detect_faces(self, frame: np.ndarray) -> List[Any]:
        return self.recog_pipeline.recognizer.detect(frame)
        
    def check_already_registered(self, frame: np.ndarray, box: np.ndarray) -> Optional[str]:
        faces = self.recog_pipeline.process_frame(frame)
        main_face = next((f for f in faces if np.array_equal(f.bbox, box)), None)
        if main_face:
            class_id, _ = self.classify_pipeline.predict_with_score(main_face.embedding)
            if class_id is not None and "UNKNOWN" not in str(class_id):
                return class_id
        return None
        
    def process_face_sample(self, class_id: str, frame_raw: np.ndarray, face: Any) -> Dict[str, Any]:
        """
        Processes a single face detection for registration pose requirements.
        """
        curr_pose = self.current_pose
        det_pose = get_pose_name(face.pose)
        
        result = {
            "req_pose": curr_pose,
            "det_pose": det_pose,
            "status": "WRONG_POSE"
        }
        
        if det_pose == curr_pose:
            if not self.is_same_person(face.embedding, curr_pose):
                result["status"] = "DIFFERENT_PERSON"
            elif self.is_diverse(face.embedding):
                self.collect_sample(class_id, frame_raw, face, curr_pose)
                result["status"] = "COLLECTED"
            else:
                result["status"] = "NOT_DIVERSE"
                
        return result
        
    @property
    def current_pose(self) -> Optional[str]:
        """Returns the currently requested pose name."""
        if self.current_pose_idx < len(self.poses):
            return self.poses[self.current_pose_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """Returns True if all required images are collected."""
        return self.current_pose_idx >= len(self.poses)

    def is_same_person(self, embedding: np.ndarray, current_pose_name: str) -> bool:
        """Checks if embedding belongs to the same person as already collected ones."""
        if not any(self.embeddings_by_pose.values()):
            return True
            
        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0:
            return False
            
        for pose_key, pose_embeddings in self.embeddings_by_pose.items():
            if not pose_embeddings:
                continue
                
            threshold = self.same_pose_threshold if pose_key == current_pose_name else self.cross_pose_threshold
            
            best_similarity = -1.0
            for existing in pose_embeddings:
                exist_norm = np.linalg.norm(existing)
                if exist_norm == 0: continue
                
                similarity = np.dot(embedding, existing) / (emb_norm * exist_norm)
                if similarity > best_similarity:
                    best_similarity = similarity
                    
            if best_similarity != -1.0 and best_similarity < threshold:
                return False
        return True

    def is_diverse(self, embedding: np.ndarray) -> bool:
        """Checks if embedding is diverse enough from already collected ones."""
        flat_embeddings = [e for e_list in self.embeddings_by_pose.values() for e in e_list]
        if not flat_embeddings:
            return True
            
        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0:
            return False
            
        for existing in flat_embeddings:
            exist_norm = np.linalg.norm(existing)
            if exist_norm == 0: continue
            
            similarity = np.dot(embedding, existing) / (emb_norm * exist_norm)
            if similarity > self.similar_embedding_threshold:
                return False
        return True

    def add_sample(self, real_embedding: np.ndarray, masked_embedding: Optional[np.ndarray] = None) -> bool:
        """Adds a sample to the current pose collection."""
        curr_pose = self.current_pose
        if not curr_pose:
            return False
            
        self.embeddings_by_pose[curr_pose].append(real_embedding)
        if masked_embedding is not None:
            self.embeddings_by_pose[curr_pose].append(masked_embedding)
            
        self.total_collected += 1
        return True

    def collect_sample(self, class_id: str, frame_raw: np.ndarray, face: Any, pose: str) -> None:
        """Helper to collect and save a single sample."""
        masked_frame = add_virtual_mask(frame_raw, face)
        masked_detections = self.recog_pipeline.recognizer.detect(masked_frame)
        masked_emb = masked_detections[0].embedding if masked_detections else None
        
        self.add_sample(face.embedding, masked_emb)
        
        if debug_config.SAVE_DEBUG_IMAGES:
            self._save_debug(class_id, frame_raw, face.bbox, pose, masked_frame, masked_detections)
            
        self.pose_counts[pose] += 1
        if self.pose_counts[pose] >= self.images_per_pose:
            self.increment_pose()

    def _save_debug(self, class_id: str, frame: np.ndarray, box: np.ndarray, pose: str, masked_frame: np.ndarray, masked_detections: Any) -> None:
        class_dir = os.path.join(debug_config.DEBUG_IMAGES_DIR, class_id)
        os.makedirs(class_dir, exist_ok=True)
        x1, y1, x2, y2 = box.astype(int)
        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        
        ts = int(time.time())
        p_name = pose.replace(' ', '_')
        cv2.imwrite(os.path.join(class_dir, f"{p_name}_{self.pose_counts[pose]}_{ts}.jpg"), frame[y1:y2, x1:x2])
        if masked_detections:
            cv2.imwrite(os.path.join(class_dir, f"{p_name}_{self.pose_counts[pose]}_{ts}_masked.jpg"), masked_frame[y1:y2, x1:x2])

    def increment_pose(self) -> None:
        """Move to the next pose."""
        self.current_pose_idx += 1

    def save(self, class_id: str) -> None:
        """Save collected embeddings to the database."""
        raw_embeddings = [emb for emb_list in self.embeddings_by_pose.values() for emb in emb_list]
        if raw_embeddings:
            self.classify_pipeline.fit(class_id, raw_embeddings)
            logging.info(f"Saved {len(raw_embeddings)} embeddings for '{class_id}'")
