# ==============================================================================
#                           SECTION: UPDATE FACE SERVICE
# ==============================================================================
"""
Business logic for updating face embeddings (adding samples to existing ID).
"""

import logging
from typing import Optional, List, Any
import numpy as np
import os
import time
import cv2
from recog.face_recognition import FaceRecognizer
from utils.pose_utils import POSES, get_pose_name
from utils.mask_utils import add_virtual_mask
import services.debug_config as debug_config

class UpdateFaceService:
    """Service for adding new faces to an existing ID."""
    
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
        Initialize update service.
        """
        self.recog_pipeline = recog_pipeline
        self.classify_pipeline = classify_pipeline
        
        self.images_per_pose = images_per_pose
        self.same_pose_threshold = same_pose_threshold
        self.cross_pose_threshold = cross_pose_threshold
        self.similar_embedding_threshold = similar_embedding_threshold
        
        self.poses = POSES
        self.max_update_images = len(self.poses) * self.images_per_pose
        self.embeddings_by_pose = {p: [] for p in self.poses}
        self.current_pose_idx = 0
        self.total_collected_session = 0
        self.pose_counts = {p: 0 for p in self.poses}
        
    def get_pose_count(self, pose: str) -> int:
        return self.pose_counts.get(pose, 0)
        
    def detect_faces(self, frame: np.ndarray) -> List[Any]:
        return self.recog_pipeline.recognizer.detect(frame)
        
    def process_face_sample(self, class_id: str, frame_raw: np.ndarray, face: Any) -> dict:
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
        
        self._existing_db_list = []

    def load_existing_vectors(self, class_id: str) -> None:
        """Loads existing vectors from DB for diversity checking."""
        db_vectors_array = self.classify_pipeline.classifier.get_vectors_by_id(class_id)
        self._existing_db_list = [db_vectors_array[i] for i in range(len(db_vectors_array))] if len(db_vectors_array) > 0 else []

    @property
    def current_pose(self) -> Optional[str]:
        if self.current_pose_idx < len(self.poses):
            return self.poses[self.current_pose_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_pose_idx >= len(self.poses)

    def is_same_person(self, embedding: np.ndarray, current_pose_name: str) -> bool:
        """Check if embedding belongs to the same person contextually across poses."""
        if not any(self.embeddings_by_pose.values()) and not self._existing_db_list:
            return True
        
        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0: return False
        
        # Check against session frames
        for pose_key, pose_embeddings in self.embeddings_by_pose.items():
            if not pose_embeddings: continue
            threshold = self.same_pose_threshold if pose_key == current_pose_name else self.cross_pose_threshold
            best_similarity = -1.0
            for existing in pose_embeddings:
                exist_norm = np.linalg.norm(existing)
                if exist_norm == 0: continue
                similarity = np.dot(embedding, existing) / (emb_norm * exist_norm)
                if similarity > best_similarity: best_similarity = similarity
            if best_similarity != -1.0 and best_similarity < threshold: return False
            
        # Check against DB (rough check as CROSS_POSE since we don't know DB poses easily here)
        for existing in self._existing_db_list:
            exist_norm = np.linalg.norm(existing)
            if exist_norm == 0: continue
            # Check length compatibility (might need transform)
            emb_to_check = embedding
            if len(embedding) != len(existing):
                emb_to_check = self.classify_pipeline.transform(embedding.reshape(1, -1))[0]
            similarity = np.dot(emb_to_check, existing) / (np.linalg.norm(emb_to_check) * exist_norm)
            if similarity < self.cross_pose_threshold: return False

        return True

    def is_diverse(self, embedding: np.ndarray) -> bool:
        """Check if embedding is diverse enough from DB and session."""
        # Check against DB
        emb_to_check = embedding
        if self._existing_db_list and len(embedding) != len(self._existing_db_list[0]):
            emb_to_check = self.classify_pipeline.transform(embedding.reshape(1, -1))[0]
            
        for existing in self._existing_db_list:
            exist_norm = np.linalg.norm(existing)
            if exist_norm == 0: continue
            if (np.dot(emb_to_check, existing) / (np.linalg.norm(emb_to_check) * exist_norm)) > self.similar_embedding_threshold:
                return False
                
        # Check against session
        flat_session = [e for e_list in self.embeddings_by_pose.values() for e in e_list]
        emb_norm = np.linalg.norm(embedding)
        for existing in flat_session:
            exist_norm = np.linalg.norm(existing)
            if exist_norm == 0: continue
            if (np.dot(embedding, existing) / (emb_norm * exist_norm)) > self.similar_embedding_threshold:
                return False
                
        return True

    def add_sample(self, real_embedding: np.ndarray, masked_embedding: Optional[np.ndarray] = None) -> None:
        curr_pose = self.current_pose
        if curr_pose:
            self.embeddings_by_pose[curr_pose].append(real_embedding)
            if masked_embedding is not None:
                self.embeddings_by_pose[curr_pose].append(masked_embedding)
            self.total_collected_session += 1

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
        self.current_pose_idx += 1

    def save(self, class_id: str) -> None:
        raw_embeddings = [emb for emb_list in self.embeddings_by_pose.values() for emb in emb_list]
        if raw_embeddings:
            self.classify_pipeline.fit(class_id, raw_embeddings)
            logging.info(f"Appended {len(raw_embeddings)} embeddings for '{class_id}'")
