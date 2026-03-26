# ==============================================================================
#                           SECTION: REGISTRATION UI
# ==============================================================================
"""
Interactive UI for face registration.
"""

import cv2
import time
import logging
import numpy as np
from typing import Any
from services.registration_service import RegistrationService
import ui.debug_config as debug_config
import ui.colors as colors

class RegistrationUI:
    """UI for interactive face registration."""
    
    def __init__(
        self,
        recog_pipeline,
        classify_pipeline,
        window_name: str = "Registration",
        color_authorizing: tuple = colors.YELLOW,
        color_success: tuple = colors.GREEN,
        color_failure: tuple = colors.RED,
        images_per_pose: int = 3,
    ):
        """
        Initialize Registration UI.
        """
        self.service = RegistrationService(
            recog_pipeline=recog_pipeline,
            classify_pipeline=classify_pipeline,
            images_per_pose=images_per_pose
        )
        self.window_name = window_name
        self.color_authorizing = color_authorizing
        self.color_success = color_success
        self.color_failure = color_failure

    def run(self, class_id: str, camera: Any) -> None:
        """Runs the registration UI loop."""
        logging.info(f"Starting registration for class ID: {class_id}")
        last_warning_time = 0
        last_hint_time = 0
        
        while True:
            frame = camera.capture_frame()
            if frame is None:
                cv2.waitKey(100)
                continue
                
            frame_raw = frame.copy()
            detections = self.service.detect_faces(frame)
            
            if detections:
                main_face = detections[0]
                box = main_face.bbox
                emb = main_face.embedding
                
                if debug_config.SHOW_FACE_LANDMARKS and getattr(main_face, 'landmarks', None) is not None:
                    self._draw_landmarks(frame, main_face.landmarks)
                
                ### 1. Check if already known
                db_id = self.service.check_already_registered(frame, box)
                
                if db_id:
                    current_time = time.time()
                    if current_time - last_warning_time > 2.0:
                        logging.warning(f"Face already registered as {db_id}!")
                        last_warning_time = current_time
                    self._draw_warning(frame, box, f"ALREADY {db_id}")
                
                ### 2. Handle Collection
                elif not self.service.is_complete:
                    res = self.service.process_face_sample(class_id, frame_raw, main_face)
                    
                    self._draw_hud(frame, res["req_pose"], res["det_pose"])
                    
                    if res["status"] == "DIFFERENT_PERSON":
                        self._draw_error(frame, box, "DIFFERENT PERSON!")
                    elif res["status"] == "NOT_DIVERSE":
                        if time.time() - last_hint_time > 2.0:
                            logging.info("Change angle slightly for diversity")
                            last_hint_time = time.time()
                    
                    self._draw_bbox(frame, box, self.service.total_collected, self.service.max_registration_images)
                else:
                    ### 3. Complete
                    self.service.save(class_id)
                    logging.info("Registration complete!")
                    cv2.waitKey(2000)
                    break
            
            cv2.imshow(self.window_name, frame)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
                
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)

    def _draw_hud(self, frame: np.ndarray, req_pose: str, det_pose: str) -> None:
        count = self.service.get_pose_count(req_pose)
        cv2.putText(frame, f"Yeu cau: {req_pose} ({count}/{self.service.images_per_pose})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors.CYAN, 2)
        pose_color = colors.GREEN if det_pose == req_pose else colors.RED
        cv2.putText(frame, f"Hien tai: {det_pose}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pose_color, 2)

    def _draw_bbox(self, frame: np.ndarray, box: np.ndarray, count: int, total: int) -> None:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.color_authorizing, 2)
        cv2.putText(frame, f"{count}/{total}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_authorizing, 2)

    def _draw_warning(self, frame, box, msg):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.color_failure, 2)
        cv2.putText(frame, msg, (box[0], box[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_failure, 2)

    def _draw_error(self, frame, box, msg):
        self._draw_warning(frame, box, msg)

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        for i, lmk in enumerate(landmarks.astype(int)):
            cv2.circle(frame, tuple(lmk), 2, colors.GREEN, -1)
            cv2.putText(frame, str(i), tuple(lmk), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors.RED, 1)
