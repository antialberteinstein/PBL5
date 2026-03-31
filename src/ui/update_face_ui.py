# ==============================================================================
#                           SECTION: UPDATE FACE UI
# ==============================================================================
"""
Interactive UI for adding new face samples to an existing ID.
"""

import cv2
import time
import logging
import numpy as np
from typing import Any
from services.update_face_service import UpdateFaceService
import ui.debug_config as debug_config
import ui.colors as colors

class UpdateFaceUI:
    """UI for interactive face updating."""
    
    def __init__(
        self,
        recog_pipeline,
        classify_pipeline,
        window_name: str = "Update Face",
        color_authorizing: tuple = colors.YELLOW,
        color_success: tuple = colors.GREEN,
        color_failure: tuple = colors.RED,
        images_per_pose: int = 3,
    ):
        """
        Initialize Update Face UI.
        """
        self.service = UpdateFaceService(
            recog_pipeline=recog_pipeline,
            classify_pipeline=classify_pipeline,
            images_per_pose=images_per_pose
        )
        self.window_name = window_name
        self.color_authorizing = color_authorizing
        self.color_success = color_success
        self.color_failure = color_failure

    def run(self, class_id: str, camera: Any) -> None:
        import threading
        
        logging.info(f"Starting Face Update for class ID: {class_id} via Background Thread")
        self.service.load_existing_vectors(class_id)
        
        self._stop_event = threading.Event()
        self._latest_frame = None
        
        self._ui_state = {
            "main_face": None,
            "res": None,
            "is_complete": False,
        }
        self._state_lock = threading.Lock()
        
        def _pipeline_worker():
            last_hint_time = 0
            
            while not self._stop_event.is_set():
                if self._latest_frame is not None:
                    frame_raw = self._latest_frame.copy()
                    
                    detections = self.service.detect_faces(frame_raw)
                    main_face = detections[0] if detections else None
                    res = None
                    is_complete = self.service.is_complete
                    
                    if main_face:
                        if not is_complete:
                            res = self.service.process_face_sample(class_id, frame_raw, main_face)
                            is_complete = self.service.is_complete
                            
                            if res["status"] == "NOT_DIVERSE":
                                if time.time() - last_hint_time > 2.0:
                                    logging.info("Change angle slightly for diversity")
                                    last_hint_time = time.time()
                                    
                        if is_complete and not self._ui_state["is_complete"]:
                            self.service.save(class_id)
                            logging.info(f"Face update complete for {class_id}!")
                    
                    with self._state_lock:
                        self._ui_state["main_face"] = main_face
                        self._ui_state["res"] = res
                        self._ui_state["is_complete"] = is_complete
                        
                time.sleep(0.1)

        worker = threading.Thread(target=_pipeline_worker, daemon=True)
        worker.start()
        
        while True:
            frame = camera.capture_frame()
            if frame is None:
                cv2.waitKey(100)
                continue
                
            self._latest_frame = frame
            
            with self._state_lock:
                main_face = self._ui_state["main_face"]
                res = self._ui_state["res"]
                is_complete = self._ui_state["is_complete"]
            
            if main_face:
                box = main_face.bbox
                
                if debug_config.SHOW_FACE_LANDMARKS and getattr(main_face, 'landmarks', None) is not None:
                    self._draw_landmarks(frame, main_face.landmarks)
                
                if not is_complete and res:
                    self._draw_hud(frame, res["req_pose"], res["det_pose"])
                    if res["status"] == "DIFFERENT_PERSON":
                        self._draw_error(frame, box, "DIFFERENT PERSON!")
                    self._draw_bbox(frame, box, self.service.total_collected_session, self.service.max_update_images)
            
            cv2.imshow(self.window_name, frame)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27 or is_complete:
                self._stop_event.set()
                break
                
        worker.join(timeout=1.0)
        cv2.destroyAllWindows()
        if is_complete:
            cv2.waitKey(1000)
            
        import sys
        if sys.platform == "darwin":
            for _ in range(30): cv2.waitKey(1)

    def _draw_hud(self, frame, req_pose, det_pose):
        count = self.service.get_pose_count(req_pose)
        cv2.putText(frame, f"Yeu cau: {req_pose} ({count}/{self.service.images_per_pose})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors.CYAN, 2)
        pose_color = colors.GREEN if det_pose == req_pose else colors.RED
        cv2.putText(frame, f"Hien tai: {det_pose}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pose_color, 2)

    def _draw_bbox(self, frame, box, count, total):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.color_authorizing, 2)
        cv2.putText(frame, f"UPDATE: {count}/{total}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_authorizing, 2)

    def _draw_error(self, frame, box, msg):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.color_failure, 2)
        cv2.putText(frame, msg, (box[0], box[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_failure, 2)

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        for i, lmk in enumerate(landmarks.astype(int)):
            cv2.circle(frame, tuple(lmk), 2, colors.GREEN, -1)
            cv2.putText(frame, str(i), tuple(lmk), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors.RED, 1)
