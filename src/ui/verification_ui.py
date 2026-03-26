# ==============================================================================
#                           SECTION: VERIFICATION UI
# ==============================================================================
"""
Real-time UI for face verification.
"""

import cv2
import logging
import sys
import numpy as np
from typing import Any
from services.verification_service import VerificationService
import ui.debug_config as debug_config
import ui.colors as colors

class VerificationUI:
    """UI for real-time face verification."""
    
    def __init__(
        self,
        recog_pipeline,
        classify_pipeline,
        window_name: str = "Verification",
        color_success: tuple = colors.GREEN,
        color_failure: tuple = colors.RED,
    ):
        """
        Initialize Verification UI.
        """
        self.service = VerificationService(
            recog_pipeline=recog_pipeline,
            classify_pipeline=classify_pipeline
        )
        self.window_name = window_name
        self.color_success = color_success
        self.color_failure = color_failure

    def run(self, camera: Any) -> None:
        logging.info("Starting verification UI")
        
        while True:
            frame = camera.capture_frame()
            if frame is None:
                cv2.waitKey(100)
                continue
            
            try:
                results = self.service.verify(frame)
                for res in results:
                    self._draw_result(frame, res)
                    
                    if debug_config.SHOW_FACE_LANDMARKS and res.get("landmarks") is not None:
                        self._draw_landmarks(frame, res["landmarks"])
                        
                    if res["is_known"]:
                        camera.send_result(f"MATCH: {res['class_id']} ({res['score']:.2f})")
            except Exception as e:
                logging.error(f"Verification UI error: {e}")
            
            cv2.imshow(self.window_name, frame)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27: break
            
        cv2.destroyAllWindows()
        if sys.platform == "darwin":
            # Extra wait for macOS OpenCV window cleanup
            for _ in range(30): cv2.waitKey(1)

    def _draw_result(self, frame, res):
        box = res["bbox"].astype(int)
        color = self.color_success if res["is_known"] else self.color_failure
        
        # Draw BBox
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw ID and Score
        score_val = res["score"]
        score_str = f"{score_val:.2f}" if score_val != float('inf') else "inf"
        text = f"{res['class_id']}: {score_str}" if res["is_known"] else f"UNKNOWN: {score_str}"
        cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw Pose
        cv2.putText(frame, res["pose_name"], (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.CYAN, 2)

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        for i, lmk in enumerate(landmarks.astype(int)):
            cv2.circle(frame, tuple(lmk), 2, colors.GREEN, -1)
            cv2.putText(frame, str(i), tuple(lmk), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors.RED, 1)
