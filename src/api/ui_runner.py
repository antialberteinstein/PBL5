"""
Main-thread UI runner for OpenCV windows.

On macOS, cv2.imshow/cv2.waitKey MUST run on the main thread.
API handlers submit UI tasks to a queue; this runner consumes them
on the main thread so OpenCV works correctly.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class UITaskType(Enum):
    REGISTER = auto()
    UPDATE = auto()
    VERIFY = auto()


@dataclass
class UITask:
    """A unit of work submitted by an API endpoint."""

    task_type: UITaskType
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    done_event: threading.Event = field(default_factory=threading.Event)


# Singleton queue shared between the API thread and the main thread.
_task_queue: queue.Queue[UITask] = queue.Queue(maxsize=1)


def submit_task(task: UITask) -> None:
    """Submit a UI task (called from the API/uvicorn thread)."""
    _task_queue.put(task)


def run_ui_loop(
    recog_pipeline: Any,
    classify_pipeline: Any,
    camera: Any,
) -> None:
    """
    Blocking loop that runs on the **main thread**.
    Waits for UITask items from the queue and executes them
    with the corresponding UI class.
    """
    from ui.registration_ui import RegistrationUI
    from ui.update_face_ui import UpdateFaceUI
    from ui.verification_ui import VerificationUI

    logging.info("UI runner started on main thread — waiting for tasks …")

    while True:
        try:
            task: UITask = _task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        logging.info("UI runner received task: %s", task.task_type.name)

        try:
            if task.task_type == UITaskType.REGISTER:
                ui = RegistrationUI(recog_pipeline, classify_pipeline)
                ui.run(task.params["class_id"], camera)

                task.result = {
                    "status": "completed" if ui.service.is_complete else "incomplete",
                    "class_id": task.params["class_id"],
                    "total_collected": ui.service.total_collected,
                    "max_required": ui.service.max_registration_images,
                }

            elif task.task_type == UITaskType.UPDATE:
                ui = UpdateFaceUI(recog_pipeline, classify_pipeline)
                ui.run(task.params["class_id"], camera)

                task.result = {
                    "status": "completed" if ui.service.is_complete else "incomplete",
                    "class_id": task.params["class_id"],
                    "total_collected": ui.service.total_collected_session,
                    "max_required": ui.service.max_update_images,
                }

            elif task.task_type == UITaskType.VERIFY:
                ui = VerificationUI(recog_pipeline, classify_pipeline)
                ui.run(camera)

                task.result = {
                    "status": "completed",
                }

        except Exception as exc:
            logging.exception("UI task failed: %s", exc)
            task.error = str(exc)

        finally:
            task.done_event.set()
            _task_queue.task_done()
