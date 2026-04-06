"""FastAPI app exposing register, update, and verify endpoints.

When an endpoint is called, it submits a UI task to the main-thread
UI runner so the OpenCV window is displayed. The API thread blocks
until the UI window is closed (user presses 'q'/ESC or the flow
completes).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict

if __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.api_config import get_camera_client, get_pipelines
from api.ui_runner import UITask, UITaskType, submit_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up pipelines on startup to reduce first-request latency.
    get_pipelines()
    get_camera_client()
    yield


app = FastAPI(title="Face Recognition API", lifespan=lifespan)


class RegisterRequest(BaseModel):
    class_id: str = Field(..., min_length=1)


class UpdateRequest(BaseModel):
    class_id: str = Field(..., min_length=1)


@app.post("/register")
def register_face(payload: RegisterRequest) -> Dict[str, Any]:
    """
    Register a new face.

    Opens an OpenCV UI window on the main thread for interactive
    face registration. Blocks until the window closes.
    """
    task = UITask(
        task_type=UITaskType.REGISTER,
        params={"class_id": payload.class_id},
    )
    submit_task(task)

    # Block the API thread until the UI runner finishes this task.
    task.done_event.wait()

    if task.error:
        raise HTTPException(status_code=500, detail=task.error)

    return task.result


@app.post("/update")
def update_face(payload: UpdateRequest) -> Dict[str, Any]:
    """
    Update an existing face with new samples.

    Opens an OpenCV UI window on the main thread for interactive
    face updating. Blocks until the window closes.
    """
    task = UITask(
        task_type=UITaskType.UPDATE,
        params={"class_id": payload.class_id},
    )
    submit_task(task)

    task.done_event.wait()

    if task.error:
        raise HTTPException(status_code=500, detail=task.error)

    return task.result


@app.post("/verify")
def verify_face() -> Dict[str, Any]:
    """
    Verify faces in real-time.

    Opens an OpenCV UI window on the main thread for interactive
    face verification. Blocks until the window closes.
    """
    task = UITask(task_type=UITaskType.VERIFY)
    submit_task(task)

    task.done_event.wait()

    if task.error:
        raise HTTPException(status_code=500, detail=task.error)

    return task.result


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start uvicorn in a **daemon thread** so the main thread stays
    free for the OpenCV UI runner."""
    import uvicorn

    server_thread = threading.Thread(
        target=uvicorn.run,
        args=("api.app:app",),
        kwargs={"host": host, "port": port, "log_level": "info"},
        daemon=True,
    )
    server_thread.start()
    logging.info("Uvicorn started on http://%s:%d (daemon thread)", host, port)


if __name__ == "__main__":
    from api.api_config import get_pipelines, get_camera_client
    from api.ui_runner import run_ui_loop

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ### 1. Initialize shared resources
    recog_pipeline, classify_pipeline = get_pipelines()
    camera = get_camera_client()

    ### 2. Start Uvicorn API server in background thread
    start_server()

    ### 3. Run OpenCV UI event loop on main thread (blocks forever)
    run_ui_loop(recog_pipeline, classify_pipeline, camera)
