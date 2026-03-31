"""FastAPI app exposing register, update, and verify endpoints."""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

if __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.api_config import get_camera_client, get_camera_lock, get_pipelines
from services.registration_service import RegistrationService
from services.update_face_service import UpdateFaceService
from services.verification_service import VerificationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up pipelines on startup to reduce first-request latency.
    get_pipelines()
    get_camera_client()
    yield


app = FastAPI(title="Face Recognition API", lifespan=lifespan)


class RegisterRequest(BaseModel):
    class_id: str = Field(..., min_length=1)
    timeout_sec: float = Field(30.0, gt=0, le=300)


class UpdateRequest(BaseModel):
    class_id: str = Field(..., min_length=1)
    timeout_sec: float = Field(30.0, gt=0, le=300)


class VerifyResponse(BaseModel):
    results: List[Dict[str, Any]]


def _serialize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    serialized = dict(res)
    if "bbox" in serialized and hasattr(serialized["bbox"], "tolist"):
        serialized["bbox"] = serialized["bbox"].tolist()
    if "pose" in serialized and hasattr(serialized["pose"], "tolist"):
        serialized["pose"] = serialized["pose"].tolist()
    if "landmarks" in serialized and hasattr(serialized["landmarks"], "tolist"):
        serialized["landmarks"] = serialized["landmarks"].tolist()
    return serialized


@app.post("/register")
def register_face(payload: RegisterRequest) -> Dict[str, Any]:
    recog_pipeline, classify_pipeline = get_pipelines()
    camera = get_camera_client()
    lock = get_camera_lock()

    service = RegistrationService(recog_pipeline, classify_pipeline)

    start_time = time.time()
    last_status: Optional[Dict[str, Any]] = None

    with lock:
        while time.time() - start_time < payload.timeout_sec:
            frame = camera.capture_frame()
            if frame is None:
                continue

            detections = service.detect_faces(frame)
            main_face = detections[0] if detections else None
            if main_face is None:
                continue

            db_id = service.check_already_registered(frame, main_face.bbox)
            if db_id:
                return {
                    "status": "already_registered",
                    "class_id": db_id,
                }

            last_status = service.process_face_sample(payload.class_id, frame, main_face)
            if service.is_complete:
                service.save(payload.class_id)
                return {
                    "status": "completed",
                    "class_id": payload.class_id,
                    "total_collected": service.total_collected,
                    "max_required": service.max_registration_images,
                }

    return {
        "status": "incomplete",
        "class_id": payload.class_id,
        "total_collected": service.total_collected,
        "max_required": service.max_registration_images,
        "last": last_status,
    }


@app.post("/update")
def update_face(payload: UpdateRequest) -> Dict[str, Any]:
    recog_pipeline, classify_pipeline = get_pipelines()
    camera = get_camera_client()
    lock = get_camera_lock()

    service = UpdateFaceService(recog_pipeline, classify_pipeline)
    service.load_existing_vectors(payload.class_id)

    start_time = time.time()
    last_status: Optional[Dict[str, Any]] = None

    with lock:
        while time.time() - start_time < payload.timeout_sec:
            frame = camera.capture_frame()
            if frame is None:
                continue

            detections = service.detect_faces(frame)
            main_face = detections[0] if detections else None
            if main_face is None:
                continue

            last_status = service.process_face_sample(payload.class_id, frame, main_face)
            if service.is_complete:
                service.save(payload.class_id)
                return {
                    "status": "completed",
                    "class_id": payload.class_id,
                    "total_collected": service.total_collected_session,
                    "max_required": service.max_update_images,
                }

    return {
        "status": "incomplete",
        "class_id": payload.class_id,
        "total_collected": service.total_collected_session,
        "max_required": service.max_update_images,
        "last": last_status,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify_face() -> VerifyResponse:
    recog_pipeline, classify_pipeline = get_pipelines()
    camera = get_camera_client()
    lock = get_camera_lock()

    service = VerificationService(recog_pipeline, classify_pipeline)

    with lock:
        frame = camera.capture_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame from camera")
        results = service.verify(frame)

    return VerifyResponse(results=[_serialize_result(res) for res in results])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
