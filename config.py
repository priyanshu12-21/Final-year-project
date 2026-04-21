"""Central configuration for the Face Recognition Management System."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV may be unavailable during import checks
    cv2 = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "face_models"
DATASET_DIR = DATA_DIR / "dataset"
ENCODINGS_FILE = DATA_DIR / "encodings.pickle"
DATABASE_PATH = DATA_DIR / "database.db"
UPLOAD_FOLDER = DATA_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

LEGACY_MODEL_DIR = BASE_DIR / "face_models"
LEGACY_DATASET_DIR = BASE_DIR / "dataset"
LEGACY_ENCODINGS_FILE = BASE_DIR / "encodings.pickle"
LEGACY_DATABASE_PATH = BASE_DIR / "database.db"


def _prefer_existing(primary: Path, legacy: Path) -> Path:
    """Use the new structured location when present, otherwise fall back safely."""
    if primary.exists():
        return primary
    return legacy


ACTIVE_MODEL_DIR = _prefer_existing(MODEL_DIR, LEGACY_MODEL_DIR)
ACTIVE_DATASET_DIR = _prefer_existing(DATASET_DIR, LEGACY_DATASET_DIR)
ACTIVE_ENCODINGS_FILE = _prefer_existing(ENCODINGS_FILE, LEGACY_ENCODINGS_FILE)
ACTIVE_DATABASE_PATH = _prefer_existing(DATABASE_PATH, LEGACY_DATABASE_PATH)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_MATCH_TOLERANCE = 0.5

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")
IOT_DEVICE_TOKEN = os.environ.get("IOT_DEVICE_TOKEN", "change-this-iot-token")
CLASS_START_TIME = "10:00"
PRESENT_UNTIL_MINUTES = 10
LATE_UNTIL_MINUTES = 20


def _normalize_remote_camera_url(value: str | None) -> str | None:
    """Keep only safe absolute HTTP(S) URLs for remote camera integration."""
    normalized = (value or "").strip()
    if not normalized:
        return None

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return normalized


REMOTE_CAMERA_NAME = os.environ.get("REMOTE_CAMERA_NAME", "Raspberry Pi Camera").strip() or "Raspberry Pi Camera"
REMOTE_CAMERA_STREAM_URL = _normalize_remote_camera_url(os.environ.get("REMOTE_CAMERA_STREAM_URL"))
REMOTE_CAMERA_SNAPSHOT_URL = _normalize_remote_camera_url(os.environ.get("REMOTE_CAMERA_SNAPSHOT_URL"))

HAAR_CASCADE_PATH = (
    Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if cv2 is not None
    else Path("haarcascade_frontalface_default.xml")
)
SHAPE_PREDICTOR_PATH = ACTIVE_MODEL_DIR / "shape_predictor_5_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = ACTIVE_MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat"
