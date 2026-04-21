import os
import pickle
import sqlite3
import shutil
import sys
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np

from config import (
    ACTIVE_DATASET_DIR,
    ACTIVE_ENCODINGS_FILE,
    ACTIVE_MODEL_DIR,
    FACE_MATCH_TOLERANCE,
    FACE_RECOGNITION_MODEL_PATH as CONFIG_FACE_RECOGNITION_MODEL_PATH,
    HAAR_CASCADE_PATH as CONFIG_HAAR_CASCADE_PATH,
    SHAPE_PREDICTOR_PATH as CONFIG_SHAPE_PREDICTOR_PATH,
    SUPPORTED_IMAGE_EXTENSIONS,
)

try:
    import dlib
    DLIB_IMPORT_ERROR = None
except Exception as exc:
    dlib = None
    DLIB_IMPORT_ERROR = exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = str(ACTIVE_DATASET_DIR)
ENCODINGS_FILE = str(ACTIVE_ENCODINGS_FILE)
MODEL_DIR = str(ACTIVE_MODEL_DIR)
HAAR_CASCADE_PATH = str(CONFIG_HAAR_CASCADE_PATH)
SHAPE_PREDICTOR_PATH = str(CONFIG_SHAPE_PREDICTOR_PATH)
FACE_RECOGNITION_MODEL_PATH = str(CONFIG_FACE_RECOGNITION_MODEL_PATH)
MIN_FACE_SIZE = 80
MATCH_MARGIN = 0.03


class FaceSystemError(Exception):
    """Base error for the new face recognition system."""


class FaceTrainingError(FaceSystemError):
    """Raised when training cannot build any usable face encodings."""

    def __init__(self, message, skipped_reasons=None):
        super().__init__(message)
        self.skipped_reasons = skipped_reasons or []


def get_face_runtime_status():
    """Report whether native face-recognition dependencies are available."""
    if dlib is None:
        install_hint = "Install a Windows-compatible dlib build, then restart the app." if sys.platform.startswith("win") else "Install dlib, then restart the app."
        return {
            "available": False,
            "message": (
                "Face recognition is unavailable because the 'dlib' package could not be imported. "
                f"{install_hint} Original error: {DLIB_IMPORT_ERROR}"
            ),
        }
    return {"available": True, "message": ""}


def ensure_dlib_available():
    """Raise a user-friendly error when dlib is not installed correctly."""
    status = get_face_runtime_status()
    if not status["available"]:
        raise FaceSystemError(status["message"])


def make_safe_folder_name(student_name):
    """Convert a student name into a simple folder name."""
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in student_name.strip())
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return safe_name.strip("_") or "student"


def get_student_identity(student):
    """Normalize student identity input into an id/name pair."""
    if isinstance(student, dict):
        return student.get("id"), student.get("name")
    if hasattr(student, "keys"):
        return student["id"] if "id" in student.keys() else None, student["name"] if "name" in student.keys() else None
    if isinstance(student, (tuple, list)) and len(student) >= 2:
        return student[0], student[1]
    if isinstance(student, int):
        return student, None
    if isinstance(student, str):
        return None, student
    return None, None


def get_legacy_student_dataset_dir(student_name):
    """Return the original name-based dataset folder used by older versions."""
    return os.path.join(DATASET_DIR, make_safe_folder_name(student_name))


def get_student_dataset_dir(student):
    """Return the primary folder that stores one student's face images."""
    student_id, student_name = get_student_identity(student)
    if student_id is not None:
        return os.path.join(DATASET_DIR, f"student_{int(student_id)}")
    return get_legacy_student_dataset_dir(student_name or "student")


def get_student_dataset_dirs(student):
    """Return the current dataset folder plus any legacy folder for compatibility."""
    student_id, student_name = get_student_identity(student)
    primary_dir = get_student_dataset_dir(student)
    dataset_dirs = [primary_dir]

    if student_name:
        legacy_dir = get_legacy_student_dataset_dir(student_name)
        if legacy_dir not in dataset_dirs:
            dataset_dirs.append(legacy_dir)

    return dataset_dirs


def migrate_legacy_student_dataset(student):
    """Move older name-based dataset files into the new stable student-id folder."""
    student_id, student_name = get_student_identity(student)
    if student_id is None or not student_name:
        return get_student_dataset_dir(student)

    primary_dir = get_student_dataset_dir(student)
    legacy_dir = get_legacy_student_dataset_dir(student_name)

    if os.path.abspath(primary_dir) == os.path.abspath(legacy_dir):
        return primary_dir
    if not os.path.isdir(legacy_dir):
        return primary_dir

    os.makedirs(primary_dir, exist_ok=True)
    for file_name in os.listdir(legacy_dir):
        source_path = os.path.join(legacy_dir, file_name)
        if not os.path.isfile(source_path):
            continue

        target_path = os.path.join(primary_dir, file_name)
        if os.path.exists(target_path):
            continue
        shutil.move(source_path, target_path)

    if os.path.isdir(legacy_dir) and not os.listdir(legacy_dir):
        os.rmdir(legacy_dir)

    return primary_dir


def ensure_face_directories():
    """Create the dataset and model folders if they do not already exist."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def decode_image_bytes(image_bytes):
    """Turn uploaded image bytes into an OpenCV image."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise FaceSystemError("The image could not be read. Please use a clear JPG or PNG file.")
    return frame


def normalize_image_extension(filename):
    """Choose a safe file extension supported by OpenCV and the app."""
    image_extension = os.path.splitext(filename or "capture.jpg")[1].lower() or ".jpg"
    if image_extension not in SUPPORTED_IMAGE_EXTENSIONS:
        return ".jpg"
    return image_extension


def encode_frame_for_storage(frame, image_extension):
    """Re-encode the frame so saved dataset files are always readable."""
    if image_extension in {".jpg", ".jpeg"}:
        success, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif image_extension == ".png":
        success, encoded_image = cv2.imencode(".png", frame)
    elif image_extension == ".bmp":
        success, encoded_image = cv2.imencode(".bmp", frame)
    elif image_extension == ".webp":
        webp_params = [int(cv2.IMWRITE_WEBP_QUALITY), 95] if hasattr(cv2, "IMWRITE_WEBP_QUALITY") else []
        success, encoded_image = cv2.imencode(".webp", frame, webp_params)
    else:
        success, encoded_image = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if not success:
        raise FaceSystemError("The image could not be prepared for saving.")
    return encoded_image.tobytes()


def face_box_area(face_box):
    """Return the detected face area for sorting and filtering."""
    _, _, width, height = face_box
    return int(width) * int(height)


def detect_faces_with_dlib(frame):
    """Detect faces using dlib for better accuracy when the models are available."""
    ensure_dlib_available()
    face_detector, _, _ = get_dlib_models()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = face_detector(rgb_frame, 1)
    return [
        (face.left(), face.top(), face.width(), face.height())
        for face in detections
        if face.width() >= MIN_FACE_SIZE and face.height() >= MIN_FACE_SIZE
    ]


def detect_faces(frame):
    """Detect faces with dlib first and fall back to Haar when needed."""
    try:
        dlib_faces = detect_faces_with_dlib(frame)
    except FaceSystemError:
        dlib_faces = []

    if dlib_faces:
        return sorted(dlib_faces, key=face_box_area, reverse=True)

    haar_faces = detect_faces_with_haar(frame)
    return sorted(haar_faces, key=face_box_area, reverse=True)


def validate_single_face_frame(frame):
    """Ensure the training or recognition image contains exactly one clear face."""
    face_boxes = detect_faces(frame)
    if not face_boxes:
        raise FaceSystemError("No face was detected. Please use a clear photo with one visible face.")
    if len(face_boxes) > 1:
        raise FaceSystemError("Multiple faces were detected. Please keep only one face in the image.")

    _, _, width, height = face_boxes[0]
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        raise FaceSystemError("The detected face is too small. Please move closer to the camera.")

    return face_boxes[0]


def save_student_image(student, image_bytes, filename=None, require_single_face=False):
    """Save one validated image inside the student's dataset folder."""
    ensure_face_directories()
    student_folder = migrate_legacy_student_dataset(student)
    os.makedirs(student_folder, exist_ok=True)

    frame = decode_image_bytes(image_bytes)
    if require_single_face:
        validate_single_face_frame(frame)

    image_extension = normalize_image_extension(filename)
    normalized_bytes = encode_frame_for_storage(frame, image_extension)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    final_filename = f"{timestamp}{image_extension}"
    image_path = os.path.join(student_folder, final_filename)

    with open(image_path, "wb") as image_file:
        image_file.write(normalized_bytes)

    return image_path


def list_student_image_paths(student):
    """Return every saved image for one student."""
    student_folders = get_student_dataset_dirs(student)
    image_paths = []
    seen_paths = set()
    for student_folder in student_folders:
        if not os.path.isdir(student_folder):
            continue

        for file_name in sorted(os.listdir(student_folder)):
            file_extension = os.path.splitext(file_name)[1].lower()
            image_path = os.path.join(student_folder, file_name)
            if file_extension in SUPPORTED_IMAGE_EXTENSIONS and image_path not in seen_paths:
                image_paths.append(image_path)
                seen_paths.add(image_path)
    return image_paths


@lru_cache(maxsize=1)
def get_haar_cascade():
    """Load the Haar Cascade used for fast face detection."""
    cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if cascade.empty():
        raise FaceSystemError("OpenCV Haar Cascade could not be loaded.")
    return cascade


@lru_cache(maxsize=1)
def get_dlib_models():
    """Load the dlib model files needed for face encoding."""
    ensure_dlib_available()
    missing_files = [
        model_path
        for model_path in (SHAPE_PREDICTOR_PATH, FACE_RECOGNITION_MODEL_PATH)
        if not os.path.exists(model_path)
    ]
    if missing_files:
        missing_names = ", ".join(os.path.basename(path) for path in missing_files)
        raise FaceSystemError(
            "Required dlib model files are missing. "
            f"Please place these files inside '{MODEL_DIR}': {missing_names}"
        )

    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    return face_detector, shape_predictor, face_encoder


def detect_faces_with_haar(frame):
    """Detect face boxes using OpenCV."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    cascade = get_haar_cascade()
    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    return list(faces)


def build_dlib_rectangle(face_box):
    """Convert an OpenCV face box into a dlib rectangle."""
    x_pos, y_pos, width, height = face_box
    return dlib.rectangle(
        int(x_pos),
        int(y_pos),
        int(x_pos + width),
        int(y_pos + height),
    )


def encode_face_from_box(frame, face_box):
    """Create one numeric face encoding from a detected face box."""
    _, shape_predictor, face_encoder = get_dlib_models()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_rectangle = build_dlib_rectangle(face_box)
    facial_landmarks = shape_predictor(rgb_frame, face_rectangle)
    face_descriptor = face_encoder.compute_face_descriptor(rgb_frame, facial_landmarks)
    return np.array(face_descriptor, dtype=np.float64)


def load_saved_encodings(encodings_file=ENCODINGS_FILE):
    """Load the trained face encodings from disk."""
    if not os.path.exists(encodings_file):
        raise FaceSystemError("The encodings file was not found. Please run train_model.py first.")

    with open(encodings_file, "rb") as file:
        known_data = pickle.load(file)

    if not known_data.get("encodings"):
        raise FaceSystemError("The encodings file is empty. Please train the model again.")
    if not known_data.get("student_ids"):
        raise FaceSystemError(
            "The encodings file is from an older format. Please train the model again to refresh it."
        )

    return known_data


def find_best_match(known_data, unknown_encoding, tolerance=FACE_MATCH_TOLERANCE):
    """Compare one unknown face encoding against all saved encodings."""
    known_encodings = np.array(known_data["encodings"])
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    best_per_student = {}

    for index, distance in enumerate(distances):
        student_id = known_data["student_ids"][index]
        current_best = best_per_student.get(student_id)
        numeric_distance = float(distance)
        if current_best is None or numeric_distance < current_best["distance"]:
            best_per_student[student_id] = {
                "index": index,
                "distance": numeric_distance,
            }

    ranked_matches = sorted(best_per_student.values(), key=lambda item: item["distance"])
    if not ranked_matches:
        return None, None

    best_match = ranked_matches[0]
    second_best_distance = ranked_matches[1]["distance"] if len(ranked_matches) > 1 else None

    if best_match["distance"] > tolerance:
        return None, best_match["distance"]
    if second_best_distance is not None and (second_best_distance - best_match["distance"]) < MATCH_MARGIN:
        return None, best_match["distance"]

    return best_match["index"], best_match["distance"]


def train_model_from_dataset(conn, student_id=None, encodings_file=ENCODINGS_FILE):
    """Read dataset images, create dlib encodings, and save them to a pickle file."""
    ensure_face_directories()
    get_dlib_models()

    query = "SELECT id, name FROM students"
    parameters = ()
    if student_id is not None:
        query += " WHERE id=?"
        parameters = (student_id,)
    query += " ORDER BY name"

    student_rows = conn.execute(query, parameters).fetchall()
    if not student_rows:
        raise FaceTrainingError("No student records were found for training.")

    known_encodings = []
    known_names = []
    known_student_ids = []
    skipped_reasons = []
    images_scanned = 0

    for student_row in student_rows:
        image_paths = list_student_image_paths(student_row)
        if not image_paths:
            skipped_reasons.append(f"{student_row['name']}: no images found in the dataset folder.")
            continue

        for image_path in image_paths:
            images_scanned += 1
            frame = cv2.imread(image_path)
            if frame is None:
                skipped_reasons.append(f"{os.path.basename(image_path)}: image could not be opened.")
                continue

            face_boxes = detect_faces(frame)
            if len(face_boxes) != 1:
                skipped_reasons.append(
                    f"{os.path.basename(image_path)}: expected 1 face but found {len(face_boxes)}."
                )
                continue

            try:
                face_encoding = encode_face_from_box(frame, face_boxes[0])
            except Exception as exc:
                skipped_reasons.append(f"{os.path.basename(image_path)}: encoding failed ({exc}).")
                continue

            known_encodings.append(face_encoding)
            known_names.append(student_row["name"])
            known_student_ids.append(student_row["id"])

    if not known_encodings:
        raise FaceTrainingError(
            "Training failed because no valid single-face images were found.",
            skipped_reasons=skipped_reasons,
        )

    with open(encodings_file, "wb") as file:
        pickle.dump(
            {
                "encodings": known_encodings,
                "names": known_names,
                "student_ids": known_student_ids,
            },
            file,
        )

    unique_students = len(set(known_student_ids))
    return {
        "images_scanned": images_scanned,
        "images_encoded": len(known_encodings),
        "images_skipped": max(images_scanned - len(known_encodings), 0),
        "students_trained": unique_students,
        "skipped_reasons": skipped_reasons,
    }


def recognize_faces_in_frame(frame, tolerance=FACE_MATCH_TOLERANCE, encodings_file=ENCODINGS_FILE):
    """Recognize every face found in a camera frame."""
    get_dlib_models()
    known_data = load_saved_encodings(encodings_file=encodings_file)
    face_boxes = detect_faces(frame)
    results = []

    for face_box in face_boxes:
        try:
            unknown_encoding = encode_face_from_box(frame, face_box)
            match_index, best_distance = find_best_match(known_data, unknown_encoding, tolerance=tolerance)
        except Exception:
            match_index = None
            best_distance = None

        if match_index is None:
            results.append(
                {
                    "face_box": face_box,
                    "student_id": None,
                    "name": "Unknown",
                    "matched": False,
                    "distance": best_distance,
                }
            )
            continue

        results.append(
            {
                "face_box": face_box,
                "student_id": known_data["student_ids"][match_index],
                "name": known_data["names"][match_index],
                "matched": True,
                "distance": best_distance,
            }
        )

    return results


def recognize_single_face(image_bytes, tolerance=FACE_MATCH_TOLERANCE, encodings_file=ENCODINGS_FILE):
    """Recognize one uploaded image and return a simple result dictionary."""
    frame = decode_image_bytes(image_bytes)
    results = recognize_faces_in_frame(frame, tolerance=tolerance, encodings_file=encodings_file)

    if not results:
        return {"success": False, "status": "no_face", "message": "No face was detected."}
    if len(results) > 1:
        return {"success": False, "status": "multiple_faces", "message": "Please use an image with only one face."}

    first_result = results[0]
    if not first_result["matched"]:
        return {"success": False, "status": "unknown", "message": "The face did not match any student."}

    return {
        "success": True,
        "status": "matched",
        "student_id": first_result["student_id"],
        "name": first_result["name"],
        "distance": first_result["distance"],
    }


def save_face_attendance(db_path, student_id, status="Present", marked_at=None):
    """Store one attendance record for a recognized face."""
    marked_at = marked_at or datetime.now()
    attendance_date = marked_at.strftime("%Y-%m-%d")
    attendance_time = marked_at.strftime("%H:%M:%S")

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    existing_row = connection.execute(
        "SELECT id, status, time FROM attendance WHERE student_id=? AND date=?",
        (student_id, attendance_date),
    ).fetchone()

    if existing_row:
        connection.close()
        return {
            "created": False,
            "date": attendance_date,
            "time": existing_row["time"],
            "status": existing_row["status"],
        }

    connection.execute(
        "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
        (student_id, attendance_date, attendance_time, status),
    )
    connection.commit()
    connection.close()

    return {
        "created": True,
        "date": attendance_date,
        "time": attendance_time,
        "status": status,
    }
