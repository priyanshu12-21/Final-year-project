3"""Quick readiness check for database and face-recognition dependencies."""

from __future__ import annotations

import pickle
from pathlib import Path

from config import (
    ACTIVE_DATABASE_PATH,
    ACTIVE_ENCODINGS_FILE,
    ACTIVE_MODEL_DIR,
    FACE_RECOGNITION_MODEL_PATH,
    SHAPE_PREDICTOR_PATH,
)
from database import ensure_database_ready, get_db
from face_system import get_face_runtime_status


def describe_path(path_value) -> tuple[bool, str]:
    path = Path(path_value)
    if not path.exists():
        return False, f"Missing: {path}"
    if path.is_dir():
        return True, f"Found directory: {path}"
    return True, f"Found: {path} ({path.stat().st_size} bytes)"


def main() -> int:
    print("System readiness check")
    print("-" * 24)

    face_status = get_face_runtime_status()
    print(f"dlib import status : {'OK' if face_status['available'] else 'FAILED'}")
    if face_status["message"]:
        print(f"  {face_status['message']}")

    for label, path_value in [
        ("database file", ACTIVE_DATABASE_PATH),
        ("model folder", ACTIVE_MODEL_DIR),
        ("shape predictor", SHAPE_PREDICTOR_PATH),
        ("face encoder model", FACE_RECOGNITION_MODEL_PATH),
        ("encodings file", ACTIVE_ENCODINGS_FILE),
    ]:
        ok, message = describe_path(path_value)
        print(f"{label:18}: {'OK' if ok else 'FAILED'}")
        print(f"  {message}")

    ensure_database_ready()
    conn = get_db()
    try:
        student_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        attendance_count = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        grade_count = conn.execute("SELECT COUNT(*) FROM grade_levels").fetchone()[0]
        subject_count = conn.execute("SELECT COUNT(*) FROM grade_subjects").fetchone()[0]
        print("database schema     : OK")
        print(
            f"  students={student_count}, attendance={attendance_count}, "
            f"grades={grade_count}, grade_subjects={subject_count}"
        )
    finally:
        conn.close()

    encodings_path = Path(ACTIVE_ENCODINGS_FILE)
    if encodings_path.exists():
        try:
            with encodings_path.open("rb") as file:
                known_data = pickle.load(file)
            encoding_count = len(known_data.get("encodings", []))
            student_count = len(set(known_data.get("student_ids", [])))
            print("trained encodings   : OK")
            print(f"  encodings={encoding_count}, students={student_count}")
        except Exception as exc:
            print("trained encodings   : FAILED")
            print(f"  Could not read encodings file: {exc}")
    else:
        print("trained encodings   : WARNING")
        print("  No encodings file yet. Train the model after collecting student images.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
