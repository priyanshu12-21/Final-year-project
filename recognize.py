"""Real-time face recognition using the same pipeline as the Flask app."""

import argparse
from datetime import datetime
import sqlite3

import cv2
import numpy as np

from config import ACTIVE_DATABASE_PATH
from database import get_attendance_rule
from face_system import FaceSystemError, recognize_faces_in_frame, save_face_attendance


DATABASE_PATH = str(ACTIVE_DATABASE_PATH)
FRAME_SKIP = 2


def calculate_attendance_status(class_rule, current_time=None):
    """Mirror the app's attendance timing rules for CLI recognition."""
    current_time = current_time or datetime.now()
    class_start = datetime.strptime(class_rule["class_start_time"], "%H:%M").time()
    class_start_at = datetime.combine(current_time.date(), class_start)
    present_until = class_start_at.timestamp() + (class_rule["present_until_minutes"] * 60)
    late_until = class_start_at.timestamp() + (class_rule["late_until_minutes"] * 60)
    current_timestamp = current_time.timestamp()

    if current_timestamp <= present_until:
        return "Present"
    if current_timestamp <= late_until:
        return "Late"
    return "Very Late"


def parse_args():
    """Read optional camera settings from the command line."""
    parser = argparse.ArgumentParser(description="Real-time face recognition attendance")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index to use, for example 0 or 1")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf", "default"],
        help="OpenCV camera backend to use",
    )
    return parser.parse_args()


def configure_camera(camera):
    """Apply safer webcam settings for smoother and cleaner video on Windows."""
    if hasattr(cv2, "CAP_PROP_FOURCC"):
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    if hasattr(cv2, "CAP_PROP_FRAME_HEIGHT"):
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if hasattr(cv2, "CAP_PROP_FPS"):
        camera.set(cv2.CAP_PROP_FPS, 30)


def camera_frame_is_usable(frame):
    """Basic check to reject empty or corrupted-looking frames."""
    if frame is None or frame.size == 0:
        return False
    return float(np.std(frame)) > 3.0


def get_backend_sequence(backend_name):
    """Convert a simple backend name into OpenCV backend values."""
    if backend_name == "dshow":
        return [("CAP_DSHOW", cv2.CAP_DSHOW)] if hasattr(cv2, "CAP_DSHOW") else []
    if backend_name == "msmf":
        return [("CAP_MSMF", cv2.CAP_MSMF)] if hasattr(cv2, "CAP_MSMF") else []
    if backend_name == "default":
        return [("DEFAULT", None)]

    backends_to_try = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends_to_try.append(("CAP_DSHOW", cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends_to_try.append(("CAP_MSMF", cv2.CAP_MSMF))
    backends_to_try.append(("DEFAULT", None))
    return backends_to_try


def open_camera(preferred_indexes=None, backend_name="auto"):
    """Try camera indexes and OpenCV backends to improve Windows webcam support."""
    backends_to_try = get_backend_sequence(backend_name)
    camera_indexes = preferred_indexes or [0, 1, 2]

    for camera_index in camera_indexes:
        for backend_label, backend_value in backends_to_try:
            camera = cv2.VideoCapture(camera_index) if backend_value is None else cv2.VideoCapture(camera_index, backend_value)
            if not camera.isOpened():
                camera.release()
                continue

            configure_camera(camera)
            ok, test_frame = camera.read()
            if ok and camera_frame_is_usable(test_frame):
                print(f"Camera opened with index {camera_index} using backend: {backend_label}")
                return camera

            print(f"Rejected camera index {camera_index} with backend {backend_label} due to unusable frame.")
            camera.release()

    return None


def main():
    """Start webcam recognition using the shared face pipeline."""
    args = parse_args()
    camera_indexes = [args.camera_index] if args.camera_index is not None else [0, 1, 2]
    camera = open_camera(preferred_indexes=camera_indexes, backend_name=args.backend)

    if camera is None:
        print("Could not open webcam.")
        raise SystemExit(1)

    print("Real-time face recognition started.")
    print("Press Q to quit.")

    attendance_connection = sqlite3.connect(DATABASE_PATH)
    attendance_connection.execute("PRAGMA foreign_keys = ON")
    attendance_connection.row_factory = sqlite3.Row
    attendance_rule = get_attendance_rule(attendance_connection)
    attendance_connection.close()

    frame_count = 0
    latest_results = []
    marked_students = set()

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read frame from webcam.")
            break

        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            try:
                latest_results = recognize_faces_in_frame(frame)
            except FaceSystemError as exc:
                print(f"Recognition failed: {exc}")
                break

        display_frame = frame.copy()
        for result in latest_results:
            x_pos, y_pos, width, height = result["face_box"]

            if result["matched"]:
                attendance_status = calculate_attendance_status(attendance_rule)
                box_color = (0, 255, 0)
                label_text = f"{result['name']} - {attendance_status}"
                if result["student_id"] not in marked_students:
                    attendance_result = save_face_attendance(
                        DATABASE_PATH,
                        result["student_id"],
                        status=attendance_status,
                    )
                    if attendance_result["created"]:
                        marked_students.add(result["student_id"])
                        print(
                            f"Attendance saved for {result['name']} at "
                            f"{attendance_result['time']} with status {attendance_result['status']}."
                        )
            else:
                box_color = (0, 0, 255)
                label_text = "Unknown"

            cv2.rectangle(display_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), box_color, 2)
            cv2.putText(
                display_frame,
                label_text,
                (x_pos, max(y_pos - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
            )

        cv2.imshow("Recognize Faces", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
