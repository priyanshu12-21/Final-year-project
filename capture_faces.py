import argparse
import os

import cv2

from face_system import FaceSystemError, detect_faces, ensure_face_directories, get_student_dataset_dir, save_student_image


def parse_arguments():
    """Read simple command-line options for image capture."""
    parser = argparse.ArgumentParser(description="Capture face images for one student.")
    parser.add_argument("--student-name", required=True, help="Student name used for the dataset folder.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="How many images to save. The default is 20.",
    )
    return parser.parse_args()


def main():
    """Open the camera and save face images for the selected student."""
    arguments = parse_arguments()
    ensure_face_directories()

    student_folder = get_student_dataset_dir(arguments.student_name)
    os.makedirs(student_folder, exist_ok=True)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    print("Press C to save the current face image.")
    print("Press Q to quit.")
    print(f"Images will be saved in: {student_folder}")

    saved_count = 0

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read a frame from the camera.")
            break

        face_boxes = detect_faces(frame)
        display_frame = frame.copy()

        for x_pos, y_pos, width, height in face_boxes:
            cv2.rectangle(display_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), (255, 255, 0), 2)

        cv2.putText(
            display_frame,
            f"Saved: {saved_count}/{arguments.target_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Capture Student Faces", display_frame)
        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == ord("q"):
            break

        if key_pressed == ord("c"):
            if len(face_boxes) != 1:
                print("Please keep exactly one face in the frame before saving.")
                continue

            success, encoded_frame = cv2.imencode(".jpg", frame)
            if not success:
                print("The frame could not be encoded. Please try again.")
                continue

            try:
                image_path = save_student_image(
                    arguments.student_name,
                    encoded_frame.tobytes(),
                    filename=f"face_{saved_count + 1:02d}.jpg",
                    require_single_face=True,
                )
            except FaceSystemError as exc:
                print(f"Could not save image: {exc}")
                continue

            saved_count += 1
            print(f"Saved image {saved_count}: {image_path}")

            if saved_count >= arguments.target_count:
                print("Target image count reached.")
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
