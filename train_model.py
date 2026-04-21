"""Train face encodings using the shared application database and dataset paths."""

from database import get_db
from face_system import ENCODINGS_FILE, FaceTrainingError, train_model_from_dataset


def main():
    """Build the shared encodings file used by the web app."""
    conn = get_db()
    try:
        result = train_model_from_dataset(conn)
    except FaceTrainingError as exc:
        print(str(exc))
        for reason in exc.skipped_reasons[:10]:
            print(f"[SKIP] {reason}")
        raise SystemExit(1)
    finally:
        conn.close()

    print("Training completed successfully.")
    print(f"Images scanned : {result['images_scanned']}")
    print(f"Images encoded : {result['images_encoded']}")
    print(f"Images skipped : {result['images_skipped']}")
    print(f"Students trained : {result['students_trained']}")
    print(f"Saved encodings : {ENCODINGS_FILE}")

    for reason in result["skipped_reasons"][:10]:
        print(f"[SKIP] {reason}")


if __name__ == "__main__":
    main()
