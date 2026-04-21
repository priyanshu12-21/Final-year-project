from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify, send_file
import os
import sqlite3
import csv
import io
import base64
import hashlib
import hmac
import secrets
from functools import wraps
from datetime import date, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    IOT_DEVICE_TOKEN,
    REMOTE_CAMERA_NAME,
    REMOTE_CAMERA_SNAPSHOT_URL,
    REMOTE_CAMERA_STREAM_URL,
    SECRET_KEY,
)
from database import (
    DEFAULT_SUBJECTS,
    GRADE_LEVELS,
    assign_subjects_to_student,
    create_notification,
    ensure_database_ready,
    get_attendance_rule,
    get_attendance_status_for_now,
    get_db,
    update_student_image_path,
)
from face_system import (
    DATASET_DIR,
    FaceSystemError,
    FaceTrainingError,
    get_face_runtime_status,
    get_student_dataset_dirs,
    list_student_image_paths,
    recognize_single_face,
    save_student_image,
    train_model_from_dataset,
)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 390000


def get_remote_camera_config():
    """Expose the optional Raspberry Pi camera settings to templates and routes."""
    is_enabled = bool(REMOTE_CAMERA_STREAM_URL or REMOTE_CAMERA_SNAPSHOT_URL)
    preview_url = REMOTE_CAMERA_STREAM_URL or url_for("remote_camera_snapshot")

    return {
        "enabled": is_enabled,
        "name": REMOTE_CAMERA_NAME,
        "stream_url": REMOTE_CAMERA_STREAM_URL,
        "snapshot_url": REMOTE_CAMERA_SNAPSHOT_URL,
        "preview_url": preview_url if is_enabled else None,
        "has_stream": bool(REMOTE_CAMERA_STREAM_URL),
        "has_snapshot": bool(REMOTE_CAMERA_SNAPSHOT_URL),
    }


def fetch_remote_camera_snapshot():
    """Read one still image from the configured Raspberry Pi snapshot URL."""
    if not REMOTE_CAMERA_SNAPSHOT_URL:
        raise FaceSystemError("Remote camera snapshot URL is not configured.")

    request_headers = {
        "User-Agent": "FaceRecognitionManagementSystem/1.0",
        "Cache-Control": "no-cache",
    }
    remote_request = Request(REMOTE_CAMERA_SNAPSHOT_URL, headers=request_headers)

    try:
        with urlopen(remote_request, timeout=8) as response:
            content_type = response.headers.get("Content-Type", "image/jpeg")
            image_bytes = response.read()
    except HTTPError as exc:
        raise FaceSystemError(f"Remote camera returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise FaceSystemError(f"Could not reach the remote camera: {exc.reason}.") from exc
    except Exception as exc:
        raise FaceSystemError(f"Could not read the remote camera snapshot: {exc}.") from exc

    if not image_bytes:
        raise FaceSystemError("The remote camera snapshot was empty.")

    return image_bytes, content_type


def hash_password(password):
    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            PASSWORD_SCHEME,
            str(PASSWORD_ITERATIONS),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived_key).decode("ascii"),
        ]
    )


def verify_password(stored_password, provided_password):
    if not stored_password:
        return False
    if not stored_password.startswith(f"{PASSWORD_SCHEME}$"):
        return hmac.compare_digest(stored_password, provided_password)

    try:
        _, iteration_text, salt_b64, hash_b64 = stored_password.split("$", 3)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected_hash = base64.b64decode(hash_b64.encode("ascii"))
        derived_key = hashlib.pbkdf2_hmac(
            "sha256",
            provided_password.encode("utf-8"),
            salt,
            int(iteration_text),
        )
    except Exception:
        return False

    return hmac.compare_digest(expected_hash, derived_key)


def upgrade_password_if_needed(conn, table_name, user_id, stored_password, provided_password):
    # Older records may still store plain text passwords.
    # This helper upgrades them the next time the user logs in successfully.
    if stored_password.startswith(f"{PASSWORD_SCHEME}$"):
        return
    if verify_password(stored_password, provided_password):
        conn.execute(
            f"UPDATE {table_name} SET password=? WHERE id=?",
            (hash_password(provided_password), user_id),
        )
        conn.commit()


def get_iot_request_token():
    """Read the token used by Raspberry Pi or ESP32 camera devices."""
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-IoT-Token", "").strip()


def iot_token_is_valid():
    """Validate the configured IoT device token."""
    configured_token = (IOT_DEVICE_TOKEN or "").strip()
    provided_token = get_iot_request_token()
    if not configured_token or configured_token == "change-this-iot-token":
        return False
    return bool(provided_token) and hmac.compare_digest(configured_token, provided_token)


def get_face_dataset_summary(conn):
    """Build a simple summary of how many saved face images each student has."""
    student_rows = conn.execute("SELECT id, name FROM students ORDER BY name").fetchall()
    summary_rows = []

    for student_row in student_rows:
        image_paths = list_student_image_paths(student_row)

        summary_rows.append(
            {
                "student_id": student_row["id"],
                "student_name": student_row["name"],
                "image_count": len(image_paths),
                "last_uploaded": (
                    datetime.fromtimestamp(max(os.path.getmtime(path) for path in image_paths)).strftime("%Y-%m-%d %H:%M:%S")
                    if image_paths
                    else None
                ),
            }
        )

    return [row for row in summary_rows if row["image_count"] > 0]


def get_face_dataset_images(conn, limit=18):
    """Collect the latest saved dataset images for preview and delete actions."""
    student_rows = conn.execute("SELECT id, name FROM students ORDER BY name").fetchall()
    dataset_images = []

    for student_row in student_rows:
        for image_path in list_student_image_paths(student_row):
            file_name = os.path.basename(image_path)

            dataset_images.append(
                {
                    "student_id": student_row["id"],
                    "student_name": student_row["name"],
                    "file_name": file_name,
                    "created_at": datetime.fromtimestamp(os.path.getmtime(image_path)).strftime("%Y-%m-%d %H:%M:%S"),
                    "relative_path": os.path.relpath(image_path, DATASET_DIR).replace("\\", "/"),
                }
            )

    dataset_images.sort(key=lambda image_item: image_item["created_at"], reverse=True)
    return dataset_images[:limit]


def resolve_dataset_image_path(relative_path):
    """Turn a relative dataset path into a safe absolute path inside the dataset folder."""
    absolute_path = os.path.abspath(os.path.join(DATASET_DIR, relative_path))
    dataset_root = os.path.abspath(DATASET_DIR)
    if os.path.commonpath([dataset_root, absolute_path]) != dataset_root:
        raise ValueError("Invalid image path.")
    return absolute_path


def create_pending_fee_notifications(conn, student_id, student_name):
    """Create fee reminder notifications after attendance when pending fees exist."""
    pending_fee_rows = conn.execute(
        """
        SELECT amount, due_date, description
        FROM fees
        WHERE student_id=? AND status='Pending'
        ORDER BY due_date IS NULL, due_date, id
        """,
        (student_id,),
    ).fetchall()

    if not pending_fee_rows:
        return 0

    total_pending_amount = sum(float(row["amount"] or 0) for row in pending_fee_rows)
    earliest_due_date = next((row["due_date"] for row in pending_fee_rows if row["due_date"]), None)
    due_text = f" Earliest due date: {earliest_due_date}." if earliest_due_date else ""

    create_notification(
        conn,
        "teacher",
        "Pending fee reminder",
        (
            f"{student_name} has {len(pending_fee_rows)} pending fee record(s) "
            f"totalling NPR {total_pending_amount:.2f}.{due_text}"
        ),
        student_id,
    )
    create_notification(
        conn,
        "student",
        "Pending fee reminder",
        (
            f"You have {len(pending_fee_rows)} pending fee record(s) "
            f"totalling NPR {total_pending_amount:.2f}.{due_text}"
        ),
        student_id,
    )
    return len(pending_fee_rows)

def build_student_recommendations(att_row, marks_rows, fee_rows):
    recommendations = []
    total_classes = att_row["total"] or 0
    present_classes = att_row["present"] or 0
    attendance_pct = ((present_classes / total_classes) * 100) if total_classes else None

    if attendance_pct is not None and attendance_pct < 80:
        recommendations.append({
            "title": "Improve attendance",
            "message": f"Your attendance is {attendance_pct:.0f}%. Try to stay above 80% for better academic performance."
        })

    subject_scores = {}
    for row in marks_rows:
        if not row["total_marks"]:
            continue
        pct = (row["marks_obtained"] / row["total_marks"]) * 100
        subject_scores.setdefault(row["subject"], []).append(pct)

    weak_subjects = []
    for subject, values in subject_scores.items():
        avg = sum(values) / len(values)
        if avg < 60:
            weak_subjects.append((subject, avg))

    for subject, avg in sorted(weak_subjects, key=lambda item: item[1]):
        recommendations.append({
            "title": f"Focus on {subject}",
            "message": f"Your average in {subject} is {avg:.0f}%. Consider revision, tutoring, or extra practice in this subject."
        })

    pending_fees = [row for row in fee_rows if row["status"] == "Pending"]
    if pending_fees:
        recommendations.append({
            "title": "Clear pending fees",
            "message": f"You have {len(pending_fees)} pending fee record(s). Paying on time helps avoid future issues."
        })

    if not recommendations:
        recommendations.append({
            "title": "Keep going",
            "message": "Your recent attendance and performance look stable. Continue maintaining this progress."
        })

    return recommendations

def build_teacher_student_rows(conn):
    student_rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    student_subject_rows = conn.execute(
        """
        SELECT student_id, subject_name
        FROM student_subjects
        ORDER BY subject_name
        """
    ).fetchall()
    grade_subject_rows = conn.execute(
        """
        SELECT grade, subject_name
        FROM grade_subjects
        ORDER BY grade, subject_name
        """
    ).fetchall()

    subject_map = {}
    for row in student_subject_rows:
        subject_map.setdefault(row["student_id"], []).append(row["subject_name"])

    grade_subject_options = {}
    for row in grade_subject_rows:
        grade_subject_options.setdefault(row["grade"], []).append(row["subject_name"])

    students = []
    for row in student_rows:
        current_subjects = subject_map.get(row["id"], [])
        grade_subjects = grade_subject_options.get(row["year"], [])
        subject_options = sorted(set(current_subjects + grade_subjects + DEFAULT_SUBJECTS))
        students.append({
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "course": row["course"],
            "year": row["year"],
            "phone": row["phone"],
            "current_subjects": current_subjects,
            "subject_options": subject_options,
        })

    return students

def build_grouped_student_options(student_rows):
    grouped = {}
    for row in student_rows:
        grade_value = row["year"] if row["year"] is not None else "-"
        course_value = row["course"] or "No Section"
        label = f"Grade {grade_value} - {course_value}"
        grouped.setdefault(label, []).append(
            {
                "id": row["id"],
                "name": row["name"],
                "year": row["year"],
                "course": row["course"],
            }
        )

    return [
        {"label": label, "students": sorted(rows, key=lambda item: item["name"].lower())}
        for label, rows in sorted(grouped.items(), key=lambda item: item[0])
    ]

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def teacher_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") not in {"teacher", "admin"}:
            flash("Access denied.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

def student_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "student":
            flash("Student access only.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access only.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

def csv_response(filename, header, rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

def forecast_next_percentage(percentages):
    if not percentages:
        return None
    if len(percentages) == 1:
        return round(percentages[0], 1)

    total_x = sum(range(1, len(percentages) + 1))
    total_y = sum(percentages)
    x_mean = total_x / len(percentages)
    y_mean = total_y / len(percentages)

    numerator = 0
    denominator = 0
    for index, value in enumerate(percentages, start=1):
        numerator += (index - x_mean) * (value - y_mean)
        denominator += (index - x_mean) ** 2

    slope = (numerator / denominator) if denominator else 0
    intercept = y_mean - (slope * x_mean)
    prediction = intercept + (slope * (len(percentages) + 1))
    return round(max(0, min(100, prediction)), 1)

def build_marks_analytics(records):
    if not records:
        return {
            "overall_prediction": None,
            "trend_label": "No data",
            "chart_points": [],
            "chart_lines": [],
            "forecast_line": None,
            "y_axis": [0, 25, 50, 75, 100],
            "subject_predictions": [],
        }

    ordered = sorted(records, key=lambda row: row["id"])
    performance = []
    for index, row in enumerate(ordered, start=1):
        pct = round((row["marks_obtained"] / row["total_marks"]) * 100, 1) if row["total_marks"] else 0
        performance.append({
            "index": index,
            "label": f"{row['subject']} - {row['exam_type'] or 'Exam'}",
            "percentage": pct,
        })

    percentages = [item["percentage"] for item in performance]
    overall_prediction = forecast_next_percentage(percentages)
    current_average = round(sum(percentages) / len(percentages), 1)
    trend_delta = round(overall_prediction - percentages[-1], 1) if overall_prediction is not None else 0
    if trend_delta >= 5:
        trend_label = "Improving"
    elif trend_delta <= -5:
        trend_label = "Needs attention"
    else:
        trend_label = "Stable"

    width = 760
    height = 240
    left_pad = 44
    right_pad = 24
    top_pad = 16
    bottom_pad = 34
    usable_width = width - left_pad - right_pad
    usable_height = height - top_pad - bottom_pad
    step_count = max(len(performance), 2)
    step_x = usable_width / (step_count - 1)

    chart_points = []
    for idx, item in enumerate(performance):
        x = left_pad + (idx * step_x if len(performance) > 1 else usable_width / 2)
        y = top_pad + ((100 - item["percentage"]) / 100) * usable_height
        chart_points.append({
            "x": round(x, 1),
            "y": round(y, 1),
            "label": item["label"],
            "percentage": item["percentage"],
            "short_label": str(item["index"]),
        })

    chart_lines = []
    for idx in range(len(chart_points) - 1):
        chart_lines.append({
            "x1": chart_points[idx]["x"],
            "y1": chart_points[idx]["y"],
            "x2": chart_points[idx + 1]["x"],
            "y2": chart_points[idx + 1]["y"],
        })

    forecast_line = None
    if chart_points and overall_prediction is not None:
        last_point = chart_points[-1]
        next_x = left_pad + (len(chart_points) * step_x if len(chart_points) > 1 else usable_width)
        next_x = min(width - right_pad, next_x)
        next_y = top_pad + ((100 - overall_prediction) / 100) * usable_height
        forecast_line = {
            "x1": last_point["x"],
            "y1": last_point["y"],
            "x2": round(next_x, 1),
            "y2": round(next_y, 1),
            "prediction": overall_prediction,
        }

    subject_groups = {}
    for row in ordered:
        pct = round((row["marks_obtained"] / row["total_marks"]) * 100, 1) if row["total_marks"] else 0
        subject_groups.setdefault(row["subject"], []).append(pct)

    subject_predictions = []
    for subject_name, values in sorted(subject_groups.items()):
        predicted = forecast_next_percentage(values)
        average = round(sum(values) / len(values), 1)
        if predicted is not None and predicted >= average + 5:
            subject_trend = "Upward"
        elif predicted is not None and predicted <= average - 5:
            subject_trend = "Downward"
        else:
            subject_trend = "Steady"
        subject_predictions.append({
            "subject": subject_name,
            "exam_count": len(values),
            "average": average,
            "predicted": predicted,
            "trend": subject_trend,
        })

    return {
        "overall_prediction": overall_prediction,
        "current_average": current_average,
        "trend_label": trend_label,
        "chart_points": chart_points,
        "chart_lines": chart_lines,
        "forecast_line": forecast_line,
        "y_axis": [0, 25, 50, 75, 100],
        "subject_predictions": subject_predictions,
    }

# ─── AUTH ───────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]
        conn = get_db()
        if role == "student":
            user = conn.execute("SELECT * FROM students WHERE email=?", (email,)).fetchone()
            table_name = "students"
        elif role == "teacher":
            user = conn.execute("SELECT * FROM teachers WHERE email=?", (email,)).fetchone()
            table_name = "teachers"
        else:
            user = conn.execute("SELECT * FROM admins WHERE email=?", (email,)).fetchone()
            table_name = "admins"
        if user and verify_password(user["password"], password):
            upgrade_password_if_needed(conn, table_name, user["id"], user["password"], password)
            conn.close()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["role"] = role
            return redirect(url_for("dashboard"))
        conn.close()
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]
        password_hash = hash_password(password)
        conn = get_db()
        try:
            if role == "student":
                phone = request.form.get("phone", "")
                course = request.form.get("course", "")
                year = int(request.form.get("year", 1))
                student_image = request.files.get("student_image")
                if year not in GRADE_LEVELS:
                    flash("Grade must be between 1 and 12.", "error")
                    return redirect(url_for("signup"))
                cursor = conn.execute("INSERT INTO students (name, email, password, phone, course, year) VALUES (?,?,?,?,?,?)",
                                      (name, email, password_hash, phone, course, year))
                assign_subjects_to_student(conn, cursor.lastrowid, year)
                student = {"id": cursor.lastrowid, "name": name}

                if student_image and student_image.filename:
                    image_bytes = student_image.read()
                    if not image_bytes:
                        conn.rollback()
                        flash("Student face photo was empty. Please choose a valid image.", "error")
                        return redirect(url_for("signup"))
                    save_student_image(student, image_bytes, student_image.filename, require_single_face=True)
                    update_student_image_path(conn, student)
            elif role == "teacher":
                subject = request.form.get("subject", "")
                assigned_year = request.form.get("assigned_year", "").strip()
                assigned_course = request.form.get("assigned_course", "").strip()
                normalized_year = int(assigned_year) if assigned_year.isdigit() and int(assigned_year) in GRADE_LEVELS else None
                normalized_course = assigned_course or None
                conn.execute(
                    """
                    INSERT INTO teachers (name, email, password, subject, assigned_year, assigned_course)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (name, email, password_hash, subject, normalized_year, normalized_course),
                )
            elif role == "admin":
                conn.execute(
                    "INSERT INTO admins (name, email, password) VALUES (?,?,?)",
                    (name, email, password_hash)
                )
            else:
                flash("Invalid role selected.", "error")
                return redirect(url_for("signup"))
            conn.commit()
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.rollback()
            flash("Email already registered.", "error")
        except FaceSystemError as exc:
            conn.rollback()
            flash(f"Could not save student face photo: {exc}", "error")
        finally:
            conn.close()
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── DASHBOARD ──────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    if session["role"] == "student":
        sid = session["user_id"]
        student = conn.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
        att = conn.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('Present','Late','Very Late') THEN 1 ELSE 0 END) as present
            FROM attendance
            WHERE student_id=?
            """,
            (sid,)
        ).fetchone()
        marks = conn.execute("SELECT * FROM marks WHERE student_id=? ORDER BY id DESC", (sid,)).fetchall()
        fees = conn.execute("SELECT * FROM fees WHERE student_id=?", (sid,)).fetchall()
        recommendations = build_student_recommendations(att, marks, fees)
        notifications = conn.execute(
            """
            SELECT * FROM notifications
            WHERE audience IN ('all', 'student') AND (student_id IS NULL OR student_id=?)
            ORDER BY id DESC
            LIMIT 6
            """,
            (sid,)
        ).fetchall()
        conn.close()
        return render_template(
            "dashboard_student.html",
            student=student,
            att=att,
            marks=marks,
            fees=fees,
            recommendations=recommendations,
            notifications=notifications,
        )
    elif session["role"] == "teacher":
        students_count = conn.execute("SELECT COUNT(*) as c FROM students").fetchone()["c"]
        today_att = conn.execute("SELECT COUNT(*) as c FROM attendance WHERE date=?", (str(date.today()),)).fetchone()["c"]
        pending_fees = conn.execute("SELECT COUNT(*) as c FROM fees WHERE status='Pending'").fetchone()["c"]
        students = build_teacher_student_rows(conn)
        late_students = conn.execute(
            "SELECT COUNT(*) as c FROM attendance WHERE status IN ('Late','Very Late') AND date=?",
            (str(date.today()),)
        ).fetchone()["c"]
        low_performers = conn.execute(
            """
            SELECT s.name, s.year, ROUND(AVG((m.marks_obtained * 100.0) / NULLIF(m.total_marks, 0)), 1) as avg_score
            FROM students s
            JOIN marks m ON m.student_id = s.id
            GROUP BY s.id, s.name, s.year
            HAVING avg_score < 60
            ORDER BY avg_score ASC
            LIMIT 5
            """
        ).fetchall()
        notifications = conn.execute(
            """
            SELECT * FROM notifications
            WHERE audience IN ('all', 'teacher')
            ORDER BY id DESC
            LIMIT 8
            """
        ).fetchall()

        conn.close()
        return render_template(
            "dashboard_teacher.html",
            students_count=students_count,
            today_att=today_att,
            pending_fees=pending_fees,
            students=students,
            late_students=late_students,
            low_performers=low_performers,
            notifications=notifications,
            grade_levels=GRADE_LEVELS,
        )
    else:
        students_count = conn.execute("SELECT COUNT(*) as c FROM students").fetchone()["c"]
        today_att = conn.execute("SELECT COUNT(*) as c FROM attendance WHERE date=?", (str(date.today()),)).fetchone()["c"]
        pending_fees = conn.execute("SELECT COUNT(*) as c FROM fees WHERE status='Pending'").fetchone()["c"]
        students = conn.execute(
            """
            SELECT
                s.id,
                s.name,
                s.email,
                s.phone,
                s.course,
                s.year,
                s.created_at,
                COUNT(DISTINCT ss.subject_name) as subject_count,
                COUNT(DISTINCT f.id) as fee_count
            FROM students s
            LEFT JOIN student_subjects ss ON ss.student_id = s.id
            LEFT JOIN fees f ON f.student_id = s.id
            GROUP BY s.id, s.name, s.email, s.phone, s.course, s.year, s.created_at
            ORDER BY s.name
            """
        ).fetchall()
        teachers = conn.execute(
            "SELECT id, name, email, subject, created_at FROM teachers ORDER BY name"
        ).fetchall()
        admins = conn.execute(
            "SELECT id, name, email, created_at FROM admins ORDER BY name"
        ).fetchall()
        notifications = conn.execute(
            """
            SELECT * FROM notifications
            ORDER BY id DESC
            LIMIT 8
            """
        ).fetchall()
        conn.close()
        return render_template(
            "dashboard_teacher.html",
            students_count=students_count,
            teachers_count=len(teachers),
            admins_count=len(admins),
            today_att=today_att,
            pending_fees=pending_fees,
            students=students,
            teachers=teachers,
            admins=admins,
            notifications=notifications,
        )

# ─── STUDENT PROFILE ────────────────────────────────────

@app.route("/admin/notifications", methods=["POST"])
@login_required
@admin_required
def admin_create_notification():
    audience = request.form.get("audience", "").strip()
    title = request.form.get("title", "").strip()
    message = request.form.get("message", "").strip()
    student_id_value = request.form.get("student_id", "").strip()

    if audience not in {"all", "teacher", "student"}:
        flash("Please choose a valid notification audience.", "error")
        return redirect(url_for("dashboard"))

    if not title or not message:
        flash("Notification title and message are required.", "error")
        return redirect(url_for("dashboard"))

    normalized_student_id = None
    if student_id_value:
        if not student_id_value.isdigit():
            flash("Please choose a valid student for the targeted notification.", "error")
            return redirect(url_for("dashboard"))
        normalized_student_id = int(student_id_value)

    if audience != "student" and normalized_student_id is not None:
        flash("Targeted student notifications must use the Student audience.", "error")
        return redirect(url_for("dashboard"))

    conn = get_db()
    try:
        if normalized_student_id is not None:
            student = conn.execute(
                "SELECT id, name FROM students WHERE id=?",
                (normalized_student_id,),
            ).fetchone()
            if not student:
                flash("Selected student was not found.", "error")
                return redirect(url_for("dashboard"))

        create_notification(conn, audience, title, message, normalized_student_id)
        conn.commit()
    finally:
        conn.close()

    flash("Notification created successfully.", "success")
    return redirect(url_for("dashboard"))

@app.route("/profile")
@login_required
def profile():
    conn = get_db()
    if session["role"] == "student":
        user = conn.execute("SELECT * FROM students WHERE id=?", (session["user_id"],)).fetchone()
    elif session["role"] == "teacher":
        user = conn.execute("SELECT * FROM teachers WHERE id=?", (session["user_id"],)).fetchone()
    else:
        user = conn.execute("SELECT * FROM admins WHERE id=?", (session["user_id"],)).fetchone()
    conn.close()
    return render_template("profile.html", user=user)

@app.route("/teacher-students")
@login_required
@teacher_required
def teacher_students():
    edit_student = request.args.get("edit", "").strip()
    edit_student_id = int(edit_student) if edit_student.isdigit() else None

    conn = get_db()
    students = build_teacher_student_rows(conn)
    conn.close()
    return render_template(
        "teacher_students.html",
        students=students,
        grade_levels=GRADE_LEVELS,
        edit_student_id=edit_student_id,
    )

@app.route("/dashboard/student/<int:student_id>/update", methods=["POST"])
@login_required
@teacher_required
def update_student_dashboard_details(student_id):
    grade_value = request.form.get("year", "").strip()

    if not grade_value.isdigit() or int(grade_value) not in GRADE_LEVELS:
        flash("Please choose a valid grade.", "error")
        return redirect(url_for("dashboard"))

    grade = int(grade_value)
    submitted_subjects = []
    for subject_name in request.form.getlist("subject_names"):
        cleaned = subject_name.strip()
        if cleaned and cleaned not in submitted_subjects:
            submitted_subjects.append(cleaned)

    conn = get_db()
    student = conn.execute(
        "SELECT id, name FROM students WHERE id=?",
        (student_id,)
    ).fetchone()

    if not student:
        conn.close()
        flash("Student not found.", "error")
        return redirect(url_for("dashboard"))

    if not submitted_subjects:
        submitted_subjects = [
            row["subject_name"]
            for row in conn.execute(
                "SELECT subject_name FROM grade_subjects WHERE grade=? ORDER BY subject_name",
                (grade,)
            ).fetchall()
        ]

    conn.execute("UPDATE students SET year=? WHERE id=?", (grade, student_id))
    conn.execute("DELETE FROM student_subjects WHERE student_id=?", (student_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO student_subjects (student_id, subject_name) VALUES (?, ?)",
        [(student_id, subject_name) for subject_name in submitted_subjects]
    )
    conn.commit()
    conn.close()

    flash(f"Updated grade and subjects for {student['name']}.", "success")
    return redirect(url_for("dashboard"))

# ─── ATTENDANCE ─────────────────────────────────────────

@app.route("/attendance")
@login_required
def attendance():
    # This page shows attendance records and the manual attendance form.
    conn = get_db()
    if session["role"] == "student":
        records = conn.execute("SELECT * FROM attendance WHERE student_id=? ORDER BY date DESC", (session["user_id"],)).fetchall()
        conn.close()
        return render_template("attendance_student.html", records=records)
    else:
        student_rows = conn.execute("SELECT * FROM students").fetchall()
        students = [
            {
                "id": row["id"],
                "name": row["name"],
                "year": row["year"],
                "course": row["course"],
            }
            for row in student_rows
        ]
        records = conn.execute("""
            SELECT a.*, s.id as student_id, s.name as student_name, s.year
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.date DESC
        """).fetchall()
        conn.close()
        return render_template(
            "attendance_teacher.html",
            students=students,
            records=records,
            today=str(date.today()),
            current_time=datetime.now().strftime("%H:%M"),
        )

@app.route("/attendance/report/download")
@login_required
@teacher_required
def download_attendance_records_report():
    selected_grade = request.args.get("grade", "").strip()
    selected_student = request.args.get("student_id", "").strip()
    normalized_grade = int(selected_grade) if selected_grade.isdigit() and int(selected_grade) in GRADE_LEVELS else None
    normalized_student = int(selected_student) if selected_student.isdigit() else None

    conn = get_db()
    query = """
        SELECT a.date, a.time, a.status, s.id as student_id, s.name as student_name, s.year, s.course
        FROM attendance a
        JOIN students s ON a.student_id = s.id
    """
    conditions = []
    parameters = []

    if normalized_grade is not None:
        conditions.append("s.year=?")
        parameters.append(normalized_grade)
    if normalized_student is not None:
        conditions.append("s.id=?")
        parameters.append(normalized_student)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY a.date DESC, s.name"

    rows = conn.execute(query, tuple(parameters)).fetchall()
    conn.close()

    data = [
        (
            row["student_id"],
            row["student_name"],
            row["year"],
            row["course"] or "",
            row["date"],
            row["time"] or "",
            row["status"],
        )
        for row in rows
    ]

    file_suffix = []
    if normalized_grade is not None:
        file_suffix.append(f"grade_{normalized_grade}")
    if normalized_student is not None:
        file_suffix.append(f"student_{normalized_student}")
    filename = "attendance_records"
    if file_suffix:
        filename += "_" + "_".join(file_suffix)
    filename += ".csv"

    return csv_response(
        filename,
        ["Student ID", "Student Name", "Grade", "Course", "Date", "Time", "Status"],
        data
    )

@app.route("/attendance/mark", methods=["POST"])
@login_required
@teacher_required
def mark_attendance():
    # This route lets a teacher mark one student's attendance manually.
    student_id = request.form["student_id"]
    att_date = request.form["date"]
    status = request.form["status"]
    marked_time = request.form.get("time", "").strip() or datetime.now().strftime("%H:%M:%S")
    conn = get_db()
    student = conn.execute("SELECT name FROM students WHERE id=?", (student_id,)).fetchone()
    existing = conn.execute("SELECT id FROM attendance WHERE student_id=? AND date=?", (student_id, att_date)).fetchone()
    if existing:
        conn.execute(
            "UPDATE attendance SET status=?, time=? WHERE student_id=? AND date=?",
            (status, marked_time, student_id, att_date),
        )
    else:
        conn.execute(
            "INSERT INTO attendance (student_id, date, time, status) VALUES (?,?,?,?)",
            (student_id, att_date, marked_time, status),
        )
    if status in ("Late", "Very Late", "Absent"):
        create_notification(
            conn,
            "teacher",
            f"{status} attendance alert",
            f"{student['name']} was marked {status.lower()} on {att_date}.",
            int(student_id)
        )
        create_notification(
            conn,
            "student",
            "Attendance updated",
            f"Your attendance for {att_date} was marked as {status}.",
            int(student_id)
        )
    create_pending_fee_notifications(conn, int(student_id), student["name"])
    conn.commit()
    conn.close()
    flash("Attendance marked!", "success")
    return redirect(url_for("attendance"))


@app.route("/attendance/mark-absent", methods=["POST"])
@login_required
@teacher_required
def mark_absent_students():
    conn = get_db()
    att_date = request.form.get("date") or str(date.today())
    students = conn.execute("SELECT id, name FROM students ORDER BY name").fetchall()
    existing_rows = conn.execute(
        "SELECT student_id FROM attendance WHERE date=?",
        (att_date,),
    ).fetchall()
    marked_student_ids = {row["student_id"] for row in existing_rows}

    absent_students = [student for student in students if student["id"] not in marked_student_ids]
    for student in absent_students:
        conn.execute(
            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, 'Absent')",
            (student["id"], att_date, "00:00:00"),
        )
        create_notification(
            conn,
            "teacher",
            "Absent attendance marked",
            f"{student['name']} was automatically marked absent for {att_date}.",
            student["id"],
        )
        create_notification(
            conn,
            "student",
            "Absent attendance alert",
            f"You were marked absent for {att_date}. Please contact your teacher if this is incorrect.",
            student["id"],
        )

    conn.commit()
    conn.close()
    if absent_students:
        flash(f"Marked {len(absent_students)} student(s) absent for {att_date}.", "success")
    else:
        flash(f"All students already have attendance records for {att_date}.", "success")
    return redirect(url_for("attendance"))

@app.route("/recognize", methods=["POST"])
@login_required
def recognize_attendance():
    # This route receives one uploaded photo, recognizes the student, and marks attendance.
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"success": False, "message": "Image file is required."}), 400

    image_bytes = image_file.read()
    try:
        result = recognize_single_face(image_bytes)
    except FaceSystemError as exc:
        return jsonify({"success": False, "status": "system_error", "message": str(exc)}), 500

    if not result["success"]:
        status_code = 404 if result["status"] == "unknown" else 400
        return jsonify(result), status_code

    now_value = datetime.now()
    today = now_value.strftime("%Y-%m-%d")
    current_time = now_value.strftime("%H:%M:%S")
    conn = get_db()
    rule = get_attendance_rule(conn)
    student = None
    if result.get("student_id"):
        student = conn.execute(
            "SELECT id, name FROM students WHERE id=?",
            (result["student_id"],)
        ).fetchone()
    if student is None:
        student = conn.execute(
            "SELECT id, name FROM students WHERE lower(name)=lower(?)",
            (result["name"],)
        ).fetchone()

    if student is None:
        conn.close()
        return jsonify({
            "success": False,
            "status": "student_not_found",
            "message": f"Recognized '{result['name']}', but no matching student exists in the database."
        }), 404

    if session.get("role") == "student":
        conn.close()
        return jsonify({
            "success": False,
            "status": "student_access_removed",
            "message": "Students can no longer mark attendance with face recognition. Please use the attendance view page."
        }), 403

    if session.get("role") == "teacher":
        teacher = conn.execute(
            "SELECT subject, assigned_year, assigned_course FROM teachers WHERE id=?",
            (session.get("user_id"),)
        ).fetchone()
        teacher_subject = (teacher["subject"] if teacher else "").strip()
        teacher_year = teacher["assigned_year"] if teacher else None
        teacher_course = (teacher["assigned_course"] if teacher and teacher["assigned_course"] else "").strip()
        if teacher_year is None or not teacher_course:
            conn.close()
            return jsonify({
                "success": False,
                "status": "class_assignment_missing",
                "message": "Your teacher account is missing an assigned class. Ask the admin to set your grade and section first."
            }), 403

        class_match = conn.execute(
            """
            SELECT 1
            FROM students
            WHERE id=? AND year=? AND lower(COALESCE(course, ''))=lower(?)
            LIMIT 1
            """,
            (student["id"], teacher_year, teacher_course)
        ).fetchone()
        if not class_match:
            conn.close()
            return jsonify({
                "success": False,
                "status": "class_access_denied",
                "message": (
                    f"You can only mark attendance for students in Grade {teacher_year} - "
                    f"{teacher_course}. Your subject is {teacher_subject or 'not set'}."
                )
            }), 403

    existing = conn.execute(
        "SELECT id, status FROM attendance WHERE student_id=? AND date=?",
        (student["id"], today)
    ).fetchone()

    if existing:
        conn.close()
        return jsonify({
            "success": True,
            "status": "already_marked",
            "name": student["name"],
            "attendance_status": existing["status"],
            "recognized": True,
            "time": None,
            "message": f"Attendance already marked for {student['name']} today."
        })

    attendance_status = get_attendance_status_for_now(rule)
    conn.execute(
        "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
        (student["id"], today, current_time, attendance_status)
    )
    create_notification(
        conn,
        "student",
        "Attendance marked automatically",
        f"Your attendance for {today} was marked as {attendance_status} through face recognition.",
        student["id"]
    )
    create_notification(
        conn,
        "teacher",
        "Face attendance marked",
        f"{student['name']} was marked as {attendance_status} through face recognition.",
        student["id"]
    )
    pending_fee_count = create_pending_fee_notifications(conn, student["id"], student["name"])
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "status": "marked",
        "name": student["name"],
        "attendance_status": attendance_status,
        "recognized": True,
        "pending_fee_reminders": pending_fee_count,
        "time": current_time,
        "message": (
            f"Attendance marked successfully for {student['name']} as {attendance_status}."
            + (
                f" Fee reminder generated for {pending_fee_count} pending fee record(s)."
                if pending_fee_count
                else ""
            )
        )
    })

# ─── MARKS ──────────────────────────────────────────────

@app.route("/api/iot/status", methods=["GET"])
def iot_status():
    """Simple readiness endpoint for Raspberry Pi or ESP32 camera devices."""
    if not iot_token_is_valid():
        return jsonify({"success": False, "message": "Unauthorized IoT device."}), 401

    face_runtime_status = get_face_runtime_status()
    encodings_candidates = [
        os.path.join(os.path.dirname(__file__), "encodings.pickle"),
        os.path.join(os.path.dirname(__file__), "data", "encodings.pickle"),
    ]
    return jsonify(
        {
            "success": True,
            "face_runtime_available": face_runtime_status["available"],
            "face_runtime_message": face_runtime_status["message"],
            "encodings_ready": any(os.path.exists(path) for path in encodings_candidates),
        }
    )


@app.route("/api/iot/recognize", methods=["POST"])
def iot_recognize_attendance():
    """Accept a still image from an IoT camera device and mark attendance automatically."""
    if not iot_token_is_valid():
        return jsonify({"success": False, "message": "Unauthorized IoT device."}), 401

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"success": False, "message": "Image file is required."}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"success": False, "message": "The uploaded IoT image was empty."}), 400

    try:
        result = recognize_single_face(image_bytes)
    except FaceSystemError as exc:
        return jsonify({"success": False, "status": "system_error", "message": str(exc)}), 500

    if not result["success"]:
        status_code = 404 if result["status"] == "unknown" else 400
        return jsonify(result), status_code

    now_value = datetime.now()
    today = now_value.strftime("%Y-%m-%d")
    current_time = now_value.strftime("%H:%M:%S")
    device_name = request.form.get("device_name", "").strip() or "IoT camera"

    conn = get_db()
    try:
        rule = get_attendance_rule(conn)
        student = conn.execute(
            "SELECT id, name FROM students WHERE id=?",
            (result["student_id"],),
        ).fetchone() if result.get("student_id") else None

        if student is None:
            student = conn.execute(
                "SELECT id, name FROM students WHERE lower(name)=lower(?)",
                (result["name"],),
            ).fetchone()

        if student is None:
            return jsonify(
                {
                    "success": False,
                    "status": "student_not_found",
                    "message": f"Recognized '{result['name']}', but no matching student exists in the database.",
                }
            ), 404

        existing = conn.execute(
            "SELECT id, status, time FROM attendance WHERE student_id=? AND date=?",
            (student["id"], today),
        ).fetchone()
        if existing:
            return jsonify(
                {
                    "success": True,
                    "status": "already_marked",
                    "name": student["name"],
                    "attendance_status": existing["status"],
                    "recognized": True,
                    "time": existing["time"],
                    "device_name": device_name,
                    "message": f"Attendance was already marked for {student['name']} today.",
                }
            )

        attendance_status = get_attendance_status_for_now(rule)
        conn.execute(
            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
            (student["id"], today, current_time, attendance_status),
        )
        create_notification(
            conn,
            "student",
            "Attendance marked automatically",
            f"Your attendance for {today} was marked as {attendance_status} through {device_name}.",
            student["id"],
        )
        create_notification(
            conn,
            "teacher",
            "IoT attendance marked",
            f"{student['name']} was marked as {attendance_status} through {device_name}.",
            student["id"],
        )
        pending_fee_count = create_pending_fee_notifications(conn, student["id"], student["name"])
        conn.commit()
    finally:
        conn.close()

    return jsonify(
        {
            "success": True,
            "status": "marked",
            "name": student["name"],
            "attendance_status": attendance_status,
            "recognized": True,
            "pending_fee_reminders": pending_fee_count,
            "device_name": device_name,
            "time": current_time,
            "message": (
                f"Attendance marked successfully for {student['name']} as {attendance_status} through {device_name}."
                + (
                    f" Fee reminder generated for {pending_fee_count} pending fee record(s)."
                    if pending_fee_count
                    else ""
                )
            ),
        }
    )


@app.route("/marks")
@login_required
def marks():
    conn = get_db()
    if session["role"] == "student":
        records = conn.execute(
            "SELECT * FROM marks WHERE student_id=? ORDER BY id DESC",
            (session["user_id"],)
        ).fetchall()
        analytics = build_marks_analytics(records)
        conn.close()
        return render_template("marks_student.html", records=records, analytics=analytics)
    else:
        selected_grade = request.args.get("grade", "").strip()
        selected_subject = request.args.get("subject", "").strip()
        selected_student = request.args.get("student_id", "").strip()
        recent_grade = request.args.get("recent_grade", "").strip()
        recent_student = request.args.get("recent_student", "").strip()
        selected_exam_type = request.args.get("exam_type", "Mid Term").strip() or "Mid Term"
        total_marks_input = request.args.get("total_marks", "100").strip() or "100"
        normalized_grade = int(selected_grade) if selected_grade.isdigit() and int(selected_grade) in GRADE_LEVELS else None
        normalized_student = int(selected_student) if selected_student.isdigit() else None
        normalized_recent_grade = int(recent_grade) if recent_grade.isdigit() and int(recent_grade) in GRADE_LEVELS else None
        normalized_recent_student = int(recent_student) if recent_student.isdigit() else None

        students_query = "SELECT * FROM students"
        students_params = []
        if normalized_grade:
            students_query += " WHERE year=?"
            students_params.append(normalized_grade)
            if normalized_student:
                students_query += " AND id=?"
                students_params.append(normalized_student)
        students_query += " ORDER BY name"
        students = conn.execute(students_query, tuple(students_params)).fetchall()
        grade_students = []
        if normalized_grade:
            grade_students = conn.execute(
                "SELECT id, name, course, year FROM students WHERE year=? ORDER BY name",
                (normalized_grade,)
            ).fetchall()
        records_query = """
            SELECT m.*, s.name as student_name, s.year as student_grade
            FROM marks m
            JOIN students s ON m.student_id = s.id
        """
        records_params = []
        if normalized_recent_grade:
            records_query += " WHERE s.year=?"
            records_params.append(normalized_recent_grade)
        records_query += " ORDER BY m.id DESC"
        records = conn.execute(records_query, tuple(records_params)).fetchall()

        recent_grade_students = []
        if normalized_recent_grade:
            recent_grade_students = conn.execute(
                "SELECT id, name, course, year FROM students WHERE year=? ORDER BY name",
                (normalized_recent_grade,)
            ).fetchall()

        recent_student_marks = []
        recent_student_info = None
        if normalized_recent_student:
            recent_student_info = conn.execute(
                "SELECT id, name, course, year FROM students WHERE id=?",
                (normalized_recent_student,)
            ).fetchone()
            recent_student_marks = conn.execute(
                """
                SELECT subject, exam_type, marks_obtained, total_marks
                FROM marks
                WHERE student_id=?
                ORDER BY subject, id DESC
                """,
                (normalized_recent_student,)
            ).fetchall()
        exam_type_rows = conn.execute(
            "SELECT exam_name FROM exam_types ORDER BY id"
        ).fetchall()
        exam_type_options = [row["exam_name"] for row in exam_type_rows]
        if normalized_grade:
            subject_rows = conn.execute(
                "SELECT subject_name FROM grade_subjects WHERE grade=? ORDER BY subject_name",
                (normalized_grade,)
            ).fetchall()
        else:
            subject_rows = conn.execute(
                "SELECT DISTINCT subject_name FROM grade_subjects ORDER BY subject_name"
            ).fetchall()
        subject_options = [row["subject_name"] for row in subject_rows]

        sheet_rows = []
        if selected_subject:
            marks_query = """
                SELECT student_id, marks_obtained, total_marks
                FROM marks
                WHERE subject=? AND exam_type=?
            """
            marks_params = [selected_subject, selected_exam_type]
            if normalized_grade:
                marks_query += " AND student_id IN (SELECT id FROM students WHERE year=?)"
                marks_params.append(normalized_grade)

            existing_marks = conn.execute(marks_query, tuple(marks_params)).fetchall()
            marks_by_student = {row["student_id"]: row for row in existing_marks}

            for student in students:
                existing = marks_by_student.get(student["id"])
                sheet_rows.append({
                    "student_id": student["id"],
                    "student_name": student["name"],
                    "course": student["course"],
                    "year": student["year"],
                    "marks_obtained": existing["marks_obtained"] if existing else "",
                    "total_marks": existing["total_marks"] if existing else total_marks_input,
                })

        conn.close()
        return render_template(
            "marks_teacher.html",
            students=students,
            records=records,
            sheet_rows=sheet_rows,
            grade_levels=GRADE_LEVELS,
            subject_options=subject_options,
            selected_grade=normalized_grade,
            selected_student=normalized_student,
            selected_subject=selected_subject,
            selected_exam_type=selected_exam_type,
            exam_type_options=exam_type_options,
            total_marks_input=total_marks_input,
            grade_students=grade_students,
            recent_grade=normalized_recent_grade,
            recent_student=normalized_recent_student,
            recent_grade_students=recent_grade_students,
            recent_student_marks=recent_student_marks,
            recent_student_info=recent_student_info,
        )

@app.route("/marks/add", methods=["POST"])
@login_required
@teacher_required
def add_marks():
    conn = get_db()
    conn.execute("INSERT INTO marks (student_id, subject, marks_obtained, total_marks, exam_type) VALUES (?,?,?,?,?)",
                 (request.form["student_id"], request.form["subject"],
                  request.form["marks_obtained"], request.form["total_marks"], request.form["exam_type"]))
    conn.commit()
    conn.close()
    flash("Marks added!", "success")
    return redirect(url_for("marks"))

@app.route("/marks/sheet", methods=["POST"])
@login_required
@teacher_required
def save_marks_sheet():
    selected_grade = request.form.get("grade", "").strip()
    subject = request.form.get("subject", "").strip()
    exam_type = request.form.get("exam_type", "").strip()
    default_total_marks = request.form.get("total_marks", "").strip()
    normalized_grade = int(selected_grade) if selected_grade.isdigit() and int(selected_grade) in GRADE_LEVELS else None

    if not subject or not exam_type or not default_total_marks:
        flash("Subject, exam type, and total marks are required.", "error")
        return redirect(url_for("marks", grade=selected_grade, subject=subject, exam_type=exam_type, total_marks=default_total_marks))

    try:
        default_total_marks_value = float(default_total_marks)
    except ValueError:
        flash("Total marks must be a valid number.", "error")
        return redirect(url_for("marks", grade=selected_grade, subject=subject, exam_type=exam_type, total_marks=default_total_marks))

    conn = get_db()
    if normalized_grade:
        subject_exists = conn.execute(
            "SELECT 1 FROM grade_subjects WHERE grade=? AND subject_name=?",
            (normalized_grade, subject),
        ).fetchone()
        if not subject_exists:
            conn.close()
            flash("That subject is not configured for the selected grade.", "error")
            return redirect(url_for("marks", grade=selected_grade, subject=subject, exam_type=exam_type, total_marks=default_total_marks))

    saved_count = 0

    for key, value in request.form.items():
        if not key.startswith("marks_"):
            continue

        student_id = key.replace("marks_", "", 1)
        marks_value = value.strip()
        if marks_value == "":
            continue

        try:
            marks_obtained = float(marks_value)
        except ValueError:
            continue
        if marks_obtained > default_total_marks_value:
            continue

        existing = conn.execute(
            "SELECT id FROM marks WHERE student_id=? AND subject=? AND exam_type=? ORDER BY id DESC LIMIT 1",
            (student_id, subject, exam_type),
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE marks SET marks_obtained=?, total_marks=? WHERE id=?",
                (marks_obtained, default_total_marks_value, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO marks (student_id, subject, marks_obtained, total_marks, exam_type) VALUES (?,?,?,?,?)",
                (student_id, subject, marks_obtained, default_total_marks_value, exam_type),
            )
        if default_total_marks_value and ((marks_obtained / default_total_marks_value) * 100) < 50:
            student = conn.execute("SELECT name FROM students WHERE id=?", (student_id,)).fetchone()
            if student:
                create_notification(
                    conn,
                    "teacher",
                    "Low marks alert",
                    f"{student['name']} scored below 50% in {subject} ({exam_type}).",
                    int(student_id)
                )
                create_notification(
                    conn,
                    "student",
                    "Marks updated",
                    f"Your {subject} marks for {exam_type} were updated. Extra revision is recommended.",
                    int(student_id)
                )
        saved_count += 1

    conn.commit()
    conn.close()
    flash(f"Marks sheet saved for {saved_count} student(s).", "success")
    return redirect(url_for("marks", grade=selected_grade, subject=subject, exam_type=exam_type, total_marks=default_total_marks))

# ─── SUBJECTS ───────────────────────────────────────────

@app.route("/subjects")
@login_required
def subjects():
    conn = get_db()
    if session["role"] == "student":
        student = conn.execute(
            "SELECT * FROM students WHERE id=?",
            (session["user_id"],)
        ).fetchone()
        subject_rows = conn.execute(
            """
            SELECT
                ss.id,
                ss.subject_name,
                COUNT(m.id) as exam_count,
                AVG((m.marks_obtained * 100.0) / NULLIF(m.total_marks, 0)) as average_percentage,
                MAX((m.marks_obtained * 100.0) / NULLIF(m.total_marks, 0)) as best_percentage
            FROM student_subjects ss
            LEFT JOIN marks m
                ON m.subject = ss.subject_name
               AND m.student_id = ?
            WHERE ss.student_id = ?
            GROUP BY ss.id, ss.subject_name
            ORDER BY ss.subject_name
            """,
            (session["user_id"], session["user_id"])
        ).fetchall()
        conn.close()
        return render_template("subjects_student.html", student=student, subjects=subject_rows)
    else:
        selected_grade = request.args.get("grade", "").strip()
        normalized_grade = int(selected_grade) if selected_grade.isdigit() and int(selected_grade) in GRADE_LEVELS else GRADE_LEVELS[0]
        subject_rows = conn.execute(
            """
            SELECT
                gs.id,
                gs.grade,
                gs.subject_name,
                COUNT(DISTINCT s.id) as student_count,
                COUNT(m.id) as marks_entries
            FROM grade_subjects gs
            LEFT JOIN students s ON s.year = gs.grade
            LEFT JOIN marks m ON m.subject = gs.subject_name AND m.student_id = s.id
            WHERE gs.grade = ?
            GROUP BY gs.id, gs.grade, gs.subject_name
            ORDER BY gs.subject_name
            """,
            (normalized_grade,)
        ).fetchall()
        students_in_grade = conn.execute(
            "SELECT COUNT(*) as c FROM students WHERE year=?",
            (normalized_grade,)
        ).fetchone()["c"]
        teacher_rows = conn.execute(
            """
            SELECT id, name, email, subject, created_at
            FROM teachers
            ORDER BY
                CASE
                    WHEN lower(subject) IN (
                        SELECT lower(subject_name)
                        FROM grade_subjects
                        WHERE grade=?
                    ) THEN 0
                    ELSE 1
                END,
                name
            """,
            (normalized_grade,)
        ).fetchall()
        conn.close()
        return render_template(
            "subjects_teacher.html",
            subjects=subject_rows,
            grade_levels=GRADE_LEVELS,
            selected_grade=normalized_grade,
            students_in_grade=students_in_grade,
            default_subjects=DEFAULT_SUBJECTS,
            teachers=teacher_rows,
        )

@app.route("/subjects/add", methods=["POST"])
@login_required
@teacher_required
def add_subject():
    selected_grade = request.form.get("grade", "").strip()
    subject_name = request.form.get("subject_name", "").strip()
    normalized_grade = int(selected_grade) if selected_grade.isdigit() and int(selected_grade) in GRADE_LEVELS else None

    if not normalized_grade or not subject_name:
        flash("Grade and subject name are required.", "error")
        return redirect(url_for("subjects", grade=selected_grade))

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO grade_subjects (grade, subject_name) VALUES (?, ?)",
            (normalized_grade, subject_name)
        )
        student_rows = conn.execute("SELECT id FROM students WHERE year=?", (normalized_grade,)).fetchall()
        conn.executemany(
            "INSERT OR IGNORE INTO student_subjects (student_id, subject_name) VALUES (?, ?)",
            [(row["id"], subject_name) for row in student_rows]
        )
        conn.commit()
        flash("Subject added successfully.", "success")
    except sqlite3.IntegrityError:
        flash("That subject already exists for the selected grade.", "error")
    finally:
        conn.close()

    return redirect(url_for("subjects", grade=normalized_grade))

@app.route("/subjects/delete/<int:subject_id>", methods=["POST"])
@login_required
@teacher_required
def delete_subject(subject_id):
    selected_grade = request.form.get("grade", "").strip()
    conn = get_db()
    subject_row = conn.execute(
        "SELECT grade, subject_name FROM grade_subjects WHERE id=?",
        (subject_id,)
    ).fetchone()
    if subject_row:
        conn.execute("DELETE FROM grade_subjects WHERE id=?", (subject_id,))
        conn.execute(
            """
            DELETE FROM student_subjects
            WHERE subject_name=?
              AND student_id IN (SELECT id FROM students WHERE year=?)
            """,
            (subject_row["subject_name"], subject_row["grade"])
        )
    conn.commit()
    conn.close()
    flash("Subject removed successfully.", "success")
    return redirect(url_for("subjects", grade=selected_grade))

# ─── FEES ───────────────────────────────────────────────

@app.route("/fees")
@login_required
def fees():
    conn = get_db()
    if session["role"] == "student":
        records = conn.execute("SELECT * FROM fees WHERE student_id=?", (session["user_id"],)).fetchall()
        conn.close()
        return render_template("fees_student.html", records=records)
    else:
        students = conn.execute("SELECT * FROM students").fetchall()
        records = conn.execute("""
            SELECT f.*, s.name as student_name FROM fees f
            JOIN students s ON f.student_id = s.id ORDER BY f.due_date
        """).fetchall()
        conn.close()
        return render_template("fees_teacher.html", students=students, records=records)

@app.route("/fees/add", methods=["POST"])
@login_required
@teacher_required
def add_fee():
    conn = get_db()
    student_id = request.form["student_id"]
    amount = request.form["amount"]
    due_date = request.form["due_date"]
    description = request.form["description"]
    conn.execute("INSERT INTO fees (student_id, amount, due_date, status, description) VALUES (?,?,?,?,?)",
                 (student_id, amount, due_date, "Pending", description))
    student = conn.execute("SELECT name FROM students WHERE id=?", (student_id,)).fetchone()
    if student:
        create_notification(
            conn,
            "teacher",
            "Fee reminder created",
            f"Fee record added for {student['name']} with due date {due_date or 'not set'}.",
            int(student_id)
        )
        create_notification(
            conn,
            "student",
            "New fee reminder",
            f"A new fee record of NPR {amount} has been added{f' due on {due_date}' if due_date else ''}.",
            int(student_id)
        )
    conn.commit()
    conn.close()
    flash("Fee record added!", "success")
    return redirect(url_for("fees"))

@app.route("/fees/pay/<int:fee_id>")
@login_required
def pay_fee(fee_id):
    conn = get_db()
    if session.get("role") == "student":
        fee = conn.execute(
            "SELECT id FROM fees WHERE id=? AND student_id=?",
            (fee_id, session["user_id"])
        ).fetchone()
        if not fee:
            conn.close()
            flash("Access denied.", "error")
            return redirect(url_for("fees"))
    fee_row = conn.execute(
        "SELECT student_id, amount, description FROM fees WHERE id=?",
        (fee_id,)
    ).fetchone()
    conn.execute("UPDATE fees SET status='Paid' WHERE id=?", (fee_id,))
    if fee_row:
        create_notification(
            conn,
            "teacher",
            "Fee paid",
            f"Payment received for {fee_row['description'] or 'Fee'} amount NPR {fee_row['amount']}.",
            fee_row["student_id"]
        )
        create_notification(
            conn,
            "student",
            "Fee payment received",
            f"Your payment of NPR {fee_row['amount']} has been marked as paid.",
            fee_row["student_id"]
        )
    conn.commit()
    conn.close()
    flash("Fee marked as paid!", "success")
    return redirect(url_for("fees"))

@app.route("/face-admin")
@login_required
@admin_required
def face_admin():
    # This page helps the teacher manage face images and train the model.
    conn = get_db()
    student_rows = conn.execute("SELECT id, name, year, course FROM students ORDER BY name").fetchall()
    students = [
        {
            "id": row["id"],
            "name": row["name"],
            "year": row["year"],
            "course": row["course"],
        }
        for row in student_rows
    ]
    student_groups = build_grouped_student_options(students)
    attendance_rule = get_attendance_rule(conn)
    dataset_summary = get_face_dataset_summary(conn)
    dataset_images = get_face_dataset_images(conn)
    conn.close()
    face_runtime_status = get_face_runtime_status()
    return render_template(
        "face_admin.html",
        students=students,
        student_groups=student_groups,
        dataset_summary=dataset_summary,
        attendance_rule=attendance_rule,
        dataset_images=dataset_images,
        face_runtime_status=face_runtime_status,
        remote_camera=get_remote_camera_config(),
    )

@app.route("/face-camera")
@login_required
@teacher_required
def face_camera():
    conn = get_db()
    teacher = conn.execute(
        "SELECT subject, assigned_year, assigned_course FROM teachers WHERE id=?",
        (session.get("user_id"),)
    ).fetchone()
    teacher_subject = (teacher["subject"] if teacher and teacher["subject"] else "").strip()
    teacher_year = teacher["assigned_year"] if teacher else None
    teacher_course = (teacher["assigned_course"] if teacher and teacher["assigned_course"] else "").strip()
    allowed_students = conn.execute(
        """
        SELECT id, name, year, course
        FROM students
        WHERE year=? AND lower(COALESCE(course, ''))=lower(?)
        ORDER BY name
        """,
        (teacher_year, teacher_course)
    ).fetchall() if teacher_year is not None and teacher_course else []
    student_groups = build_grouped_student_options(allowed_students)
    conn.close()
    face_runtime_status = get_face_runtime_status()
    return render_template(
        "face_camera.html",
        teacher_subject=teacher_subject,
        teacher_year=teacher_year,
        teacher_course=teacher_course,
        allowed_student_groups=student_groups,
        allowed_students_count=len(allowed_students),
        face_runtime_status=face_runtime_status,
        remote_camera=get_remote_camera_config(),
    )


@app.route("/remote-camera/snapshot")
@login_required
def remote_camera_snapshot():
    """Proxy one still image from the configured Raspberry Pi camera."""
    try:
        image_bytes, content_type = fetch_remote_camera_snapshot()
    except FaceSystemError as exc:
        return jsonify({"success": False, "message": str(exc)}), 502

    return Response(
        image_bytes,
        mimetype=content_type,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )

@app.route("/face-admin/upload", methods=["POST"])
@login_required
@admin_required
def face_admin_upload():
    student_id = request.form.get("student_id", "").strip()
    files = request.files.getlist("images")

    if not student_id or not student_id.isdigit():
        flash("Please select a student.", "error")
        return redirect(url_for("face_admin"))

    valid_files = [file for file in files if file and file.filename]
    if not valid_files:
        flash("Please choose at least one image.", "error")
        return redirect(url_for("face_admin"))

    conn = get_db()
    student = conn.execute(
        "SELECT id, name FROM students WHERE id=?",
        (int(student_id),)
    ).fetchone()
    if not student:
        conn.close()
        flash("Selected student was not found.", "error")
        return redirect(url_for("face_admin"))

    saved_count = 0
    for file in valid_files:
        filename = os.path.basename(file.filename)
        image_bytes = file.read()
        if not filename or not image_bytes:
            continue
        try:
            save_student_image(student, image_bytes, filename, require_single_face=True)
            update_student_image_path(conn, student)
        except FaceSystemError as exc:
            conn.close()
            flash(f"Could not save {filename}: {exc}", "error")
            return redirect(url_for("face_admin"))
        saved_count += 1
    conn.commit()
    conn.close()

    flash(f"{saved_count} image(s) uploaded for {student['name']} into the dataset folder.", "success")
    return redirect(url_for("face_admin"))


@app.route("/face-admin/capture", methods=["POST"])
@login_required
@admin_required
def face_admin_capture():
    # This route saves a webcam photo for one selected student.
    student_id = request.form.get("student_id", "").strip()
    image_file = request.files.get("image")

    if not student_id or not student_id.isdigit():
        return jsonify({"success": False, "message": "Please select a student before saving a webcam photo."}), 400

    if image_file is None:
        return jsonify({"success": False, "message": "No webcam image was received."}), 400

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"success": False, "message": "The captured image was empty. Please try again."}), 400

    conn = get_db()
    student = conn.execute(
        "SELECT id, name FROM students WHERE id=?",
        (int(student_id),)
    ).fetchone()
    if not student:
        conn.close()
        return jsonify({"success": False, "message": "Selected student was not found."}), 404

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"webcam_{student['id']}_{timestamp}.jpg"
    try:
        save_student_image(student, image_bytes, filename, require_single_face=True)
        update_student_image_path(conn, student)
    except FaceSystemError as exc:
        conn.close()
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        conn.close()
        return jsonify({"success": False, "message": f"Server error while saving webcam image: {exc}"}), 500
    conn.commit()
    conn.close()

    return jsonify(
        {
            "success": True,
            "message": f"Webcam photo saved for {student['name']}.",
            "student_name": student["name"],
            "file_name": filename,
        }
    )

@app.route("/face-admin/train", methods=["POST"])
@login_required
@admin_required
def face_admin_train():
    # This route builds the encodings.pickle file from the saved dataset images.
    selected_student_id = request.form.get("student_id", "").strip()
    normalized_student_id = int(selected_student_id) if selected_student_id.isdigit() else None
    student_name = None
    conn = get_db()

    if normalized_student_id is not None:
        student_row = conn.execute(
            "SELECT name FROM students WHERE id=?",
            (normalized_student_id,)
        ).fetchone()
        if not student_row:
            conn.close()
            flash("Selected student was not found.", "error")
            return redirect(url_for("face_admin"))
        student_name = student_row["name"]

    try:
        result = train_model_from_dataset(conn, student_id=normalized_student_id)
        flash_message = (
            f"Training complete. Encoded {result['images_encoded']} image(s) for "
            f"{result['students_trained']} student(s) from {result['images_scanned']} saved image(s)."
        )
        if student_name:
            flash_message = f"Training complete for {student_name}. " + flash_message[len("Training complete. "):]
        if result["images_skipped"]:
            flash_message += f" Skipped {result['images_skipped']} invalid or unusable image(s)."
        flash(flash_message, "success")
        for reason in result["skipped_reasons"][:5]:
            flash(f"Skipped image: {reason}", "error")
    except FaceTrainingError as exc:
        flash(str(exc), "error")
        for reason in exc.skipped_reasons[:5]:
            flash(f"Skipped image: {reason}", "error")
    except Exception as exc:
        flash(f"Training failed: {exc}", "error")
    finally:
        conn.close()
    return redirect(url_for("face_admin"))

@app.route("/face-admin/image")
@login_required
@admin_required
def face_admin_image():
    # This route shows one saved dataset image in the browser preview.
    relative_path = request.args.get("path", "").strip()
    if not relative_path:
        return ("Image path is required.", 400)

    try:
        image_path = resolve_dataset_image_path(relative_path)
    except ValueError:
        return ("Invalid image path.", 400)

    if not os.path.exists(image_path):
        return ("Image not found.", 404)

    return send_file(image_path)


@app.route("/face-admin/delete", methods=["POST"])
@login_required
@admin_required
def face_admin_delete():
    # This route deletes one dataset image from the student's folder.
    relative_path = request.form.get("path", "").strip()
    if not relative_path:
        flash("Image path is required.", "error")
        return redirect(url_for("face_admin"))

    try:
        image_path = resolve_dataset_image_path(relative_path)
    except ValueError:
        flash("Invalid image path.", "error")
        return redirect(url_for("face_admin"))

    if not os.path.exists(image_path):
        flash("Image not found.", "error")
        return redirect(url_for("face_admin"))

    deleted_name = os.path.basename(image_path)
    os.remove(image_path)
    conn = get_db()
    try:
        deleted_dir = os.path.dirname(image_path)
        student_rows = conn.execute("SELECT id, name FROM students").fetchall()
        for student_row in student_rows:
            candidate_dirs = {
                os.path.abspath(dataset_dir)
                for dataset_dir in get_student_dataset_dirs(student_row)
            }
            if os.path.abspath(deleted_dir) in candidate_dirs:
                update_student_image_path(conn, student_row)
                conn.commit()
                break
    finally:
        conn.close()
    flash(f"Deleted dataset image {deleted_name}.", "success")
    return redirect(url_for("face_admin"))

@app.route("/face-admin/attendance-rule", methods=["POST"])
@login_required
@admin_required
def update_attendance_rule():
    class_start_time = request.form.get("class_start_time", "").strip()
    present_until_minutes = request.form.get("present_until_minutes", "").strip()
    late_until_minutes = request.form.get("late_until_minutes", "").strip()

    try:
        datetime.strptime(class_start_time, "%H:%M")
        present_minutes = int(present_until_minutes)
        late_minutes = int(late_until_minutes)
    except ValueError:
        flash("Please enter a valid class start time and minute limits.", "error")
        return redirect(url_for("face_admin"))

    if present_minutes < 0 or late_minutes < present_minutes:
        flash("Late limit must be greater than or equal to present limit.", "error")
        return redirect(url_for("face_admin"))

    conn = get_db()
    conn.execute(
        """
        UPDATE attendance_rules
        SET class_start_time=?, present_until_minutes=?, late_until_minutes=?
        WHERE id=1
        """,
        (class_start_time, present_minutes, late_minutes)
    )
    conn.commit()
    conn.close()
    flash("Attendance timing rules updated.", "success")
    return redirect(url_for("face_admin"))

@app.route("/student/report/download")
@login_required
@student_required
def download_student_report():
    student_id = session["user_id"]
    conn = get_db()

    student = conn.execute(
        "SELECT id, name, email, course, year FROM students WHERE id=?",
        (student_id,)
    ).fetchone()
    marks_rows = conn.execute(
        """
        SELECT subject, exam_type, marks_obtained, total_marks
        FROM marks
        WHERE student_id=?
        ORDER BY subject, id DESC
        """,
        (student_id,)
    ).fetchall()
    attendance_rows = conn.execute(
        """
        SELECT date, status
        FROM attendance
        WHERE student_id=?
        ORDER BY date DESC
        """,
        (student_id,)
    ).fetchall()
    fee_rows = conn.execute(
        """
        SELECT description, amount, due_date, status
        FROM fees
        WHERE student_id=?
        ORDER BY due_date
        """,
        (student_id,)
    ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Student Report"])
    writer.writerow(["Student ID", student["id"]])
    writer.writerow(["Name", student["name"]])
    writer.writerow(["Email", student["email"]])
    writer.writerow(["Course", student["course"] or "-"])
    writer.writerow(["Grade", student["year"] or "-"])
    writer.writerow([])

    writer.writerow(["Marks"])
    writer.writerow(["Subject", "Exam Type", "Marks Obtained", "Total Marks"])
    if marks_rows:
        for row in marks_rows:
            writer.writerow([row["subject"], row["exam_type"] or "-", row["marks_obtained"], row["total_marks"]])
    else:
        writer.writerow(["No marks records found"])
    writer.writerow([])

    writer.writerow(["Attendance"])
    writer.writerow(["Date", "Status"])
    if attendance_rows:
        for row in attendance_rows:
            writer.writerow([row["date"], row["status"]])
    else:
        writer.writerow(["No attendance records found"])
    writer.writerow([])

    writer.writerow(["Fees"])
    writer.writerow(["Description", "Amount", "Due Date", "Status"])
    if fee_rows:
        for row in fee_rows:
            writer.writerow([row["description"] or "Fee", row["amount"], row["due_date"] or "-", row["status"]])
    else:
        writer.writerow(["No fee records found"])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=student_report_{student_id}.csv"}
    )

@app.route("/student/report/marks")
@login_required
@student_required
def download_marks_report():
    student_id = session["user_id"]
    conn = get_db()
    student = conn.execute(
        "SELECT id, name FROM students WHERE id=?",
        (student_id,)
    ).fetchone()
    rows = conn.execute(
        """
        SELECT subject, exam_type, marks_obtained, total_marks
        FROM marks
        WHERE student_id=?
        ORDER BY subject, id DESC
        """,
        (student_id,)
    ).fetchall()
    conn.close()
    data = [
        (
            student["id"],
            student["name"],
            row["subject"],
            row["exam_type"] or "-",
            row["marks_obtained"],
            row["total_marks"],
        )
        for row in rows
    ]
    return csv_response(
        f"marks_{student_id}.csv",
        ["Student ID", "Student Name", "Subject", "Exam Type", "Marks Obtained", "Total Marks"],
        data
    )

@app.route("/student/report/attendance")
@login_required
@student_required
def download_attendance_report():
    student_id = session["user_id"]
    conn = get_db()
    student = conn.execute(
        "SELECT id, name FROM students WHERE id=?",
        (student_id,)
    ).fetchone()
    rows = conn.execute(
        """
        SELECT date, status
        FROM attendance
        WHERE student_id=?
        ORDER BY date DESC
        """,
        (student_id,)
    ).fetchall()
    conn.close()
    data = [
        (
            student["id"],
            student["name"],
            row["date"],
            row["status"],
        )
        for row in rows
    ]
    return csv_response(
        f"attendance_{student_id}.csv",
        ["Student ID", "Student Name", "Date", "Status"],
        data
    )

@app.route("/student/report/fees")
@login_required
@student_required
def download_fees_report():
    student_id = session["user_id"]
    conn = get_db()
    student = conn.execute(
        "SELECT id, name FROM students WHERE id=?",
        (student_id,)
    ).fetchone()
    rows = conn.execute(
        """
        SELECT description, amount, due_date, status
        FROM fees
        WHERE student_id=?
        ORDER BY due_date
        """,
        (student_id,)
    ).fetchall()
    conn.close()
    data = [
        (
            student["id"],
            student["name"],
            row["description"] or "Fee",
            row["amount"],
            row["due_date"] or "-",
            row["status"],
        )
        for row in rows
    ]
    return csv_response(
        f"fees_{student_id}.csv",
        ["Student ID", "Student Name", "Description", "Amount", "Due Date", "Status"],
        data
    )

@app.route("/reports")
@login_required
@teacher_required
def reports():
    conn = get_db()
    grade = request.args.get("grade", "").strip()
    normalized_grade = int(grade) if grade.isdigit() and int(grade) in GRADE_LEVELS else None

    students_query = "SELECT id, name, year FROM students"
    students_params = []
    if normalized_grade:
        students_query += " WHERE year=?"
        students_params.append(normalized_grade)
    students_query += " ORDER BY name"
    students = conn.execute(students_query, tuple(students_params)).fetchall()

    attendance_summary_query = """
        SELECT s.name, s.year,
               COUNT(a.id) as total_classes,
               SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) as present_count,
               SUM(CASE WHEN a.status='Late' THEN 1 ELSE 0 END) as late_count
        FROM students s
        LEFT JOIN attendance a ON a.student_id = s.id
    """
    marks_summary_query = """
        SELECT s.name, s.year,
               ROUND(AVG((m.marks_obtained * 100.0) / NULLIF(m.total_marks, 0)), 1) as avg_marks
        FROM students s
        LEFT JOIN marks m ON m.student_id = s.id
    """
    fees_summary_query = """
        SELECT s.name, s.year,
               SUM(CASE WHEN f.status='Pending' THEN f.amount ELSE 0 END) as pending_amount
        FROM students s
        LEFT JOIN fees f ON f.student_id = s.id
    """
    if normalized_grade:
        attendance_summary_query += " WHERE s.year=?"
        marks_summary_query += " WHERE s.year=?"
        fees_summary_query += " WHERE s.year=?"
    attendance_summary_query += " GROUP BY s.id, s.name, s.year ORDER BY s.name"
    marks_summary_query += " GROUP BY s.id, s.name, s.year ORDER BY s.name"
    fees_summary_query += " GROUP BY s.id, s.name, s.year ORDER BY s.name"

    attendance_summary = conn.execute(attendance_summary_query, ((normalized_grade,) if normalized_grade else ())).fetchall()
    marks_summary = conn.execute(marks_summary_query, ((normalized_grade,) if normalized_grade else ())).fetchall()
    fees_summary = conn.execute(fees_summary_query, ((normalized_grade,) if normalized_grade else ())).fetchall()
    conn.close()

    return render_template(
        "reports_teacher.html",
        grade_levels=GRADE_LEVELS,
        selected_grade=normalized_grade,
        students=students,
        attendance_summary=attendance_summary,
        marks_summary=marks_summary,
        fees_summary=fees_summary,
    )

# ─── ABOUT ──────────────────────────────────────────────

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    ensure_database_ready()
    # Allow easy switching between normal HTTP and local HTTPS for camera access.
    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "5000"))
    use_https = os.environ.get("APP_USE_HTTPS", "").lower() in {"1", "true", "yes", "on"}
    ssl_context = "adhoc" if use_https else None

    app.run(
        host=host,
        port=port,
        debug=True,
        use_reloader=False,
        ssl_context=ssl_context,
    )
