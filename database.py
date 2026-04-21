"""Database helpers and schema management for the Face Recognition Management System."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from config import (
    ACTIVE_DATABASE_PATH,
    CLASS_START_TIME,
    LATE_UNTIL_MINUTES,
    PRESENT_UNTIL_MINUTES,
)
from face_system import ensure_face_directories, get_student_dataset_dir, list_student_image_paths, save_student_image


DB = str(ACTIVE_DATABASE_PATH)
GRADE_LEVELS = list(range(1, 13))
DEFAULT_SUBJECTS = ["Math", "Science", "Social", "English", "Nepali", "Computer"]
DEFAULT_EXAM_TYPES = ["Mid Term", "Final", "Unit Test", "Assignment", "Practical"]


def table_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def table_exists(conn, table_name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_column(conn, table_name, column_name, definition):
    columns = table_columns(conn, table_name)
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def migrate_legacy_schema(conn):
    # Add any columns that older database versions may be missing.
    student_columns = table_columns(conn, "students")
    if "phone" not in student_columns:
        ensure_column(conn, "students", "phone", "TEXT")
    if "course" not in student_columns:
        ensure_column(conn, "students", "course", "TEXT")
    if "year" not in student_columns:
        ensure_column(conn, "students", "year", "INTEGER DEFAULT 1")
    if "created_at" not in student_columns:
        ensure_column(conn, "students", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if "image_path" not in student_columns:
        ensure_column(conn, "students", "image_path", "TEXT")

    teacher_columns = table_columns(conn, "teachers")
    if "subject" not in teacher_columns:
        ensure_column(conn, "teachers", "subject", "TEXT")
    if "assigned_year" not in teacher_columns:
        ensure_column(conn, "teachers", "assigned_year", "INTEGER")
    if "assigned_course" not in teacher_columns:
        ensure_column(conn, "teachers", "assigned_course", "TEXT")
    if "created_at" not in teacher_columns:
        ensure_column(conn, "teachers", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    marks_columns = table_columns(conn, "marks")
    if "marks_obtained" not in marks_columns:
        ensure_column(conn, "marks", "marks_obtained", "REAL")
    if "total_marks" not in marks_columns:
        ensure_column(conn, "marks", "total_marks", "REAL DEFAULT 100")
    if "exam_type" not in marks_columns:
        ensure_column(conn, "marks", "exam_type", "TEXT")
    if "created_at" not in marks_columns:
        ensure_column(conn, "marks", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    marks_columns = table_columns(conn, "marks")
    if "marks" in marks_columns:
        conn.execute(
            """
            UPDATE marks
            SET marks_obtained = COALESCE(marks_obtained, marks),
                total_marks = COALESCE(total_marks, 100)
            WHERE marks_obtained IS NULL OR total_marks IS NULL
            """
        )

    fees_columns = table_columns(conn, "fees")
    if "amount" not in fees_columns:
        ensure_column(conn, "fees", "amount", "REAL")
    if "due_date" not in fees_columns:
        ensure_column(conn, "fees", "due_date", "TEXT")
    if "status" not in fees_columns:
        ensure_column(conn, "fees", "status", "TEXT DEFAULT 'Pending'")
    if "description" not in fees_columns:
        ensure_column(conn, "fees", "description", "TEXT")

    fees_columns = table_columns(conn, "fees")
    if "total_fee" in fees_columns:
        conn.execute("UPDATE fees SET amount = COALESCE(amount, total_fee) WHERE amount IS NULL")
    if "due" in fees_columns:
        conn.execute("UPDATE fees SET due_date = COALESCE(due_date, due) WHERE due_date IS NULL")
    if "paid" in fees_columns:
        conn.execute(
            """
            UPDATE fees
            SET status = CASE
                WHEN paid IN (1, '1', 'true', 'True', 'paid', 'Paid') THEN 'Paid'
                ELSE COALESCE(status, 'Pending')
            END
            WHERE status IS NULL OR status = ''
            """
        )

    conn.execute("UPDATE marks SET total_marks = 100 WHERE total_marks IS NULL")
    conn.execute("UPDATE fees SET status = 'Pending' WHERE status IS NULL OR status = ''")
    conn.execute("UPDATE students SET year = 1 WHERE year IS NULL")

    attendance_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='attendance'"
    ).fetchone()
    if attendance_schema and attendance_schema["sql"] and "Very Late" not in attendance_schema["sql"]:
        conn.executescript(
            """
            ALTER TABLE attendance RENAME TO attendance_old;

            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT,
                status TEXT CHECK(status IN ('Present','Absent','Late','Very Late')) DEFAULT 'Absent',
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            INSERT INTO attendance (id, student_id, date, time, status)
            SELECT id, student_id, date, NULL, status
            FROM attendance_old;

            DROP TABLE attendance_old;
            """
        )

    attendance_columns = table_columns(conn, "attendance")
    if "time" not in attendance_columns:
        ensure_column(conn, "attendance", "time", "TEXT")


def update_student_image_path(conn, student):
    """Save or clear the dataset folder path in the student record based on actual saved images."""
    image_folder = get_student_dataset_dir(student) if list_student_image_paths(student) else None
    conn.execute("UPDATE students SET image_path=? WHERE id=?", (image_folder, student["id"]))


def migrate_legacy_face_dataset(conn):
    """
    Move old face photos out of the legacy database table and into dataset folders.
    After migration, the old table is removed so the new system stays clean.
    """
    if not table_exists(conn, "face_datasets"):
        return

    legacy_rows = conn.execute(
        """
        SELECT student_id, student_name, file_name, image_data
        FROM face_datasets
        ORDER BY id
        """
    ).fetchall()

    for legacy_row in legacy_rows:
        if not legacy_row["image_data"]:
            continue
        save_student_image(legacy_row, legacy_row["image_data"], legacy_row["file_name"])
        conn.execute(
            "UPDATE students SET image_path=? WHERE id=?",
            (get_student_dataset_dir(legacy_row), legacy_row["student_id"]),
        )

    conn.execute("DROP TABLE IF EXISTS face_datasets")


def assign_subjects_to_student(conn, student_id, grade):
    subject_rows = conn.execute(
        "SELECT subject_name FROM grade_subjects WHERE grade=? ORDER BY subject_name",
        (grade,),
    ).fetchall()
    conn.executemany(
        "INSERT OR IGNORE INTO student_subjects (student_id, subject_name) VALUES (?, ?)",
        [(student_id, row["subject_name"]) for row in subject_rows],
    )


def sync_student_subjects(conn):
    student_rows = conn.execute("SELECT id, year FROM students").fetchall()
    for row in student_rows:
        assign_subjects_to_student(conn, row["id"], row["year"])


def get_attendance_rule(conn):
    rule = conn.execute(
        """
        SELECT class_start_time, present_until_minutes, late_until_minutes
        FROM attendance_rules
        WHERE id=1
        """
    ).fetchone()
    if rule:
        return rule
    conn.execute(
        """
        INSERT OR IGNORE INTO attendance_rules (id, class_start_time, present_until_minutes, late_until_minutes)
        VALUES (?, ?, ?, ?)
        """,
        (1, CLASS_START_TIME, PRESENT_UNTIL_MINUTES, LATE_UNTIL_MINUTES),
    )
    conn.commit()
    return conn.execute(
        """
        SELECT class_start_time, present_until_minutes, late_until_minutes
        FROM attendance_rules
        WHERE id=1
        """
    ).fetchone()


def get_attendance_status_for_now(rule, now_value=None):
    now_value = now_value or datetime.now()
    class_start = datetime.strptime(rule["class_start_time"], "%H:%M").time()
    start_dt = datetime.combine(now_value.date(), class_start)
    present_limit = start_dt + timedelta(minutes=rule["present_until_minutes"])
    late_limit = start_dt + timedelta(minutes=rule["late_until_minutes"])

    if now_value <= present_limit:
        return "Present"
    if now_value <= late_limit:
        return "Late"
    return "Very Late"


def create_notification(conn, audience, title, message, student_id=None):
    conn.execute(
        """
        INSERT INTO notifications (audience, title, message, student_id)
        VALUES (?, ?, ?, ?)
        """,
        (audience, title, message, student_id),
    )


def initialize_database(conn):
    # This function creates the project tables if they do not exist yet.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT,
            course TEXT,
            year INTEGER DEFAULT 1,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            subject TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT,
            status TEXT CHECK(status IN ('Present','Absent','Late','Very Late')) DEFAULT 'Absent',
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            marks_obtained REAL NOT NULL,
            total_marks REAL NOT NULL,
            exam_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS fees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            due_date TEXT,
            status TEXT CHECK(status IN ('Pending','Paid')) DEFAULT 'Pending',
            description TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS grade_levels (
            grade INTEGER PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS grade_subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grade INTEGER NOT NULL,
            subject_name TEXT NOT NULL,
            UNIQUE (grade, subject_name),
            FOREIGN KEY (grade) REFERENCES grade_levels(grade)
        );

        CREATE TABLE IF NOT EXISTS exam_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS student_subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_name TEXT NOT NULL,
            UNIQUE (student_id, subject_name),
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audience TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            student_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS attendance_rules (
            id INTEGER PRIMARY KEY,
            class_start_time TEXT NOT NULL DEFAULT '10:00',
            present_until_minutes INTEGER NOT NULL DEFAULT 10,
            late_until_minutes INTEGER NOT NULL DEFAULT 20
        );

        CREATE TABLE IF NOT EXISTS schema_notes (
            object_name TEXT PRIMARY KEY,
            object_type TEXT NOT NULL,
            description TEXT NOT NULL,
            key_fields TEXT,
            saved_data TEXT
        );
        """
    )

    # Upgrade legacy tables before creating indexes and views that depend on
    # the newer columns. This keeps older database files compatible.
    migrate_legacy_schema(conn)
    migrate_legacy_face_dataset(conn)
    sync_student_subjects(conn)

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date);
        CREATE INDEX IF NOT EXISTS idx_marks_student_subject ON marks(student_id, subject);
        CREATE INDEX IF NOT EXISTS idx_fees_student_status ON fees(student_id, status);
        CREATE INDEX IF NOT EXISTS idx_student_subjects_student ON student_subjects(student_id);

        DROP VIEW IF EXISTS student_directory;
        CREATE VIEW student_directory AS
        SELECT
            id AS student_id,
            name AS student_name,
            email,
            phone,
            course,
            year AS grade,
            image_path,
            created_at AS registered_at
        FROM students;

        DROP VIEW IF EXISTS teacher_directory;
        CREATE VIEW teacher_directory AS
        SELECT
            id AS teacher_id,
            name AS teacher_name,
            email,
            subject AS assigned_subject,
            created_at AS registered_at
        FROM teachers;

        DROP VIEW IF EXISTS attendance_records;
        CREATE VIEW attendance_records AS
        SELECT
            a.id AS record_id,
            a.student_id,
            s.name AS student_name,
            s.year AS grade,
            s.course,
            a.date AS attendance_date,
            COALESCE(a.time, '-') AS attendance_time,
            a.status AS attendance_status
        FROM attendance a
        JOIN students s ON s.id = a.student_id;

        DROP VIEW IF EXISTS marks_records;
        CREATE VIEW marks_records AS
        SELECT
            m.id AS record_id,
            m.student_id,
            s.name AS student_name,
            s.year AS grade,
            s.course,
            m.subject AS subject_name,
            COALESCE(m.exam_type, '-') AS exam_type,
            m.marks_obtained,
            m.total_marks,
            ROUND((m.marks_obtained * 100.0) / NULLIF(m.total_marks, 0), 2) AS percentage,
            m.created_at AS saved_at
        FROM marks m
        JOIN students s ON s.id = m.student_id;

        DROP VIEW IF EXISTS fee_records;
        CREATE VIEW fee_records AS
        SELECT
            f.id AS record_id,
            f.student_id,
            s.name AS student_name,
            s.year AS grade,
            s.course,
            COALESCE(f.description, 'Fee Payment') AS fee_title,
            f.amount AS amount_npr,
            f.due_date,
            CASE
                WHEN f.status = 'Pending' THEN 'Unpaid'
                ELSE f.status
            END AS fee_status
        FROM fees f
        JOIN students s ON s.id = f.student_id;

        DROP VIEW IF EXISTS grade_subject_map;
        CREATE VIEW grade_subject_map AS
        SELECT
            gs.id AS record_id,
            gs.grade,
            gs.subject_name
        FROM grade_subjects gs;

        DROP VIEW IF EXISTS student_subject_map;
        CREATE VIEW student_subject_map AS
        SELECT
            ss.id AS record_id,
            ss.student_id,
            s.name AS student_name,
            s.year AS grade,
            ss.subject_name
        FROM student_subjects ss
        JOIN students s ON s.id = ss.student_id;

        DROP VIEW IF EXISTS notification_log;
        CREATE VIEW notification_log AS
        SELECT
            n.id AS notification_id,
            n.audience,
            COALESCE(s.name, 'General') AS student_name,
            n.title,
            n.message,
            n.created_at
        FROM notifications n
        LEFT JOIN students s ON s.id = n.student_id;

        DROP VIEW IF EXISTS attendance_settings;
        CREATE VIEW attendance_settings AS
        SELECT
            id AS rule_id,
            class_start_time,
            present_until_minutes,
            late_until_minutes
        FROM attendance_rules;
        """
    )

    conn.executemany(
        "INSERT OR IGNORE INTO grade_levels (grade) VALUES (?)",
        [(grade,) for grade in GRADE_LEVELS],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO grade_subjects (grade, subject_name) VALUES (?, ?)",
        [(grade, subject) for grade in GRADE_LEVELS for subject in DEFAULT_SUBJECTS],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO exam_types (exam_name) VALUES (?)",
        [(exam_type,) for exam_type in DEFAULT_EXAM_TYPES],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO attendance_rules (id, class_start_time, present_until_minutes, late_until_minutes)
        VALUES (1, '10:00', 10, 20)
        """
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO schema_notes (object_name, object_type, description, key_fields, saved_data)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("students", "table", "Main student login and profile records.", "id, email, year", "Student name, login email, password, grade, contact details, and the dataset folder path in image_path."),
            ("teachers", "table", "Teacher login and profile records.", "id, email", "Teacher name, login email, password, assigned subject."),
            ("admins", "table", "Admin login and management records.", "id, email", "Admin name, login email, password, and created date."),
            ("attendance", "table", "Raw attendance entries saved by manual marking or face recognition.", "id, student_id, date", "Student attendance date, marked time, and status for each day."),
            ("marks", "table", "Raw exam and marks entries for students.", "id, student_id, subject", "Subject marks, exam type, total marks, saved time."),
            ("fees", "table", "Fee payment records for each student.", "id, student_id, due_date", "Amount, due date, fee status, fee description."),
            ("grade_subjects", "table", "Subjects assigned to each grade.", "id, grade, subject_name", "Default subject list for every grade."),
            ("student_subjects", "table", "Subject list assigned to each student.", "id, student_id, subject_name", "Personal subject mapping per student."),
            ("exam_types", "table", "Exam type options used in marks entry.", "id, exam_name", "Mid Term, Final, Unit Test and other exam labels."),
            ("attendance_rules", "table", "Time rules used to mark Present, Late, and Very Late.", "id", "Class start time and attendance cutoffs."),
            ("notifications", "table", "Internal app notifications for teacher and student dashboards.", "id, audience, student_id", "Notification title, message, target audience."),
            ("student_directory", "view", "Clean student list for easy database browsing.", "student_id", "Student identity, grade, contact, registration date, and image folder."),
            ("attendance_records", "view", "Readable attendance view with student names and grade.", "record_id", "Attendance date, time, and status joined with student details."),
            ("marks_records", "view", "Readable marks view with student names and percentage.", "record_id", "Subject marks, exam type, percentage, saved time."),
            ("fee_records", "view", "Readable fee view with student names and clear status labels.", "record_id", "Fee title, amount in NPR, due date, paid or unpaid status."),
            ("student_subject_map", "view", "Readable student-subject mapping.", "record_id", "Student-wise subject assignments."),
            ("grade_subject_map", "view", "Readable grade-subject mapping.", "record_id", "Grade-wise subject assignments."),
            ("notification_log", "view", "Readable notification history.", "notification_id", "Notification audience, student name, title, message, created time."),
            ("attendance_settings", "view", "Readable attendance timing configuration.", "rule_id", "Class start time and present/late minute limits."),
        ],
    )
    ensure_face_directories()
    conn.commit()


def get_db():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    initialize_database(conn)
    return conn


def ensure_database_ready():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    initialize_database(conn)
    conn.close()
