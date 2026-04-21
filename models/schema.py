"""SQLite schema for the school management system with face recognition."""

from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "database.db"

GRADE_LEVELS = list(range(1, 13))
DEFAULT_SUBJECTS = ["Math", "Science", "Social", "English", "Nepali", "Computer"]
DEFAULT_EXAM_TYPES = ["Mid Term", "Final", "Unit Test", "Assignment", "Practical"]
SCHEMA_NOTES = [
    (
        "students",
        "table",
        "Stores student login, profile, class, and face dataset path information.",
        "id, email, year, course",
        "Student identity, hashed password, contact details, grade, section, and image dataset folder path.",
    ),
    (
        "teachers",
        "table",
        "Stores teacher login, assigned subject, and assigned class details.",
        "id, email, subject, assigned_year, assigned_course",
        "Teacher profile, hashed password, assigned subject, assigned grade and section, and creation timestamp.",
    ),
    (
        "admins",
        "table",
        "Stores administrator login records.",
        "id, email",
        "Admin identity, hashed password, and creation timestamp.",
    ),
    (
        "attendance",
        "table",
        "Stores daily attendance entries produced manually or by face recognition.",
        "id, student_id, date",
        "Attendance date, time, and status for each student.",
    ),
    (
        "marks",
        "table",
        "Stores assessment results per student and subject.",
        "id, student_id, subject, exam_type",
        "Obtained marks, total marks, exam type, and created timestamp.",
    ),
    (
        "fees",
        "table",
        "Stores fee dues and payment states for students.",
        "id, student_id, due_date, status",
        "Fee amount, due date, paid or pending status, and description.",
    ),
    (
        "grade_levels",
        "table",
        "Lookup table for supported school grades from 1 to 12.",
        "grade",
        "One row per valid grade level.",
    ),
    (
        "grade_subjects",
        "table",
        "Maps each grade level to its default subjects.",
        "id, grade, subject_name",
        "Subject catalogue by grade with one unique row per grade-subject pair.",
    ),
    (
        "exam_types",
        "table",
        "Lookup table for allowed exam labels.",
        "id, exam_name",
        "Mid Term, Final, Unit Test, Assignment, Practical, and future labels.",
    ),
    (
        "student_subjects",
        "table",
        "Maps individual students to their enrolled subjects.",
        "id, student_id, subject_name",
        "Subject allocation per student.",
    ),
    (
        "notifications",
        "table",
        "Stores announcements for all users or targeted students.",
        "id, audience, student_id",
        "Notification title, body, audience, optional student target, and created timestamp.",
    ),
    (
        "attendance_rules",
        "table",
        "Stores timing rules used to classify attendance as Present, Late, or Very Late.",
        "id",
        "Single settings row with class start time and minute cutoffs.",
    ),
    (
        "schema_notes",
        "table",
        "Stores documentation notes for tables and views for viva explanation.",
        "object_name, object_type",
        "Object description, key fields, and the type of data saved there.",
    ),
    (
        "student_directory",
        "view",
        "Readable student listing for browsing registered students.",
        "student_id",
        "Student name, email, phone, section, grade, image path, and registration time.",
    ),
    (
        "teacher_directory",
        "view",
        "Readable teacher listing for browsing teacher accounts.",
        "teacher_id",
        "Teacher name, email, assigned subject, and registration time.",
    ),
    (
        "attendance_records",
        "view",
        "Readable attendance history joined with student details.",
        "record_id, student_id, attendance_date",
        "Student name, grade, section, date, time, and attendance status.",
    ),
    (
        "marks_records",
        "view",
        "Readable marks history joined with student details and percentage.",
        "record_id, student_id, subject_name",
        "Student name, grade, section, subject, exam type, marks, total, and percentage.",
    ),
    (
        "fee_records",
        "view",
        "Readable fee history joined with student details.",
        "record_id, student_id, due_date",
        "Student identity, fee amount, due date, description, and user-friendly payment status.",
    ),
    (
        "grade_subject_map",
        "view",
        "Readable mapping between grades and their default subjects.",
        "record_id, grade",
        "One row per grade-subject combination.",
    ),
    (
        "student_subject_map",
        "view",
        "Readable mapping between students and their subjects.",
        "record_id, student_id",
        "Student name, grade, section, and subject assignment.",
    ),
    (
        "notification_log",
        "view",
        "Readable notification history with optional student names.",
        "notification_id",
        "Audience, title, message, related student, and created time.",
    ),
    (
        "attendance_settings",
        "view",
        "Readable view of the active attendance timing configuration.",
        "rule_id",
        "Class start time and present/late minute thresholds.",
    ),
]


def initialize_database(conn: sqlite3.Connection) -> None:
    """Create or refresh the complete SQLite schema and default seed data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            phone TEXT,
            course TEXT,
            year INTEGER NOT NULL DEFAULT 1 CHECK (year BETWEEN 1 AND 12),
            image_path TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            subject TEXT NOT NULL,
            assigned_year INTEGER,
            assigned_course TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Absent'
                CHECK (status IN ('Present', 'Absent', 'Late', 'Very Late')),
            FOREIGN KEY (student_id) REFERENCES students(id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            marks_obtained REAL NOT NULL,
            total_marks REAL NOT NULL DEFAULT 100 CHECK (total_marks >= 0),
            exam_type TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS fees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending'
                CHECK (status IN ('Pending', 'Paid')),
            description TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS grade_levels (
            grade INTEGER PRIMARY KEY CHECK (grade BETWEEN 1 AND 12)
        );

        CREATE TABLE IF NOT EXISTS grade_subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grade INTEGER NOT NULL,
            subject_name TEXT NOT NULL,
            UNIQUE (grade, subject_name),
            FOREIGN KEY (grade) REFERENCES grade_levels(grade)
                ON UPDATE CASCADE
                ON DELETE CASCADE
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
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audience TEXT NOT NULL CHECK (audience IN ('all', 'student', 'teacher')),
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            student_id INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
                ON UPDATE CASCADE
                ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS attendance_rules (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            class_start_time TEXT NOT NULL DEFAULT '10:00',
            present_until_minutes INTEGER NOT NULL DEFAULT 10 CHECK (present_until_minutes >= 0),
            late_until_minutes INTEGER NOT NULL DEFAULT 20 CHECK (late_until_minutes >= present_until_minutes)
        );

        CREATE TABLE IF NOT EXISTS schema_notes (
            object_name TEXT PRIMARY KEY,
            object_type TEXT NOT NULL CHECK (object_type IN ('table', 'view')),
            description TEXT NOT NULL,
            key_fields TEXT,
            saved_data TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_attendance_student_date
        ON attendance(student_id, date);

        CREATE INDEX IF NOT EXISTS idx_marks_student_subject
        ON marks(student_id, subject);

        CREATE INDEX IF NOT EXISTS idx_fees_student_status
        ON fees(student_id, status);

        CREATE INDEX IF NOT EXISTS idx_student_subjects_student
        ON student_subjects(student_id);

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
            a.time AS attendance_time,
            a.status AS attendance_status
        FROM attendance AS a
        JOIN students AS s ON s.id = a.student_id;

        DROP VIEW IF EXISTS marks_records;
        CREATE VIEW marks_records AS
        SELECT
            m.id AS record_id,
            m.student_id,
            s.name AS student_name,
            s.year AS grade,
            s.course,
            m.subject AS subject_name,
            m.marks_obtained,
            m.total_marks,
            m.exam_type,
            ROUND((m.marks_obtained * 100.0) / NULLIF(m.total_marks, 0), 2) AS percentage,
            m.created_at
        FROM marks AS m
        JOIN students AS s ON s.id = m.student_id;

        DROP VIEW IF EXISTS fee_records;
        CREATE VIEW fee_records AS
        SELECT
            f.id AS record_id,
            f.student_id,
            s.name AS student_name,
            s.year AS grade,
            s.course,
            f.amount,
            f.due_date,
            CASE WHEN f.status = 'Pending' THEN 'Unpaid' ELSE f.status END AS fee_status,
            f.description
        FROM fees AS f
        JOIN students AS s ON s.id = f.student_id;

        DROP VIEW IF EXISTS grade_subject_map;
        CREATE VIEW grade_subject_map AS
        SELECT
            gs.id AS record_id,
            gs.grade,
            gs.subject_name
        FROM grade_subjects AS gs;

        DROP VIEW IF EXISTS student_subject_map;
        CREATE VIEW student_subject_map AS
        SELECT
            ss.id AS record_id,
            ss.student_id,
            s.name AS student_name,
            s.year AS grade,
            s.course,
            ss.subject_name
        FROM student_subjects AS ss
        JOIN students AS s ON s.id = ss.student_id;

        DROP VIEW IF EXISTS notification_log;
        CREATE VIEW notification_log AS
        SELECT
            n.id AS notification_id,
            n.audience,
            n.title,
            n.message,
            n.student_id,
            s.name AS student_name,
            n.created_at
        FROM notifications AS n
        LEFT JOIN students AS s ON s.id = n.student_id;

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
        [(exam_name,) for exam_name in DEFAULT_EXAM_TYPES],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO attendance_rules (
            id,
            class_start_time,
            present_until_minutes,
            late_until_minutes
        )
        VALUES (1, '10:00', 10, 20)
        """
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO schema_notes (
            object_name,
            object_type,
            description,
            key_fields,
            saved_data
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        SCHEMA_NOTES,
    )
    conn.commit()


def get_connection() -> sqlite3.Connection:
    """Return a ready-to-use connection to data/database.db."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    initialize_database(conn)
    return conn
