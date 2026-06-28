"""
Face Recognition Attendance System
A complete production-quality desktop application built with PySide6.
Single-file application with modern UI, animations, and computer vision.
"""

import sys
import os
import cv2
import sqlite3
import pickle
import hashlib
import logging
import threading
import io
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# Force qtawesome to bind to PySide6 BEFORE importing it
os.environ.setdefault('QT_API', 'pyside6')
import qtawesome as qta
from qt_material import apply_stylesheet

from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize, QPoint, QRect, QRectF,
    QDate, QTime, QDateTime, QAbstractTableModel, QModelIndex, QThread,
    Signal, QObject, QSortFilterProxyModel, QRegularExpression
)
from PySide6.QtGui import (
    QFont, QPixmap, QImage, QPainter, QColor, QIcon, QLinearGradient,
    QBrush, QPen, QFontDatabase, QAction, QPalette, QCursor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QStackedWidget,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QDateEdit, QFileDialog, QMessageBox, QDialog, QFormLayout,
    QDialogButtonBox, QScrollArea, QProgressBar, QGraphicsDropShadowEffect,
    QSpinBox, QCheckBox, QSizePolicy, QAbstractItemView, QMenu,
    QSystemTrayIcon, QToolTip, QTabWidget, QTextEdit, QSplashScreen,
    QGroupBox, QSlider, QListWidget, QListWidgetItem, QStatusBar
)

# Try to import face_recognition (optional, gracefully degrade)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

# OpenCV LBPH face recognizer — works without dlib/face_recognition
try:
    LBPH_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
    LBPH_AVAILABLE = True
except Exception:
    LBPH_AVAILABLE = False

# Global LBPH recognizer state (used across instances)
class FaceRecognizerStore:
    """Shared OpenCV LBPH recognizer instance for the running app."""
    recognizer = None
    label_map = {}      # int -> (student_id, full_name)
    trained = False

    @classmethod
    def get(cls):
        if cls.recognizer is None and LBPH_AVAILABLE:
            cls.recognizer = cv2.face.LBPHFaceRecognizer_create()
        return cls.recognizer

    @classmethod
    def predict(cls, gray_face):
        """Return (student_id, name, confidence) or (None, None, None)."""
        rec = cls.get()
        if rec is None or not cls.trained:
            return None, None, None
        try:
            label, conf = rec.predict(gray_face)
            if label in cls.label_map and conf < 70:  # LBPH: lower is better, 70 is a strict threshold
                sid, name = cls.label_map[label]
                # Convert LBPH distance to a 0-100 confidence.
                # Genuine matches usually have distance 25-55. Map so that
                # distance 30 -> 100%, distance 70 -> 30%, with smooth falloff.
                if conf <= 30:
                    confidence = 100.0
                else:
                    confidence = max(0.0, min(100.0, 100.0 * (70.0 - float(conf)) / 40.0))
                return sid, name, confidence
            return None, None, conf
        except Exception:
            return None, None, None

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ----------------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FaceAttendance')

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
APP_NAME = "Face Recognition Attendance System"
APP_VERSION = "1.0.0"
APP_DIR = Path(__file__).parent.resolve()
DB_PATH = APP_DIR / "attendance.db"
FACES_DIR = APP_DIR / "faces"
EXPORTS_DIR = APP_DIR / "exports"
ICONS_DIR = APP_DIR / "icons"

for d in (FACES_DIR, EXPORTS_DIR, ICONS_DIR):
    d.mkdir(exist_ok=True)

# Color palette
COLORS = {
    'primary': '#5B6CFF',
    'primary_dark': '#3D4DB7',
    'secondary': '#00C9A7',
    'accent': '#FF6B9D',
    'warning': '#FFB946',
    'danger': '#FF5470',
    'success': '#00C9A7',
    'info': '#3D9CF2',
    'bg_dark': '#0F1419',
    'bg_medium': '#161B26',
    'bg_light': '#1E2533',
    'surface': '#252B3A',
    'text_primary': '#FFFFFF',
    'text_secondary': '#A4B0C5',
    'text_muted': '#6B7896',
    'border': '#2D3548',
}

LIGHT_COLORS = {
    'primary': '#5B6CFF',
    'primary_dark': '#3D4DB7',
    'bg_dark': '#F4F6FB',
    'bg_medium': '#FFFFFF',
    'bg_light': '#FFFFFF',
    'surface': '#F9FAFD',
    'text_primary': '#1A1F36',
    'text_secondary': '#4A5570',
    'text_muted': '#8B95AB',
    'border': '#E1E5EE',
}

# ----------------------------------------------------------------------------
# Database Layer
# ----------------------------------------------------------------------------
class Database:
    """SQLite database manager for the attendance system."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_db()
        self.seed_admin()

    def get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    department TEXT,
                    level TEXT,
                    phone TEXT,
                    email TEXT,
                    image_path TEXT,
                    face_encoding BLOB,
                    registered_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL,
                    FOREIGN KEY(student_id) REFERENCES students(student_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    type TEXT NOT NULL,
                    read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def seed_admin(self):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM admins")
            if cur.fetchone()[0] == 0:
                pwd = self.hash_password("admin123")
                cur.execute(
                    "INSERT INTO admins (username, password_hash, created_at) VALUES (?, ?, ?)",
                    ("admin", pwd, datetime.now().isoformat())
                )
                conn.commit()

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def verify_admin(self, username: str, password: str) -> bool:
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM admins WHERE username = ?", (username,))
            row = cur.fetchone()
            if not row:
                return False
            return row['password_hash'] == self.hash_password(password)

    # Students CRUD
    def add_student(self, data: dict, encoding=None) -> bool:
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                enc_blob = pickle.dumps(encoding) if encoding is not None else None
                cur.execute("""
                    INSERT INTO students (student_id, full_name, department, level, phone, email, image_path, face_encoding, registered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['student_id'], data['full_name'], data.get('department', ''),
                    data.get('level', ''), data.get('phone', ''), data.get('email', ''),
                    data.get('image_path', ''), enc_blob, datetime.now().isoformat()
                ))
                conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Student add failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Student add error: {e}")
            return False

    def update_student(self, student_pk: int, data: dict, encoding=None):
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                if encoding is not None:
                    enc_blob = pickle.dumps(encoding)
                    cur.execute("""
                        UPDATE students SET full_name=?, department=?, level=?, phone=?, email=?, face_encoding=?
                        WHERE id=?
                    """, (data['full_name'], data.get('department', ''), data.get('level', ''),
                          data.get('phone', ''), data.get('email', ''), enc_blob, student_pk))
                else:
                    cur.execute("""
                        UPDATE students SET full_name=?, department=?, level=?, phone=?, email=?
                        WHERE id=?
                    """, (data['full_name'], data.get('department', ''), data.get('level', ''),
                          data.get('phone', ''), data.get('email', ''), student_pk))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Student update error: {e}")
            return False

    def delete_student(self, student_pk: int) -> bool:
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT student_id FROM students WHERE id=?", (student_pk,))
                row = cur.fetchone()
                if row:
                    cur.execute("DELETE FROM attendance WHERE student_id=?", (row['student_id'],))
                cur.execute("DELETE FROM students WHERE id=?", (student_pk,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Student delete error: {e}")
            return False

    def get_students(self, search: str = ""):
        with self.get_conn() as conn:
            cur = conn.cursor()
            if search:
                like = f"%{search}%"
                cur.execute("""
                    SELECT * FROM students
                    WHERE full_name LIKE ? OR student_id LIKE ? OR department LIKE ? OR email LIKE ?
                    ORDER BY full_name
                """, (like, like, like, like))
            else:
                cur.execute("SELECT * FROM students ORDER BY full_name")
            return cur.fetchall()

    def get_student_by_id(self, student_pk: int):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM students WHERE id=?", (student_pk,))
            return cur.fetchone()

    def get_all_students(self):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM students")
            return cur.fetchall()

    def get_encodings(self):
        """Return list of (student_pk, student_id, full_name, encoding, image_path)."""
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, student_id, full_name, face_encoding, image_path FROM students")
            rows = cur.fetchall()
        out = []
        for r in rows:
            if r['face_encoding']:
                try:
                    enc = pickle.loads(r['face_encoding'])
                    out.append((r['id'], r['student_id'], r['full_name'], enc, r['image_path']))
                except Exception:
                    pass
        return out

    # Attendance
    def mark_attendance(self, student_id: str, status: str = "Present", confidence: float = 0.0) -> bool:
        today = date.today().isoformat()
        now = datetime.now().strftime("%H:%M:%S")
        with self.get_conn() as conn:
            cur = conn.cursor()
            # Defensive FK check: ensure the student actually exists before inserting.
            # Prevents FOREIGN KEY constraint failures if the recognizer returns a
            # stale/unknown student_id (e.g. a label whose owner was deleted).
            cur.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
            if not cur.fetchone():
                logger.warning(f"mark_attendance: unknown student_id '{student_id}' - skipped")
                return False
            cur.execute("SELECT id FROM attendance WHERE student_id=? AND date=?", (student_id, today))
            if cur.fetchone():
                return False
            cur.execute("""
                INSERT INTO attendance (student_id, date, time, status, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (student_id, today, now, status, confidence))
            conn.commit()
        return True

    def get_attendance(self, filter_date: str = None, filter_month: str = None,
                       student_id: str = None):
        with self.get_conn() as conn:
            cur = conn.cursor()
            q = """
                SELECT a.*, s.full_name, s.department
                FROM attendance a
                LEFT JOIN students s ON s.student_id = a.student_id
                WHERE 1=1
            """
            params = []
            if filter_date:
                q += " AND a.date = ?"
                params.append(filter_date)
            if filter_month:
                q += " AND a.date LIKE ?"
                params.append(f"{filter_month}%")
            if student_id:
                q += " AND a.student_id = ?"
                params.append(student_id)
            q += " ORDER BY a.date DESC, a.time DESC"
            cur.execute(q, params)
            return cur.fetchall()

    def get_today_attendance(self):
        return self.get_attendance(filter_date=date.today().isoformat())

    def attendance_stats(self):
        today = date.today().isoformat()
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS c FROM students")
            total_students = cur.fetchone()['c']
            cur.execute("SELECT COUNT(DISTINCT student_id) AS c FROM attendance WHERE date=?", (today,))
            present_today = cur.fetchone()['c']
            absent_today = max(0, total_students - present_today)
            cur.execute("SELECT COUNT(*) AS c FROM attendance")
            total_records = cur.fetchone()['c']
        rate = (present_today / total_students * 100) if total_students else 0.0
        return {
            'total_students': total_students,
            'present_today': present_today,
            'absent_today': absent_today,
            'attendance_rate': rate,
            'total_records': total_records
        }

    def monthly_stats(self, year: int = None, month: int = None):
        if year is None:
            year = date.today().year
        if month is None:
            month = date.today().month
        prefix = f"{year:04d}-{month:02d}"
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT date, COUNT(DISTINCT student_id) AS c
                FROM attendance WHERE date LIKE ?
                GROUP BY date
            """, (f"{prefix}%",))
            rows = {r['date']: r['c'] for r in cur.fetchall()}
        days_in_month = (date(year, month % 12 + 1, 1) - timedelta(days=1)).day if month != 12 else 31
        result = []
        for d in range(1, days_in_month + 1):
            ds = f"{prefix}-{d:02d}"
            result.append((ds, rows.get(ds, 0)))
        return result

    # Settings
    def get_setting(self, key: str, default: str = ""):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key=?", (key,))
            row = cur.fetchone()
            return row['value'] if row else default

    def set_setting(self, key: str, value: str):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    # Notifications
    def add_notification(self, title: str, message: str, ntype: str = "info"):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO notifications (title, message, type, created_at)
                VALUES (?, ?, ?, ?)
            """, (title, message, ntype, datetime.now().isoformat()))
            conn.commit()

    def get_notifications(self, limit: int = 20):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?", (limit,))
            return cur.fetchall()

    def mark_notifications_read(self):
        with self.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE notifications SET read=1")
            conn.commit()

# ----------------------------------------------------------------------------
# Camera Worker (Thread)
# ----------------------------------------------------------------------------
class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.msleep(30)
                continue
            self.frame_ready.emit(frame)
            self.msleep(33)
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(2000)

# ----------------------------------------------------------------------------
# Custom Widgets
# ----------------------------------------------------------------------------
class RoundedWidget(QFrame):
    """Base widget with rounded corners and shadow."""
    def __init__(self, radius: int = 12, bg: str = None, parent=None):
        super().__init__(parent)
        self.radius = radius
        if bg:
            self.setStyleSheet(f"background-color: {bg}; border-radius: {radius}px;")
        else:
            self.setStyleSheet(f"border-radius: {radius}px;")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # default no-op; subclasses override
        super().paintEvent(event)


class ModernCard(QFrame):
    """A premium card widget with hover animation."""
    def __init__(self, title: str = "", value: str = "", icon_name: str = "fa5s.chart-bar",
                 color: str = COLORS['primary'], parent=None):
        super().__init__(parent)
        self.color = color
        self.setObjectName("ModernCard")
        self.setStyleSheet(f"""
            #ModernCard {{
                background-color: {COLORS['surface']};
                border-radius: 16px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        self.setMinimumHeight(130)
        self.setMinimumWidth(220)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color=color).pixmap(28, 28))
        icon_label.setStyleSheet("background: transparent;")
        icon_label.setFixedSize(36, 36)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_container = QFrame()
        icon_container.setStyleSheet(f"""
            background-color: {color}22;
            border-radius: 10px;
        """)
        icon_container.setFixedSize(44, 44)
        il = QVBoxLayout(icon_container)
        il.setContentsMargins(0, 0, 0, 0)
        il.addWidget(icon_label, alignment=Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
        """)

        top.addWidget(icon_container)
        top.addWidget(title_label)
        top.addStretch()

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 30px;
            font-weight: 800;
            background: transparent;
        """)

        self.sub_label = QLabel("")
        self.sub_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 12px;
            background: transparent;
        """)

        layout.addLayout(top)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)
        layout.addStretch()

        self._animation = None

    def setValue(self, value: str):
        self.value_label.setText(str(value))

    def setSubText(self, text: str):
        self.sub_label.setText(text)

    def enterEvent(self, event):
        self._animate_scale(1.03)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_scale(1.0)
        super().leaveEvent(event)

    def _animate_scale(self, scale: float):
        if self._animation:
            self._animation.stop()
        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        rect = self.geometry()
        new_w = int(rect.width() * scale)
        new_h = int(rect.height() * scale)
        new_x = rect.x() - (new_w - rect.width()) // 2
        new_y = rect.y() - (new_h - rect.height()) // 2
        anim.setEndValue(QRect(new_x, new_y, new_w, new_h))
        anim.start()
        self._animation = anim


class AnimatedButton(QPushButton):
    """Premium animated button with icon support."""
    def __init__(self, text: str = "", icon_name: str = None, color: str = None,
                 style: str = "primary", parent=None):
        super().__init__(text, parent)
        self.color = color or COLORS['primary']
        self.style = style
        if icon_name:
            self.setIcon(qta.icon(icon_name, color='white'))
            self.setIconSize(QSize(18, 18))
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)
        self._apply_style()
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(15)
        self._shadow.setOffset(0, 4)
        self._shadow.setColor(QColor(self.color))
        self._shadow.setEnabled(False)
        self.setGraphicsEffect(self._shadow)
        self._animation = None

    def _apply_style(self):
        if self.style == "primary":
            bg = self.color
            hover = COLORS['primary_dark']
            fg = 'white'
        elif self.style == "success":
            bg = COLORS['success']
            hover = '#00A88C'
            fg = 'white'
        elif self.style == "danger":
            bg = COLORS['danger']
            hover = '#E63D5A'
            fg = 'white'
        elif self.style == "ghost":
            bg = 'transparent'
            hover = COLORS['surface']
            fg = COLORS['text_primary']
        elif self.style == "outline":
            bg = 'transparent'
            hover = f"{self.color}33"
            fg = self.color
        else:
            bg = self.color
            hover = COLORS['primary_dark']
            fg = 'white'

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: {'none' if self.style != 'outline' else f'2px solid {self.color}'};
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:pressed {{
                padding-top: 12px;
                padding-bottom: 8px;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['text_muted']};
                color: {COLORS['text_secondary']};
            }}
        """)

    def enterEvent(self, event):
        if self.style in ("primary", "success", "danger"):
            self._shadow.setEnabled(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._shadow.setEnabled(False)
        super().leaveEvent(event)


class RoundedLineEdit(QLineEdit):
    """Custom rounded line edit with icon support."""
    def __init__(self, placeholder: str = "", icon_name: str = None, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(44)
        self.icon_name = icon_name
        if icon_name:
            action = QAction(qta.icon(icon_name, color=COLORS['text_muted']), "", self)
            self.addAction(action, QLineEdit.LeadingPosition)
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 10px;
                padding: 8px 14px;
                font-size: 13px;
                selection-background-color: {COLORS['primary']};
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['primary']};
            }}
        """)


class SearchBox(RoundedLineEdit):
    def __init__(self, placeholder: str = "Search...", parent=None):
        super().__init__(placeholder, "fa5s.search", parent)
        self.setMinimumWidth(280)


class ModernTable(QTableWidget):
    """Custom styled table with selection and animations."""
    def __init__(self, headers: list, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                gridline-color: {COLORS['border']};
                font-size: 13px;
                selection-background-color: {COLORS['primary']}55;
                selection-color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_secondary']};
                padding: 12px 8px;
                border: none;
                border-bottom: 2px solid {COLORS['border']};
                font-weight: 700;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 10px 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['primary']}33;
            }}
        """)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        h = self.horizontalHeader()
        h.setStretchLastSection(True)
        h.setSectionResizeMode(QHeaderView.Interactive)
        h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def populate(self, data: list):
        self.setRowCount(0)
        for row_data in data:
            r = self.rowCount()
            self.insertRow(r)
            for c, val in enumerate(row_data):
                item = QTableWidgetItem(str(val) if val is not None else "")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(r, c, item)


class NotificationWidget(QFrame):
    """Animated notification toast."""
    closed = Signal()

    def __init__(self, title: str, message: str, ntype: str = "info", parent=None):
        super().__init__(parent)
        color_map = {
            'success': COLORS['success'],
            'error': COLORS['danger'],
            'warning': COLORS['warning'],
            'info': COLORS['info'],
        }
        icon_map = {
            'success': 'fa5s.check-circle',
            'error': 'fa5s.times-circle',
            'warning': 'fa5s.exclamation-triangle',
            'info': 'fa5s.info-circle',
        }
        self.color = color_map.get(ntype, COLORS['info'])
        icon_name = icon_map.get(ntype, 'fa5s.info-circle')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(360)

        container = QFrame(self)
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 12px;
                border-left: 5px solid {self.color};
            }}
        """)
        shadow = QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 80))
        container.setGraphicsEffect(shadow)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color=self.color).pixmap(26, 26))
        icon_label.setStyleSheet("background: transparent;")
        layout.addWidget(icon_label)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 13px; background: transparent;")
        msg_label = QLabel(message)
        msg_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;")
        msg_label.setWordWrap(True)
        text_layout.addWidget(title_label)
        text_layout.addWidget(msg_label)
        layout.addLayout(text_layout, 1)

        close_btn = QPushButton()
        close_btn.setIcon(qta.icon('fa5s.times', color=COLORS['text_muted']))
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
            }}
        """)
        close_btn.clicked.connect(self.close_animation)
        layout.addWidget(close_btn)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(container)

    def show_animated(self):
        self.show()
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_anim.setDuration(280)
        self._opacity_anim.setStartValue(0)
        self._opacity_anim.setEndValue(1)
        self._opacity_anim.start()
        QTimer.singleShot(5000, self.close_animation)

    def close_animation(self):
        try:
            self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
            self._opacity_anim.setDuration(280)
            self._opacity_anim.setStartValue(1)
            self._opacity_anim.setEndValue(0)
            self._opacity_anim.finished.connect(self.close)
            self._opacity_anim.finished.connect(self.closed.emit)
            self._opacity_anim.start()
        except Exception:
            self.close()
            self.closed.emit()


class LoadingSpinner(QWidget):
    """Custom CSS-style loading spinner."""
    def __init__(self, size: int = 40, color: str = COLORS['primary'], parent=None):
        super().__init__(parent)
        self.size = size
        self.color = color
        self.angle = 0
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(30)

    def _rotate(self):
        self.angle = (self.angle + 8) % 360
        self.update()

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.translate(self.size / 2, self.size / 2)
            p.rotate(self.angle)
            pen = QPen(QColor(self.color))
            pen.setWidth(4)
            pen.setCapStyle(Qt.RoundCap)
            for i in range(8):
                alpha = int(255 * (i + 1) / 8)
                c = QColor(self.color)
                c.setAlpha(alpha)
                pen.setColor(c)
                p.setPen(pen)
                p.drawLine(int(self.size * 0.25), 0, int(self.size * 0.4), 0)
                p.rotate(45)
            p.end()
        except Exception as e:
            logger.error(f"Spinner paint error: {e}")


class AnimatedSidebar(QFrame):
    """Sidebar with animated navigation buttons."""
    page_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AnimatedSidebar")
        self.setFixedWidth(240)
        self.setStyleSheet(f"""
            #AnimatedSidebar {{
                background-color: {COLORS['bg_medium']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)
        self.buttons = []
        self.active_index = 0
        self.indicator = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 20, 14, 20)
        layout.setSpacing(6)

        # Logo area
        logo_layout = QHBoxLayout()
        logo_icon = QLabel()
        logo_icon.setPixmap(qta.icon('fa5s.fingerprint', color=COLORS['primary']).pixmap(34, 34))
        logo_icon.setStyleSheet("background: transparent;")
        logo_text = QLabel("FaceAttend")
        logo_text.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 18px;
            font-weight: 800;
            background: transparent;
        """)
        logo_layout.addWidget(logo_icon)
        logo_layout.addWidget(logo_text)
        logo_layout.addStretch()
        layout.addLayout(logo_layout)
        layout.addSpacing(20)

        # Menu items
        self.menu_items = [
            ("Dashboard", "fa5s.tachometer-alt"),
            ("Students", "fa5s.user-graduate"),
            ("Register Face", "fa5s.camera"),
            ("Live Recognition", "fa5s.eye"),
            ("Attendance", "fa5s.calendar-check"),
            ("Reports", "fa5s.file-pdf"),
            ("Statistics", "fa5s.chart-line"),
            ("Settings", "fa5s.cog"),
        ]
        for idx, (text, icon) in enumerate(self.menu_items):
            btn = self._create_menu_button(text, icon, idx)
            self.buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Bottom info
        info = QFrame()
        info.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            border-radius: 10px;
        """)
        il = QVBoxLayout(info)
        il.setContentsMargins(12, 10, 12, 10)
        info_title = QLabel(f"v{APP_VERSION}")
        info_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 12px; background: transparent;")
        info_sub = QLabel("Premium Edition")
        info_sub.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;")
        il.addWidget(info_title)
        il.addWidget(info_sub)
        layout.addWidget(info)

        self.set_active(0)

    def _create_menu_button(self, text: str, icon_name: str, idx: int) -> QPushButton:
        btn = QPushButton(f"  {text}")
        btn.setIcon(qta.icon(icon_name, color=COLORS['text_secondary']))
        btn.setIconSize(QSize(18, 18))
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(44)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                text-align: left;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
        btn.clicked.connect(lambda: self.set_active(idx))
        return btn

    def set_active(self, idx: int):
        if idx == self.active_index:
            return
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == idx)
            icon = self.menu_items[i][1]
            if i == idx:
                btn.setIcon(qta.icon(icon, color='white'))
            else:
                btn.setIcon(qta.icon(icon, color=COLORS['text_secondary']))
        self.active_index = idx
        self.page_changed.emit(idx)


class BarChartWidget(QWidget):
    """Custom bar chart widget."""
    def __init__(self, data: list = None, labels: list = None, parent=None):
        super().__init__(parent)
        self.data = data or []
        self.labels = labels or []
        self.setMinimumHeight(220)
        self.setStyleSheet(f"background: transparent;")

    def set_data(self, data: list, labels: list):
        self.data = data
        self.labels = labels
        self.update()

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            margin = 30
            chart_w = w - margin * 2
            chart_h = h - margin * 2

            if not self.data:
                p.setPen(QColor(COLORS['text_muted']))
                font = p.font()
                font.setPointSize(11)
                p.setFont(font)
                p.drawText(self.rect(), Qt.AlignCenter, "No data available")
                p.end()
                return

            max_v = max(self.data) if max(self.data) > 0 else 1
            n = len(self.data)
            bar_w = chart_w / max(n, 1) * 0.7
            gap = chart_w / max(n, 1) * 0.3

            # Grid lines
            p.setPen(QPen(QColor(COLORS['border']), 1, Qt.DashLine))
            for i in range(5):
                y = margin + chart_h * i / 4
                p.drawLine(margin, int(y), w - margin, int(y))

            # Bars
            grad = QLinearGradient(0, margin, 0, h - margin)
            grad.setColorAt(0, QColor(COLORS['primary']))
            grad.setColorAt(1, QColor(COLORS['primary_dark']))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)

            for i, v in enumerate(self.data):
                x = margin + i * (bar_w + gap) + gap / 2
                bh = (v / max_v) * chart_h
                y = h - margin - bh
                rect = QRectF(int(x), int(y), int(bar_w), int(bh))
                p.drawRoundedRect(rect, 4, 4)

                # Label
                p.setPen(QColor(COLORS['text_secondary']))
                font = p.font()
                font.setPointSize(8)
                p.setFont(font)
                label = self.labels[i] if i < len(self.labels) else ""
                p.drawText(QRectF(int(x), h - margin + 4, int(bar_w), 20),
                           Qt.AlignCenter, str(label))

            # Values on top
            p.setPen(QColor(COLORS['text_primary']))
            font = p.font()
            font.setPointSize(9)
            font.setBold(True)
            p.setFont(font)
            for i, v in enumerate(self.data):
                x = margin + i * (bar_w + gap) + gap / 2
                y = h - margin - (v / max_v) * chart_h - 18
                p.drawText(QRectF(int(x), int(y), int(bar_w), 16),
                           Qt.AlignCenter, str(v))
            p.end()
        except Exception as e:
            logger.error(f"BarChart paint error: {e}")


class PieChartWidget(QWidget):
    """Simple pie chart."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments = []
        self.setMinimumHeight(220)

    def set_data(self, segments: list):
        """segments: [(label, value, color)]"""
        self.segments = segments
        self.update()

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            size = min(w, h) - 60
            cx, cy = w // 2, h // 2
            rect = QRect(cx - size // 2, cy - size // 2, size, size)

            total = sum(s[1] for s in self.segments) or 1
            start = 0
            for label, value, color in self.segments:
                span = (value / total) * 360 * 16
                p.setBrush(QColor(color))
                p.setPen(QColor(COLORS['bg_medium']))
                p.drawPie(rect, int(start), int(span))
                start += span

            # Legend
            legend_x = 10
            legend_y = h - 30
            p.setPen(QColor(COLORS['text_primary']))
            font = p.font()
            font.setPointSize(9)
            p.setFont(font)
            for label, value, color in self.segments:
                p.setBrush(QColor(color))
                p.setPen(Qt.NoPen)
                p.drawRect(legend_x, legend_y + 4, 12, 12)
                p.setPen(QColor(COLORS['text_primary']))
                p.drawText(legend_x + 18, legend_y + 14, f"{label}: {value}")
                legend_x += 130
            p.end()
        except Exception as e:
            logger.error(f"PieChart paint error: {e}")


# ----------------------------------------------------------------------------
# Login Window
# ----------------------------------------------------------------------------
class LoginWindow(QWidget):
    """Modern login window with animations."""

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.setWindowTitle(f"{APP_NAME} - Login")
        self.setFixedSize(960, 600)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel - branding
        left = QFrame()
        left.setFixedWidth(440)
        left.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLORS['primary']}, stop:1 {COLORS['primary_dark']});
            border-top-left-radius: 20px;
            border-bottom-left-radius: 20px;
        """)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(40, 40, 40, 40)
        left_layout.setSpacing(20)

        icon = QLabel()
        icon.setPixmap(qta.icon('fa5s.fingerprint', color='white').pixmap(80, 80))
        icon.setStyleSheet("background: transparent;")
        icon.setAlignment(Qt.AlignCenter)
        left_layout.addStretch()
        left_layout.addWidget(icon, alignment=Qt.AlignCenter)

        title = QLabel("Face Recognition\nAttendance System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: white;
            font-size: 26px;
            font-weight: 800;
            background: transparent;
        """)
        left_layout.addWidget(title)

        sub = QLabel("Secure, modern, and reliable attendance tracking powered by AI")
        sub.setAlignment(Qt.AlignCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet("""
            color: rgba(255, 255, 255, 0.85);
            font-size: 13px;
            background: transparent;
        """)
        left_layout.addWidget(sub)
        left_layout.addStretch()

        version = QLabel(f"v{APP_VERSION}  •  Premium Edition")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 11px; background: transparent;")
        left_layout.addWidget(version)

        # Right panel - login form
        right = QFrame()
        right.setStyleSheet(f"""
            background-color: {COLORS['bg_medium']};
            border-top-right-radius: 20px;
            border-bottom-right-radius: 20px;
        """)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(60, 60, 60, 60)
        right_layout.setSpacing(20)

        # Title
        welcome = QLabel("Welcome Back")
        welcome.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 28px; font-weight: 800; background: transparent;")
        right_layout.addWidget(welcome)

        subtitle = QLabel("Sign in to continue to your dashboard")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; background: transparent;")
        right_layout.addWidget(subtitle)

        right_layout.addSpacing(20)

        # Form
        self.username_input = RoundedLineEdit("Enter username", "fa5s.user")
        self.username_input.setText("admin")
        right_layout.addWidget(self.username_input)

        self.password_input = RoundedLineEdit("Enter password", "fa5s.lock")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setText("admin123")
        right_layout.addWidget(self.password_input)

        self.show_pwd = QCheckBox("Show password")
        self.show_pwd.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_secondary']};
                font-size: 12px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: 16px; height: 16px;
                border-radius: 4px;
                border: 2px solid {COLORS['border']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['primary']};
                border: 2px solid {COLORS['primary']};
            }}
        """)
        self.show_pwd.toggled.connect(lambda c: self.password_input.setEchoMode(QLineEdit.Normal if c else QLineEdit.Password))
        right_layout.addWidget(self.show_pwd)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px; background: transparent;")
        right_layout.addWidget(self.error_label)

        right_layout.addSpacing(10)

        self.login_btn = AnimatedButton("Sign In", "fa5s.sign-in-alt", style="primary")
        self.login_btn.setMinimumHeight(48)
        self.login_btn.clicked.connect(self.do_login)
        right_layout.addWidget(self.login_btn)

        right_layout.addStretch()

        hint = QLabel("Default credentials: admin / admin123")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;")
        right_layout.addWidget(hint)

        # Window controls
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addStretch()
        for icon, color in (('fa5s.window-minimize', COLORS['text_muted']),
                            ('fa5s.times', COLORS['danger'])):
            btn = QPushButton()
            btn.setIcon(qta.icon(icon, color=color))
            btn.setFixedSize(28, 28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    border-radius: 14px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['surface']};
                }}
            """)
            if 'times' in icon:
                btn.clicked.connect(self.close)
            else:
                btn.clicked.connect(self.showMinimized)
            controls.addWidget(btn)
        right_layout.addLayout(controls)

        main_layout.addWidget(left)
        main_layout.addWidget(right, 1)

        self._drag_pos = None
        self.main_window = None

        # Fade in
        self.setWindowOpacity(0)
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(400)
        self._anim.setStartValue(0)
        self._anim.setEndValue(1)
        self._anim.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.do_login()

    def do_login(self):
        u = self.username_input.text().strip()
        p = self.password_input.text()
        if not u or not p:
            self.error_label.setText("Please fill in all fields")
            return
        if self.db.verify_admin(u, p):
            self.error_label.setText("")
            self.open_main()
        else:
            self.error_label.setText("Invalid username or password")

    def open_main(self):
        self.main_window = MainWindow(self.db)
        self.main_window.show()
        self.close()


# ----------------------------------------------------------------------------
# Page Widgets
# ----------------------------------------------------------------------------
class PageHeader(QFrame):
    """Standard page header."""
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 24px; font-weight: 800; background: transparent;")
        s = QLabel(subtitle)
        s.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; background: transparent;")
        text_layout.addWidget(t)
        text_layout.addWidget(s)

        layout.addLayout(text_layout)
        layout.addStretch()


class DashboardPage(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        self.header = PageHeader("Dashboard", f"Welcome back, here's what's happening today.")
        layout.addWidget(self.header)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.card_total = ModernCard("Total Students", "0", "fa5s.user-graduate", COLORS['primary'])
        self.card_present = ModernCard("Present Today", "0", "fa5s.user-check", COLORS['success'])
        self.card_absent = ModernCard("Absent Today", "0", "fa5s.user-times", COLORS['danger'])
        self.card_rate = ModernCard("Attendance Rate", "0%", "fa5s.percentage", COLORS['warning'])
        stats_row.addWidget(self.card_total)
        stats_row.addWidget(self.card_present)
        stats_row.addWidget(self.card_absent)
        stats_row.addWidget(self.card_rate)
        layout.addLayout(stats_row)

        # Charts row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        # Bar chart card
        bar_card = QFrame()
        bar_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        bl = QVBoxLayout(bar_card)
        bl.setContentsMargins(20, 16, 20, 16)
        bh = QLabel("Attendance This Month")
        bh.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        bl.addWidget(bh)
        self.bar_chart = BarChartWidget()
        bl.addWidget(self.bar_chart)
        charts_row.addWidget(bar_card, 2)

        # Pie chart card
        pie_card = QFrame()
        pie_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        pl = QVBoxLayout(pie_card)
        pl.setContentsMargins(20, 16, 20, 16)
        ph = QLabel("Today's Status")
        ph.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        pl.addWidget(ph)
        self.pie_chart = PieChartWidget()
        pl.addWidget(self.pie_chart)
        charts_row.addWidget(pie_card, 1)

        layout.addLayout(charts_row)

        # Recent activity card
        recent_card = QFrame()
        recent_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        rl = QVBoxLayout(recent_card)
        rl.setContentsMargins(20, 16, 20, 16)
        rh = QLabel("Recent Activity")
        rh.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        rl.addWidget(rh)
        self.recent_table = ModernTable(["Time", "Student ID", "Name", "Status", "Confidence"])
        rl.addWidget(self.recent_table)

        layout.addWidget(recent_card, 1)

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(15000)
        self.refresh()

    def refresh(self):
        stats = self.db.attendance_stats()
        self.card_total.setValue(str(stats['total_students']))
        self.card_present.setValue(str(stats['present_today']))
        self.card_absent.setValue(str(stats['absent_today']))
        self.card_rate.setValue(f"{stats['attendance_rate']:.1f}%")
        self.card_total.setSubText("Registered students")
        self.card_present.setSubText("Marked present today")
        self.card_absent.setSubText("Not yet checked in")
        self.card_rate.setSubText("Today's attendance rate")

        # Pie chart
        self.pie_chart.set_data([
            ("Present", stats['present_today'], COLORS['success']),
            ("Absent", stats['absent_today'], COLORS['danger']),
        ])

        # Bar chart - last 14 days
        today_d = date.today()
        labels = []
        values = []
        for i in range(13, -1, -1):
            d = today_d - timedelta(days=i)
            rows = self.db.get_attendance(filter_date=d.isoformat())
            labels.append(d.strftime("%d"))
            values.append(len(set(r['student_id'] for r in rows)))
        self.bar_chart.set_data(values, labels)

        # Recent activity
        rows = self.db.get_attendance()[:10]
        data = []
        for r in rows:
            data.append([
                f"{r['date']} {r['time']}",
                r['student_id'],
                r['full_name'] or 'Unknown',
                r['status'],
                f"{r['confidence']:.1f}%" if r['confidence'] else '-'
            ])
        self.recent_table.populate(data)


class StudentsPage(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        header = PageHeader("Students", "Manage student records and information")
        header_layout.addWidget(header)
        header_layout.addStretch()
        self.add_btn = AnimatedButton("Add Student", "fa5s.plus", style="primary")
        self.add_btn.clicked.connect(self.add_student_dialog)
        header_layout.addWidget(self.add_btn)
        layout.addLayout(header_layout)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.search_box = SearchBox("Search by name, ID, department...")
        self.search_box.textChanged.connect(self.refresh)
        toolbar.addWidget(self.search_box)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table
        self.table = ModernTable(["ID", "Student ID", "Full Name", "Department", "Level", "Phone", "Email", "Registered"])
        layout.addWidget(self.table, 1)

        # Action buttons
        actions = QHBoxLayout()
        actions.addStretch()
        self.edit_btn = AnimatedButton("Edit", "fa5s.edit", style="outline", color=COLORS['info'])
        self.edit_btn.clicked.connect(self.edit_student_dialog)
        self.delete_btn = AnimatedButton("Delete", "fa5s.trash", style="outline", color=COLORS['danger'])
        self.delete_btn.clicked.connect(self.delete_student)
        self.refresh_btn = AnimatedButton("Refresh", "fa5s.sync", style="ghost")
        self.refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_btn)
        actions.addWidget(self.edit_btn)
        actions.addWidget(self.delete_btn)
        layout.addLayout(actions)

        self.refresh()

    def refresh(self):
        search = self.search_box.text().strip()
        rows = self.db.get_students(search)
        data = []
        for r in rows:
            data.append([
                r['id'],
                r['student_id'],
                r['full_name'],
                r['department'] or '-',
                r['level'] or '-',
                r['phone'] or '-',
                r['email'] or '-',
                r['registered_at'][:10] if r['registered_at'] else '-'
            ])
        self.table.populate(data)
        self.table.resizeColumnsToContents()

    def get_selected_pk(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return int(item.text()) if item else None

    def add_student_dialog(self):
        dlg = StudentDialog(self.db, self)
        if dlg.exec():
            self.refresh()

    def edit_student_dialog(self):
        pk = self.get_selected_pk()
        if not pk:
            QMessageBox.warning(self, "Warning", "Please select a student to edit")
            return
        student = self.db.get_student_by_id(pk)
        if student:
            dlg = StudentDialog(self.db, self, student=dict(student))
            if dlg.exec():
                self.refresh()

    def delete_student(self):
        pk = self.get_selected_pk()
        if not pk:
            QMessageBox.warning(self, "Warning", "Please select a student to delete")
            return
        student = self.db.get_student_by_id(pk)
        if not student:
            return
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete {student['full_name']}?\nThis will also remove their attendance records.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.db.delete_student(pk):
                self.db.add_notification("Student Deleted", f"{student['full_name']} was removed", "warning")
                self.refresh()


class StudentDialog(QDialog):
    """Add/Edit student dialog."""
    def __init__(self, db: Database, parent=None, student: dict = None):
        super().__init__(parent)
        self.db = db
        self.student = student
        self.is_edit = student is not None
        self.setWindowTitle("Edit Student" if self.is_edit else "Add Student")
        self.setMinimumWidth(500)
        self.setModal(True)
        self.setStyleSheet(f"background-color: {COLORS['bg_medium']};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Edit Student" if self.is_edit else "Add New Student")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 20px; font-weight: 800; background: transparent;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignRight)

        label_style = f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; background: transparent;"

        self.student_id_input = RoundedLineEdit("e.g. CS2024001")
        self.full_name_input = RoundedLineEdit("Full name")
        self.department_input = RoundedLineEdit("Department")
        self.level_input = QComboBox()
        self.level_input.addItems(["", "100", "200", "300", "400", "500", "600", "PG"])
        self.level_input.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 10px;
                padding: 8px 14px;
                font-size: 13px;
                min-height: 24px;
            }}
            QComboBox:focus {{ border: 2px solid {COLORS['primary']}; }}
            QComboBox::drop-down {{ border: none; width: 30px; }}
        """)
        self.phone_input = RoundedLineEdit("Phone number")
        self.email_input = RoundedLineEdit("Email address")

        form.addRow(self._label("Student ID *"), self.student_id_input)
        form.addRow(self._label("Full Name *"), self.full_name_input)
        form.addRow(self._label("Department"), self.department_input)
        form.addRow(self._label("Level"), self.level_input)
        form.addRow(self._label("Phone"), self.phone_input)
        form.addRow(self._label("Email"), self.email_input)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = AnimatedButton("Cancel", style="ghost")
        cancel_btn.clicked.connect(self.reject)
        save_btn = AnimatedButton("Save", "fa5s.save", style="primary")
        save_btn.clicked.connect(self.save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        if self.is_edit:
            self.student_id_input.setText(student['student_id'])
            self.student_id_input.setEnabled(False)
            self.full_name_input.setText(student['full_name'])
            self.department_input.setText(student['department'] or '')
            self.level_input.setCurrentText(student['level'] or '')
            self.phone_input.setText(student['phone'] or '')
            self.email_input.setText(student['email'] or '')

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; background: transparent;")
        return lbl

    def save(self):
        sid = self.student_id_input.text().strip()
        name = self.full_name_input.text().strip()
        if not sid or not name:
            QMessageBox.warning(self, "Required", "Student ID and Full Name are required")
            return
        data = {
            'student_id': sid,
            'full_name': name,
            'department': self.department_input.text().strip(),
            'level': self.level_input.currentText(),
            'phone': self.phone_input.text().strip(),
            'email': self.email_input.text().strip(),
        }
        if self.is_edit:
            if self.db.update_student(self.student['id'], data):
                QMessageBox.information(self, "Success", "Student updated successfully")
                self.accept()
            else:
                QMessageBox.critical(self, "Error", "Failed to update student")
        else:
            if self.db.add_student(data):
                self.db.add_notification("Student Added", f"{name} has been registered", "success")
                QMessageBox.information(self, "Success", "Student added successfully")
                self.accept()
            else:
                QMessageBox.critical(self, "Error", "Failed to add student. Student ID may already exist.")


class FaceRegistrationPage(QWidget):
    """Capture face images and generate encodings."""

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        self.captures = []
        self.encoding = None
        self.camera_worker = None
        self.lbph_training_data = []  # list of (face_img_gray, label_int)
        self.lbph_label_map = {}     # int -> (student_id, full_name)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(PageHeader("Face Registration", "Capture multiple images to register a student's face"))

        content = QHBoxLayout()
        content.setSpacing(16)

        # Left: camera
        cam_card = QFrame()
        cam_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(16, 16, 16, 16)

        cam_header = QHBoxLayout()
        cam_title = QLabel("Camera Feed")
        cam_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        cam_status = QLabel("● OFFLINE")
        cam_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; font-weight: 700; background: transparent;")
        self.cam_status_label = cam_status
        cam_header.addWidget(cam_title)
        cam_header.addStretch()
        cam_header.addWidget(cam_status)
        cl.addLayout(cam_header)

        self.cam_view = QLabel()
        self.cam_view.setMinimumSize(640, 480)
        self.cam_view.setAlignment(Qt.AlignCenter)
        self.cam_view.setStyleSheet(f"background-color: {COLORS['bg_dark']}; border-radius: 12px; color: {COLORS['text_muted']}; font-size: 14px;")
        self.cam_view.setText("Camera not started\n\nClick 'Start Camera' to begin")
        cl.addWidget(self.cam_view)

        cam_controls = QHBoxLayout()
        self.start_btn = AnimatedButton("Start Camera", "fa5s.video", style="primary")
        self.start_btn.clicked.connect(self.start_camera)
        self.capture_btn = AnimatedButton("Capture", "fa5s.camera", style="success")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_image)
        self.stop_btn = AnimatedButton("Stop", "fa5s.stop-circle", style="danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_camera)
        cam_controls.addWidget(self.start_btn)
        cam_controls.addWidget(self.capture_btn)
        cam_controls.addWidget(self.stop_btn)
        cam_controls.addStretch()
        cl.addLayout(cam_controls)

        content.addWidget(cam_card, 2)

        # Right: form & captures
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(16, 16, 16, 16)

        student_label = QLabel("Select Student")
        student_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        rl.addWidget(student_label)

        self.student_combo = QComboBox()
        self.student_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 10px;
                padding: 8px 14px;
                font-size: 13px;
                min-height: 24px;
            }}
            QComboBox:focus {{ border: 2px solid {COLORS['primary']}; }}
            QComboBox::drop-down {{ border: none; width: 30px; }}
        """)
        rl.addWidget(self.student_combo)

        progress_label = QLabel("Capture Progress")
        progress_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 700; background: transparent; margin-top: 12px;")
        rl.addWidget(progress_label)

        self.progress = QProgressBar()
        self.progress.setMaximum(5)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m images")
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 8px;
                text-align: center;
                min-height: 22px;
                font-weight: 700;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['success']};
                border-radius: 8px;
            }}
        """)
        rl.addWidget(self.progress)

        self.captures_label = QLabel("No captures yet")
        self.captures_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent;")
        self.captures_label.setWordWrap(True)
        rl.addWidget(self.captures_label)

        rl.addSpacing(8)
        self.register_btn = AnimatedButton("Register Face", "fa5s.user-plus", style="primary")
        self.register_btn.clicked.connect(self.register_face)
        rl.addWidget(self.register_btn)

        rl.addStretch()
        content.addWidget(right_panel, 1)

        layout.addLayout(content, 1)

        self.refresh_students()

    def refresh_students(self):
        self.student_combo.clear()
        self.student_combo.addItem("-- Select Student --", None)
        for s in self.db.get_students():
            self.student_combo.addItem(f"{s['student_id']} - {s['full_name']}", s['id'])

    def start_camera(self):
        if self.camera_worker and self.camera_worker.isRunning():
            return
        self.camera_worker = CameraWorker(0)
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.cam_status_label.setText("● LIVE")
        self.cam_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: 700; background: transparent;")

    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.cam_view.setText("Camera stopped")
        self.cam_status_label.setText("● OFFLINE")
        self.cam_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; font-weight: 700; background: transparent;")

    def update_frame(self, frame: np.ndarray):
        self._last_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(self.cam_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_view.setPixmap(pix)

    def capture_image(self):
        if not hasattr(self, '_last_frame'):
            return
        if self.progress.value() >= self.progress.maximum():
            return
        self.captures.append(self._last_frame.copy())
        # Try to detect face for LBPH training (always)
        try:
            gray = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            rects = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(rects):
                # Take the largest face
                rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
                x, y, w, h = rects[0]
                # Pad a bit
                pad = int(0.1 * max(w, h))
                x = max(0, x - pad); y = max(0, y - pad)
                w = min(gray.shape[1] - x, w + 2 * pad)
                h = min(gray.shape[0] - y, h + 2 * pad)
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                sid_combo = self.student_combo.currentData()
                if sid_combo is not None:
                    label = int(sid_combo)
                    if label not in self.lbph_label_map:
                        student = self.db.get_student_by_id(label)
                        if student:
                            self.lbph_label_map[label] = (student['student_id'], student['full_name'])
                    self.lbph_training_data.append((face_img, label))
        except Exception as e:
            logger.error(f"Capture face detect error: {e}")
        self.progress.setValue(len(self.captures))
        self.captures_label.setText(f"Captured {len(self.captures)} image(s)")
        if self.progress.value() >= self.progress.maximum():
            self.capture_btn.setEnabled(False)
            QMessageBox.information(self, "Ready", "You have captured enough images. Click 'Register Face' to save.")

    def register_face(self):
        student_pk = self.student_combo.currentData()
        if not student_pk:
            QMessageBox.warning(self, "Required", "Please select a student")
            return
        if len(self.captures) < 1:
            QMessageBox.warning(self, "Required", "Please capture at least 1 image")
            return

        student = self.db.get_student_by_id(student_pk)
        if not student:
            return

        encoding = None
        if FACE_RECOGNITION_AVAILABLE:
            try:
                # Take average encoding across captures
                encodings = []
                for img in self.captures:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb)
                    if locs:
                        encs = face_recognition.face_encodings(rgb, locs)
                        if encs:
                            encodings.append(encs[0])
                if not encodings:
                    QMessageBox.warning(self, "No Face",
                                        "No faces detected in captures. Please retake with a clear face.")
                    return
                encoding = np.mean(encodings, axis=0)
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                QMessageBox.critical(self, "Error", f"Face encoding failed: {e}")
                return

        # Save primary image
        img_dir = FACES_DIR / student['student_id']
        img_dir.mkdir(exist_ok=True)
        img_path = img_dir / "primary.jpg"
        cv2.imwrite(str(img_path), self.captures[0])
        # Save thumbnails
        for i, img in enumerate(self.captures):
            cv2.imwrite(str(img_dir / f"capture_{i+1}.jpg"), img)

        # Save LBPH training faces for this student
        lbph_dir = FACES_DIR / "_lbph"
        lbph_dir.mkdir(exist_ok=True)
        for i, (face_img, lbl) in enumerate(self.lbph_training_data):
            if lbl == student_pk:
                cv2.imwrite(str(lbph_dir / f"{student['student_id']}_{i}.jpg"), face_img)

        # Retrain global LBPH model
        try:
            self._retrain_lbph()
        except Exception as e:
            logger.error(f"LBPH retrain error: {e}")

        data = dict(student)
        if self.db.update_student(student_pk, data, encoding=encoding):
            self.db.add_notification("Face Registered", f"{student['full_name']} face enrolled", "success")
            QMessageBox.information(self, "Success", "Face registered successfully")
            self.captures = []
            self.progress.setValue(0)
            self.captures_label.setText("No captures yet")
            self.capture_btn.setEnabled(True)
        else:
            QMessageBox.critical(self, "Error", "Failed to save registration")

    def _retrain_lbph(self):
        """Retrain global LBPH recognizer using all saved training faces."""
        if not LBPH_AVAILABLE:
            return
        lbph_dir = FACES_DIR / "_lbph"
        if not lbph_dir.exists():
            return
        faces = []
        labels = []
        label_map = {}
        # Build label map from all students
        students = self.db.get_all_students()
        sid_to_pk = {s['student_id']: s['id'] for s in students}

        for f in sorted(lbph_dir.glob("*.jpg")):
            try:
                # Filename: STUDENTID_IDX.jpg
                parts = f.stem.split("_")
                sid = parts[0]
                if sid not in sid_to_pk:
                    continue
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(sid_to_pk[sid])
                student = next((s for s in students if s['student_id'] == sid), None)
                if student:
                    label_map[sid_to_pk[sid]] = (student['student_id'], student['full_name'])
            except Exception as e:
                logger.error(f"LBPH load error: {e}")
                continue
        if not faces:
            FaceRecognizerStore.trained = False
            return
        try:
            rec = FaceRecognizerStore.get()
            if rec is None:
                return
            rec.train(faces, np.array(labels))
            FaceRecognizerStore.label_map = label_map
            FaceRecognizerStore.trained = True
            logger.info(f"LBPH trained on {len(faces)} faces across {len(label_map)} students")
        except Exception as e:
            logger.error(f"LBPH train error: {e}")

    def hideEvent(self, event):
        self.stop_camera()
        super().hideEvent(event)


class LiveRecognitionPage(QWidget):
    """Live face recognition with attendance marking."""

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        self.camera_worker = None
        self.known_encodings = []
        self.known_meta = []
        self.recognition_active = False
        self.last_recognition = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Header with controls
        header_layout = QHBoxLayout()
        header_layout.addWidget(PageHeader("Live Recognition", "Real-time face recognition and automatic attendance"))
        header_layout.addStretch()

        self.start_btn = AnimatedButton("Start Recognition", "fa5s.play", style="success")
        self.start_btn.clicked.connect(self.toggle_recognition)
        self.stop_btn = AnimatedButton("Stop", "fa5s.stop", style="danger")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        header_layout.addWidget(self.start_btn)
        header_layout.addWidget(self.stop_btn)
        layout.addLayout(header_layout)

        content = QHBoxLayout()
        content.setSpacing(16)

        # Camera card
        cam_card = QFrame()
        cam_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(16, 16, 16, 16)

        ch = QHBoxLayout()
        cam_title = QLabel("Live Feed")
        cam_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        self.status_label = QLabel("● OFFLINE")
        self.status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; font-weight: 700; background: transparent;")
        ch.addWidget(cam_title)
        ch.addStretch()
        ch.addWidget(self.status_label)
        cl.addLayout(ch)

        self.cam_view = QLabel()
        self.cam_view.setMinimumSize(640, 480)
        self.cam_view.setAlignment(Qt.AlignCenter)
        self.cam_view.setStyleSheet(f"background-color: {COLORS['bg_dark']}; border-radius: 12px; color: {COLORS['text_muted']}; font-size: 14px;")
        self.cam_view.setText("Camera not started\n\nClick 'Start Recognition' to begin")
        cl.addWidget(self.cam_view)

        # Tolerance slider
        tol_layout = QHBoxLayout()
        tol_label = QLabel("Recognition Sensitivity:")
        tol_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;")
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(30)
        self.tolerance_slider.setMaximum(60)
        self.tolerance_slider.setValue(45)
        self.tolerance_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {COLORS['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['primary']};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
        """)
        self.tol_value = QLabel("0.45")
        self.tol_value.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 12px; background: transparent; min-width: 40px;")
        self.tolerance_slider.valueChanged.connect(lambda v: self.tol_value.setText(f"{v/100:.2f}"))
        tol_layout.addWidget(tol_label)
        tol_layout.addWidget(self.tolerance_slider, 1)
        tol_layout.addWidget(self.tol_value)
        cl.addLayout(tol_layout)

        content.addWidget(cam_card, 2)

        # Right panel: recognized person info + logs
        right = QFrame()
        right.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)

        info_title = QLabel("Detected Person")
        info_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        rl.addWidget(info_title)

        self.info_card = QFrame()
        self.info_card.setStyleSheet(f"background-color: {COLORS['bg_light']}; border-radius: 12px;")
        il = QVBoxLayout(self.info_card)
        il.setContentsMargins(16, 16, 16, 16)
        il.setSpacing(6)

        self.info_avatar = QLabel()
        self.info_avatar.setFixedSize(80, 80)
        self.info_avatar.setAlignment(Qt.AlignCenter)
        self.info_avatar.setStyleSheet(f"""
            background-color: {COLORS['primary']};
            color: white;
            border-radius: 40px;
            font-size: 32px;
            font-weight: 800;
        """)
        self.info_avatar.setText("?")

        self.info_name = QLabel("Waiting...")
        self.info_name.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 18px; font-weight: 800; background: transparent;")
        self.info_id = QLabel("-")
        self.info_id.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; background: transparent;")
        self.info_conf = QLabel("Confidence: -")
        self.info_conf.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; background: transparent;")
        self.info_status = QLabel("Status: -")
        self.info_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; background: transparent;")

        il.addWidget(self.info_avatar, alignment=Qt.AlignCenter)
        il.addWidget(self.info_name, alignment=Qt.AlignCenter)
        il.addWidget(self.info_id, alignment=Qt.AlignCenter)
        il.addWidget(self.info_conf, alignment=Qt.AlignCenter)
        il.addWidget(self.info_status, alignment=Qt.AlignCenter)
        rl.addWidget(self.info_card)

        log_title = QLabel("Session Log")
        log_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent; margin-top: 10px;")
        rl.addWidget(log_title)

        self.log_list = QListWidget()
        self.log_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                font-size: 12px;
                padding: 6px;
            }}
            QListWidget::item {{
                padding: 6px 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        rl.addWidget(self.log_list, 1)

        content.addWidget(right, 1)
        layout.addLayout(content, 1)

        # Auto-load encodings
        self.refresh_encodings()

    def refresh_encodings(self):
        data = self.db.get_encodings()
        self.known_encodings = [d[3] for d in data]
        self.known_meta = [(d[0], d[1], d[2], d[4]) for d in data]

    def toggle_recognition(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.refresh_encodings()
        # If using LBPH fallback, retrain from saved faces
        if not FACE_RECOGNITION_AVAILABLE and LBPH_AVAILABLE:
            try:
                lbph_dir = FACES_DIR / "_lbph"
                if lbph_dir.exists():
                    faces = []
                    labels = []
                    label_map = {}
                    students = self.db.get_all_students()
                    sid_to_pk = {s['student_id']: s['id'] for s in students}
                    for f in sorted(lbph_dir.glob("*.jpg")):
                        sid = f.stem.split("_")[0]
                        if sid not in sid_to_pk:
                            continue
                        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        img = cv2.resize(img, (200, 200))
                        faces.append(img)
                        labels.append(sid_to_pk[sid])
                        student = next((s for s in students if s['student_id'] == sid), None)
                        if student:
                            label_map[sid_to_pk[sid]] = (student['student_id'], student['full_name'])
                    if faces:
                        rec = FaceRecognizerStore.get()
                        rec.train(faces, np.array(labels))
                        FaceRecognizerStore.label_map = label_map
                        FaceRecognizerStore.trained = True
                        logger.info(f"LBPH auto-trained on {len(faces)} faces")
            except Exception as e:
                logger.error(f"LBPH startup train error: {e}")
        if FACE_RECOGNITION_AVAILABLE and not self.known_encodings:
            QMessageBox.warning(self, "No Data", "No registered faces found. Please register students first.")
            return
        self.recognition_active = True
        self.camera_worker = CameraWorker(0)
        self.camera_worker.frame_ready.connect(self.process_frame)
        self.camera_worker.start()
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Running...")
        self.stop_btn.setEnabled(True)
        self.status_label.setText("● LIVE")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; font-weight: 700; background: transparent;")
        self.log_list.addItem(QListWidgetItem(f"[{datetime.now().strftime('%H:%M:%S')}] Recognition started"))

    def stop_camera(self):
        self.recognition_active = False
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Recognition")
        self.stop_btn.setEnabled(False)
        self.cam_view.setText("Camera stopped")
        self.status_label.setText("● OFFLINE")
        self.status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; font-weight: 700; background: transparent;")

    def process_frame(self, frame: np.ndarray):
        # Reduce frame size for speed
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        face_locations = []   # (top, right, bottom, left) in small coords
        face_names = []
        face_confidences = []
        face_ids = []

        # Always use Haar cascade for face detection (works without dlib)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        rects = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Try face_recognition first if available
        if FACE_RECOGNITION_AVAILABLE and self.known_encodings:
            try:
                fr_locs = face_recognition.face_locations(rgb)
                if fr_locs:
                    face_encodings = face_recognition.face_encodings(rgb, fr_locs)
                    tolerance = self.tolerance_slider.value() / 100.0
                    for enc, loc in zip(face_encodings, fr_locs):
                        distances = face_recognition.face_distance(self.known_encodings, enc)
                        best_idx = int(np.argmin(distances)) if len(distances) else None
                        if best_idx is not None and distances[best_idx] < tolerance:
                            pk, sid, name, _ = self.known_meta[best_idx]
                            conf = max(0.0, (1 - distances[best_idx]) * 100)
                            face_locations.append(loc)
                            face_names.append(name)
                            face_confidences.append(conf)
                            face_ids.append(sid)
                        else:
                            face_locations.append(loc)
                            face_names.append("UNKNOWN")
                            face_confidences.append(0.0)
                            face_ids.append(None)
            except Exception as e:
                logger.error(f"Recognition error: {e}")

        # Fallback: Haar + LBPH
        if not face_locations:
            use_lbph = (LBPH_AVAILABLE and FaceRecognizerStore.trained
                        and not FACE_RECOGNITION_AVAILABLE)
            for (x, y, w, h) in rects:
                loc = (y, x + w, y + h, x)  # (top, right, bottom, left)
                face_locations.append(loc)
                if use_lbph:
                    pad = int(0.1 * max(w, h))
                    x0 = max(0, x - pad); y0 = max(0, y - pad)
                    w0 = min(gray.shape[1] - x0, w + 2 * pad)
                    h0 = min(gray.shape[0] - y0, h + 2 * pad)
                    face_img = gray[y0:y0+h0, x0:x0+w0]
                    face_img = cv2.resize(face_img, (200, 200))
                    try:
                        sid, name, conf = FaceRecognizerStore.predict(face_img)
                    except Exception as e:
                        logger.debug(f"predict unpack error: {e}")
                        sid, name, conf = None, None, None
                    if sid and name:
                        face_names.append(name)
                        # predict() returns a 0-100 confidence already
                        face_confidences.append(max(0.0, min(100.0, conf)))
                        face_ids.append(sid)
                    else:
                        face_names.append("UNKNOWN")
                        face_confidences.append(0.0)
                        face_ids.append(None)
                else:
                    face_names.append("UNKNOWN")
                    face_confidences.append(0.0)
                    face_ids.append(None)

        # Scale back to display coordinates (display is full frame, small is half)
        scale = 2
        for (top, right, bottom, left), name, conf, sid in zip(face_locations, face_names, face_confidences, face_ids):
            top *= scale; right *= scale; bottom *= scale; left *= scale
            if name == "UNKNOWN":
                color = (90, 90, 240)  # BGR red-ish
                display_text = "UNKNOWN"
            else:
                color = (90, 200, 90)
                display_text = f"{name} ({conf:.0f}%)"
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display, (left, top - th - 10), (left + tw + 8, top), color, -1)
            cv2.putText(display, display_text, (left + 4, top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Mark attendance and update info
            if name != "UNKNOWN" and sid:
                now = datetime.now()
                key = (sid, now.strftime("%Y-%m-%d"))
                last = self.last_recognition.get(key)
                if not last or (now - last).total_seconds() > 5:
                    self.last_recognition[key] = now
                    try:
                        marked = self.db.mark_attendance(sid, "Present", conf)
                    except Exception as e:
                        logger.debug(f"mark_attendance error: {e}")
                        marked = False
                    if marked:
                        self.log_list.addItem(QListWidgetItem(
                            f"[{now.strftime('%H:%M:%S')}] ✓ {name} ({sid}) marked present ({conf:.1f}%)"
                        ))
                        self.update_info(name, sid, conf, "Present")
                        self.db.add_notification("Attendance Marked", f"{name} is present", "success")
                    else:
                        # Either duplicate, unknown student, or DB error - still show the live match
                        self.update_info(name, sid, conf, "Already marked / unknown")
                else:
                    self.update_info(name, sid, conf, "Already marked")
            else:
                self.update_info("UNKNOWN", "-", 0.0, "Unknown face")

        # Show frame
        rgb_d = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_d.shape
        img = QImage(rgb_d.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(self.cam_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_view.setPixmap(pix)

    def update_info(self, name, sid, conf, status):
        self.info_name.setText(name)
        self.info_id.setText(f"ID: {sid}")
        self.info_conf.setText(f"Confidence: {conf:.1f}%")
        self.info_status.setText(f"Status: {status}")
        if name != "UNKNOWN":
            initial = name[0].upper() if name else "?"
            self.info_avatar.setText(initial)
            self.info_avatar.setStyleSheet(f"""
                background-color: {COLORS['success']};
                color: white;
                border-radius: 40px;
                font-size: 32px;
                font-weight: 800;
            """)
        else:
            self.info_avatar.setText("?")
            self.info_avatar.setStyleSheet(f"""
                background-color: {COLORS['danger']};
                color: white;
                border-radius: 40px;
                font-size: 32px;
                font-weight: 800;
            """)

    def hideEvent(self, event):
        self.stop_camera()
        super().hideEvent(event)


class AttendancePage(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(PageHeader("Attendance", "View and manage attendance records"))

        # Tabs
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: transparent;
                border: none;
            }}
            QTabBar::tab {{
                background: {COLORS['surface']};
                color: {COLORS['text_secondary']};
                padding: 10px 22px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 4px;
                font-weight: 600;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['primary']};
                color: white;
            }}
        """)
        layout.addWidget(tabs)

        # Daily tab
        daily = QWidget()
        dl = QVBoxLayout(daily)
        dl.setContentsMargins(0, 12, 0, 0)
        daily_toolbar = QHBoxLayout()
        dl.addLayout(daily_toolbar)
        daily_label = QLabel("Date:")
        daily_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; background: transparent;")
        self.daily_date = QDateEdit()
        self.daily_date.setDate(QDate.currentDate())
        self.daily_date.setCalendarPopup(True)
        self.daily_date.setStyleSheet(f"""
            QDateEdit {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 24px;
            }}
        """)
        daily_btn = AnimatedButton("View", "fa5s.eye", style="primary")
        daily_btn.clicked.connect(self.refresh_daily)
        daily_toolbar.addWidget(daily_label)
        daily_toolbar.addWidget(self.daily_date)
        daily_toolbar.addWidget(daily_btn)
        daily_toolbar.addStretch()

        self.daily_table = ModernTable(["ID", "Student ID", "Name", "Department", "Date", "Time", "Status", "Confidence"])
        dl.addWidget(self.daily_table, 1)
        tabs.addTab(daily, "Daily")

        # Monthly tab
        monthly = QWidget()
        ml = QVBoxLayout(monthly)
        ml.setContentsMargins(0, 12, 0, 0)
        monthly_toolbar = QHBoxLayout()
        ml.addLayout(monthly_toolbar)
        ml1 = QLabel("Month:")
        ml1.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; background: transparent;")
        self.monthly_month = QComboBox()
        for i in range(1, 13):
            self.monthly_month.addItem(date(2000, i, 1).strftime("%B"), i)
        self.monthly_month.setCurrentIndex(date.today().month - 1)
        self.monthly_month.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 24px;
            }}
        """)
        ml2 = QLabel("Year:")
        ml2.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; background: transparent;")
        self.monthly_year = QSpinBox()
        self.monthly_year.setRange(2020, 2099)
        self.monthly_year.setValue(date.today().year)
        self.monthly_year.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 24px;
            }}
        """)
        mbtn = AnimatedButton("View", "fa5s.eye", style="primary")
        mbtn.clicked.connect(self.refresh_monthly)
        monthly_toolbar.addWidget(ml1)
        monthly_toolbar.addWidget(self.monthly_month)
        monthly_toolbar.addWidget(ml2)
        monthly_toolbar.addWidget(self.monthly_year)
        monthly_toolbar.addWidget(mbtn)
        monthly_toolbar.addStretch()

        self.monthly_table = ModernTable(["ID", "Student ID", "Name", "Department", "Date", "Time", "Status", "Confidence"])
        ml.addWidget(self.monthly_table, 1)
        tabs.addTab(monthly, "Monthly")

        # History tab
        history = QWidget()
        hl = QVBoxLayout(history)
        hl.setContentsMargins(0, 12, 0, 0)
        htoolbar = QHBoxLayout()
        hl.addLayout(htoolbar)
        hl_search = SearchBox("Search by name or student ID...")
        hl_search.textChanged.connect(self.refresh_history)
        htoolbar.addWidget(hl_search)
        htoolbar.addStretch()
        export_btn = AnimatedButton("Export Excel", "fa5s.file-excel", style="success")
        export_btn.clicked.connect(self.export_excel)
        htoolbar.addWidget(export_btn)
        pdf_btn = AnimatedButton("Export PDF", "fa5s.file-pdf", style="danger")
        pdf_btn.clicked.connect(self.export_pdf)
        htoolbar.addWidget(pdf_btn)

        self.history_table = ModernTable(["ID", "Student ID", "Name", "Department", "Date", "Time", "Status", "Confidence"])
        hl.addWidget(self.history_table, 1)
        tabs.addTab(history, "History")

        self.refresh_daily()
        self.refresh_monthly()
        self.refresh_history()

    def refresh_daily(self):
        d = self.daily_date.date().toString("yyyy-MM-dd")
        rows = self.db.get_attendance(filter_date=d)
        data = []
        for r in rows:
            data.append([
                r['id'], r['student_id'], r['full_name'] or 'Unknown',
                r['department'] or '-', r['date'], r['time'],
                r['status'], f"{r['confidence']:.1f}%" if r['confidence'] else '-'
            ])
        self.daily_table.populate(data)
        self.daily_table.resizeColumnsToContents()

    def refresh_monthly(self):
        year = self.monthly_year.value()
        month = self.monthly_month.currentData()
        prefix = f"{year:04d}-{month:02d}"
        rows = self.db.get_attendance(filter_month=prefix)
        data = []
        for r in rows:
            data.append([
                r['id'], r['student_id'], r['full_name'] or 'Unknown',
                r['department'] or '-', r['date'], r['time'],
                r['status'], f"{r['confidence']:.1f}%" if r['confidence'] else '-'
            ])
        self.monthly_table.populate(data)
        self.monthly_table.resizeColumnsToContents()

    def refresh_history(self):
        search = self.history_table.parent().findChild(QLineEdit).text().strip() if False else ""
        # Find the search box via parent
        search_widget = self.findChild(SearchBox)
        if search_widget:
            search = search_widget.text().strip()
        rows = self.db.get_attendance(student_id=None)
        if search:
            rows = [r for r in rows if search.lower() in (r['full_name'] or '').lower() or search in r['student_id']]
        data = []
        for r in rows:
            data.append([
                r['id'], r['student_id'], r['full_name'] or 'Unknown',
                r['department'] or '-', r['date'], r['time'],
                r['status'], f"{r['confidence']:.1f}%" if r['confidence'] else '-'
            ])
        self.history_table.populate(data)
        self.history_table.resizeColumnsToContents()

    def export_excel(self):
        rows = self.db.get_attendance()
        if not rows:
            QMessageBox.information(self, "No Data", "No records to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel File",
                                              str(EXPORTS_DIR / f"attendance_{date.today()}.xlsx"),
                                              "Excel Files (*.xlsx)")
        if not path:
            return
        try:
            df = pd.DataFrame([dict(r) for r in rows])
            df.to_excel(path, index=False)
            QMessageBox.information(self, "Success", f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def export_pdf(self):
        if not REPORTLAB_AVAILABLE:
            QMessageBox.warning(self, "Missing Library",
                                "reportlab not installed. Install it with: pip install reportlab")
            return
        rows = self.db.get_attendance()
        if not rows:
            QMessageBox.information(self, "No Data", "No records to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF File",
                                              str(EXPORTS_DIR / f"attendance_{date.today()}.pdf"),
                                              "PDF Files (*.pdf)")
        if not path:
            return
        try:
            doc = SimpleDocTemplate(path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = [
                Paragraph(f"<b>Attendance Report</b>", styles['Title']),
                Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
                Spacer(1, 12)
            ]
            data = [["Student ID", "Name", "Department", "Date", "Time", "Status"]]
            for r in rows:
                data.append([
                    r['student_id'], r['full_name'] or '-', r['department'] or '-',
                    r['date'], r['time'], r['status']
                ])
            t = Table(data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(COLORS['primary'].lstrip('#'))),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ]))
            elements.append(t)
            doc.build(elements)
            QMessageBox.information(self, "Success", f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")


class ReportsPage(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(PageHeader("Reports & Export", "Generate and export attendance reports"))

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(12)

        def make_filter(label):
            l = QLabel(label)
            l.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; background: transparent;")
            return l

        filter_row.addWidget(make_filter("From:"))
        self.from_date = QDateEdit()
        self.from_date.setDate(QDate.currentDate().addDays(-30))
        self.from_date.setCalendarPopup(True)
        self._style_date(self.from_date)
        filter_row.addWidget(self.from_date)

        filter_row.addWidget(make_filter("To:"))
        self.to_date = QDateEdit()
        self.to_date.setDate(QDate.currentDate())
        self.to_date.setCalendarPopup(True)
        self._style_date(self.to_date)
        filter_row.addWidget(self.to_date)

        generate_btn = AnimatedButton("Generate Report", "fa5s.file-alt", style="primary")
        generate_btn.clicked.connect(self.generate)
        filter_row.addWidget(generate_btn)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # Stats summary
        self.summary_layout = QHBoxLayout()
        self.summary_layout.setSpacing(16)
        layout.addLayout(self.summary_layout)

        # Report table
        self.report_table = ModernTable(["Student ID", "Name", "Department", "Days Present", "Total Records"])
        layout.addWidget(self.report_table, 1)

        # Export buttons
        export_row = QHBoxLayout()
        export_row.addStretch()
        excel_btn = AnimatedButton("Export to Excel", "fa5s.file-excel", style="success")
        excel_btn.clicked.connect(self.export_excel)
        pdf_btn = AnimatedButton("Export to PDF", "fa5s.file-pdf", style="danger")
        pdf_btn.clicked.connect(self.export_pdf)
        export_row.addWidget(excel_btn)
        export_row.addWidget(pdf_btn)
        layout.addLayout(export_row)

    def _style_date(self, w):
        w.setStyleSheet(f"""
            QDateEdit {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 24px;
            }}
        """)

    def generate(self):
        # Clear summary
        for i in reversed(range(self.summary_layout.count())):
            item = self.summary_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        students = self.db.get_all_students()
        rows_data = []
        total_present = 0
        total_absent = 0
        total_days = 0

        for s in students:
            sid = s['student_id']
            atts = self.db.get_attendance(student_id=sid)
            days_present = len(set(a['date'] for a in atts))
            total_present += days_present
            rows_data.append([
                sid, s['full_name'], s['department'] or '-', days_present, len(atts)
            ])

        self.report_table.populate(rows_data)
        self.report_table.resizeColumnsToContents()

        # Summary cards
        c1 = ModernCard("Total Students", str(len(students)), "fa5s.users", COLORS['primary'])
        c2 = ModernCard("Total Records", str(sum(r[4] for r in rows_data)), "fa5s.list", COLORS['info'])
        c3 = ModernCard("Total Present Days", str(total_present), "fa5s.user-check", COLORS['success'])
        c4 = ModernCard("Average per Student",
                        f"{(total_present / len(students)):.1f}" if students else "0",
                        "fa5s.chart-pie", COLORS['warning'])
        self.summary_layout.addWidget(c1)
        self.summary_layout.addWidget(c2)
        self.summary_layout.addWidget(c3)
        self.summary_layout.addWidget(c4)

    def export_excel(self):
        if self.report_table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "Generate report first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel File",
                                              str(EXPORTS_DIR / f"report_{date.today()}.xlsx"),
                                              "Excel Files (*.xlsx)")
        if not path:
            return
        data = []
        for r in range(self.report_table.rowCount()):
            row = []
            for c in range(self.report_table.columnCount()):
                item = self.report_table.item(r, c)
                row.append(item.text() if item else '')
            data.append(row)
        df = pd.DataFrame(data, columns=["Student ID", "Name", "Department", "Days Present", "Total Records"])
        try:
            df.to_excel(path, index=False)
            QMessageBox.information(self, "Success", f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def export_pdf(self):
        if not REPORTLAB_AVAILABLE:
            QMessageBox.warning(self, "Missing Library", "Install reportlab: pip install reportlab")
            return
        if self.report_table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "Generate report first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF",
                                              str(EXPORTS_DIR / f"report_{date.today()}.pdf"),
                                              "PDF Files (*.pdf)")
        if not path:
            return
        try:
            doc = SimpleDocTemplate(path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = [
                Paragraph("<b>Attendance Summary Report</b>", styles['Title']),
                Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
                Spacer(1, 12)
            ]
            data = [["Student ID", "Name", "Department", "Days Present", "Total Records"]]
            for r in range(self.report_table.rowCount()):
                row = []
                for c in range(self.report_table.columnCount()):
                    item = self.report_table.item(r, c)
                    row.append(item.text() if item else '')
                data.append(row)
            t = Table(data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(COLORS['primary'].lstrip('#'))),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            elements.append(t)
            doc.build(elements)
            QMessageBox.information(self, "Success", f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")


class StatisticsPage(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(PageHeader("Statistics", "Detailed attendance analytics"))

        # Top stats
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.sc_total = ModernCard("Total Students", "0", "fa5s.users", COLORS['primary'])
        self.sc_present = ModernCard("Total Present Days", "0", "fa5s.user-check", COLORS['success'])
        self.sc_rate = ModernCard("Average Rate", "0%", "fa5s.chart-line", COLORS['warning'])
        self.sc_records = ModernCard("Total Records", "0", "fa5s.database", COLORS['info'])
        stats_row.addWidget(self.sc_total)
        stats_row.addWidget(self.sc_present)
        stats_row.addWidget(self.sc_rate)
        stats_row.addWidget(self.sc_records)
        layout.addLayout(stats_row)

        # Charts
        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        bar_card = QFrame()
        bar_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        bl = QVBoxLayout(bar_card)
        bl.setContentsMargins(20, 16, 20, 16)
        bh = QLabel("Daily Attendance - Last 30 Days")
        bh.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        bl.addWidget(bh)
        self.bar_chart = BarChartWidget()
        bl.addWidget(self.bar_chart)
        charts_row.addWidget(bar_card, 2)

        pie_card = QFrame()
        pie_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        pl = QVBoxLayout(pie_card)
        pl.setContentsMargins(20, 16, 20, 16)
        ph = QLabel("Today's Distribution")
        ph.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        pl.addWidget(ph)
        self.pie_chart = PieChartWidget()
        pl.addWidget(self.pie_chart)
        charts_row.addWidget(pie_card, 1)

        layout.addLayout(charts_row, 1)

        # Refresh
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(20000)
        self.refresh()

    def refresh(self):
        stats = self.db.attendance_stats()
        self.sc_total.setValue(str(stats['total_students']))
        self.sc_present.setValue(str(stats['present_today']))
        self.sc_rate.setValue(f"{stats['attendance_rate']:.1f}%")
        self.sc_records.setValue(str(stats['total_records']))

        self.pie_chart.set_data([
            ("Present", stats['present_today'], COLORS['success']),
            ("Absent", stats['absent_today'], COLORS['danger']),
        ])

        today_d = date.today()
        labels = []
        values = []
        for i in range(29, -1, -1):
            d = today_d - timedelta(days=i)
            rows = self.db.get_attendance(filter_date=d.isoformat())
            labels.append(d.strftime("%d"))
            values.append(len(set(r['student_id'] for r in rows)))
        self.bar_chart.set_data(values, labels)


class SettingsPage(QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(PageHeader("Settings", "Application preferences and configuration"))

        content = QHBoxLayout()
        content.setSpacing(16)

        # Theme settings
        theme_card = QFrame()
        theme_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        tl = QVBoxLayout(theme_card)
        tl.setContentsMargins(20, 20, 20, 20)
        tt = QLabel("Appearance")
        tt.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        tl.addWidget(tt)
        tl.addSpacing(8)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark (Default)", "Light"])
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
        """)
        tl.addWidget(QLabel("Theme:"))
        tl.addWidget(self.theme_combo)
        tl.addSpacing(16)

        # Camera index
        tl.addWidget(QLabel("Camera Index:"))
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 5)
        self.camera_index.setValue(int(self.db.get_setting('camera_index', '0')))
        self.camera_index.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
        """)
        tl.addWidget(self.camera_index)
        tl.addSpacing(16)

        # Auto-start attendance
        self.auto_start = QCheckBox("Auto-mark attendance on recognition")
        self.auto_start.setChecked(self.db.get_setting('auto_mark', '1') == '1')
        self.auto_start.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; background: transparent;")
        tl.addWidget(self.auto_start)
        tl.addSpacing(8)
        self.show_confidence = QCheckBox("Show confidence in attendance")
        self.show_confidence.setChecked(self.db.get_setting('show_conf', '1') == '1')
        self.show_confidence.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; background: transparent;")
        tl.addWidget(self.show_confidence)

        tl.addStretch()
        save_btn = AnimatedButton("Save Settings", "fa5s.save", style="primary")
        save_btn.clicked.connect(self.save_settings)
        tl.addWidget(save_btn)
        content.addWidget(theme_card, 1)

        # About
        about_card = QFrame()
        about_card.setStyleSheet(f"background-color: {COLORS['surface']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        al = QVBoxLayout(about_card)
        al.setContentsMargins(20, 20, 20, 20)
        at = QLabel("About")
        at.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: 700; background: transparent;")
        al.addWidget(at)
        al.addSpacing(8)

        about_text = QLabel(
            f"<b style='color:{COLORS['text_primary']}'>{APP_NAME}</b><br>"
            f"<span style='color:{COLORS['text_secondary']}'>Version: {APP_VERSION}</span><br><br>"
            f"<span style='color:{COLORS['text_secondary']}'>"
            "A modern face recognition attendance system built with PySide6, "
            "OpenCV, and the face_recognition library. Features real-time "
            "recognition, automatic attendance, and comprehensive reporting."
            "</span><br><br>"
            f"<span style='color:{COLORS['text_secondary']}'><b>Default Admin:</b> admin / admin123</span>"
        )
        about_text.setWordWrap(True)
        about_text.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; background: transparent;")
        al.addWidget(about_text)

        al.addSpacing(12)
        sys_info = QLabel(
            f"<span style='color:{COLORS['text_secondary']}'>"
            f"<b>System Info:</b><br>"
            f"face_recognition: {'Available' if FACE_RECOGNITION_AVAILABLE else 'Not Available'}<br>"
            f"reportlab (PDF): {'Available' if REPORTLAB_AVAILABLE else 'Not Available'}<br>"
            f"OpenCV: {cv2.__version__}<br>"
            f"Database: {DB_PATH.name}<br>"
            f"Total Students: {self.db.attendance_stats()['total_students']}"
            "</span>"
        )
        sys_info.setWordWrap(True)
        sys_info.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; background: transparent;")
        al.addWidget(sys_info)
        al.addStretch()

        content.addWidget(about_card, 1)
        layout.addLayout(content, 1)

    def save_settings(self):
        self.db.set_setting('camera_index', str(self.camera_index.value()))
        self.db.set_setting('auto_mark', '1' if self.auto_start.isChecked() else '0')
        self.db.set_setting('show_conf', '1' if self.show_confidence.isChecked() else '0')
        QMessageBox.information(self, "Saved", "Settings saved successfully")


# ----------------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 850)
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']};")

        # Central widget
        central = QWidget()
        central.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = AnimatedSidebar()
        self.sidebar.page_changed.connect(self.switch_page)
        main_layout.addWidget(self.sidebar)

        # Right side: header + content
        right_side = QWidget()
        right_side.setStyleSheet("background: transparent;")
        rs_layout = QVBoxLayout(right_side)
        rs_layout.setContentsMargins(0, 0, 0, 0)
        rs_layout.setSpacing(0)

        # Top header
        self.top_bar = self._create_top_bar()
        rs_layout.addWidget(self.top_bar)

        # Stacked content
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        rs_layout.addWidget(self.stack, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border']};
                font-size: 12px;
                padding: 4px;
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        main_layout.addWidget(right_side, 1)

        # Pages
        self.pages = []
        self.dashboard_page = DashboardPage(db)
        self.students_page = StudentsPage(db)
        self.register_page = FaceRegistrationPage(db)
        self.recognition_page = LiveRecognitionPage(db)
        self.attendance_page = AttendancePage(db)
        self.reports_page = ReportsPage(db)
        self.statistics_page = StatisticsPage(db)
        self.settings_page = SettingsPage(db)

        for page in [self.dashboard_page, self.students_page, self.register_page,
                     self.recognition_page, self.attendance_page, self.reports_page,
                     self.statistics_page, self.settings_page]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(f"""
                QScrollArea {{
                    background-color: {COLORS['bg_dark']};
                    border: none;
                }}
                QScrollBar:vertical {{
                    background: {COLORS['bg_medium']};
                    width: 10px;
                    border-radius: 5px;
                }}
                QScrollBar::handle:vertical {{
                    background: {COLORS['border']};
                    border-radius: 5px;
                    min-height: 30px;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: {COLORS['primary']};
                }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                    border: none; background: none; height: 0;
                }}
            """)
            wrapper = QWidget()
            wrapper.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(24, 24, 24, 24)
            wl.addWidget(page)
            scroll.setWidget(wrapper)
            self.stack.addWidget(scroll)
            self.pages.append(scroll)

        # Clock timer
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self.update_clock)
        self._clock_timer.start(1000)
        self.update_clock()

    def _create_top_bar(self):
        bar = QFrame()
        bar.setFixedHeight(70)
        bar.setStyleSheet(f"""
            background-color: {COLORS['bg_medium']};
            border-bottom: 1px solid {COLORS['border']};
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        # Page title
        self.page_title = QLabel("Dashboard")
        self.page_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 16px; font-weight: 700; background: transparent;")

        layout.addWidget(self.page_title)
        layout.addStretch()

        # Search button (global)
        search_btn = QPushButton()
        search_btn.setIcon(qta.icon('fa5s.search', color=COLORS['text_secondary']))
        search_btn.setFixedSize(40, 40)
        search_btn.setCursor(Qt.PointingHandCursor)
        search_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface']};
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
            }}
        """)
        layout.addWidget(search_btn)

        # Notifications button
        self.notif_btn = QPushButton()
        self.notif_btn.setIcon(qta.icon('fa5s.bell', color=COLORS['text_secondary']))
        self.notif_btn.setFixedSize(40, 40)
        self.notif_btn.setCursor(Qt.PointingHandCursor)
        self.notif_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['surface']};
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_light']};
            }}
        """)
        self.notif_btn.clicked.connect(self.show_notifications)
        layout.addWidget(self.notif_btn)

        # Clock
        clock_widget = QFrame()
        clock_widget.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            border-radius: 10px;
        """)
        cl = QHBoxLayout(clock_widget)
        cl.setContentsMargins(12, 0, 12, 0)
        cl.setSpacing(8)

        clock_icon = QLabel()
        clock_icon.setPixmap(qta.icon('fa5s.clock', color=COLORS['primary']).pixmap(16, 16))
        clock_icon.setStyleSheet("background: transparent;")
        self.clock_label = QLabel("--:--:--")
        self.clock_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 13px; background: transparent;")
        self.date_label = QLabel("-")
        self.date_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;")
        clock_v = QVBoxLayout()
        clock_v.setSpacing(0)
        clock_v.addWidget(self.clock_label)
        clock_v.addWidget(self.date_label)
        cl.addWidget(clock_icon)
        cl.addLayout(clock_v)
        layout.addWidget(clock_widget)

        # User profile
        user_widget = QFrame()
        user_widget.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            border-radius: 10px;
        """)
        ul = QHBoxLayout(user_widget)
        ul.setContentsMargins(8, 4, 14, 4)
        ul.setSpacing(10)

        avatar = QLabel("A")
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"""
            background-color: {COLORS['primary']};
            color: white;
            border-radius: 16px;
            font-size: 14px;
            font-weight: 800;
        """)

        info = QVBoxLayout()
        info.setSpacing(0)
        name_lbl = QLabel("Administrator")
        name_lbl.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 12px; background: transparent;")
        role_lbl = QLabel("Super Admin")
        role_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px; background: transparent;")
        info.addWidget(name_lbl)
        info.addWidget(role_lbl)
        ul.addWidget(avatar)
        ul.addLayout(info)

        # Logout button
        logout_btn = QPushButton()
        logout_btn.setIcon(qta.icon('fa5s.sign-out-alt', color=COLORS['danger']))
        logout_btn.setFixedSize(32, 32)
        logout_btn.setCursor(Qt.PointingHandCursor)
        logout_btn.setToolTip("Logout")
        logout_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']}33;
            }}
        """)
        logout_btn.clicked.connect(self.logout)
        ul.addWidget(logout_btn)

        layout.addWidget(user_widget)

        return bar

    def update_clock(self):
        now = datetime.now()
        self.clock_label.setText(now.strftime("%H:%M:%S"))
        self.date_label.setText(now.strftime("%a, %b %d"))

    def switch_page(self, idx: int):
        titles = ["Dashboard", "Students", "Face Registration", "Live Recognition",
                  "Attendance", "Reports", "Statistics", "Settings"]
        if idx < len(titles):
            self.page_title.setText(titles[idx])

        # Animate transition
        current = self.stack.currentWidget()
        new_widget = self.pages[idx]
        if current is new_widget:
            return

        # Fade out current
        fade_out = QPropertyAnimation(current, b"windowOpacity")
        fade_out.setDuration(120)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.finished.connect(lambda: self._do_switch(idx))
        fade_out.start()
        self._fade_out = fade_out

    def _do_switch(self, idx):
        self.stack.setCurrentIndex(idx)
        new_widget = self.pages[idx]
        new_widget.setWindowOpacity(0)
        fade_in = QPropertyAnimation(new_widget, b"windowOpacity")
        fade_in.setDuration(160)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.start()
        self._fade_in = fade_in
        self.status_bar.showMessage(f"Viewing: {self.page_title.text()}")

    def show_notifications(self):
        notifs = self.db.get_notifications(10)
        dlg = QDialog(self)
        dlg.setWindowTitle("Notifications")
        dlg.resize(500, 500)
        dlg.setStyleSheet(f"background-color: {COLORS['bg_medium']};")
        l = QVBoxLayout(dlg)
        l.setContentsMargins(20, 20, 20, 20)
        title = QLabel("Notifications")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 18px; font-weight: 800; background: transparent;")
        l.addWidget(title)

        if not notifs:
            empty = QLabel("No notifications")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 40px; background: transparent;")
            l.addWidget(empty)
        else:
            for n in notifs:
                card = QFrame()
                card.setStyleSheet(f"""
                    background-color: {COLORS['surface']};
                    border-radius: 10px;
                    border-left: 4px solid {COLORS['primary']};
                """)
                cl = QVBoxLayout(card)
                cl.setContentsMargins(14, 10, 14, 10)
                t = QLabel(n['title'])
                t.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 13px; background: transparent;")
                m = QLabel(n['message'])
                m.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;")
                m.setWordWrap(True)
                ts = QLabel(n['created_at'][:19].replace('T', ' '))
                ts.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; background: transparent;")
                cl.addWidget(t)
                cl.addWidget(m)
                cl.addWidget(ts)
                l.addWidget(card)

        l.addStretch()
        dlg.exec()
        self.db.mark_notifications_read()

    def logout(self):
        reply = QMessageBox.question(self, "Logout", "Are you sure you want to logout?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            self.login = LoginWindow(self.db)
            self.login.show()

    def keyPressEvent(self, event):
        # Keyboard shortcuts
        if event.modifiers() == Qt.ControlModifier:
            shortcuts = {
                Qt.Key_1: 0, Qt.Key_2: 1, Qt.Key_3: 2, Qt.Key_4: 3,
                Qt.Key_5: 4, Qt.Key_6: 5, Qt.Key_7: 6, Qt.Key_8: 7,
            }
            if event.key() in shortcuts:
                self.switch_page(shortcuts[event.key()])
        elif event.key() == Qt.Key_F5:
            self.dashboard_page.refresh()


# ----------------------------------------------------------------------------
# Splash Screen
# ----------------------------------------------------------------------------
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(520, 320)

        # Build container with gradient background
        container = QFrame(self)
        container.setGeometry(0, 0, 520, 320)
        container.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {COLORS['primary']}, stop:1 {COLORS['primary_dark']});
                border-radius: 20px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 100))
        container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(12)

        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.fingerprint', color='white').pixmap(80, 80))
        icon_label.setStyleSheet("background: transparent;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label, alignment=Qt.AlignCenter)

        title = QLabel("Face Recognition Attendance")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; font-size: 20px; font-weight: 800; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("Loading application...")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255,255,255,0.85); font-size: 12px; background: transparent;")
        layout.addWidget(subtitle)

        # Spinner
        self.spinner = LoadingSpinner(36, 'white', self)
        spinner_holder = QFrame()
        spinner_holder.setStyleSheet("background: transparent;")
        sl = QVBoxLayout(spinner_holder)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.spinner, alignment=Qt.AlignCenter)
        layout.addWidget(spinner_holder)

        layout.addStretch()

        version = QLabel(f"v{APP_VERSION}  •  Premium Edition")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 11px; background: transparent;")
        layout.addWidget(version)

        # Center on screen
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move((geo.width() - self.width()) // 2,
                      (geo.height() - self.height()) // 2)

        self.setWindowOpacity(0)


# ----------------------------------------------------------------------------
# Application Entry
# ----------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")

    # Splash
    splash = SplashScreen()
    splash.show()
    anim = QPropertyAnimation(splash, b"windowOpacity")
    anim.setDuration(400)
    anim.setStartValue(0)
    anim.setEndValue(1)
    anim.start()

    # Init database
    db = Database()

    # Process events to show splash
    app.processEvents()
    QTimer.singleShot(1500, lambda: finish_splash(splash, db))

    return app.exec()


def finish_splash(splash, db):
    splash.close()
    splash.deleteLater()
    login = LoginWindow(db)
    login.show()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Fatal error: {e}")
        sys.exit(1)
