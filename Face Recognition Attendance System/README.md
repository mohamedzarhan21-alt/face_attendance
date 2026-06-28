# Face Recognition Attendance System

A premium desktop application for tracking student attendance using real-time face recognition. Built with **Python 3.11 + PySide6**, OpenCV, and a single self-contained source file so deployment is trivial.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [First-Time Setup](#first-time-setup)
- [Usage Walkthrough](#usage-walkthrough)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [How Recognition Works](#how-recognition-works)
- [Exporting Reports](#exporting-reports)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

---

## Features

### Core
- **Admin login** with hashed password storage (SHA-256)
- **Student management** — add, edit, delete, search by ID / name / department
- **Face registration** — capture multiple images per student with live preview
- **Live face recognition** — automatic attendance marking, no duplicates
- **Attendance history** — daily and monthly views with filters
- **Excel & PDF export** of attendance and student data
- **Statistics dashboard** with bar charts and pie charts
- **Recent activity log** and in-app notification panel
- **Dark / light theme** with one-click toggle
- **Settings page** for tolerance, theme, and admin password change
- **Animated UI** — page transitions, sidebar collapse, hover effects, fade-ins, sliding panels

### UI Highlights
- Custom widgets: `AnimatedButton`, `ModernCard`, `StatisticCard`, `AnimatedSidebar`, `SearchBox`, `RoundedLineEdit`, `ModernTable`, `CameraWidget`, `NotificationWidget`, `LoadingSpinner`
- SVG / qtawesome icons throughout
- Responsive layout that scales with the window

---

## Tech Stack

| Layer            | Library                                |
|------------------|----------------------------------------|
| GUI              | PySide6 (Qt 6 for Python)              |
| Styling          | qt-material, qtawesome                 |
| Vision           | OpenCV (`opencv-python`)               |
| Face recognition | OpenCV LBPH (fallback) — dlib optional |
| Database         | SQLite (built into Python)             |
| Data             | numpy, pandas                          |
| Reports          | openpyxl (Excel), reportlab (PDF)      |
| Serialization    | pickle, hashlib                        |

The application is implemented in a **single Python file** (`face_attendance.py`) for portability.

---

## Requirements

- **Python 3.11** (3.10+ should also work; 3.11 was used during development)
- **Windows 10/11** (tested) — should also run on macOS / Linux with a webcam
- **Webcam** (built-in or USB)
- ~500 MB free disk space

> **Note about `dlib` / `face_recognition`**: these libraries require a C++ compiler (MSVC on Windows) and are difficult to install on plain Python. The application auto-detects them; if absent, it gracefully falls back to **OpenCV's LBPH recognizer**, which works well for controlled classroom lighting.

---

## Installation

```powershell
# 1. (Optional) create a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. Install required packages
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install PySide6 qt-material qtawesome opencv-python numpy pandas openpyxl reportlab
```

> If you want to use the `face_recognition` library and have MSVC build tools:
> ```powershell
> py -3.11 -m pip install cmake
> py -3.11 -m pip install dlib
> py -3.11 -m pip install face_recognition
> ```

---

## Running

```powershell
cd "E:\Face Recognition Attendance System"
py -3.11 face_attendance.py
```

A splash screen appears, then the login window.

**Default admin credentials** (created on first run):
- Username: `admin`
- Password: `admin123`

> Change the password immediately from **Settings → Change Password** after first login.

---

## First-Time Setup

1. Launch the app and log in with `admin / admin123`.
2. Go to **Students → Add Student**. Enter:
   - Student ID (e.g. `2301205`)
   - Full name
   - Department / class
   - Email (optional)
3. Open **Face Registration**:
   - Pick the student from the list
   - Click **Start Camera**
   - Click **Capture** 4–6 times with the student facing the camera at slight angles
   - Click **Save & Train** — the LBPH model is retrained immediately
4. Open **Live Recognition** and click **Start Camera** to test.

---

## Usage Walkthrough

### Dashboard
KPI cards: total students, present today, absent today, total attendance records, plus a recent-activity feed and quick charts.

### Students Page
- Search by ID, name, or department (live filter)
- Edit / delete via row buttons (with confirmation)
- Add student via the **+ Add** button (animated side panel)

### Face Registration
- Select a student → **Start Camera** → position face inside the guide box → **Capture**
- Multiple captures improve recognition accuracy
- **Save & Train** persists images to `faces/<student_id>_<n>.jpg` and retrains the LBPH model

### Live Recognition
- Click **Start Camera** — recognized faces show name + confidence, unknown faces show **UNKNOWN**
- Attendance is auto-marked the first time a face is seen each day (5-second debounce per student)
- "Already marked" prevents duplicate entries
- The right panel shows a live activity log and current-match info

### Attendance Page
- Filter by date / month / student
- Table updates live
- **Refresh** reloads from DB

### Reports / Export
- **Export to Excel** — `exports/attendance_<date>.xlsx`
- **Export to PDF** — `exports/attendance_<date>.pdf`
- Both include header, table, and timestamp

### Statistics
- Bar chart: top 5 most-present students
- Pie chart: present / absent ratio (today or selected month)

### Settings
- Toggle dark / light theme
- Adjust recognition tolerance (when `face_recognition` is available)
- Change admin password
- Clear notifications

---

## Project Structure

```
Face Recognition Attendance System/
├── face_attendance.py        # The entire application (single file, ~3400 lines)
├── attendance.db             # SQLite database (created on first run)
├── faces/                    # Captured face images
│   ├── 2301205_0.jpg
│   ├── 2301205_1.jpg
│   └── _lbph/                # (optional) cached training images for the LBPH model
├── exports/                  # Generated Excel / PDF reports
├── icons/                    # Cached SVG icons (auto-created)
├── attendance_system.log     # Rotating app log
└── README.md                 # This file
```

---

## Database Schema

```sql
admins (id, username, password_hash, created_at)

students (
  id INTEGER PRIMARY KEY,
  student_id TEXT UNIQUE,       -- public ID, e.g. "2301205"
  full_name TEXT,
  department TEXT,
  email TEXT,
  phone TEXT,
  face_encoding BLOB,           -- pickled encoding (if face_recognition available)
  image_path TEXT,              -- path to primary face image
  created_at TEXT
)

attendance (
  id INTEGER PRIMARY KEY,
  student_id TEXT NOT NULL,
  date TEXT NOT NULL,
  time TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence REAL,
  FOREIGN KEY(student_id) REFERENCES students(student_id)
)

settings (key TEXT PRIMARY KEY, value TEXT)
notifications (id, title, message, type, created_at, is_read)
```

A unique constraint on `(student_id, date)` is enforced in code to prevent duplicate attendance.

---

## How Recognition Works

The app uses a **two-tier pipeline**:

1. **Detection** — always uses OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`). Fast and dependency-free.
2. **Identification** — picks the best available method:
   - **`face_recognition` (dlib)** if installed → 128-d face embeddings + Euclidean distance
   - **LBPH fallback** (always available) → histogram of local binary patterns, trained on the registered images

Confidence display:
- For LBPH, the raw distance is converted to a 0–100% score (distance ≤ 30 → 100%, threshold 70 → 0%)
- For dlib, `(1 − distance) × 100` is used

The model is **retrained automatically**:
- Whenever new face images are saved (Face Registration → Save & Train)
- When the Live Recognition page opens (auto-train from all images in `faces/_lbph`)

---

## Exporting Reports

| Button           | Output                                                     |
|------------------|------------------------------------------------------------|
| Export Excel     | `exports/attendance_<YYYY-MM-DD>.xlsx` (multi-sheet)       |
| Export PDF       | `exports/attendance_<YYYY-MM-DD>.pdf` (formatted table)    |
| Export Students  | `exports/students_<YYYY-MM-DD>.xlsx`                       |

`openpyxl` and `reportlab` are required for these features. The app detects missing libs and disables the corresponding button.

---

## Troubleshooting

| Problem                                  | Solution                                                                 |
|------------------------------------------|--------------------------------------------------------------------------|
| `ModuleNotFoundError: PySide6`           | `py -3.11 -m pip install PySide6`                                        |
| Camera doesn't open                      | Close other apps using the webcam (Zoom, Skype, browser). Try index 0/1. |
| All faces show as UNKNOWN                | Re-register the student with better-lit, frontal captures.                |
| `sqlite3.IntegrityError` in log          | Already fixed — `mark_attendance` validates FKs before insert.           |
| Confidence stuck below 70%               | Adjust lighting; re-register with more images; threshold is 70 distance. |
| App crashes on startup                   | Check `attendance_system.log` for the traceback.                          |
| `qtawesome` uses wrong Qt binding        | The app sets `QT_API=pyside6` automatically — no action needed.           |
| Splash screen QPainter error             | Already fixed — splash uses QWidget layouts, not QPainter pixmaps.        |

### Reset the database

```powershell
del attendance.db
```

The app will recreate it on next launch. (You will lose all students and attendance — back up first.)

---

## Known Limitations

- **Single admin** — multi-user role support is not implemented.
- **LBPH is sensitive to lighting** — performance drops in very dim or backlit scenes. A controlled classroom environment is recommended.
- **No anti-spoofing** — a printed photo will be accepted. For real security, add liveness detection (blink / depth).
- **Camera index 0 only** by default. Multi-camera setups require editing the code.
- **No network sync** — database is local SQLite. For multi-terminal setups, swap in a server-based DB.
- **`dlib` is not bundled** because of MSVC dependency. The OpenCV LBPH fallback is used.

---

## License

Internal / educational use. Customize freely.

---

## Credits

- Icons by [qtawesome](https://github.com/spyder-ide/qtawesome) (Font Awesome 5)
- Material styling by [qt-material](https://github.com/UN-GCPDS/qt-material)
- Built with [PySide6](https://doc.qt.io/qtforpython-6/) and [OpenCV](https://opencv.org/)
