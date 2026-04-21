# Face Recognition Management System

This project can run on Windows. The main app now starts even if the native face-recognition dependency is not ready yet, so you can use the normal Flask pages and database features first.

## Windows Setup

1. Install Python 3.10 or 3.11.
2. Open PowerShell in [yo](D:\year 3\fyp\fynal year project (Face recognition management system )\yo).
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install the basic app dependencies:

```powershell
pip install Flask numpy opencv-python cmake
```

5. Install `dlib`.

On Windows this is the only part that is often different from macOS. If `pip install dlib` fails, install Visual Studio Build Tools with C++ workload, then try again inside the same virtual environment.

```powershell
pip install dlib
```

6. Start the app:

```powershell
python .\app.py
```

7. Open `http://127.0.0.1:5000`

## Notes

- The database file is [database.db](D:\year 3\fyp\fynal year project (Face recognition management system )\yo\database.db).
- Face model files must stay inside [yo\face_models](D:\year 3\fyp\fynal year project (Face recognition management system )\yo\face_models).
- If browser camera access is blocked, you can run the app with local HTTPS:

```powershell
$env:APP_USE_HTTPS="true"
python .\app.py
```

- If `dlib` is missing, the app will still open, and the face pages will show a clear dependency message instead of crashing.
# Final-year-project
# Final-year-project
