@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe trace_demo.py
echo.
echo Done. Open trace_demo.json at https://ui.perfetto.dev
pause
