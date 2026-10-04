@echo off
cd /d "%~dp0"
echo Show rock / paper / scissors to the webcam. Press q in the window to quit.
".venv\Scripts\python.exe" EX_03_PC_Webcam_RPS_YOLO_ONNX.py %*
pause
