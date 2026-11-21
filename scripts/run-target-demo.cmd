@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Create the .venv first. See README.md.
  pause
  exit /b 1
)
if not exist "data\demo\target_demo.avi" (
  ".venv\Scripts\python.exe" -m tools.generate_demo
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" main.py --source data/demo/target_demo.avi --save-video
if errorlevel 1 pause
