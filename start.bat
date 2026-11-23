@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
if not exist "checkpoints\nano.pt" (
  echo No Nano checkpoint found. Training the included sample model first...
  call train.bat
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m ui.server --checkpoint checkpoints/nano.pt
pause
