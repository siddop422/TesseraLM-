@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\evaluate.py --checkpoint checkpoints\nano.pt --prompt "A language model learns" --max-new-tokens 80
pause
