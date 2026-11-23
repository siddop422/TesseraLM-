@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m training.train --data examples/seed_corpus.txt --config configs/nano.json --steps 101 --output checkpoints/nano.pt
pause
