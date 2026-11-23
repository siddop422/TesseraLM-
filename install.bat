@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.11, 3.12, 3.13, or 3.14 from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3.14 -m venv .venv 2>nul
  if errorlevel 1 py -3.13 -m venv .venv 2>nul
  if errorlevel 1 py -3.12 -m venv .venv 2>nul
  if errorlevel 1 py -3.11 -m venv .venv 2>nul
)
if not exist ".venv\Scripts\python.exe" (
  echo Python 3.11-3.14 was not found.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
echo Setup complete. Run start.bat to train Nano on first launch and open the chat.
pause
exit /b 0
:failed
echo Installation failed. Check the error above.
pause
exit /b 1
