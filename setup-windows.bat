@echo off
REM Smart Vision Assistant - setup for Windows
REM Run from the repository root: setup-windows.bat

cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment .venv ...
    python -m venv .venv || py -3 -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create .venv. Check that Python 3.11+ is installed.
        exit /b 1
    )
)

echo Installing packages (the first time can take several minutes) ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: package installation failed. Copy the full message above.
    exit /b 1
)

echo.
echo Setup finished. Checking the environment ...
.venv\Scripts\python.exe scripts\00_check_environment.py
