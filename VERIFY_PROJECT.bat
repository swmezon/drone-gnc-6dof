@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
set PYTHONPATH=src
python -m pytest -q
if errorlevel 1 exit /b 1
python scripts\run_waypoint_demo.py
if errorlevel 1 exit /b 1
python scripts\run_monte_carlo.py
if errorlevel 1 exit /b 1
echo.
echo Verification completed successfully.
pause
