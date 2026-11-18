@echo off
REM Laptop 1 (Windows): double-click to start the PathSnap API.
cd /d "%~dp0.."
ipconfig | findstr IPv4
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
