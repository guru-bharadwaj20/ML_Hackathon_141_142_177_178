@echo off
REM Laptop 2 (Windows): double-click, then type laptop 1 IP.
cd /d "%~dp0.."
set /p IP=Laptop 1 IP: 
set PATHSNAP_API_URL=http://%IP%:8000
curl -s %PATHSNAP_API_URL%/health
python -m streamlit run ui\app.py
pause
