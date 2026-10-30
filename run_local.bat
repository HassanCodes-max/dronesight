@echo off
rem Starts DroneSight on this computer. Close both windows to stop it.
cd /d "%~dp0"
python -m dronesight.app.seed demo/flight.mp4 "Sample flight"
start "DroneSight worker" python -m dronesight.app.worker
start "" http://127.0.0.1:8000
python -m uvicorn dronesight.app.main:app --port 8000 --no-server-header
