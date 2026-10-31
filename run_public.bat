@echo off
rem Puts DroneSight online from this computer through a temporary Cloudflare link.
rem The link appears in this window and changes every time. Close all three windows to go offline.
rem Data stays outside this folder because it is in OneDrive, and uploads must not sync there.
cd /d "%~dp0"
set DRONESIGHT_ENV=production
set DRONESIGHT_CLIENT_IP_HEADER=Cf-Connecting-Ip
set DRONESIGHT_DATA=%LOCALAPPDATA%\DroneSight
python -m dronesight.app.seed demo/flight.mp4 "Sample flight"
start "DroneSight worker" python -m dronesight.app.worker
start "DroneSight server" python -m uvicorn dronesight.app.main:app --host 127.0.0.1 --port 8000 --no-server-header --limit-concurrency 64
cloudflared tunnel --url http://127.0.0.1:8000
