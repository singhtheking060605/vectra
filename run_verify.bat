@echo off
cd /d "%~dp0"
taskkill /F /IM db.exe >nul 2>&1
start "" "%~dp0db.exe"
ping -n 3 127.0.0.1 >nul
test_http_client.exe
taskkill /F /IM db.exe >nul 2>&1
