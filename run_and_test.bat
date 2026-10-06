@echo off
taskkill /F /IM db.exe >nul 2>&1
start /B "" "%~dp0db.exe" > server_out.log 2> server_err.log
ping -n 2 127.0.0.1 >nul
test_http_client.exe
echo ExitCode = %ERRORLEVEL%
taskkill /F /IM db.exe >nul 2>&1
echo --- SERVER OUT ---
type server_out.log
echo --- SERVER ERR ---
type server_err.log
