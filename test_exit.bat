@echo off
cd /d "%~dp0"
echo [TEST] Running db.exe directly...
db.exe
echo [TEST] db.exe exited with ERRORLEVEL: %ERRORLEVEL%
