@echo off
title "VECTRA - Vector Database Engine & Dashboard"
echo ========================================================
echo   Launching VECTRA C++ Vector Database (Milestone 2)
echo ========================================================

tasklist /FI "IMAGENAME eq db.exe" 2>NUL | find /I /N "db.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] VECTRA C++ Engine is already running on port 8080.
) else (
    echo [INFO] Starting VECTRA C++ Server with AVX2 SIMD acceleration...
    start "" /B db.exe
    ping -n 2 127.0.0.1 >nul
)

echo [INFO] Opening Live Dashboard in your default browser...
start http://localhost:8080
echo ========================================================
echo   Engine Ready at: http://localhost:8080
echo ========================================================
pause

