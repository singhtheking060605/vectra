@echo off
cd /d "%~dp0"
title VECTRA - Vector Database Engine and Dashboard

echo ========================================================
echo   Launching VECTRA C++ Vector Database (AVX2 SIMD)
echo ========================================================

set "M=C:\Users\AKASHS~1\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64"
set "PATH=%M%\bin;%PATH%;C:\Windows\system32;C:\Windows"

set "PORT=8081"
if not "%~1"=="" set "PORT=%~1"

echo [INFO] Cleaning up previous db.exe instances...
taskkill /F /IM db.exe >nul 2>&1

echo [INFO] Opening Live Dashboard in your default browser on port %PORT%...
start http://localhost:%PORT%

echo [INFO] Starting VECTRA C++ Engine on port %PORT%...
echo [INFO] (Keep this terminal window open while using the app)
echo ========================================================
echo.

"%~dp0db.exe" %PORT%

echo.
echo ========================================================
echo [STOPPED] VECTRA Engine process has terminated.
echo ========================================================
pause
