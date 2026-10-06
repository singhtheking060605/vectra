@echo off
cd /d "%~dp0"
set "M=C:\Users\ARSHSH~1\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64"
set "PATH=%M%\bin;%PATH%"
echo [VECTRA] Starting on port 8081...
db.exe 8081
