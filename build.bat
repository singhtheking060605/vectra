@echo off
cd /d "%~dp0"
set "M=C:\Users\AKASHS~1\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64"
set "PATH=%M%\bin;C:\Windows\system32;C:\Windows"
taskkill /F /IM db.exe >nul 2>&1

echo [INFO] Compiling main.cpp...
"%M%\bin\g++.exe" -std=c++17 -O3 -mavx2 -mfma -mstackrealign -c main.cpp -o main.o
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] G++ Compilation Failed!
    exit /b 1
)

echo [INFO] Linking db.exe with full C++ runtime...
"%M%\bin\ld.exe" -m i386pep -Bstatic -o db.exe "%M%\x86_64-w64-mingw32\lib\crt2.o" "%M%\lib\gcc\x86_64-w64-mingw32\16.1.0\crtbegin.o" main.o -L"%M%\lib\gcc\x86_64-w64-mingw32\16.1.0" -L"%M%\x86_64-w64-mingw32\lib" -L"%M%\lib" -lstdc++ -lmingw32 -lgcc -lgcc_eh -lpthread -lmingwex -lucrt -lkernel32 -lws2_32 -ladvapi32 -lshell32 -luser32 "%M%\lib\gcc\x86_64-w64-mingw32\16.1.0\crtend.o"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] LD Link Failed!
    exit /b 1
)

echo [SUCCESS] db.exe built successfully.

