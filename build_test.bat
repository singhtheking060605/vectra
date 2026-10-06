@echo off
set "M=C:\Users\ARSHSH~1\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64"

"%M%\bin\g++.exe" -std=c++17 -O2 -mavx2 -mfma -c test_all_cpp.cpp -o test_all_cpp.o
if %ERRORLEVEL% NEQ 0 exit /b 1

"%M%\bin\ld.exe" -m i386pep -Bdynamic -o test_all_cpp.exe "%M%\x86_64-w64-mingw32\lib\crt2.o" "%M%\lib\gcc\x86_64-w64-mingw32\16.2.0\crtbegin.o" test_all_cpp.o -L"%M%\lib\gcc\x86_64-w64-mingw32\16.2.0" -L"%M%\x86_64-w64-mingw32\lib" -L"%M%\lib" -lstdc++ -lmingw32 -lgcc_s -lgcc -lpthread -lmingwex -lucrt -lkernel32 -lws2_32 -ladvapi32 -lshell32 -luser32 "%M%\lib\gcc\x86_64-w64-mingw32\16.2.0\crtend.o"
if %ERRORLEVEL% NEQ 0 exit /b 1

test_all_cpp.exe
