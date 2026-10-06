Dim shell, mingw, vectra, cmd
Set shell  = CreateObject("WScript.Shell")

mingw  = "C:\Users\ARSHSH~1\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
vectra = "C:\Users\Arsh Sharma\OneDrive\Desktop\Vector DB\vectra"

' Build the command: set PATH then run db.exe
cmd = "cmd /c ""set PATH=" & mingw & ";%PATH% & cd /d """ & vectra & """ & db.exe 8081"""

' Run with windowStyle=1 (normal window), bWaitOnReturn=False (fire and forget)
shell.Run cmd, 1, False

WScript.Quit
