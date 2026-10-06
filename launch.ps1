$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = "c:\Users\Arsh Sharma\OneDrive\Desktop\Vector DB\vectra\run_server.bat"
$pinfo.WorkingDirectory = "c:\Users\Arsh Sharma\OneDrive\Desktop\Vector DB\vectra"
$pinfo.UseShellExecute = $true
$pinfo.WindowStyle = "Normal"
$proc = [System.Diagnostics.Process]::Start($pinfo)
Write-Host "Server launched with PID: $($proc.Id)"
