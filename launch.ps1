$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = "C:\Users\Akash Singh\OneDrive\Desktop\Vectra\run_server.bat"
$pinfo.WorkingDirectory = "C:\Users\Akash Singh\OneDrive\Desktop\Vectra"
$pinfo.UseShellExecute = $true
$pinfo.WindowStyle = "Normal"
$proc = [System.Diagnostics.Process]::Start($pinfo)
Write-Host "Server launched with PID: $($proc.Id)"
