# Runs the Web UI in this console window so Ctrl+C stops it. Nothing is left behind.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Project Python environment missing. Run uv sync --locked first.'
}
$port = 8765
if ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    Write-Output "A service is already listening on port $port. Opening it instead of starting a second one."
    Start-Process "http://127.0.0.1:$port"
    exit
}

Write-Output "Starting the Web UI on http://127.0.0.1:$port"
Write-Output 'The browser opens by itself. Press Ctrl+C in this window to stop the server.'
Push-Location $projectRoot
try {
    # No exit code is inspected here: a Ctrl+C stop surfaces as STATUS_CONTROL_C_EXIT
    # in the wrapper, which is not an error. The server prints its own failures.
    & $pythonExe main.py webui
}
finally {
    Pop-Location
    Write-Output ''
    Write-Output 'Web UI stopped. Training, if it was running, was not affected.'
}
