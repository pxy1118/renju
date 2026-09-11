# Runs the Web UI in this console window so Ctrl+C stops it. Nothing is left behind.
# -Share serves the LAN and prints an invite link instead of only opening loopback.
param([switch]$Share)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Project Python environment missing. Run uv sync --locked first.'
}
$port = 8765
$listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    # Open whatever address the running service actually answers on: a shared
    # service may be bound to a LAN address that loopback cannot reach.
    $bound = @($listening.LocalAddress | Where-Object { $_ -and $_ -ne '0.0.0.0' -and $_ -ne '::' })[0]
    if (-not $bound) { $bound = '127.0.0.1' }
    Write-Output "A service is already listening on port $port. Opening it instead of starting a second one."
    Start-Process "http://${bound}:$port"
    exit
}

$arguments = @('main.py', 'webui', '--port', $port)
$browse = "http://127.0.0.1:$port"
if ($Share) {
    # The default-route address is the one a guest on the same network can reach,
    # unlike the virtual adapters a machine with VMware/Hyper-V also owns.
    $lan = (Find-NetRoute -RemoteIPAddress 8.8.8.8 -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress
    if (-not $lan -or $lan -like '127.*') {
        throw 'Could not determine a LAN address. Run: python main.py webui --host <your LAN IP> --share'
    }
    $arguments += @('--host', $lan, '--share')
    $browse = "http://${lan}:$port"
    Write-Output "Sharing on the LAN. The invite link is printed below; friends open it to get their own table."
    Write-Output 'Guests need this machine reachable from their network; allow it if Windows Firewall asks.'
}

Write-Output "Starting the Web UI on $browse"
Write-Output 'The browser opens by itself. Press Ctrl+C in this window to stop the server.'
Push-Location $projectRoot
try {
    # No exit code is inspected here: a Ctrl+C stop surfaces as STATUS_CONTROL_C_EXIT
    # in the wrapper, which is not an error. The server prints its own failures.
    & $pythonExe @arguments
}
finally {
    Pop-Location
    Write-Output ''
    Write-Output 'Web UI stopped. Training, if it was running, was not affected.'
}
