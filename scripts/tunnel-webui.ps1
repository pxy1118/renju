# Expose the local Web UI through a Cloudflare quick tunnel.
#
# The URL changes every time cloudflared starts, so this script starts the
# tunnel, prints the new https URL as soon as Cloudflare assigns it, and stays in
# the foreground: Ctrl+C closes the tunnel. Start the Web UI with sharing and a
# trusted host first, for example:
#
#   .\.venv\Scripts\python.exe main.py webui --host 192.168.1.3 --share --password <口令> `
#       --trusted-host .trycloudflare.com
#
# A quick tunnel has no authentication of its own: whoever learns the URL can
# reach the page. Always pair it with --password.
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$cloudflared = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
if (-not (Test-Path -LiteralPath $cloudflared)) {
    $found = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $found) { throw 'cloudflared not found. Install it or fix the path in this script.' }
    $cloudflared = $found.Source
}
if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    throw "Nothing is listening on port $Port. Start the Web UI first."
}

$log = Join-Path $env:TEMP 'renju-cloudflared.log'
Remove-Item $log -ErrorAction SilentlyContinue
$tunnel = Start-Process -FilePath $cloudflared -ArgumentList 'tunnel', '--url', "http://127.0.0.1:$Port" `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
Write-Output "cloudflared started (PID $($tunnel.Id)); waiting for the public URL…"
Write-Output 'Press Ctrl+C to close the tunnel.'

try {
    $url = $null
    foreach ($attempt in 1..60) {
        Start-Sleep -Milliseconds 500
        foreach ($file in @($log, "$log.err")) {
            if (Test-Path $file) {
                $match = Select-String -Path $file -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches |
                    ForEach-Object { $_.Matches } | Select-Object -Last 1
                if ($match) { $url = $match.Value; break }
            }
        }
        if ($url) { break }
    }
    if ($url) {
        Write-Output ''
        Write-Output "公网地址： $url"
        Write-Output "邀请链接： $url/?k=<服务启动时打印的邀请串>"
        Write-Output '（页面里的“邀请棋友”面板给的是局域网链接，公网请把域名换成上面这个。）'
    }
    else {
        Write-Output "未在 30 秒内解析到公网地址，请查看 $log 与 $log.err。"
    }
    Wait-Process -Id $tunnel.Id
}
finally {
    if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue }
}
