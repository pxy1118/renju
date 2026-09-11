[CmdletBinding()]
param(
    [string]$RuntimeDir
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RapfiRoot = Join-Path $ProjectRoot "external\rapfi"
$SourceDir = Join-Path $RapfiRoot "Rapfi"
$NetworksDir = Join-Path $RapfiRoot "Networks"
$BuildDir = Join-Path $ProjectRoot "artifacts\rapfi-build"
if (-not $RuntimeDir) {
    $RuntimeDir = Join-Path $ProjectRoot "external\rapfi-runtime"
}

if (-not (Test-Path -LiteralPath (Join-Path $SourceDir "CMakeLists.txt"))) {
    throw "Rapfi submodule is missing. Run: git submodule update --init --recursive"
}
if (-not (Test-Path -LiteralPath (Join-Path $NetworksDir "config-example\config.toml"))) {
    throw "Rapfi Networks submodule is missing. Run: git submodule update --init --recursive"
}

New-Item -ItemType Directory -Force -Path $BuildDir, $RuntimeDir | Out-Null
& cmake -S $SourceDir -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
    -DUSE_SSE=ON -DUSE_AVX2=ON -DUSE_AVX512=OFF -DUSE_BMI2=OFF -DUSE_VNNI=OFF
if ($LASTEXITCODE -ne 0) { throw "Rapfi CMake configuration failed." }

& cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Rapfi build failed." }

$Engine = Join-Path $BuildDir "Release\pbrain-rapfi.exe"
if (-not (Test-Path -LiteralPath $Engine)) {
    throw "Rapfi build completed but pbrain-rapfi.exe was not found."
}

Copy-Item -LiteralPath $Engine -Destination (Join-Path $RuntimeDir "pbrain-rapfi.exe") -Force
Copy-Item -LiteralPath (Join-Path $NetworksDir "config-example\config.toml") `
    -Destination (Join-Path $RuntimeDir "config.toml") -Force
Get-ChildItem -LiteralPath (Join-Path $NetworksDir "classical") -File -Filter "*.bin" |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $RuntimeDir -Force }
Get-ChildItem -LiteralPath (Join-Path $NetworksDir "mix9svq") -File -Filter "*.lz4" |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $RuntimeDir -Force }

$RuntimeDir = (Resolve-Path -LiteralPath $RuntimeDir).Path
$Engine = Join-Path $RuntimeDir "pbrain-rapfi.exe"
Write-Host "Rapfi runtime prepared outside the tracked source tree."
Write-Host "Engine: $Engine"
Write-Host "Engine directory: $RuntimeDir"
Write-Host "Teacher smoke command:"
Write-Host "python main.py teacher-generate --rule freestyle --engine `"$Engine`" --engine-dir `"$RuntimeDir`" --output data/teacher/freestyle-pilot --positions 500"
