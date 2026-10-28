# NVIDIA RTX comparison against ntc-cli. Not part of run_all.
# Builds RTXNTC if ntc-cli is not already on PATH or in external\RTXNTC.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path .venv\Scripts\python.exe)) {
    py -3 -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install numpy pillow
$env:PYTHONPATH = ".;src"
& $py bench/download_materials.py --preset rtxntc --dest data/materials
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cli = $env:NTC_CLI
if (-not $cli) {
    $cmd = Get-Command ntc-cli -ErrorAction SilentlyContinue
    if ($cmd) { $cli = $cmd.Source }
}
if (-not $cli -and (Test-Path external\RTXNTC\build\Release\ntc-cli.exe)) {
    $cli = "external\RTXNTC\build\Release\ntc-cli.exe"
}
if (-not $cli) {
    if (-not (Test-Path external\RTXNTC\.git)) {
        git clone --recursive --depth 1 https://github.com/NVIDIA-RTX/RTXNTC.git external\RTXNTC
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    cmake -S external\RTXNTC -B external\RTXNTC\build -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    cmake --build external\RTXNTC\build --config Release --target ntc-cli
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $found = Get-ChildItem -Path external\RTXNTC\build -Recurse -Filter ntc-cli.exe | Select-Object -First 1
    if (-not $found) { Write-Error "ntc-cli.exe was not produced"; exit 1 }
    $cli = $found.FullName
}
& $py bench/rtxntc_compare.py --ntc-cli $cli @args
exit $LASTEXITCODE
