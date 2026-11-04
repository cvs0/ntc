# Throughput measurement for the RX 9070 (or any CUDA/ROCm torch already installed).
# Does not install torch and does not write results/train_max.
# From the repo root, in the same environment as the c33648b train_max run:
#
#   powershell -ExecutionPolicy Bypass -File bench\train_speed.ps1
#
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = "src;."
New-Item -ItemType Directory -Force -Path results\train_speed | Out-Null
python bench\train_speed.py `
    --material data\materials\metal_plate `
    --size 256 `
    --g16 `
    --steps 200 `
    --seeds 4 `
    --profile-steps 8 `
    --probe-steps 400 `
    --out results\train_speed\rx9070.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "profile and speedup: results\train_speed\rx9070.json"
Write-Host "full sweep (separate directory, does not resume results\train_max):"
Write-Host "  python bench\train_max_v2.py --out results\train_max_v2 --materials data\materials --probe-steps 400"
