# Max-quality sweep for the RX 9070. Same torch selection as train_sweep.ps1.
# Writes results/train_max/. Does not change train_sweep.ps1 or run_all.ps1.
# Install the ROCm 7.2.1 wheels first. See the comment in train_sweep.ps1.
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$candidates = @(
    ".\.venv-rocm\Scripts\python.exe",
    ".\.venv\Scripts\python.exe",
    ".\.venv-dml\Scripts\python.exe"
)
$py = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py" } else { $py = "python" }
}
$env:PYTHONPATH = "src"
if ($py -eq "py") {
    & py -3.12 -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) {
        & py -3.12 bench/download_materials.py --preset max --dest data/materials
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & py -3.12 -u bench/train_max.py --out results/train_max @args
        exit $LASTEXITCODE
    }
    & py -3 bench/download_materials.py --preset max --dest data/materials
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & py -3 -u bench/train_max.py --out results/train_max @args
    exit $LASTEXITCODE
}
& $py bench/download_materials.py --preset max --dest data/materials
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -u bench/train_max.py --out results/train_max @args
exit $LASTEXITCODE
