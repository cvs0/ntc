# Longer training on the desktop GPU.
# Output: results/train_sweep/  (json + .ntc). Copy that folder back.
#
# RX 9070 (gfx1201) Windows wheels, ROCm 7.2.1, Python 3.12.
# AMD lists Adrenalin 26.2.2 as the driver for these wheels. A WDDM driver
# string is not the same number; if torch.cuda.is_available() is false, use
# the DirectML fallback below and the script records that backend.
#
#   py -3.12 -m venv .venv-rocm
#   .\.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
#   .\.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
#     https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
#
# DirectML fallback (do not install torch before torch-directml; 0.2.5.dev240914
# pins torch==2.4.1). Python 3.11 is the safer interpreter for that pin.
#
#   py -3.11 -m venv .venv-dml
#   .\.venv-dml\Scripts\python.exe -m pip install torch-directml numpy pillow
#
# CPU scaling for this sweep is in the docstring of bench/train_sweep.py.
# A GPU wall time is not known until this script prints it.
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
        & py -3.12 -u bench/train_sweep.py --out results/train_sweep @args
        exit $LASTEXITCODE
    }
    & py -3 -u bench/train_sweep.py --out results/train_sweep @args
    exit $LASTEXITCODE
}
& $py -u bench/train_sweep.py --out results/train_sweep @args
exit $LASTEXITCODE
