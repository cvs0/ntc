# Build, test, and benchmark on Windows.
# Writes results/hw/<gpu-slug>/ and screenshots/<gpu-slug>/.
# Does not force lavapipe or a software GL. RTXNTC is probed and skipped when
# ntc-cli is absent or the GPU is not NVIDIA.
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$py = $null
foreach ($c in @(".\.venv-rocm\Scripts\python.exe", ".\.venv\Scripts\python.exe", ".\.venv-dml\Scripts\python.exe")) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { & py -3.12 -m venv .venv } else { & py -3 -m venv .venv }
    } else {
        python -m venv .venv
    }
    $py = ".\.venv\Scripts\python.exe"
}
& $py -m pip install --upgrade pip
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "torch is missing. CPU wheel follows. For an RX 9070 see bench/train_sweep.ps1."
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install numpy pillow pytest ruff
$env:PYTHONPATH = "src"
& $py -m ruff check src tests bench
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path "build\CMakeCache.txt")) {
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path build) { Remove-Item -Recurse -Force build }
        cmake -S . -B build -G "Visual Studio 17 2022" -A x64
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ctest --test-dir build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py bench/download_materials.py --dest data/materials
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$vk = "build\ntc_vk.exe"
$gl = "build\ntc_gl.exe"
if (-not (Test-Path $vk)) { $vk = "build\Release\ntc_vk.exe" }
if (-not (Test-Path $gl)) { $gl = "build\Release\ntc_gl.exe" }
$shader = "build\shaders"
if (-not (Test-Path $shader)) { $shader = "build\Release\shaders" }
& $py bench/gpu_bench.py --encode data/materials --vk-bin $vk --gl-bin $gl --shader-dir $shader --gl-shader-dir shaders --warmup 8 --iters 30
exit $LASTEXITCODE
