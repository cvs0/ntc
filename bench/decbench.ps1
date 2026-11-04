# Decode microbenchmark, one command (Windows). Builds ntc_decbench, writes
# the synthetic shapes to build\decbench\, checks every decoder variant
# against the CPU reference, then times them against the BC baselines.
#   powershell -ExecutionPolicy Bypass -File bench\decbench.ps1 [ntc_decbench options] [extra .ntc files]
# Output: stdout table and results\decbench\<gpu>.json.
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$py = $null
foreach ($c in @(".\.venv\Scripts\python.exe", ".\.venv-rocm\Scripts\python.exe", ".\.venv-dml\Scripts\python.exe")) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) { $py = "python" }
if (-not (Test-Path "build\CMakeCache.txt")) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $configured = $false
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) {
            cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DVCPKG_APPLOCAL_DEPS=OFF
            if ($LASTEXITCODE -eq 0) { $configured = $true }
        }
    }
    if (-not $configured) {
        cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVCPKG_APPLOCAL_DEPS=OFF
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
cmake --build build --config Release --target ntc_decbench
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = "src"
& $py bench/make_bench_ntc.py --out build/decbench
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$exe = "build\ntc_decbench.exe"
if (-not (Test-Path $exe)) { $exe = "build\Release\ntc_decbench.exe" }
New-Item -ItemType Directory -Force -Path results\decbench | Out-Null
$gpu = "unknown"
$probe = & $exe --check-only --variants none build\decbench\synth_ship256.ntc 2>$null | Select-String -Pattern "^using (.*) driver" | Select-Object -First 1
if ($probe) { $gpu = ($probe.Matches[0].Groups[1].Value -replace "[^A-Za-z0-9]", "_") }
$files = @("build\decbench\synth_ship256.ntc", "build\decbench\synth_b10_2k.ntc", "build\decbench\synth_b15_2k.ntc", "build\decbench\synth_wide_2k.ntc")
if (Test-Path "results\train_sweep\metal_g16_s4000_foot.ntc") { $files = @("results\train_sweep\metal_g16_s4000_foot.ntc") + $files }
& $exe --check-mips 2 --json "results\decbench\$gpu.json" @args @files
exit $LASTEXITCODE
