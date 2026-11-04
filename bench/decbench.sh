#!/bin/sh
# Decode microbenchmark, one command (Linux). Builds ntc_decbench, writes the
# synthetic shapes to build/decbench/, checks every decoder variant against
# the CPU reference, then times them against the BC baselines.
#   bench/decbench.sh [ntc_decbench options] [extra .ntc files]
# Output: stdout table and results/decbench/<gpu>.json.
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
if [ ! -f build/CMakeCache.txt ]; then
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build --target ntc_decbench
PYTHONPATH=src "$PY" bench/make_bench_ntc.py --out build/decbench
mkdir -p results/decbench
GPU="$(build/ntc_decbench --check-only --variants none build/decbench/synth_ship256.ntc 2>/dev/null \
    | sed -n 's/^using \(.*\) driver.*/\1/p' | tr -c 'A-Za-z0-9\n' '_' | head -n 1)"
[ -n "$GPU" ] || GPU=unknown
FILES="build/decbench/synth_ship256.ntc build/decbench/synth_b10_2k.ntc build/decbench/synth_b15_2k.ntc build/decbench/synth_wide_2k.ntc"
[ -f results/train_sweep/metal_g16_s4000_foot.ntc ] && FILES="results/train_sweep/metal_g16_s4000_foot.ntc $FILES"
# shellcheck disable=SC2086
exec build/ntc_decbench --check-mips 2 --json "results/decbench/$GPU.json" "$@" $FILES
