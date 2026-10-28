#!/bin/sh
# Configure, test, and benchmark. Software rendering is used when /dev/dri is
# absent or NTC_SOFTWARE=1. Timings from that path are not a performance result.
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
fi
PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip
if ! "$PY" -c "import torch" 2>/dev/null; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -m pip install numpy pillow pytest ruff
export PYTHONPATH=src
"$PY" -m ruff check src tests bench
"$PY" -m pytest -q
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
"$PY" bench/download_materials.py --dest data/materials
SOFT=""
if [ "${NTC_SOFTWARE:-}" = "1" ] || [ ! -e /dev/dri ]; then
    SOFT="--software"
fi
# shellcheck disable=SC2086
exec "$PY" bench/gpu_bench.py $SOFT --encode data/materials --warmup 8 --iters 30
