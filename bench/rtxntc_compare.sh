#!/bin/sh
# NVIDIA RTX comparison against ntc-cli. Not part of run_all.
# Builds RTXNTC if ntc-cli is not already on PATH or in external/RTXNTC.
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
"$PY" -m pip install numpy pillow
export PYTHONPATH=".:src"
"$PY" bench/download_materials.py --preset rtxntc --dest data/materials

CLI="${NTC_CLI:-}"
if [ -z "$CLI" ]; then
    if command -v ntc-cli >/dev/null 2>&1; then
        CLI="$(command -v ntc-cli)"
    elif [ -x external/RTXNTC/build/ntc-cli ]; then
        CLI="external/RTXNTC/build/ntc-cli"
    else
        if [ ! -d external/RTXNTC/.git ]; then
            git clone --recursive --depth 1 https://github.com/NVIDIA-RTX/RTXNTC.git external/RTXNTC
        fi
        cmake -S external/RTXNTC -B external/RTXNTC/build -DCMAKE_BUILD_TYPE=Release
        cmake --build external/RTXNTC/build --target ntc-cli -j
        CLI="$(find external/RTXNTC/build -type f -name ntc-cli | head -n 1)"
    fi
fi
exec "$PY" bench/rtxntc_compare.py --ntc-cli "$CLI" "$@"
