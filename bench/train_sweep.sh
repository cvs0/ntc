#!/bin/sh
# Longer training. Prints the torch backend, then writes results/train_sweep/.
# RX 9070 on Windows: install the ROCm wheels first. See bench/train_sweep.ps1.
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
fi
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
OUT="results/train_sweep"
if [ "${1:-}" != "" ] && [ "${1#-}" = "$1" ]; then
    OUT="$1"
    shift
fi
exec "$PY" -u bench/train_sweep.py --out "$OUT" "$@"
