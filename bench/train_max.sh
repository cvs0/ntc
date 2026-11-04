#!/bin/sh
# Max-quality sweep. Writes results/train_max/. See bench/train_max.py.
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
fi
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
"$PY" bench/download_materials.py --preset max --dest data/materials
exec "$PY" -u bench/train_max.py --out results/train_max "$@"
