#!/bin/bash
# Fresh Runpod PyTorch pod. One process per GPU. Shard is I/N, for example 0/5.
set -euo pipefail
if ! curl -m 10 -sI https://github.com >/dev/null; then
  echo "no outbound internet: curl -m 10 -sI https://github.com failed" >&2
  exit 1
fi
SHARD="${1:?usage: setup_pod.sh I/N}"
ROOT="${NTC_ROOT:-$HOME/ntc}"
BRANCH="${NTC_BRANCH:-cursor/ntc-rd-gpu-2cb6}"
REPO="${NTC_REPO:-https://github.com/cvs0/ntc.git}"
OUT="${NTC_OUT:-$ROOT/results/cloud5090}"

if [[ ! -d "$ROOT/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO" "$ROOT"
fi
cd "$ROOT"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# ``python bench/x.py`` puts bench/ on sys.path, not the repo root.
export PYTHONPATH="$ROOT/src:$ROOT"

python -m pip install -e ".[dev]"
MATS="$ROOT/data/materials"
mkdir -p "$MATS"
SPLIT="$MATS/cloud_split.json"
exec 9>"$MATS/.cloud_download.lock"
flock 9
if [[ -f "$SPLIT" ]]; then
  echo "skip download $SPLIT"
else
  python bench/download_materials.py --preset cloud --dest "$MATS"
fi

python - << 'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("cuda is not available")
props = torch.cuda.get_device_properties(0)
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", round(props.total_memory / (1024 ** 3), 2))
if props.total_memory < 24 * (1024 ** 3):
    raise SystemExit("expected a 32 GB card")
PY

mkdir -p "$OUT/logs"
SAFE="${SHARD//\//_}"
LOG="$OUT/logs/shard_${SAFE}.log"
PID="$OUT/logs/shard_${SAFE}.pid"
nohup python bench/train_max_v2.py \
  --profile cloud5090 \
  --shard "$SHARD" \
  --out "$OUT" \
  --materials "$ROOT/data/materials" \
  > "$LOG" 2>&1 &
echo $! | tee "$PID"
echo "log $LOG"
