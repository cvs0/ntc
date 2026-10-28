#!/bin/bash
# Fresh Runpod PyTorch pod. One process per GPU. Shard is I/N, for example 0/5.
# NTC_PROFILE picks the job list (cloud5090 or cloudfast). NTC_WORKERS > 1
# runs one trainer that starts that many chain workers on this GPU instead
# of a shard; the shard argument is then optional.
set -euo pipefail
if ! curl -m 10 -sI https://github.com >/dev/null; then
  echo "no outbound internet: curl -m 10 -sI https://github.com failed" >&2
  exit 1
fi
PROFILE="${NTC_PROFILE:-cloud5090}"
WORKERS="${NTC_WORKERS:-1}"
if [[ ! "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
  echo "NTC_WORKERS must be a positive integer, got $WORKERS" >&2
  exit 1
fi
if (( WORKERS > 1 )); then
  SHARD="${1:-0/1}"
else
  SHARD="${1:?usage: setup_pod.sh I/N}"
fi
ROOT="${NTC_ROOT:-$HOME/ntc}"
BRANCH="${NTC_BRANCH:-cursor/ntc-rd-gpu-2cb6}"
REPO="${NTC_REPO:-https://github.com/cvs0/ntc.git}"
OUT="${NTC_OUT:-$ROOT/results/$PROFILE}"
export NTC_ROOT="$ROOT"

# One lock for clone, pull, and the material download. Two shards pulling
# the same checkout at once fail with "Cannot fast-forward to multiple
# branches." The lock file lives beside the repo so it exists before clone.
mkdir -p "$(dirname "$ROOT")"
exec 9>"$(dirname "$ROOT")/.ntc_cloud_setup.lock"
flock 9

if [[ ! -d "$ROOT/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO" "$ROOT"
fi
cd "$ROOT"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
if [[ "$(git rev-parse HEAD)" == "$(git rev-parse "origin/$BRANCH")" ]]; then
  echo "HEAD matches origin/$BRANCH"
else
  git pull --ff-only origin "$BRANCH"
fi

# ``python bench/x.py`` puts bench/ on sys.path, not the repo root.
export PYTHONPATH="$ROOT/src:$ROOT"

python -m pip install -e ".[dev]"
MATS="$ROOT/data/materials"
mkdir -p "$MATS"
SPLIT="$MATS/cloud_split.json"
if [[ -f "$SPLIT" ]]; then
  echo "skip download $SPLIT"
else
  python bench/download_materials.py --preset cloud --dest "$MATS"
fi

# The nohup trainer inherits open fds. Leaving fd 9 open holds this flock
# until that process exits, and the next shard on the pod blocks forever.
exec 9>&-

python - << 'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("cuda is not available")
props = torch.cuda.get_device_properties(0)
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", round(props.total_memory / (1024 ** 3), 2))
if props.total_memory < 24 * (1024 ** 3):
    raise SystemExit("expected a card with at least 24 GB")
PY

mkdir -p "$OUT/logs"
if (( WORKERS > 1 )); then
  # Each worker gets its own Inductor and Triton cache from the trainer and
  # takes whole material chains, so warm starts stay in rate order.
  export TORCHINDUCTOR_COMPILE_THREADS=1
  LOG="$OUT/logs/workers.log"
  PID="$OUT/logs/workers.pid"
  nohup python bench/train_max_v2.py \
    --profile "$PROFILE" \
    --dense-workers "$WORKERS" \
    --out "$OUT" \
    --materials "$ROOT/data/materials" \
    9>&- > "$LOG" 2>&1 &
  echo $! | tee "$PID"
  echo "log $LOG"
  exit 0
fi
SAFE="${SHARD//\//_}"
# Inductor's compile workers and a shared /tmp/torchinductor_<user> lock
# deadlock every shard but the last one launched on the pod.
CACHE="$ROOT/.cache/shard_${SAFE}"
mkdir -p "$CACHE/inductor" "$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_COMPILE_THREADS=1
export NTC_SHARD="$SAFE"
LOG="$OUT/logs/shard_${SAFE}.log"
PID="$OUT/logs/shard_${SAFE}.pid"
nohup python bench/train_max_v2.py \
  --profile "$PROFILE" \
  --shard "$SHARD" \
  --out "$OUT" \
  --materials "$ROOT/data/materials" \
  9>&- > "$LOG" 2>&1 &
echo $! | tee "$PID"
echo "log $LOG"
