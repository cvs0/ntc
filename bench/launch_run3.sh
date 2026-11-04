#!/bin/sh
# Run 3 on a B300. Prints the GPU-hour budget, then trains.
# Do not start this on a machine that should not spend GPU time.
set -eu
cd "$(dirname "$0")/.."
python bench/train_max_v2.py --help >/dev/null
python -c "from bench.train_max_v2 import run3_report; run3_report()"
exec env NTC_PROFILE=cloudfast NTC_MAX_GPU=1 NTC_EXIT_WHEN_DONE=1 \
    python bench/train_max_v2.py \
    --profile cloudfast \
    --materials data/materials \
    --out results/run3 \
    --max-gpu \
    --exit-when-done
