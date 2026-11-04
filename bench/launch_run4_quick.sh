#!/bin/sh
# Four-job B300 check: MetalPlates013 and Tiles135D at b5 and b10.
# Same cloud flags as run 4 (prune off, no latent snap). Do not start this
# on a machine that should not spend GPU time.
set -eu
cd "$(dirname "$0")/.."
python bench/train_max_v2.py --help >/dev/null
python -c "from bench.train_max_v2 import run4_quick_report; run4_quick_report()"
echo "two chain workers share one card, so the wall clock is the sum of the four jobs"
echo "b5 trains from scratch. b10 warm-starts from that b5 checkpoint"
echo "the guard passed when each finish line has psnr_u8 within 0.5 dB of loop and the log has no ABORT line"
exec env NTC_PROFILE=cloudfast NTC_MAX_GPU=1 NTC_EXIT_WHEN_DONE=1 \
    python bench/train_max_v2.py \
    --profile cloudfast \
    --materials data/materials \
    --out results/run4_quick \
    --only MetalPlates013,Tiles135D \
    --bpp 5,10 \
    --max-gpu \
    --exit-when-done
