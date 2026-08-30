#!/usr/bin/env bash
# The one measurement that decides whether the new model can go in the paper.
#
# On the development split, per-cell energy regression with gate supervision on the
# region-complete overlay reads 0.0375 as a single model against the control's 0.0399, and
# 0.0367 against 0.0388 as a three-seed ensemble. Every one of those is one fixed split.
# This project has twice had a development-split winner evaporate under ten folds: the
# two-stage window read 400/400 paired and did not reproduce, and selective recentring
# changed sign. So nothing is written into the paper until the same construction has been
# measured out-of-sample on every event.
#
# One seed per fold, because the comparison that matters is paired against the control's
# own single-seed folds (SubNetW8CleanAuxExDnRcK*QdEma, seed 0), which already exist. An
# ensemble headline would need three seeds per fold and three times the GPU; it is the
# second question, not the first.
set -u
cd "$(dirname "$0")/.."

B="--sample minbias --window 8 --extra --dens --recenter --cleanaux --cellreg --gatesup 5.0"
V2="--overlay .scratch/cache/overlay_v2.pkl"
MAX=3          # six at once took the card to 63 GB of 81 and OOMed an arm

wait_for_slot () {
  while [ "$(pgrep -cf 'train_picocal.py --sample')" -ge "$MAX" ]; do sleep 120; done
}

mkdir -p .scratch
for k in 0 1 2 3 4 5 6 7 8 9; do
  wait_for_slot
  nohup python scripts/train_picocal.py $B $V2 --fold "$k" --nfold 10 --seeds 0 \
        > ".scratch/cvf_k${k}.log" 2>&1 &
  sleep 20
done
wait
echo CV_DONE
