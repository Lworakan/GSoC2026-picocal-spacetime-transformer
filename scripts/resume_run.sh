#!/usr/bin/env bash
# Everything still owed on the push to a single model at 0.035, in the order that answers the
# most with the least GPU. Run on the Studio from the repository root.
#
# The best development-split configuration is cellreg + gate supervision at weight 5 + the
# region-complete overlay: 0.0372 at its best seed, 0.0376 as a single-model mean, 0.0367 as a
# three-seed ensemble, ahead of the control in 15 of 15 bins at seed 0. Two questions remain, and
# they are ordered here by what a negative answer would cost.
set -u
cd "$(dirname "$0")/.."

B="--sample minbias --window 8 --extra --dens --recenter --cleanaux --cellreg --gatesup 5.0"
V2="--overlay .scratch/cache/overlay_v2.pkl"
MAX=3          # concurrent trainings; six at once took the card to 63 GB of 81 and OOMed an arm

wait_for_slot () {
  while [ "$(pgrep -cf 'train_picocal.py --sample')" -ge "$MAX" ]; do sleep 120; done
}

# 1. CROSS-VALIDATION. Every number above is one fixed split. This project has twice had a
#    development-split winner evaporate under ten folds -- the two-stage window read 400/400 and
#    did not reproduce, and selective recentring changed sign. Nothing is published before this.
for k in 0 1 2 3 4 5 6 7 8 9; do
  wait_for_slot
  nohup python scripts/train_picocal.py $B $V2 --fold "$k" --nfold 10 --seeds 0 \
        > ".scratch/cv_k${k}.log" 2>&1 &
  sleep 20
done
wait
echo CV_DONE

# 2. MORE SUPERVISION. The gate labels are the only lever that has moved the number all session
#    (0.0402 -> 0.0376) and they currently reach 39% of the training set: 28,547 synthetic events
#    against 72,554 real. Unlike real data we generate them ourselves. If the gain grows with
#    volume this is the road to 0.035; if it is flat between 2x and 3x the lever is spent and the
#    honest ceiling is about 0.037 on this sample.
OVB=".scratch/cache/overlay_b.pkl"
OVC=".scratch/cache/overlay_c.pkl"
[ -f "$OVB" ] || python scripts/make_overlay.py --regions 0 1 2 3 4 --per-event 3 --seed 7  --out "$OVB"
[ -f "$OVC" ] || python scripts/make_overlay.py --regions 0 1 2 3 4 --per-event 3 --seed 13 --out "$OVC"

wait_for_slot
nohup python scripts/train_picocal.py $B --overlay ".scratch/cache/overlay_v2.pkl,$OVB" \
      --seeds 0 1 2 > .scratch/ov2x.log 2>&1 &
wait_for_slot
nohup python scripts/train_picocal.py $B --overlay ".scratch/cache/overlay_v2.pkl,$OVB,$OVC" \
      --seeds 0 1 2 > .scratch/ov3x.log 2>&1 &
wait
echo SUPERVISION_DONE
