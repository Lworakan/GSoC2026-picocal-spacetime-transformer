# Half the cross-validation folds were trained badly, and the headline is pessimistic

Worakan Lasudee, 2026-08-29.

## What was found

Every arm in the production ensemble is systematically worse on folds 5–9
than on folds 0–4. A model trained today, on the same folds with the same
code, is not.

| arm | folds 0–4 | folds 5–9 |
|---|---|---|
| `W8…RcK{f}` (member, seed 0) | 0.0387 | **0.0403** |
| `W8…RcK{f}Rr01` (member) | 0.0388 | **0.0405** |
| `W4…RcK{f}Ac` (member) | 0.0387 | **0.0409** |
| `W8…K{f}` (no recentring, trained 2026-08-25) | 0.0389 | 0.0385 |

The folds come from a seeded random permutation (`split()` in
`run_experiments.py`), so they are exchangeable by construction, and the arm
trained this week confirms it. The break is in the training, not the data.

## The direct test

Folds 5–9 of `W8…RcK{f}` were retrained from scratch — same flags, same
events, checkpoints and prediction CSVs moved aside first so nothing could
resume or be skipped.

| fold | before | retrained | change |
|---|---|---|---|
| 0–4 | — | *(untouched)* | 0.0000 each |
| 5 | 0.0400 | 0.0388 | −0.0012 |
| 6 | 0.0401 | 0.0386 | −0.0015 |
| 7 | 0.0401 | 0.0386 | −0.0015 |
| 8 | 0.0408 | 0.0387 | −0.0021 |
| 9 | 0.0406 | 0.0382 | −0.0024 |
| **aggregate, single seed** | **0.0396** | **0.0388** | **−2.0%** |

Folds 0–4 reproduce to the digit, which is the control: the comparison is
clean and the change is confined to the folds that were retrained.

## What it means for the paper

The reported $\seff = 0.0388 \pm 0.0002$ is a five-member ensemble whose
folds 5–9 members all come from the degraded era. The number is therefore
**pessimistic**, not optimistic — the model is better than the paper claims.
Retraining the remaining members (`Rr01` and `Ac`, folds 5–9) is running; the
corrected headline will be recomputed from them.

Two things this does *not* do. It does not change any comparison in the paper
that is internal to the same era — the encoder head-to-head, the window
ladder and the negative-results table are all fixed-split, unaffected. And it
does not explain *why* those runs came out worse; the cause is not
established, only the fact and the fix.

## Why it was not caught

The ten folds were only ever read as one pooled number. No per-fold table was
made until a comparison against a freshly trained arm forced one. A protocol
that assumes exchangeable folds should check that assumption: a per-fold
column belongs in the results table, and a break of this size — 4.7%,
consistent across three arms and three seeds — is visible at a glance once it
is plotted.

Originals are preserved in `reports/pred_pre_2026-08-25/` and the old
checkpoints in `.scratch/ckpt_old/` on the Studio, so the degraded era can
still be examined if the cause is worth chasing.
