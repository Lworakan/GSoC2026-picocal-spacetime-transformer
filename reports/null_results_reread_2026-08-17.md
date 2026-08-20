# Every null result, re-read against one broken assumption

2026-08-17. Written after measuring that the window has been centred on the wrong cell in 17.6% of
15mm events. The question is not "what else can we add" but "what did all our negative results
share".

## The measurement that forces the re-reading

`event_geom` sets `seed = argmax(e)` and every window since the first experiment has been centred
there. Against the truth entry point:

| region | photon > 2 cells from seed | p90 offset |
|---|---|---|
| 15mm | **17.6%** | **8.28 cells** |
| 30mm | 11.4% | 3.08 |
| 40mm | 3.6% | 0.94 |
| 60mm | 1.3% | 0.91 |
| 120mm | 0.1% | 0.75 |

A low-energy photon loses the loudness contest to a pileup cell, and it loses it hardest where
occupancy is highest -- which is exactly where our two failing bins are. Splitting the champion's own
15mm low-E events on that offset: **0.0613 when centred within one cell, 0.2847 when off by more than
two.** (That split is not itself proof, because mis-centred events are also the pileup-heavy ones;
`--recenter` is the experiment that separates the two.)

## Why this touches everything, not only containment

The window origin is the coordinate system for every mechanism we have tested:

| mechanism | what it computes from the origin |
|---|---|
| `ref_time` | the reference time from the top-decile ENERGY cells -- if the loudest cells are pileup, **t0 is the pileup time**, and every time pull is measured against the wrong clock |
| gate | asked to find photon cells in a window whose centre may not contain the photon core |
| `--dens`, `--orho`, `--rings` | radial quantities, and "outside the window" is defined relative to the wrong centre |
| `pair_feats` (Geo, PairT) | relative geometry from a displaced origin |
| D4 augmentation and TTA | rotations and reflections about the wrong point |
| `--wlow` | upweights the sparse low-energy inner slice -- of which ~18% is mis-centred |

## The nulls, and what each one now looks like

| null result | measured | re-reading |
|---|---|---|
| Gate supervision, four ways (H7 corr 0.971 worse; `--gatesup` neutral; `--prior-teach` 30mm 0.1013 -> 0.1272; `--prior-feat` neutral) | the gate is not a fraction estimator | still the primary reading, but the gate was also being asked to work in a window that may not contain the photon |
| `--tpull` unstable: helps at W6 (0.1328 vs 0.1363), hurts at W4 (0.1826 vs 0.1628) and W7 (0.1427 vs 0.1256) | no consistent direction | **the cleanest re-reading in this table.** A time pull needs a correct t0. When the seed is pileup, t0 is pileup, so the pull is noise exactly in the events where timing should pay |
| hard time cuts worse than no cut (0.3513 -> 0.3742) | photon loss costs more than the pileup removed | same t0 problem: the cut is centred on a possibly wrong time |
| graph and pairwise encoders all lose (Geo 0.1804, GeoQp 0.1684, PairT 0.2852, EfnResidual 0.2883 against 0.1628) | architecture re-expression does not help | their entire value is relative geometry, computed from a displaced origin |
| `--wlow` hurts 15mm (+0.0144 at 0.5, +0.0240 at 1.0) while helping 30mm slightly | inverse-density weighting misfires | it upweights the slice that contains the mis-centred events, amplifying a pipeline error rather than a hard physics case |
| D4 TTA neutral | group averaging buys nothing | symmetry about the wrong centre is not the detector's symmetry |
| ring sums at W4 nearly useless (-0.0037) yet decisive on top of W8 (-0.0675) | rings extend rather than replace a window | consistent: at W4 the core is not even resolved, and the centre may be wrong as well |
| ensembling ceiling ~3% from epistemic/aleatoric ratio 0.21-0.27 | most error irreducible | **a deterministic, correctable pipeline error was being counted as aleatoric noise.** If part of that 0.73-0.79 is mis-centring, it is reducible and the ceiling was mis-estimated |
| trimmed risk badly worse (Tr10 0.2758, Tr20 0.3242) | ignoring the worst losses hurts | it teaches the model to ignore precisely the mis-centred events instead of fixing them |

## Falsifiable predictions from the re-reading

These are the point of the exercise. Each is a number `--recenter` will either produce or fail to.

1. **`--tpull` becomes consistent and positive** once t0 is computed from a correctly centred window.
   If it stays unstable, the timing story is genuinely limited and the origin was not the cause.
2. **`--wlow` stops hurting 15mm low-E**, because the upweighted slice is no longer full of
   mis-centred events.
3. **Outer regions improve slightly too** -- 40mm and 60mm are mis-centred in 3.6% and 1.3% of
   events, which is small but not zero, and they carry 68% of the events. This is the only mechanism
   in the queue that can move the AGGREGATE rather than just the weak bins.
4. **The gate's correlation with per-cell truth rises** from 0.211, because the window now contains
   the photon core. If the gate's resolution contribution still does not improve, that confirms the
   separate finding that the gate is not supposed to be a fraction estimator.
5. **The epistemic/aleatoric split shifts** towards reducible error, raising the ensembling ceiling
   above the 3% previously measured.

Prediction 3 is the one that matters for the aggregate. The arithmetic is unforgiving: the two
failing bins are only 10.5% of events, so perfecting them to 0.07 moves the aggregate from 0.0379 to
0.0372, and even forcing every bin to 0.03 gives only 0.0289. Anything that helps all regions at once
is worth more than anything that helps the weak bins alone.

## What this does not claim

Recentring is not yet measured. The 0.0613/0.2847 split conflates centring with pileup severity, the
same class of error that produced four wrong bottleneck claims this week: comparing two populations
that share an untested assumption. The queue is ordered so that `--recenter` is settled first, at
five seeds, before any of this is written into a paper.
