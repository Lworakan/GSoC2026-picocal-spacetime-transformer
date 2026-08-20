# Paper numbers — single source of truth

2026-08-18. Every table names its protocol. PRIMARY numbers are cross-validated and out-of-sample:
each event is predicted by an in-fold ensemble (5 members: 3 seeds of Rc-W8 + one Rr01 + one
allcells variant) whose members never saw that event. SECONDARY numbers are the fixed-split
ensembles used during development; they are systematically ~0.005 flattering in the weak bins and
must not be quoted as headline results.

## Table 1 — PRIMARY: per-region resolution, half-sample cross-validation (36,271 events)

| region | low-E | mid-E | high-E |
|---|---|---|---|
| 15mm | 0.0754 (n 1451) | 0.0429 | 0.0327 |
| 30mm | 0.0756 (n 2189) | 0.0402 | 0.0282 |
| 40mm | 0.0555 | 0.0309 | 0.0231 |
| 60mm | 0.0526 | 0.0308 | 0.0226 |
| 120mm | 0.0635 | 0.0375 | 0.0285 |

**Aggregate sigma_eff = 0.0379.** Baseline at the start of this campaign (fixed 9x9 argmax-seeded
window, five-seed ensemble): aggregate 0.0390, 15mm low 0.1654, 30mm low 0.0957. Improvements:
**15mm low -54%, 30mm low -21%**, every bin below 0.076, thirteen of fifteen below 0.056.
(Folds 5-9 are filling; numbers will be refreshed to full-sample CV before submission.)

## Table 2 — architecture head-to-head (paired seed 0; identical recentred-W8 windows, physics
readout, qd loss, splits; the encoder is the only difference)

| encoder | aggregate | 15mm low | 30mm low | train time/seed |
|---|---|---|---|---|
| **transformer (ours)** | **0.0402** | **0.0767** | **0.0768** | 33 min |
| ParticleNet (EdgeConv, dyn-kNN) | 0.0412 | 0.0813 | 0.0834 | 120 min |
| GravNet | 0.0432 | 0.0829 | 0.0893 | 60 min |

Additionally lost under the same protocol across the campaign: pairwise-bias attention (Geo,
0.1804 at 15mm low vs 0.1628 then-baseline), PairT (0.2852), EFN (0.2883), time-sliced attention
(LCT, aggregate 0.0442), plus CNN paths. Seven encoder families, zero wins over full attention.

## Table 3 — where the gain comes from (cumulative, 15mm low-E, five-seed protocol)

| step | 15mm low | mechanism |
|---|---|---|
| baseline W4 | 0.1654 | 9x9 argmax-seeded window |
| widen to W8 | 0.1156 | window saw 37.6% of the cluster energy |
| + recentre on cluster barycentre | 0.0711 | coverage of the energy field (NOT photon localisation: the barycentre is farther from the photon than the argmax seed, median 3.1 vs 0.4 cells, yet wins) |
| + cross-variant ensemble | 0.0691 (fixed-split) / 0.0754 (CV) | median over 10-15 members |

## Table 4 — closed doors (all measured, cite as negative results)

| family | protocols | outcome |
|---|---|---|
| gate-as-fraction-estimator | H7, --gatesup, --prior-feat, --prior-teach | worse or neutral in all 4; corr 0.211 with per-cell truth is not a defect |
| engineered timing | tpull x3 coordinates, hard cuts, time-sliced attention, front+back inverse-variance combination | all worse; raw timestamps carry the entire 20% ablation value |
| training recipe | lr x2, batch x2, cosine, coordinate jitter x2, epoch budget | all worse; the frozen recipe was already optimal |
| ensembling upgrades | learned stacking (held-out), diverse-config CV members | stacking much worse (0.0414); diversity no gain over seeds |
| post-hoc recalibration | (region x E-decile), held-out | worse (0.0386) |
| attention alternatives | literature-closed with numbers (Mamba-3/Kimi-Linear/Pamba) | parity begins ~4k tokens; we are at 81-289 |
| capacity | dim/layers sweep queued (final check) | expected null: aggregate flat 0.0379-0.044 across 7 encoder families |

## Scaling case (the path below 0.035)

Measured exponents: sigma ~ N^-0.13 (k-fold, 70%->85.5% train) and N^-0.28 (earlier point).
Ensemble curve saturates ~0.037. With 3x simulated data: 0.0322-0.0341 under both exponents.
Without new data, everything measured saturates at aggregate ~0.0378.

## Framing for the paper

The contribution is a measurement-driven diagnosis: two reconstruction-level defects (window
truncation, window mis-centring) dominate everything an architecture can express — quantified by
seven encoder families, four supervision protocols and a frozen-recipe sweep all returning null
against them. Do not claim architectural novelty; claim the diagnosis, the negative results, and
the timing ablation (20% aggregate, 24-39% weak bins).
