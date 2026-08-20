# The gate is not a photon-fraction estimator, and making it one destroys resolution

2026-08-16, Lightning A100. Two seeds per arm, champion scored on the same two seeds.
Screening numbers, not quotable.

## What was tried

`scripts/cell_info_ceiling.py` showed a plain gradient-boosted model on per-cell observables
reaching corr 0.93-0.95 with the true photon fraction while the network's emergent gate sits at
0.211. That looked like a large unexploited lever, so the estimator was fitted
(`scripts/fit_cell_prior.py`, held-out corr 0.9456) and used two ways:

- `--prior-feat` -- its output appended as a per-cell input feature
- `--prior-teach` -- its output used as the gate target on **every** event, including real
  min-bias, which is what truth-based supervision could never reach

## Result

| bin | champion | `--prior-feat` | `--prior-teach` | err |
|---|---|---|---|---|
| 15mm low | 0.1649 | 0.1684 | **0.1782** | 0.0055 |
| 30mm low | 0.1013 | 0.1051 | **0.1272** | 0.0026 |
| 40mm low | 0.0653 | 0.0642 | 0.0714 | 0.0014 |
| 60mm low | 0.0524 | 0.0523 | 0.0559 | 0.0010 |
| all | 0.0398 | 0.0402 | **0.0416** | 0.0003 |

Feeding the estimate in is neutral-to-slightly-harmful. **Teaching the gate to match it is
strongly harmful**: 30mm low-E degrades by ten times its error, and every bin gets worse.

## Why, and what it means for the project

This is the third independent measurement of the same effect:

1. **H7** (novelty ledger): exact aggregate-fraction supervision steered the gate to corr 0.971
   and *worsened* resolution 0.0466 -> 0.0491.
2. **`--gatesup`** (2026-08-16 morning): per-cell truth supervision on synthetic events, net
   effect under 2x its error.
3. **`--prior-teach`** (this run): a corr-0.95 target applied to all real events, clearly worse.

The conclusion is no longer avoidable: **the gate's low correlation with the photon fraction is
not a defect to be fixed.** The readout `a*log(1 + sum sigma(f_i) E_i) + b` does not want a
fraction estimator; it wants whatever per-cell weighting minimises energy variance, and that is a
different object. The three arms above all replaced a better object with a worse one.

Two earlier readings must be corrected as a consequence.

- The "1.9x oracle-gate headroom at 30mm low-E" was measured with the oracle inside a **sum**
  estimator (`overlay_bound.py`), not inside the trained network. It never was headroom for this
  model, for the same reason the sum's "perfect pileup removal" floor was not a floor.
- The r = 0.92 claim being wrong is now the *expected* result rather than an embarrassment. A gate
  that scored 0.92 against per-cell truth would, on this evidence, resolve energy worse than the
  one we have. The docs have been corrected to 0.211 and should not present it as a shortfall.

## Domain gap, measured

`scripts/domain_gap.py` (new) trains a classifier to separate synthetic overlay cells from real
min-bias cells on the estimator's own features: **AUC 0.931 at 15mm, 0.943 at 30mm** -- trivially
separable, so nothing learned on the overlay has a right to transfer. Per-feature AUC located the
cause in two event-level columns, log window energy (0.64-0.68) and cell count (0.71-0.71): the
overlay puts 77 cells in a window where real data has 65, because `make_overlay.py` transplants
three patches per event. Dropping both columns costs almost no predictive power (corr 0.9324 ->
0.9310 at 15mm) and cuts separability to AUC 0.714. `cell_prior_features` is now cell-level only.

That repair was **not** the reason these arms failed, though -- `--prior-teach` fails for the
mechanistic reason above, not for want of a better estimator. Refitting with the cleaner features
would be worth doing only if some future use of the prior does not overwrite the gate.

## What survives from the 0.07 plan

Ruled out by measurement: anything that replaces the learned gate with a fraction estimator, in
any of its three forms. Options A and B of `reports/bottleneck_0p07_2026-08-16.md`, which I ranked
first and second, are dead.

Still open and untouched by this result, since none of them touch the gate's semantics:

- **`--wlow`** loss weighting on the sparse slices (104 events/seed at 15mm 11-24 GeV, +0.197 bias)
- **`--orho`** outside-window pileup density subtraction, coded, incomplete at 5 seeds
- **front+back inverse-variance time combination**, expected 0.82x on per-cell sigma_t
- **a dedicated low-E expert** via the existing `--film` conditioning

The bottleneck analysis itself stands: 13 of 15 bins are already under 0.07, both failing bins are
pileup-limited, and every failing slice has clean-sample headroom below what 0.07 requires. What
this run removes is a family of solutions, not the target.
