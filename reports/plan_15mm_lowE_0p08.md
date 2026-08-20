# Plan: 15mm low-E from 0.17 to 0.08

2026-08-16. Written after the gate-supervision result came back negative, from a slice-level
diagnosis that changes what the problem is.

## The bin was never one problem

"15mm low-E = 0.17" is a tercile average that hides a factor of four. Champion, 5 seeds:

| slice | n/seed | sigma_eff | sigma_E |
|---|---|---|---|
| 11.2-23.9 GeV | 104 | **0.3148** | 5.84 GeV |
| 23.9-32.1 GeV | 104 | 0.1498 | 2.6 GeV |
| 32.1-39.6 GeV | 104 | 0.0922 | 2.6 GeV |
| 39.6-46.8 GeV | 103 | **0.0819** | 3.5 GeV |

The top quartile **already meets 0.08**. Three of four are at or under 0.15. The entire deficit
is the 11-24 GeV quartile, roughly 104 events per seed. Everything below is scoped to that slice
alone; work aimed at "15mm low-E" as a whole would mostly optimise bins that are already done.

## The target is reachable, and pileup is the whole gap

Matched region, matched energy slice (E < 24 GeV, E_med 18.4 vs 18.1 GeV), comparable counts:

| | n/seed | sigma_eff | sigma_E | median bias |
|---|---|---|---|---|
| min-bias, champion | 104 | 0.3169 | 5.84 GeV | +0.197 |
| **clean, GateHuber** | 79 | **0.0854** | **1.55 GeV** | +0.036 |
| clean, MeanResidual | 79 | 0.1036 | 1.88 GeV | +0.051 |
| clean, CleanHuberW4 | 93 | 0.2107 | 3.82 GeV | +0.039 |

A trained model on pileup-free data at this exact energy in this exact region reaches **0.0854**.
So 0.08 is not blocked by sampling, containment or detector physics — it is what this detector
already delivers without pileup. The 15mm low-E deficit is **3.8x excess absolute noise, all of
it pileup**, and the +0.197 median bias is the signature of a window whose energy is mostly not
the photon.

This corrects the earlier reading of `overlay_bound.py`, whose "perfect pileup removal" row gives
0.1870 for this slice and looked like a floor above the target. It is not a bound: every row in
that script is a *sum* estimator with a global log-linear calibration. A trained model beats it by
more than 2x on clean data. The script's docstring has been fixed to say so.

## What this rules out

- **Per-cell gate supervision.** Measured this session across three arms: net effect versus the
  champion is under 2x its error, and the synthetic overlay data hurts on its own
  (`reports/gatesup_2026-08-16.md`). Knowing the gate's target does not let the model reach it.
- **Post-hoc calibration as a resolution lever.** sigma_eff is the half-width of the smallest
  68.3% interval, so it is location-invariant: removing the +0.197 bias does not move it. Only a
  *multiplicative* miscalibration would, and that has not been shown.
- **Ensembling.** The epistemic/aleatoric ratio of 0.21-0.27 caps it near 3%, against the 74%
  reduction needed here.

## Ranked plan

**1. Loss weighting on the rare slice — `--wlow`.** Coded, never taken past screening. 104
events/seed against ~10k elsewhere, and a +0.197 bias, is the textbook signature of an unweighted
loss ignoring a sparse corner of input space. Sweep 0.5 / 1.0, 2 seeds, then 5 on the winner.
Cheapest test of the most likely cause. Success: the 11-24 GeV slice moves below 0.25.

**2. Explain the 2.5x between GateHuber and CleanHuberW4 on clean data.** Same sample, same
region, same slice: 0.0854 versus 0.2107. Whatever separates them is worth more at 15mm low-E
than anything tried under pileup, and it costs no GPU to find out — both prediction sets and both
configurations are already on disk. Do this before spending GPU on any new architecture. If the
difference is the gate, note that the min-bias champion already has one, and the lever is instead
*why the gate degrades under pileup*.

**3. Outside-window pileup subtraction — `--orho`.** Now clearly motivated: pileup is the entire
gap. Coded, incomplete at 5 seeds. The outside-window cells are nearly photon-free and their
density matched the inside-window pileup density to 1% at 15mm and 30mm, which is what makes the
estimate legitimate.

**4. Only if 1-3 stall: a dedicated low-E path.** One shared model spends its capacity on
abundant easy events. A region-and-energy-conditioned expert (the `--film` machinery exists) or a
separate 15mm low-E model is the structural version of lever 1.

## Honest statement of the target

0.08 at 15mm below 24 GeV equals the *no-pileup* performance of the best existing model. Reaching
it requires removing essentially all pileup noise, which nothing measured so far comes close to
doing. The credible first milestone is **0.31 to 0.15-0.20 in that slice**, which would bring the
15mm low-E tercile to roughly 0.11-0.13; 0.08 is the asymptote, not the next step. Above 32 GeV
the 0.08 goal is already met on min-bias, so the result is better quoted as an energy threshold —
which is how calorimeter resolution is normally reported anyway — than as a single bin average.

All numbers here are 5-seed except CleanHuberW4 (2 seeds, screening only).
