# Error budget before spending credit — where the aggregate actually comes from

2026-08-23. Run on the pooled 31-member ensemble (0.0376, development split,
10,877 test events) with `scripts/error_budget.py`. The point of this pass was
to check the plan in `plan_to_0p036_2026-08-23.md` against the residuals before
paying for GPU time. Two of its four phases did not survive.

## 1. The bottleneck is not the bin we spent the project on

For each (region, energy tercile): its population share, its own $\sigma_{\rm eff}$,
and what the **aggregate** would be if that bin were predicted perfectly.

| bin | share of events | bin $\sigma_{\rm eff}$ | aggregate if perfect |
|---|---|---|---|
| 60 mm low | 11.3% | 0.0513 | **0.0319** |
| 40 mm low | 9.6% | 0.0583 | **0.0321** |
| 30 mm low | 6.7% | 0.0699 | 0.0335 |
| 15 mm low | 3.8% | 0.0675 | 0.0354 |
| 120 mm low | 2.0% | 0.0540 | 0.0366 |

The 15 mm low-energy bin has nearly the worst resolution and is worth the least:
it is 3.8% of the sample. The aggregate is carried by 40 and 60 mm low-energy,
which are three times as populous. Every window, centring and pointing
measurement in this project targeted the inner regions.

## 2. But those bins are stochastic-dominated, so there is little to take

Fitting $\sigma(E) = \sqrt{a^2/E + b^2 + c^2/E^2}$ per region on the pooled
ensemble, and evaluating the terms at each region's low-tercile median energy:

| region | $a$ [$\sqrt{\rm GeV}$] | $b$ | low-E median | bin $\sigma$ | stochastic part |
|---|---|---|---|---|---|
| 15 mm | 0.237 | 0.000 | 32.1 GeV | 0.0675 | 0.0417 |
| 30 mm | 0.231 | 0.000 | 20.3 GeV | 0.0699 | 0.0513 |
| 40 mm | 0.163 | 0.000 | 12.0 GeV | 0.0583 | 0.0469 |
| **60 mm** | 0.062 | 0.016 | 7.5 GeV | 0.0513 | 0.0225 |
| 120 mm | 0.071 | 0.020 | 5.9 GeV | 0.0540 | 0.0291 |

At 40 mm the sampling term alone is 0.047 of the bin's 0.058 — 80% of it, and
not something a model can remove. At 60 mm it is only 0.023 of 0.051, so that
bin *does* have room above its sampling floor; the same is true at 120 mm. The
two most valuable bins in section 1 therefore split: **40 mm low is close to its
physics floor, 60 mm low is not.**

## 3. Ensembling is finished

- 28 single members: mean 0.0394, best 0.0386.
- Pooled ensemble of all 31: 0.0376 — only 4.7% below the member mean.
- Median member-to-member spread on a single event: **0.87%** of the
  prediction, against a total error of 3.76%.

The members agree with each other far more closely than they agree with the
truth. The variance component of the error is roughly $(0.87/3.76)^2 \approx 5\%$
of the budget and most of it is already removed. **Phase 3 of the plan (more
members, more diversity) is dead** — it cannot buy more than a few $10^{-4}$.

## 4. Tails are the largest single identifiable block

| cut | share of events | aggregate if those events were perfect |
|---|---|---|
| $\lvert r \rvert > 2\sigma$ | 13.9% | 0.0255 |
| $\lvert r \rvert > 3\sigma$ | 8.3% | **0.0296** |
| $\lvert r \rvert > 5\sigma$ | 4.1% | 0.0334 |

8% of events carry 21% of the aggregate. We already know the model can *flag*
them (predicted width separates them at AUC 0.93) and that removing them from
the sample does not help $\sigma_{\rm eff}$ by construction. What we cannot do
is say what they are — the hypothesis is wrong-photon matching and overlapping
clusters, and testing it needs the per-event truth flags requested from the
collaboration, not GPU time.

## 5. Residual systematics

Overall median bias $-0.0074$; per-bin bias runs from $+0.0043$ (15 mm low) to
$-0.0176$ (15 mm high) — the model systematically under-predicts high-energy
photons. Removing the per-bin median bias post-hoc gives 0.0376 → 0.0374. Real
but small; $\sigma_{\rm eff}$ is a width and absorbs shifts.
Correlation of $\lvert$residual$\rvert$ with $E_T$ is $-0.23$: some structure
remains that the conditioning does not capture.

## 6. Revised plan

| phase | keep? | why |
|---|---|---|
| Phase 1–2: two-stage under 10-fold CV (90% training data) | **keep** | the only measured $-4.4\%$ available, and it is the number that goes on the front page |
| Phase 3: pool more members | **cut** | section 3 — variance is exhausted |
| Phase 4: `--rc-regions`, snapshots | **keep the first, cut the second** | the 120 mm regression is real; snapshots are ensembling by another name |
| **new**: a 60 mm-low-targeted arm | **add** | biggest aggregate lever that is *not* at its sampling floor |
| **new**: per-bin bias correction in the calibration | **add, free** | $-0.0002$, costs nothing |

Expected landing point: $0.0388 \times 0.974$ (two-stage) $\times 0.956$
(90% training) $\approx 0.0361$, with the 60 mm arm and bias correction the
plausible best case is $\approx 0.0355$. **0.035 remains out of reach without
either the truth flags that would let us attack the 8% tail, or more data.**

## 7. Pre-test of the proposed 60 mm arm — it did not survive

Section 6 added "a 60 mm-low-targeted arm" on the strength of section 2. Before
queueing it, three checks:

**What 0.0319 actually means.** It is the aggregate if that bin were predicted
*perfectly*, which is not a target. Scaling the bin's residuals to plausible
levels instead:

| 60 mm low goes to | meaning | aggregate |
|---|---|---|
| 0.0000 | bound, unreachable | 0.0319 |
| 0.0225 | everything but sampling removed | 0.0345 |
| 0.0350 | noise term halved | 0.0361 |
| 0.0430 | a 15% improvement on the bin | 0.0370 |

**What the bin is made of.** At its median 7.5 GeV the fitted terms are
sampling $0.0226$, constant $0.0164$ and noise $c/E = 0.0387$. The bin is
**noise-dominated**: an absolute error of $c \approx 0.29$ GeV, which is the
size of the pileup fluctuation the window carries at that energy
($E_{\rm win}/E_\gamma = 1.25$ at 60 mm).

**Whether that noise is still addressable.** Residuals in this bin correlate
$+0.25$ with the pileup excess $E_{\rm win} - E_{\rm true}$ and $+0.17$ with
the observable window energy alone, which looked like an uncorrected
systematic. Removing a linear dependence on $\log E_{\rm win}$ per bin, fitted
on half the rows and applied to all, makes every bin **worse** by 7–10%. The
window energy is already a global input; the network has used it, and a
post-hoc correction on the same variable only adds noise.

Two caveats on those correlations: only 46% of test rows could be matched back
to the event cache by $(E_{\rm true}, E_T, {\rm region})$, and that matched
subset scores 0.0427 rather than 0.0376, so it is not a fair sample. The
correlation figures are indicative at best.

**Conclusion: the 60 mm arm is cut too.** The bin is the largest aggregate
lever, but its dominant term is pileup noise in absolute energy, the one
observable that would predict it is already consumed by the model, and the only
correction I could construct from existing predictions is harmful. Proposing a
GPU run for it would be spending on a hunch.

## 8. What is left that I would pay for

| item | confidence | cost | expected |
|---|---|---|---|
| two-stage configuration under 10-fold CV (trains on 90% rather than 70%) | **high** — measured $-2.6\%$ and a measured scaling exponent | ~15 GPU-h | $0.0388 \to \approx 0.0361$ |
| `--rc-regions 0 1 2 3` | medium — fixes a measured 120 mm regression | ~2 GPU-h | $-0.0002$ |
| per-bin bias removal in calibration | high, free | none | $-0.0002$ |

Everything else in the earlier plan is now cut: more ensemble members (variance
exhausted), snapshot ensembling (same mechanism), the 60 mm arm (section 7).

**Honest expectation: 0.036 is reachable, 0.035 is not.** The remaining gap to
0.035 sits in the 8% tail, and nothing in the current inputs identifies what
those events are.
