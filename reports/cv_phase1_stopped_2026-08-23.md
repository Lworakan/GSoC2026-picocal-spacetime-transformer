# Phase 1 stopped after three folds — the plan's main lever did not exist

2026-08-23. Run halted deliberately and the Studio stopped. This records why, so
the mistake is not repeated.

## The mistake

`plan_to_0p036_2026-08-23.md` projected 0.0388 → 0.0361 from two factors: the
two-stage window (−2.6%) and *training on 90% of the sample instead of 70%*
(−4.4%, from our own N^−0.18 curve).

**The second factor was already spent.** The 0.0388 headline is itself a
ten-fold cross-validated number: every one of its members already trains on 90%
of the sample. The 70% figure belongs to the development split, whose numbers
(0.0378 for the two-stage) are also computed on a different, smaller and easier
test set. I compared a development-split member (0.0395) with a
cross-validated member (0.0384) and read the difference as a data effect, when
most of it is the test set changing underneath.

## What the three completed folds actually show

Scored on the same 21,753 events (the test sets of folds 0–2), so every row is
comparable:

| construction | $\sigma_{\rm eff}$ |
|---|---|
| baseline Rc, 1 seed per fold | 0.0389 |
| aux, 1 seed per fold | 0.0385 |
| baseline Rc, 3 seeds per fold (the headline recipe) | 0.0381 |
| pool: Rc(3) + aux(1) | 0.0380 |
| pool: Rc(3) + aux(1) + Rp(1) | **0.0378** |

The auxiliary position head is worth about **1%** under cross-validation, not
3%: within the same protocol it moves a member from 0.0389 to 0.0385, which is
the same 1% it was worth on the development split. Pooling all three families
adds another 0.8% over the headline recipe.

The two-stage window does not reproduce under cross-validation at all. Per
fold, against the auxiliary member on identical events: fold 0 0.0388 vs
0.0383, fold 1 0.0387 vs 0.0385. Its development-split advantage was measured
against a different baseline on a different split and does not survive the
change of protocol. The pointer itself is unaffected — its error is 0.174 cells
per fold against 0.164 on the development split, so nothing leaked and nothing
degraded. What fails is the claim that pointing improves the energy.

## Where that leaves the target

Scaling the pooled result to the full sample: $0.0388 \times (0.0378/0.0381)
\approx \mathbf{0.0385}$. Completing the remaining seven folds of the auxiliary
run and pooling everything is worth about **1%**, for roughly four more GPU
hours.

**0.036 is not reachable on this sample.** The levers that remain after this
measurement are each worth around 1%, and the two that were supposed to be
worth 7% together turn out to be one already-spent protocol difference and one
result that does not generalise across splits.

## What is still worth buying

| item | expected | cost |
|---|---|---|
| finish folds 3–9 of the auxiliary run, pool with the existing Rc and Rp folds | 0.0388 → ~0.0385 | ~4 GPU-h |
| nothing else | — | — |

The honest paths below 0.037 are the ones outside our control: more simulated
events (measured N^−0.18), or the per-event truth flags that would let us
attack the 8% tail carrying 21% of the error.

## Consequence for the paper

The two-stage section must be re-read in this light. Its development-split
numbers are correct as measured, but the cross-validated check says the gain
does not transfer. That belongs in the paper as a stated limitation, and the
section's claim should be softened from "the one architectural mechanism that
survives" to a mechanism that improves the development protocol and does not
reproduce under cross-validation.
