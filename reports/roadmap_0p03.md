# Roadmap to 0.030 (per-bin, E>17 GeV) — status 2026-07-30

## The honest frame first

Aggregate 0.03 over the full 1-100 GeV spectrum is physically impossible at this window (design floor 10%/√E ⊕ 1% → 0.03 requires E ≥ 12.5 GeV; nb29 oracle confirms). The roadmap target is therefore **per-bin**: 0.06 / 0.045 / 0.035 / 0.032 / 0.030 / 0.030 across the six energy-quantile bins, with the fight concentrated in the four E>17 GeV bins.

## Where we stand (nb43 quant 2-seed ensemble; nb44 stack finishing now)

| Bin | 1 (low E) | 2 | 3 | 4 | 5 | 6 (high E) |
|---|---|---|---|---|---|---|
| Current | 0.0659 | 0.0479 | 0.0376 | 0.0362 | 0.0344 | 0.0387 |
| Target | 0.06 | 0.045 | 0.035 | 0.032 | 0.030 | 0.030 |
| Gap | +0.006 | +0.003 | +0.003 | +0.004 | +0.004 | **+0.009** |

Bins 1-2 sit near the oracle/design floor already. The money is in bins 3-6; bin 6 is the largest gap and (per nb37) contamination-dominated. Session ladder so far: 0.0485 → 0.0440 overall (−9%); nb44 seed 0 alone already matches the old 7-model record (0.0440).

## Phase 0 — this week (running/free)

- **nb44 quantile stack** (quant + clean-aux + 5 seeds + D4 TTA): running; new overall anchor expected ~0.042-0.043.
- **H11a mixture pooling** of the 5 seeds (pool quantile posteriors, take mixture median — not mean-of-means): zero compute, post-hoc on saved outputs.
- Exit: new per-bin anchor table; registry export + nb20 refresh so mentors see the current stack.

## Phase 1 — early August (the two highest-leverage label-free levers)

- **H9 (nb45): physics-informed time features** — CMS-HGCAL recipe: σ_t(E)-weighted reference-time fit per window, per-cell pulls (t−t0)/σ_t(E), energy-dependent soft time-gate. Cheap; targets contamination (bins 3-6) and inner regions R0/R1 directly. This is the unused physics handle.
- **H10 (nb46): masked-cell pretraining** on all unlabeled minbias windows (no Etrue needed), fine-tune with quant head (recipe: 2409.12589 + 2503.19165, ~50% gain published at exactly our fine-tune scale).
- Exit criterion each: 2 seeds, beats nb44 anchor by >0.002 overall or in any E>17 bin; verdict cells written either way.

## Phase 2 — late August (compose + distributional ceiling)

- Stack whatever survived Phase 1 into the full recipe (nb47): winner features + pretrained init + clean-aux + 5 seeds + TTA + mixture pooling.
- **H11b** flow posterior-mode head only if the residual-skew diagnostic says median is leaving width on the table.
- Realistic landing zone per the SOTA review: **0.031-0.033 in bins 3-6**. If bins 5-6 touch ≤0.032, the label-free program has done all the literature says it can.

## Phase 3 — the data lever (mentor-gated, parallel from now)

Closing the last 0.001-0.003 in bins 5-6 to flat 0.030 most likely requires per-cell truth. Two asks already prepared:
1. **Zero-bias embedding** (ATLAS-style overlay at digitization, 2102.09495): per-cell truth with REAL pileup, routine collaboration practice — cheaper than the declined truth branches.
2. **Zurich PicoCal GNN group** (CPAN Days 2025): connect via Felipe; benchmark exchange, avoid duplication.

## Phase 4 — paper (runs alongside, not after)

- Story is already sufficient WITHOUT phase 2-3 wins: N1 (label-free signal-fraction gate, adversarially validated) + N2 (seven falsified label-pressure mechanisms) + 0.0485→0.044 ladder + oracle-bounded honesty.
- 2026 submission targets (verification in flight): ML4PS @ NeurIPS 2026 (expected ~Sept deadline), ANIMMA 2027 abstract (Nov 12, 2026, confirmed open), JINST/CSBS rolling any time.
- Hard gate: Felipe/Carla approval for using LHCb simulation in a publication — raise at the next meeting.

## Standing guardrails

2 seeds minimum before any claim; <0.002 = noise; selection on val only, test reported once per notebook; GPU lock check before every launch; every model exported to the registry with predictions; no commits without explicit instruction.

## What would change this roadmap

- H9 or H10 exceeding +0.004 in bins 5-6 → 0.030 flat becomes reachable label-free; accelerate paper.
- Both flat → Phase 3 becomes the only route; paper pivots fully to the N1+N2 methods story (still publishable).
- Zero-bias embedding granted → oracle says 0.030-0.041 per-bin is on the table; that is the endgame for the thesis-grade result.
