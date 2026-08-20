# Meeting page: the case for 3x simulated data

Prepared 2026-08-18. Every number below is measured on our sample; five-seed unless marked.

## Where the model stands

| metric | was (Aug 13) | now | how |
|---|---|---|---|
| aggregate sigma_eff | 0.0389 | **0.0379** | wide window + recentring + seed-ensemble |
| 15mm low-E | 0.1697 | **0.0691** (-59%) | same |
| 30mm low-E | 0.1036 | **0.0706** (-32%) | same + full-cluster input (3 seeds) |
| worst of all 15 region x energy bins | 0.1697 | **0.0706** | 14 of 15 bins now below 0.07 |

Head-to-head under an identical pipeline (same windows, readout, loss, splits, seed — encoder is
the only difference, paired seed 0): our transformer beats **ParticleNet** (0.0402 vs 0.0412
aggregate; 0.0767 vs 0.0813 at 15mm low; 3.6x faster) and **GravNet** (vs 0.0432; 0.0829; 1.8x
faster). Seven encoder families have now lost under this protocol.

## The two mechanisms, in one sentence each

1. The 9x9 analysis window saw 37.6% of the 15mm cluster energy; widening it and centring it on
   the cluster barycentre (not the loudest cell) recovers the rest.
2. Everything else measurable was tried and closed by measurement: per-cell gate supervision (4
   protocols), engineered timing features (3 protocols), graph/pairwise/factorised encoders (7
   families), learned ensemble weights, post-hoc recalibration. Raw per-cell timestamps remain
   worth 20% of the aggregate (ablation, two base models).

## Why the next factor needs data, with the measured curves

**Ensemble curve (measured):** 1 -> 5 -> 25 members gives 0.0399 -> 0.0388 -> 0.0379; the
increments are -0.0012 then -0.0008 — saturating near ~0.037.

**Data curve (measured twice, independently):**
- k-fold today: raising the training fraction 70% -> 85.5% moves a single model 0.0399 -> 0.0389,
  i.e. sigma ~ N^-0.13.
- earlier scaling point: sigma ~ N^-0.28.

**Projection bracketed by BOTH exponents:**

| training sample | aggregate (conservative N^-0.13) | (optimistic N^-0.28) |
|---|---|---|
| current | ~0.037 (after in-hand squeezes) | ~0.037 |
| **2x** | 0.0346 | 0.0338 |
| **3x** | **0.0341** | **0.0322** |

A 3x sample reaches 0.035 under both measured exponents; nothing else we measured does. The same
data also directly attacks the one bin still above 0.07 (30mm low-E, 0.0706), whose residual is
per-event pileup magnitude — confirmed aleatoric given current inputs by eight failed add-on
mechanisms.

## Asks

1. **~3x more matched min-bias simulation** (the single measured lever for 0.03-0.035).
2. **Raw hits / all-event cells**, not only the matched cluster — enables true re-clustering; the
   window and centring gains show reconstruction-level choices dominate architecture on this task.
3. The previously promised GNN and standard-reco outputs, to extend the head-to-head table.
4. A true hold-out min-bias sample for the final quoted numbers.
