# Week report — the two things asked for on 2026-08-07

Both requests from the meeting are done: (1) resolution per calorimeter region for the best model, min-bias vs clean superimposed; (2) inference throughput in clusters/second. Everything is reproducible from scripts in this repo.

## 1. Resolution per region

Command:

```
uv run scripts/plot_resolution.py \
    reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv \
    reports/predictions/clean__CleanQuantW2.csv \
    --labels "minimum bias (best)" "clean (best, no pileup)" --by-region \
    --out reports/figures/regions.html
```

Figures: `reports/figures/regions.png` (all regions combined) and `regions_region_{15,30,40,60,120}mm.png`, one per cell pitch, each with the two curves superimposed and per-bin statistical errors 0.96*sigma_eff/sqrt(n).

sigma_eff, test split, seeds averaged:

| region (cell pitch) | n | min-bias | clean |
|---|---|---|---|
| 15mm | 1233 | 0.0632 | 0.0553 |
| 30mm | 2178 | 0.0579 | 0.0538 |
| 40mm | 3134 | 0.0357 | 0.0388 |
| 60mm | 3693 | 0.0339 | 0.0406 |
| 120mm | 646 | 0.0366 | 0.0503 |
| all | 10884 | 0.0409 | 0.0476 |

**Where it performs worst, and why.** The weak regions are the two innermost, smallest-cell ones. Splitting each region by energy tercile shows the failure is not the region as such but *low-energy photons inside it*:

| region | sigma_eff low-E | sigma_eff high-E |
|---|---|---|
| 15mm | 0.1811 | 0.0313 |
| 30mm | 0.1079 | 0.0362 |
| 40mm | 0.0627 | 0.0232 |
| 60mm | 0.0530 | 0.0226 |
| 120mm | 0.0539 | 0.0299 |

High-energy clusters are already at 0.023-0.031 everywhere, including the innermost regions. Without pileup the same 15mm low-E bin is 0.0706, so roughly 2.5x of the degradation is pileup contamination rather than granularity: at low photon energy in the densest-occupancy region the photon is a small fraction of the window energy and the per-cell signal-to-noise is low.

Reproduce the split with:

```
uv run scripts/score_regions.py reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv \
    --baseline reports/predictions/clean__CleanQuantW2.csv --bin low
```

Two caveats on the clean curve. It comes from an earlier 2-seed model (`CleanQuantW2`, 5x5 window); training the current champion configuration on the clean sample gave a *worse* 0.0563, because that sample is about 3x smaller (21k training clusters vs 51k + 30k auxiliary) and the 9x9/qd configuration was selected on min-bias. So at fixed architecture, training-set size currently dominates the pileup penalty — which is also why the clean numbers above should be read as indicative.

## 2. Inference throughput

Command (the script has `--help` with examples):

```
uv run scripts/benchmark_inference.py \
    models/SubNetW4CleanAuxQdEma_s0.pt models/SubNetW4CleanAuxQuant_s0.pt \
    models/CleanHuberW4_s0.pt --files 4 --batch-sizes 64 256 \
    --out reports/benchmark_inference.csv
```

Laptop CPU (i9-13900HX, 24 threads), torch 2.13+cpu, 3173 clusters from 4 min-bias files, median of 5 timed passes after 2 warmup passes. One cluster per event in these samples, so clusters/s equals events/s. Full table in `reports/benchmark_inference.csv`.

| model | params | batch | clusters/s | ms/cluster |
|---|---|---|---|---|
| SubNetW4CleanAuxQdEma (best) | 623238 | 64 | 4677 | 0.214 |
| SubNetW4CleanAuxQdEma (best) | 623238 | 256 | 3753 | 0.266 |
| SubNetW4CleanAuxQuant | 623238 | 64 | 4604 | 0.217 |
| CleanHuberW4 | 623238 | 64 | 4505 | 0.222 |

All three share the same architecture and parameter count and differ only in training objective, so they time out within 4% of each other; the objective does not affect inference cost.

Derived rates for the deployed configurations of the best model:

| configuration | clusters/s |
|---|---|
| single model, single pass | 4677 |
| 5-seed ensemble | 935 |
| 5-seed ensemble + D4 test-time augmentation (40 passes) | 117 |

Two measurement notes. Throughput is best at small batch (64) because the model is small enough that per-step overhead dominates; and the measurement must be made on an otherwise idle machine — taken while a training job was running, the same benchmark reported 332 clusters/s, a factor 14 lower.

A GPU number is not included: the RTX 4080 Laptop in this machine hard-powers-off when several CUDA jobs run at once, so GPU work here is restricted to one job at a time and is used for training rather than for timing claims. Running this same script on a standard lab machine would give the number worth quoting.

## 3. Accuracy versus throughput across model types

`--baselines` adds the two non-transformer model types to the same timing harness: the analytic calibrated sum `a*log(1+sum E)+b`, and a boosted-tree regressor (HistGradientBoosting, 300 iterations) on exactly the aggregate features the transformer also receives (log sum E, log seed E, log n cells, front/back ratio, lateral spread). Same machine, same clusters, same protocol.

```
uv run scripts/benchmark_inference.py models/SubNetW4CleanAuxQdEma_s0.pt \
    models/CleanHuberW4_s0.pt --files 4 --batch-sizes 64 256 --baselines \
    --out reports/benchmark_inference.csv
```

| model type | sigma_eff (min-bias) | clusters/s @64 | clusters/s @256 | size |
|---|---|---|---|---|
| calibrated sum (analytic) | 0.1837 | 5.6e6 | 1.3e7 | 2 parameters |
| BDT on aggregate features | 0.1253 | 16554 | 60035 | 18300 tree nodes |
| PairT transformer (earlier design) | 0.0624 | not timed (no checkpoint saved) | — | — |
| SubNetW4CleanAuxQdEma, 1 seed | 0.0413 | 4958 | 3605 | 623238 parameters |
| same, 5-seed ensemble | 0.0409 | 992 | 721 | 5 x 623238 |
| same, 5-seed + D4 TTA | not part of the pipeline — see note | 124 | 90 | 5 x 623238 |

Note on the TTA row: the script pipeline had no test-time augmentation when this table was
first written (`d4_apply` ran only inside the training loop), so that operating point was
hypothetical. It is implemented now (`--tta`, plus `scripts/eval_tta.py` for saved weights)
and measured neutral — aggregate 0.0389 -> 0.0391 — so the 40x cost is not paid and the
quoted configuration is the 5-seed ensemble.

What this says about the cost of accuracy: the analytic sum is effectively free but 4.5x worse than the transformer; the BDT is 3.3x faster per cluster than a single transformer pass and still 3x worse in resolution; the transformer's advantage comes from reading cells individually, and that is what costs the time. The full 0.0402 stack is 40x slower than a single pass — if throughput ever matters more than the last 0.0007 of resolution, a single seed at 4958 clusters/s is the operating point to quote.

Caveat on the baseline resolutions: they come from the earlier experiment pipeline with a slightly different cluster selection (12227 test clusters versus 10884 here), so treat them as indicative of the model class rather than as exact head-to-head numbers. The throughput figures are directly comparable — same data, same harness.

Batch-size note: the transformer is fastest at batch 64 (per-step overhead dominates for a model this small), while the vectorised baselines get faster with larger batches. Each model type should be quoted at its own best batch size.

## Files

- `scripts/plot_resolution.py` — `--by-region` mode, per-region figures
- `scripts/score_regions.py` — sigma_eff per region and per energy bin, with `--baseline` delta and `--seeds` for like-for-like comparisons
- `scripts/benchmark_inference.py` — clusters/second, saved table
- `reports/figures/regions*.png|html`, `reports/benchmark_inference.csv`
- `reports/region_deep_dive_2026-08-08.md` — error analysis, literature sweep, and the architecture experiments run on top of this understanding
