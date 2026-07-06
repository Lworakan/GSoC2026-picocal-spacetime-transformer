# PicoCal Photon Energy Regression — Experiment Plan

Permutation-invariant Set-Transformer (Deep Sets style: per-cell token embedding -> TransformerEncoder
with padding mask -> masked-mean pool -> MLP head) for LHCb PicoCal photon energy regression.
First-stage target = `log(sig_flux_eTot)`. Mentor: Felipe (LHCb).

This plan is design-only. No training has been run yet. It encodes the comparison axes the mentor asked
for, triages each experiment by cost vs paper value, and fixes the evaluation metric.

References:
- `notebooks/02_Training-ready dataset.ipynb` — cuts, dedup, NxN seed window, RAM rationale.
- `notebooks/03_Detector regions.ipynb` — region definition (cell pitch), region survival after the
  vertex cut, convergence check, single-region-first argument.
- `notebooks/04_Training plan.ipynb` — staged strategy and the experiment matrix this file details.

## Fixed dataset definition (baseline preprocessing)

Unless an experiment explicitly varies one of these:
- Cut `sig_flux_prod_vertex_z < 100` before building the dataset. Keeps 36,852 / 199,538 = 18.5% (prompt
  photons; removes material-conversion candidates).
- Drop duplicate-cluster groups. After the vertex cut this removes 0 additional clusters, so vertex-cut
  and dedup are nearly equivalent here.
- Seed = the most energetic cell we compute, not the stored `seed_cell_x,y`.
- Window = 3x3 seed window (caps tokens at ~24, median 9, vs up to 528 cells) — the RAM fix for the
  <12 GB local machine.
- Timing features OUT for the first stage.
- Target = `log(E_true)` where `E_true = sig_flux_eTot`.
- Region = R3 (60 mm). Most surviving data and a stable ranking across the 20/40/60/80/100-file
  convergence check. Kept-per-region after the cut: R0=6,609 R1=7,993 R2=9,079 R3=10,692 R4=2,479.

## Primary metric — energy resolution

For each evaluated cluster define the relative residual

    r = (E_reco - E_true) / E_true

Report the **width** of the `r` distribution two ways so the result is robust to tails:
- `sigma_eff` = half-width of the smallest interval containing 68.3% of `r` (effective resolution), and
- IQR-based width = (Q75 - Q25) / 1.349 (Gaussian-equivalent sigma, outlier-insensitive).
Also report the median of `r` as the residual bias.

Resolution is reported **per true-energy bin** and **per region** (resolution vs E is the physics-
meaningful curve; calorimeter resolution scales roughly as a/sqrt(E) (+) b). Secondary metrics: median
absolute relative error and the bias (median of `r`). The transformer predicts `log(E)`; convert back to
linear E before computing `r`.

## Baselines to beat

The transformer must beat references on the same kept R3 test set, same metric. Two tiers:

Analytic (non-learned) — answer "is ML better than equations?":
- **B0 sum-of-cells**: predicted E = sum of cell energies in the 3x3 window (the obvious physics estimator).
- **B1 seed-cell energy**: predicted E = energy of the single most energetic cell (worst-case floor).
- **B2 stored total_energy branch**: LHCb's own per-cluster `total_energy` if present (the rule-based
  number we are ultimately benchmarking against).
- **calibrated sum**: affine fit of `log(E_true)` on `log(sum-of-cells)` — corrects the mean scale/non-
  linearity (leakage, sampling) but uses no shower shape. Essentially free.

Learned (cheap ML) — answer the harder "does the transformer beat cheap ML on hand-made features?":
- **B3 BDT**: a gradient-boosted tree (sklearn `HistGradientBoostingRegressor`, CPU, trains in seconds)
  on ~6 aggregate features per cluster: total energy `sum(E)`, front/back energy ratio, number of cells,
  seed-cell energy, lateral RMS `sqrt(sum(E * r^2)/sum(E))` about the seed, and region index. This is the
  fair competitor; if the transformer cannot beat B3, the set/attention structure is not justified.

The calibrated sum and B3 form a ladder (pure sum -> calibrated sum -> BDT on features -> Deep Sets ->
transformer); each rung isolates one source of gain. Other architectures (GNN/GravNet, autoencoders) are
later model comparisons, not first-round baselines.

## Experiment matrix

| ID | What varies | Held fixed | Hypothesis | Primary metric | Est. compute cost | Priority verdict | One-line reasoning |
|----|-------------|-----------|------------|----------------|-------------------|------------------|--------------------|
| E0 | Analytic baselines B0/B1/B2 + calibrated sum | R3, 3x3, vertex cut, log(E) | Sum-of-cells is decent but biased low (leakage); transformer should beat it | sigma_eff, bias of `r` per E-bin | Near-zero (no training) | MUST-RUN | Defines the bar; without it the transformer number is uninterpretable |
| E0b | B3 BDT on ~6 aggregate features | R3, 3x3, vertex cut, log(E) | Cheap ML beats analytic; the real bar the transformer must clear | sigma_eff per E-bin & region | Seconds, CPU | MUST-RUN | The fair "is deep learning worth it?" control |
| E1 | Transformer baseline | R3, 3x3, vertex cut, no timing, log(E), POS_REF=seed | Beats E0 sum-of-cells, esp. tails | sigma_eff per E-bin & region | 1 GPU-hr-class, fits <12 GB | MUST-RUN | The headline model; everything else is measured against this |
| E2 | POS_REF: seed vs cluster position | else = E1 | Seed-relative coords give tighter resolution (shower is centred on seed) | delta sigma_eff vs E1 | One extra E1-sized run | MUST-RUN | Mentor-requested; flag already exists; cheap and likely a clear win |
| E3 | Region: R3 vs each of R0,R1,R2,R4 (single-region models) | else = E1 | Resolution differs by region; inner (fine) regions resolve better; R4 starved (2,479) | sigma_eff per region | 4 extra E1-sized runs | NICE-TO-HAVE | Strong paper figure (resolution vs region), but R4 may be too data-poor to trust |
| E4 | All-regions single model vs per-region | else = E1, shared window | One model across geometries is worse than region-specialised (8x window-span span) | sigma_eff overall & per region | 1 extra E1-sized run | MUST-RUN | Directly tests the core "one region first" design claim |
| E5 | Window 3x3 vs 5x5 | R3, else = E1 | 5x5 captures more leakage -> lower bias, at higher token count / RAM | sigma_eff, bias | 1 larger run (more tokens, may need CERN) | NICE-TO-HAVE | Real gain possible but RAM-bound locally; defer to CERN access |
| E6 | Cuts: filtered vs "all data" | model = E1 | Naive all-data is ill-posed (ambiguous duplicate targets + material photons) | sigma_eff (+ qualitative) | 1 extra run | NICE-TO-HAVE (fair version only) | See validity note below; only run the well-posed variant |
| E7 | Timing features off vs on | R3, else = E1 | Timing adds little to energy regression at first stage | delta sigma_eff | 1 extra E1-sized run | NICE-TO-HAVE | Later-stage; low marginal value until baseline is solid |
| E8 | Target log(E) vs linear E | R3, else = E1 | log(E) trains more stably and is unbiased across the dynamic range | sigma_eff, bias, train stability | 1 extra E1-sized run | NICE-TO-HAVE | Cheap, defensible methodology footnote; not a headline result |
| E9 | Per-region all-region sweep on full clusters (no window) | all regions, all cells | Upper bound if RAM were unlimited | sigma_eff | Large, OOM on <12 GB | SKIP | Caused OOM; the windowing decision is the contribution, not a thing to re-litigate |
| E10 | Architecture sweep (depth/heads/dim) | R3, else = E1 | Bigger model = marginally better | sigma_eff | Many runs | SKIP (for now) | Tuning churn with low paper value before the baseline and axes are settled |
| E11 | Deep Sets (mean-pool, no attention) vs transformer | else = E1 | Attention adds resolution over plain permutation-invariant pooling | delta sigma_eff vs E1 | One E1-sized run | MUST-RUN | Gives attention its own credit line; directly defends the architecture choice |

### Validity note for E6 ("all data" comparison)

A naive "train on all 199,538 entries" run is **ill-posed**: 66.7% of entries are in duplicate-cluster
groups where one cluster maps to several true photons with different targets (the model is asked to
predict two different energies from identical inputs), and it includes material-conversion photons the
vertex cut is meant to remove. A loss on that set is not comparable to the filtered loss.

The **fair** version of "effect of the cuts" is one of:
- (a) train on filtered data, then *evaluate* on filtered-only but report how much of the original sample
  was discarded (coverage cost of the cut), or
- (b) keep all clusters but for each duplicate group keep a single well-defined target (e.g. the highest-
  energy true photon) so the mapping is a function — then compare to the cut sample.
Only (a)/(b) belong in the paper. The naive all-data loss should be shown, if at all, only as a cautionary
"why the cut exists" illustration, not as a competing resolution number.

## Recommended order

Given <12 GB local RAM and pending CERN access (via Carla), prioritise a working baseline fast and keep
every early run inside the 3x3 / R3 envelope that is known to fit:

1. **E0 + E0b** — analytic baselines (no training) then the B3 BDT (seconds, CPU); do first so E1 is interpretable.
2. **E1** — transformer baseline on R3, 3x3, seed coords, log(E), no timing.
3. **E2** — POS_REF seed vs cluster (one extra run, likely clear win, mentor-requested).
4. **E11** — Deep Sets vs transformer (defends the attention/architecture choice).
5. **E4** — all-regions vs per-region (tests the central design claim).
5. **E8** then **E6(fair)** — methodology checks, both cheap and local.
6. **E3** — per-region resolution figure (4 runs), local-feasible but lower priority; treat R4 result as
   indicative only (2,479 clusters).
7. **E5** — 5x5 window, and any all-region full-cluster work: **defer to CERN** (RAM-bound).
8. **E7** — timing features, once the energy baseline is solid (second stage).

## Triage summary — what NOT to run, and why

- **E9 (full clusters, no window)** — SKIP. It is the exact configuration that OOMs on the <12 GB
  machine; the windowing decision is itself the contribution. Re-running it adds no paper value.
- **E10 (architecture/HP sweep)** — SKIP for now. High run count, low marginal value before the data axes
  (region, POS_REF, cuts) are settled. Revisit only if E1 fails to beat E0.
- **E6 naive all-data** — do NOT run as a comparison number; it is ill-posed. Run only the fair variant.
- **E5 / full-region runs** — do not attempt locally; they are RAM-bound. Hold for CERN access.
