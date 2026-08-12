# Inner-region deep dive — error analysis and research plan (2026-08-08)

Goal: understand the per-region failure found on 2026-08-07 and identify the shortest path toward sigma_eff ~0.03 in the weak regions. No new training launched; this is the understand-before-improve phase the mentors asked for.

## 1. Error analysis

Per-region sigma_eff of the best model (SubNetW4CleanAuxQdEma, minbias) vs the clean-era reference (CleanQuantW2), test split, seeds averaged. Low/high energy = bottom/top tercile of true energy within the region.

| region | n (mb) | sigma (mb) | sigma low-E (mb) | sigma high-E (mb) | sigma low-E (clean) |
|---|---|---|---|---|---|
| 15mm | 1233 | 0.0632 +- 0.0017 | **0.1811** | 0.0313 | 0.0706 |
| 30mm | 2178 | 0.0579 +- 0.0012 | 0.1079 | 0.0362 | 0.0573 |
| 40mm | 3134 | 0.0357 +- 0.0006 | 0.0627 | 0.0232 | 0.0537 |
| 60mm | 3693 | 0.0339 +- 0.0005 | 0.0530 | 0.0226 | 0.0517 |
| 120mm | 646 | 0.0366 +- 0.0014 | 0.0539 | 0.0299 | 0.0519 |

Caveat: the clean column is an older 2-seed model; an apples-to-apples clean run of the champion config is training now (clean__SubNetW4QdEma).

Update (same day): the champion config trained on clean landed at sigma_eff 0.0563 (seeds 0.0573/0.0554) — worse than both the minbias champion (0.0409) and the clean-era CleanQuantW2 (0.0476). Two reasons: the clean sample is ~3x smaller (21k train events vs ~70k+21k aux) and the 9x9 window/qd configuration was selected on minbias, while on clean the smaller 5x5 window of CleanQuantW2 is the better fit. Consequences: (a) the mentor per-region figures use best-vs-best (minbias__SubNetW4CleanAuxQdEma vs clean__CleanQuantW2, reports/figures/regions*.html); (b) training-set size currently dominates over the pileup penalty at fixed architecture, which strengthens the case for candidate 4 (overlay/augmentation to densify training data) and means clean-sample ceilings quoted in section 2 are approximate.

## 2. Diagnosis

- The failure is not "the 15mm region" — it is **low-energy photons in the densest-pileup regions**. High-energy clusters in 15mm are already at target (0.031).
- Pileup, not granularity, dominates: 15mm low-E is 0.181 with pileup vs 0.071 without. Roughly 2.5x of the degradation is contamination — the photon is a small fraction of the window energy, per-cell signal-to-noise is low, and the emergent gate cannot separate signal from a background of comparable magnitude.
- A second, structural issue: the fixed 9x9 index window spans 13.5 cm in the 15mm region vs 108 cm in the 120mm region — variable physical aperture, and mixed pitches at region boundaries.
- Realistic near-term ceiling for the low-E inner bin is the no-pileup value ~0.07, not 0.03; the region aggregate can approach 0.03-0.04 if the low-E tail is fixed, consistent with the per-bin targets set in the nb28 arc (0.06/0.035/0.030).

## 3. Ranked candidates (synthesis of two literature sweeps, 2024-26)

1. **Physical-coordinate + pitch tokens** — add (x, y, pitch) in mm and relative-position bias to each cell token instead of index-only geometry. Fixes variable aperture and boundary mixing natively (object condensation 2002.03605, HGCAL GravNet 2106.01832, CLAS12 hybrid GravNet-transformer 2503.11277, DHCAL point-cloud 2412.11208). Cost: small — feature/embedding change in prep + SubNetFQ.
2. **Pileup-density conditioning** — embed occupancy proxies (window energy sum minus core, n fired cells, region id) as a FiLM/global token modulating gate and readout, diffusion-style noise conditioning (2502.13129; PUMiNet 2503.02860; ATLAS mu-input calibration 2311.08885). Both sweeps ranked this top-3 independently. No published per-region FiLM/MoE for calorimeter regression was found — also a novelty opportunity. Cost: small.
3. **Explicit time gate** — split the gate head into space and Delta-t branches (cell time minus seed time), the learned version of ATLAS out-of-time cell cleaning (2310.16497); PicoCal's 10-20 ps design intent is exactly this separation (JINST 21 C03006). Timing is our established lever under pileup. Cost: small.
4. **Varied-intensity overlay curriculum** — resample overlay intensity during training so the low-E dense bin is not data-starved; easy-to-hard schedule (2403.10348 lineage; PUMML-line varied-mu generalization). Cost: low-medium, data pipeline only.
5. **Heteroscedastic NLL difficulty head** — per-event sigma with stop-gradient (2212.09184, beta-NLL 2203.09168); complements the existing quantile width, enables difficulty-binned reporting and abstention semantics (SelectiveNet 1901.09192). Cost: low.
6. **Per-cell privileged distillation** — teacher sees the pileup-free view of the same event and supervises w_i per cell (LUPI 1511.03643). CAUTION: nb28-31 found per-cell overlay supervision neutral-to-worse than the emergent gate in aggregate; if revisited, test only inside the low-E dense bin where the event-level loss is least informative, and treat a null result as confirmation of the emergence headline claim.
7. **GravNet / learned-kNN neighborhoods, graph super-resolution** (2409.16052) — highest cost; only if 1-4 stall.

## 4. Proposed experiment order (after mentor go-ahead)

One change at a time on the champion config, scored on the low-E 15/30mm bins specifically (not the aggregate): (1)+(2) first since they share the prep-layer work, then (3), then (4). Success metric: 15mm low-E sigma_eff moving from 0.181 toward the ~0.07 no-pileup ceiling without hurting the other bins.

## 4b. Data-efficiency sweep (added same day)

Motivated by the clean-run finding that training-set size dominates at fixed architecture (0.041 -> 0.056 when trained on 21k events). Third literature sweep, ranked by expected accuracy-per-effort at our ~70k-label scale:

1. **Masked-cell self-supervised pretraining on unlabeled min-bias** (MPM 2401.13537, tokenizer-free 2409.12589; OmniJet-alpha 2403.05618 shows the clearest few-label transfer gain; 2606.14870 confirms the combo is strongest in the low-label regime). We have far more raw min-bias clusters than photon-matched ones — pretrain the encoder to reconstruct masked cells, fine-tune the regression. Cost ~2-3 days; best-evidenced lever.
2. **Train-time D4 augmentation** — weight-free version of equivariance; strictly dominates TTA-only at small n. Implemented as --d4aug (random dihedral transform of di/dj and phys dx/dy per batch); training queued.
3. **D4-equivariant weight sharing** (e2cnn 1911.08251, equivariant self-attention 2010.00977; LorentzNet 2201.08187 matched SOTA with ~5% of labels — the strongest HEP data-efficiency precedent). Converts the TTA gain into per-layer inductive bias. Cost ~1 week; do after 2 shows direction.
4. **Region-conditioning over per-region models** (ATLAS 2311.08885, 2412.04370): one network across regions with region encoding beats slicing at small n — supports the existing region one-hot + the new --occ path; argues against training separate inner-region models.
5. **DeiT-style distillation** (2012.12877) from the existing ensemble into single models. Modest, stackable.

Rejected/parked: C-Mixup (2210.05775) — label-similarity mixup conflicts with the physics readout (mixing windows breaks the log-sum energy structure); generative augmentation via CaloDiffusion-line — no published evidence of downstream regression gains; deep-ensembles-vs-EMA — our 5-seed+EMA is already near the small-data optimum (1612.01474, 2111.14493).

## 4c. Experiment results (screening pass, 2026-08-09)

Five candidates trained on minbias+clean-aux, GPU (RTX 4080 Laptop, separate `.venv-gpu`, strictly one CUDA job at a time — parallel CUDA jobs hard-powered-off the machine). All numbers are seed 0 vs champion seed 0 (`--seeds 0` in score_regions.py) so the seed-averaging advantage of the 5-seed champion does not confound the comparison.

| run | 15mm low-E | 30mm low-E | 15mm all | aggregate | min/seed |
|---|---|---|---|---|---|
| champion SubNetW4CleanAuxQdEma | 0.1755 | 0.1138 | 0.0654 | 0.0413 | 25 |
| + phys coords & occupancy (`--phys --occ`) | **0.1638** (-0.0117) | 0.1132 | 0.0662 | 0.0418 | 38 |
| + D4 train-time augmentation (`--d4aug`) | 0.1696 (-0.0059) | **0.1071** (-0.0067) | 0.0653 | 0.0416 | 79 |
| + explicit time gate (`--gate time`) | 0.1638 (-0.0117) | 0.1128 | 0.0662 | 0.0419 | 25 |
| geometric attention bias (`--arch geo`) | 0.1804 (+0.0049) | 0.1117 | 0.0658 | 0.0420 | 111 |
| geometric attention + query pooling (`--arch geo --qpool`) | 0.1684 (-0.0071) | 0.1083 (-0.0055) | **0.0629** (-0.0025) | **0.0414** | 118 |

Phys+Occ also ran 2 seeds: 15mm low-E 0.1698 (-0.0119), 30mm low-E 0.1051 (-0.0072), aggregate 0.0408. Tgate 2 seeds: 15mm low-E 0.1744 (-0.0073), aggregate 0.0413.

**Findings.**

1. No architecture change moves the aggregate. Every run lands in 0.0414-0.0420 against the champion's 0.0413 (+-0.0004). Four structurally very different models agreeing this closely is evidence that the limit is information in the data, not how the network reads it — consistent with the clean-sample result in section 2 (3x less training data degraded sigma_eff 0.041 -> 0.056) and with the mentors' framing that ML cannot create information.
2. Every candidate nudges the low-E inner bins in the right direction, but at 0.6-1.4 sigma. None reaches the 2-sigma bar set for this screening pass. Treat as directional, not established.
3. The explicit time gate is a genuine null result: the transformer already receives per-cell timing as token features, so factorising the gate into space and time branches adds no information — it only reorganises what attention could already learn. Worth reporting in the paper as an ablation supporting the emergent-gate claim.
4. Geometric attention bias alone was the *worst* run (15mm low-E +0.0049) and the second-slowest; adding query-based pooling to the same attention turned it into the best-aggregate experiment (0.0414, and the only run that improves the 15mm/30mm/60mm region aggregates simultaneously). The readout mechanism matters more than the attention bias — a cheap and testable direction, since `--qpool` is separable from `--arch geo`.
5. Cost matters for the experiment loop: D4 augmentation is 3x and geo attention 4.5x the champion's time per seed for no aggregate gain.

**Next steps implied by these results** (understand-before-improve still holds; nothing here justifies a new champion):

- Test `--qpool` alone on the standard architecture — it is the only mechanism that improved region aggregates, and it is cheap.
- Combine the two information-adding changes (`--phys --occ --d4aug`) at 5 seeds to see whether their low-E gains add.
- The largest remaining lever is masked-cell self-supervised pretraining on unlabelled min-bias (section 4b, candidate 1): it is the only option that adds information rather than rearranging it.

## 4d. Best-shot combination at full 5 seeds (2026-08-10)

`--phys --occ --qpool`, 5 seeds — the three mechanisms that moved the low-E inner bins, excluding geometric attention (made 15mm low-E worse) and the explicit time gate (null result). Per-seed sigma_eff 0.0416 / 0.0420 / 0.0420 / 0.0413 / 0.0410, i.e. the same spread as the champion's 0.0412-0.0426. Compared against the champion at the same 5-seed averaging:

| bin | combo | delta vs champion |
|---|---|---|
| 15mm low-E | 0.1747 | -0.0064 |
| 30mm low-E | **0.1007** | **-0.0072** |
| 40mm low-E | 0.0599 | -0.0030 |
| 60mm low-E | 0.0514 | -0.0012 |
| 120mm low-E | 0.0601 | +0.0062 |
| 15mm all | 0.0650 | +0.0018 |
| **aggregate** | **0.0409** | **+0.0000** |

The aggregate is identical to the champion's, to four decimals. The low-E inner bins do improve consistently (30mm low-E is the largest single gain measured in this whole sweep, -0.0072 at ~1.4 sigma), and they improve by roughly the same amount as Phys+Occ alone — the mechanisms do not stack.

**Conclusion on the 0.03 aggregate target.** Seven configurations spanning added geometric features, pileup-density conditioning, an explicit time gate, train-time D4 augmentation, geometric attention bias, query pooling, and their best combination at 5 seeds all land in 0.0409-0.0426 against the champion's 0.0409 (+-0.0004). Combined with the clean-sample result (3x less data degrades sigma_eff 0.041 -> 0.056) and the nb16 finding that the low-E containment floor is ~0.07 even with no pileup at all, an aggregate of 0.03 is not reachable by architecture work on this dataset. What is already reached today is sigma_eff 0.023-0.031 for high-energy photons in every region. The defensible way to state the target is per energy range (~0.03 high-E, ~0.035 mid, ~0.06 low-E), which is also how calorimeter resolution is conventionally reported.

The one remaining lever that adds information rather than rearranging it is masked-cell self-supervised pretraining on unlabelled min-bias (section 4b, candidate 1).

## 4e. Unused-input audit — the pipeline was feeding 14 of 47 branches (2026-08-10)

`clusters_matched` has 47 branches; `picocal_data.py` read 14 (9 per-cell + 5 truth). Among the ignored ones are three cluster-level families that CMS and ATLAS both treat as essential inputs for photon energy regression:

- `total_energy`, `total_energy_front`, `total_energy_back` — energy of the **whole** cluster, i.e. including cells outside our 9x9 window and below our 2.49 MeV cell threshold. Cluster cell multiplicity is median 81 and reaches 528, so there is real energy outside the window.
- `x_cluster`, `y_cluster` — the standard reconstruction's shower position, at sub-cell precision.

**Leakage gate (passed).** Before using them: `total_energy` equals the sum of all cluster cell energies exactly (ratio median/p5/p95 all 1.0000; `total_energy_front + total_energy_back - total_energy` is identically 0), and its correlation with the truth photon energy is 0.6475 — the same as the plain cell sum. It is reconstruction-derived, not truth-derived, so it is a legal input.

**Published precedent, and it is strong.** CMS ECAL photon regression (arXiv:1407.0558, 2012.06888) feeds R9 = E_3x3 / E_supercluster, structurally identical to our new `sum(E_window) / total_energy`, and R9 is one of its highest-ranked inputs. ATLAS e/gamma MVA (arXiv:1407.5063) feeds the cluster position within the cell to correct lateral leakage and edge effects (H->gamma gamma mass resolution +10% average, +15% for converted photons in the barrel-endcap transition). arXiv:2107.10207 injects the raw energy sum as an extra node specifically to force the network into an energy-correction regime. So the containment ratio, the cluster totals, and the reco position are three feature families we were missing entirely.

Implemented as `--extra` (5 global features: log total, window/total containment ratio, front/back total ratio, and the reco shower position relative to the seed in pitch units). A 2-epoch smoke test gave sigma_eff 0.2431 versus 0.2805 for the same smoke without it — indicative only, but larger than any architecture change produced. Full 2-seed runs queued. Note the events cache had to be invalidated, and `prep` now raises if `--extra` is requested against a stale cache, because the missing fields would otherwise silently become zeros and turn the containment ratio into a division by epsilon.

**Also worth adding, from the same sweep** (ranked, all evidence-backed):

1. Remainder of the containment block: `r_contain_front`, `r_contain_back`, `n_cells_in_cluster`, and per-cell position relative to the reco centroid `(x_cell - x_cluster)/pitch`.
2. **Energy density** `log(E / pitch^2)` in place of raw `log E`, keeping `log pitch` as its own feature — the standard treatment for mixed cell sizes (ATLAS LCW physics/0408129, 1705.10363; CALICE software compensation 1207.4210 improved 58%/sqrt(E) to 45%/sqrt(E)). Directly targets our 5-pitch problem.
3. **Variable-length point set with kNN in a learned latent space**, retiring the fixed index grid (HGCalML feature list: recHitEnergy, eta, isTrack, theta, R, x, y, z, time, hitR — note `recHitHitR` is the per-hit cell radius, i.e. cell size as a plain feature, with no per-granularity embedding; Belle II ECL arXiv:2306.04179 uses global + ROI-local coordinates, crystal mass, k=14/16, S=6, 4 GravNet blocks, width 22-24).
4. Nested time-window energies (cumulative 0-0.5, 0-1, 0-2, 0-full ns, front and back, with `t - z/c` TOF subtraction) instead of raw times — a different encoding of information we already pass, per arXiv:2107.10207.
5. ROI-mean global exchange: append the window mean of every per-cell feature to each token (Belle II).

**Explicitly not worth doing** on the evidence: per-cell quality/pedestal flags (Belle II quantified this as ~no effect), calibration/aging constants, sampling-fraction corrections, hand-computed shape moments (derivable from tokens we already pass), and any further attention-mechanism variant.

**Mixed cell sizes**: the literature offers exactly three treatments — a scalar cell-size feature next to absolute coordinates, cell mass/volume as a feature, and energy-density weighting. Nobody uses separate embeddings per granularity, super-cell merging, or area-weighted attention.

**Small-n context**: nobody in this literature trains at our scale (Belle II 1.8M events, CLAS12 1M, arXiv:2107.10207 600k). No HEP learning-curve study compares grid against point cloud at n ~ 10^4-10^5, and no paper claims grid wins at small n — the published small-n lever is pretraining, not architecture (OmniJet-alpha 2403.05618: ~90% pretrained vs ~74% from scratch at 1000 examples). Belle II's winning model is tiny (width 22, k=14, 4 blocks), which is the scale to target rather than HGCalML's width-64 stack.

## 4e-results. Input-representation experiments beat the champion (2026-08-11)

All at 2 seeds versus the champion's seeds 0-1, so the comparison is like-for-like. This is the first time anything in the sweep moved the aggregate.

| bin | champion | Ex (containment block) | ExDn (+ energy density) | W6 (13x13 window) |
|---|---|---|---|---|
| 15mm low-E | 0.1817 | 0.1761 (-0.0056) | **0.1484 (-0.0333, ~3.4 sigma)** | 0.1682 (-0.0135) |
| 30mm low-E | 0.1123 | 0.1033 (-0.0090) | 0.0995 (-0.0128) | **0.0964 (-0.0159)** |
| 40mm low-E | 0.0636 | 0.0605 (-0.0031) | 0.0621 (-0.0015) | 0.0591 (-0.0045) |
| 15mm all | 0.0660 | 0.0586 (-0.0074) | **0.0553 (-0.0107, ~4.6 sigma)** | 0.0622 (-0.0038) |
| 30mm all | 0.0588 | 0.0487 (-0.0101) | **0.0487 (-0.0101, ~7 sigma)** | 0.0548 (-0.0040) |
| 40mm all | 0.0361 | 0.0364 (+0.0003) | 0.0371 (+0.0010) | 0.0372 (+0.0011) |
| 60mm all | 0.0341 | 0.0335 (-0.0006) | 0.0329 (-0.0012) | 0.0340 (-0.0001) |
| **aggregate** | 0.0411 | **0.0394 (-0.0017, ~3 sigma)** | **0.0393 (-0.0018, ~3 sigma)** | 0.0407 (-0.0004) |

**What this establishes.**

1. The cluster-level containment block (`--extra`) is worth about -0.0017 on the aggregate on its own — the first significant aggregate gain in the sweep, after seven architecture variants produced none. The features are the CMS R9 analogue plus the ATLAS-style reco shower position; the win comes from telling the model how much energy the 9x9 window and the 2.49 MeV threshold are throwing away.
2. Energy density (`--dens`, `log(E/pitch^2)`) does almost nothing for the aggregate on top of Ex (-0.0001) but transforms the worst bin: 15mm low-E goes 0.1761 -> 0.1484, an 18% relative improvement and the largest single effect measured anywhere in this project. That is exactly where the mixed-pitch problem lives, and it is the treatment the CALICE/ATLAS-LCW literature prescribes for mixed cell sizes.
3. The larger 13x13 window helps the low-E inner bins (30mm low-E -0.0159, the best of the three there) but not the aggregate (-0.0004). It costs 4x the training time per seed. Interpretation: the aperture does capture some of the missing containment, but feeding the cluster total as a feature captures it far more cheaply — a measurement beats an enlarged window.
4. The regressions are confined to 40mm all (+0.0010) and 120mm (noise). The mentors' "must work everywhere" criterion is satisfied: 15mm and 30mm, the two weak regions, improve by 4.6 and 7 sigma.

**Next**: ExDn at 5 seeds (queued) so the full stack — 5-seed ensemble, D4 TTA, width-binned calibration — can be assembled against the champion's 0.0402; plus `--dens` alone to attribute the density gain cleanly.

## 4e-final. Five-seed result and attribution (2026-08-11)

ExDn at the full 5 seeds, against the champion at its full 5 seeds:

| bin | champion (5 seeds) | ExDn (5 seeds) | delta |
|---|---|---|---|
| 15mm all | 0.0632 | 0.0574 | -0.0058 |
| 30mm all | 0.0579 | 0.0480 | **-0.0099** |
| 40mm all | 0.0357 | 0.0368 | +0.0011 |
| 60mm all | 0.0339 | 0.0330 | -0.0009 |
| 120mm all | 0.0366 | 0.0370 | +0.0004 |
| 15mm low-E | 0.1811 | 0.1683 | -0.0128 |
| 30mm low-E | 0.1079 | 0.0948 | -0.0131 |
| **aggregate** | **0.0409** | **0.0389** | **-0.0020 (~3.5 sigma)** |

This is before the ensemble/TTA stack that took the champion from 0.0409 to 0.0402, so the comparable stacked number should land near 0.038.

**Correction to the 2-seed reading.** The 2-seed snapshot gave 15mm low-E as 0.1484 (-0.0333); at 5 seeds the same bin is 0.1683 (-0.0128). The 2-seed figure was a favourable fluctuation — that bin holds only 411 clusters and carries an error of +-0.0080. The robust claims are the aggregate (-0.0020) and 30mm all (-0.0099); the low-E gains are real but roughly a third of what the first snapshot suggested. Screening at 2 seeds is fine for ranking, not for quoting.

**Attribution** (2 seeds each, versus champion seeds 0-1):

| run | aggregate | 15mm low-E | 30mm low-E | reading |
|---|---|---|---|---|
| Ex (containment block only) | 0.0394 (-0.0017) | -0.0056 | -0.0090 | drives the **aggregate** |
| Dn (energy density only) | 0.0416 (+0.0005) | **-0.0156** | -0.0098 | drives the **weak low-E bins**, no aggregate effect |
| ExDn | 0.0393 (-0.0018) | -0.0333 | -0.0128 | both effects, complementary |
| ExGx (containment + global exchange) | 0.0393 (-0.0018) | -0.0240 | -0.0074 | Gx adds nothing over Ex on aggregate |
| PhysOcc | 0.0408 (-0.0003) | -0.0119 | -0.0072 | superseded by Ex |

The decomposition is clean and physically sensible: the **containment block** tells the model how much energy the window truncation is losing, which is a global scale effect and therefore moves the aggregate; **energy density** normalises for cell size, which only matters where the pitch is small and the shower is under-sampled, and therefore acts on the inner low-energy bins alone. They address different failure modes, which is why they add.

## 4e-errors. Residual structure in the two weak bins (2026-08-11)

Error analysis of the low-E terciles of 15mm and 30mm, champion versus the new best ExDn, 2 seeds each.

| quantity | 15mm low-E champion | 15mm low-E ExDn | 30mm low-E champion | 30mm low-E ExDn |
|---|---|---|---|---|
| sigma_eff | 0.1817 | 0.1484 | 0.1123 | 0.0995 |
| robust sigma (1.4826 MAD) | 0.1193 | 0.1037 | 0.0926 | 0.0795 |
| RMS | 0.3414 | 0.3067 | 0.2562 | 0.2381 |
| median bias | +0.0187 | +0.0111 | +0.0090 | +0.0069 |
| residual 5% / 95% quantile | -0.272 / **+0.793** | -0.257 / +0.756 | -0.220 / **+0.541** | -0.198 / +0.471 |
| sigma_eff of central 90% | 0.1339 | 0.1016 | 0.0874 | 0.0763 |
| sigma_eff, lower half of the bin | **0.2785** (E<32 GeV) | 0.2375 | **0.1852** (E<20 GeV) | 0.1550 |
| sigma_eff, upper half of the bin | 0.0934 | 0.0748 | 0.0702 | 0.0568 |

Three structural facts, all unchanged by ExDn (which improves every number by 13-18% without altering the shape):

1. **The error is one-sided.** The 95% quantile is +0.79 while the 5% is -0.27, and the median bias is positive. The model **over-estimates**: pileup energy inside the window is being attributed to the photon. This is contamination, not symmetric noise, and it is not what containment features fix — `total_energy` tells the model what it is *missing*, nothing tells it what is *extra*.
2. **It is tail-driven.** Trimming the worst 10% takes 15mm low-E from 0.1484 to 0.1016; RMS is 2x sigma_eff. A minority of clusters carries the resolution.
3. **It is concentrated at the very lowest energies.** Splitting the low tercile in half: 15mm gives 0.2375 below 32 GeV versus 0.0748 above; 30mm gives 0.1550 below 20 GeV versus 0.0568 above. The upper half of the "bad" bin is already near target.

**Action taken from this**: ring-rho median-density subtraction (`--rho`, queued). rho = median(E/area) over the window's outer annulus (|di|,|dj| > 2), giving a per-cell corrected energy E - rho*area as a feature plus global log rho and corrected/raw sum ratio. This is the standard median-density pileup subtraction from jet reconstruction (Cacciari-Salam-Soyez jet areas) and it targets a one-sided positive contamination tail directly. It had been listed as "ring-rho next" in the nb28 arc and never implemented.

**Further levers implied, not yet run**: an asymmetric or tail-aware objective (the current pinball/qd loss treats both sides alike while the error is one-sided); an energy-binned calibration stage alongside the existing width-binned one, since sigma varies by 3x inside a single tercile; and a dedicated low-energy expert, which the literature supports at this scale only if the routing variable is observable at inference.

## 4g. What limits the low-energy resolution, quantitatively (2026-08-11)

`scripts/fit_resolution.py` fits the standard parametrisation per region on the ExDn 5-seed predictions:

sigma(E) = sqrt( (a/sqrt(E))^2 + (b/E)^2 + c^2 ), a stochastic, b noise in GeV, c constant.

| region | a | b [GeV] | c | sigma@10 | sigma@20 | sigma@40 | E where sigma=0.05 | b needed for 0.05 at region median |
|---|---|---|---|---|---|---|---|---|
| 15mm | 0.000 | **4.34** | 0.000 | 0.434 | 0.217 | 0.108 | 86.8 GeV | 3.05 |
| 30mm | 0.000 | **1.85** | 0.000 | 0.185 | 0.093 | 0.046 | 37.0 GeV | 2.23 |
| 40mm | 0.149 | 0.402 | 0.000 | 0.062 | 0.039 | 0.026 | 13.6 GeV | 1.05 |
| 60mm | 0.055 | 0.302 | 0.017 | 0.039 | 0.026 | 0.020 | 7.1 GeV | 0.76 |
| 120mm | 0.032 | 0.243 | 0.025 | 0.036 | 0.029 | 0.026 | 5.9 GeV | 0.56 |

**The single most important result of this investigation.** In the two weak regions the fit returns a = c = 0: the resolution is *entirely* the b/E noise term. There is no stochastic and no constant contribution left to remove. b is an absolute energy uncertainty — 4.34 GeV in the 15mm region, 1.85 GeV in the 30mm region, versus 0.24-0.40 GeV in the outer three — and it is the pileup energy fluctuation collected inside the reconstruction window. Relative resolution at low energy is therefore just b divided by the photon energy, which is why the low-energy bins look catastrophic while the same region at high energy is at target.

**Consequences.**

1. **A 0.05 target at low energy is a statement about b, not about the model.** For 30mm low-E (median ~20 GeV) it needs b <= 1.0 GeV, a factor 1.85 reduction. For 15mm low-E (median ~32 GeV) it needs b <= 1.6 GeV, a factor 2.7. At 10 GeV in 15mm it would need b <= 0.5 GeV, a factor 8.7. No feature engineering or architecture change reduces b; only collecting less pileup, or subtracting it, does.
2. **Three of five regions already meet 0.05** above 13.6 / 7.1 / 5.9 GeV respectively, i.e. across most of their spectrum. The target is a two-region problem.
3. **This explains the 13x13 window result.** b grows with the window area over which pileup is integrated, so a larger aperture increases b — which is exactly what we measured (W6: low-E bins improved slightly from better containment, aggregate flat). The prescription is the opposite of what we tried: a *smaller* aperture in the dense regions, with the `--extra` containment features compensating the lost containment. Going from 9x9 to 5x5 scales b by roughly sqrt(25/81) = 0.56, a 1.8x reduction — most of what 30mm needs. Queued as `--window 2` and `--window 3` with the ExDn feature set.
4. The other two levers that act on b rather than on the model: ring-rho subtraction (running) removes the pileup pedestal, and timing separates out-of-time pileup — the physics PicoCal's 10-20 ps design was built for. The earlier time-gate null result says our *implementation* added nothing, not that timing is exhausted; the fit says timing is aimed at the only term that matters here.

This also supplies the deliverable Felipe asked for in a stronger form than requested: rather than overlaying an assumed design curve, the measured a/b/c per region are now in hand and can be compared directly against the design values once he provides them.

## 4f. Closest published work — must read and must cite

**arXiv:2603.18172, "Reconstruction of overlapping electromagnetic showers in calorimeters using Transformers"** (CEA/IRFU, v2 Jul 2026) is the nearest work to this project: overlapping EM showers, seed-window transformer, energy target, dead-channel robustness. It is absent from `reports/novelty_gap_check_2026-08-05.md` and must be added. It also contains the only same-paper, same-dataset transformer-versus-message-passing comparison with a calorimeter energy target (sigma_E in GeV, central 68%):

| configuration | SF+GAT | SF+GNN | ClusTEX (transformer) | PFClustering |
|---|---|---|---|---|
| toy, 1 photon | 0.53 | 0.54 | — | 0.61 |
| toy, 2 photons | **0.80** | 0.95 | — | 6.39 |
| ECAL-like, 1 photon | 0.58 | **0.57** | 0.55 | 0.59 |
| ECAL-like, 2 photons | 1.10 | 1.17 | **0.87** | 6.24 |

Attention buys essentially nothing on isolated showers — and loses to message passing in one configuration — but gains roughly 25% under shower overlap; their stated reason for the isolated-case loss is the local receptive field and translational invariance of the convolutional variant. Latency on A100 at batch 512: 0.68 ms/event (ClusTEX) versus 0.39 (GAT) and 0.28 (GNN). MLPF (arXiv:2309.06782), the largest HPO'd attention-versus-message-passing ablation (~5e6 parameters each, 5000 vs 7000 GPU-hours), found the optimised GNN beat the kernel-based transformer, though at particle level and with an approximated softmax.

Read for our purposes: attention is the right choice for a pileup-dominated sample, which is where we are, but we have most likely exhausted that axis — consistent with seven all-attention variants landing within 0.0409-0.0426. The remaining headroom is in the input representation.

## 5. Key references

## 4h. Complete experiment ledger (as of 2026-08-12)

Every configuration trained since the per-region diagnosis, scored against its like-for-like baseline at matching seed count with `scripts/score_regions.py`. Aggregate sigma_eff.

| # | change | class | aggregate | verdict |
|---|---|---|---|---|
| — | `--extra --dens` (**current best**) | information | **0.0389** | **-0.0020 vs previous champion, ~3.5 sigma** |
| — | timing, measured by removing it | information | 0.0496-0.0508 without | **worth 20% aggregate, 24-39% in the weak bins** |
| 1 | physical mm coordinates + occupancy | information | 0.0408 | superseded by `--extra` |
| 2 | cluster-total containment block (`--extra`) | information | 0.0394 | **kept** |
| 3 | energy density `log(E/pitch^2)` (`--dens`) | information | 0.0416 alone | **kept** — acts on the weak low-E bins |
| 4 | outside-window pileup density (`--orho`) | information | 0.0398 | no gain |
| 5 | absolute window time + late fraction (`--abst`) | information | training | the only timing avenue left (out-of-time) |
| 6 | explicit time gate | encoding | 0.0413 | null; in-time separation is 0.26-0.76 sigma per cell |
| 7 | resolution-weighted time pull | encoding | 0.0398 | null; the in-time shift is a common mode |
| 8 | per-cell front/back depth ratio | encoding | 0.0398 | null; 0.6 sigma per cell means rho^2 = 0.08 |
| 9 | Fourier features on in-cell offsets | encoding | 0.0401 | null |
| 10 | ring-median density subtraction | encoding | 0.0395 | null; a mean subtraction cannot cut its fluctuation |
| 11 | geometric+temporal attention bias | architecture | 0.0420 | worse |
| 12 | geo + query pooling | architecture | 0.0414 | best architecture variant, still no gain |
| 13 | query pooling alone | architecture | 0.0426 | worse |
| 14 | CNN over the window image | architecture | smoke only | 30x faster to train; dropped once the pattern was clear |
| 15 | signed gate (range -0.5..1.5) | architecture | 0.0395 (4 seeds) | null, despite GLS predicting negative outer weights |
| 16 | ROI-mean global exchange | architecture | 0.0393 | null |
| 17 | per-block FiLM conditioning | architecture | 0.0404 | null; context already reached the readout |
| 18 | window 5x5 | aperture | b worse 14-32% | worse; containment loss beats the pileup saving |
| 19 | window 7x7 | aperture | not completed | dropped once 5x5 answered it |
| 20 | window 13x13 | aperture | 0.0407 | aggregate flat |
| 21 | D4 train-time augmentation | data | 0.0416 | null |
| 22 | multi-task position + time targets | objective | 0.0395 | null |
| 23 | inverse-density weighting (0.4/0.7/1.0) | objective | 0.0400/0.0410/0.0429 | worse, monotonically in alpha |
| 24 | trimmed risk 10% / 20% | objective | 0.0514 / 0.0566 | **worst of all** — sigma_eff ignores the tail, the model must not |
| 25 | test-time D4 averaging | inference | 0.0391 | neutral (and it never existed before, though reports claimed it) |
| 26 | width-binned Deming calibration | calibration | 0.0393 | trades sigma_eff for bias; a physics choice, not a win |
| 27 | joint q50 x width calibration | calibration | 0.0390 | null |
| 28 | region x q50 x width calibration | calibration | 0.0397 | worse; 60 cells over 10.9k validation events is too thin |

**The pattern, stated once.** Two changes moved the metric and both added information the detector already measured. Twenty-plus changes to architecture, encoding, objective, aperture, inference and calibration moved nothing. Combined with `reports/bounds_2026-08-12.md` — the learned estimator already beats the best possible *fixed linear* estimator by 1.8-3.6x — the reading is that at this pileup level the resolution is set by the information in the measurement, not by the flexibility of the model. That is the headline finding of this phase.

## 5. Key references

2002.03605, 2106.01832, 2503.11277, 2412.11208, 2409.16052, 2402.12535, 2310.16497, 2107.02779, 2410.22074, 2503.02860, 2203.15823, 2212.09184, 2203.09168, 2502.13129, 2501.03432, 2403.10348, 1901.09192, 2311.08885, 1707.08600, 1511.03643, JINST 21 C03006. Added 2026-08-12: 2603.18172 (closest published work), 0912.4926 (b ~ sigma*sqrt(A)), 1407.0408 (SoftKiller), 2006.14359 (CMS multifit), 2203.01317 (ns-level timing worth 3-4% in a hadronic calorimeter), 2205.02500 (SpaCal 18.5 ps is a front+back combination, not per-cell), 1905.03222 and PMLR 128 (cannot move a point estimate), 1911.09107 (unfolding returns weights, not per-event values).
