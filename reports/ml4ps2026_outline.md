# ML4PS 2026 (NeurIPS workshop) — 4-page paper outline

**Target:** ML4PS @ NeurIPS 2026 (Sydney, Dec) · deadline ~late Aug 2026 (verify) · 4 pages + refs · anonymized · non-archival
**Data:** real LHCb PicoCal single-photon + minimum-bias (94 files, ~90k clusters). **BLOCKER:** confirm with Felipe/Carla that results are publishable + co-authorship.

---

## Title candidates
- *A Spacetime Set-Transformer for Pileup-Robust Photon Energy Reconstruction in the LHCb Calorimeter*
- *Per-Cell Timing Improves Transformer-Based Calorimeter Energy Regression under Pileup*

## Abstract (draft ~130 words)
Photon energy reconstruction in high-granularity calorimeters degrades under pile-up, where each cell mixes signal and out-of-time background. We cast a calorimeter cluster as a permutation-invariant set of cells and study a Particle-Transformer-style architecture (energy-weighted / IRC-safe pooling + residual-to-calibrated-sum target) on the LHCb PicoCal single-photon + minimum-bias sample. On realistic pile-up our model reaches σ_eff ≈ 0.050, ~2.5× better than a gradient-boosted-tree baseline and ~3.6× better than the calibrated cell sum. We show that (i) adding **per-cell timing** — the variable the detector's fast timing is designed for — cuts σ_eff by ~15%, (ii) a robust (Huber) objective gives a further gain, and (iii) an error analysis attributes the residual floor to cell-selection *containment* (|residual| correlates 0.98 with captured-energy fraction), pointing to timing-aware cell selection as the next lever.

---

## 4-page structure

**1. Introduction (0.5 p)**
- Pile-up at HL-LHC / LHCb Upgrade II; calorimeter energy reco degrades.
- Sets/point-clouds for calorimeters; gap: per-cell *timing* used as a token/pooling signal for energy regression under pile-up.
- Contributions: (a) spacetime set-transformer on real min-bias, (b) timing gives measurable gain, (c) honest floor = containment.

**2. Data & task (0.5 p)**
- LHCb PicoCal, single-photon + min-bias, kNN-25 cells, 1–100 GeV, vertex cut. σ_eff metric (smallest-68% half-width).
- Note: photon is ~30% of cluster energy under pile-up (nb14) — the difficulty.

**3. Method (0.75 p)**
- Per-cell 12→15-dim tokens (energy, geometry, region one-hot, **Δt front/back + valid flag**).
- ParT pairwise-bias attention + energy-weighted EFN pooling + residual/bias target (base = calibrated sum).
- In-time gated pooling + Huber loss.

**4. Results (1.25 p) — the core**
- Table 1: baselines vs models (BDT 0.125, sum 0.184, total_E 0.358; MeanDirect/Residual/EFN 0.064–0.072; +timing 0.056; +gate+Huber 0.050).
- **Fig 1 (money plot):** σ_eff bar chart with BDT line — the ~2.5× win.
- **Fig 2:** timing ablation ladder (space → +time → +Δt-in-U → +gate+Huber).
- **Fig 3:** resolution vs energy (adaptive quantile bins) — model vs BDT vs sum.
- Two-target finding (Felipe): under pile-up bias≈direct, **pooling > target**.

**5. Error analysis & outlook (0.5 p)**
- Containment drives the floor (corr 0.98); timing-aware larger-window cell selection as next step; caveat clean vs pileup spectrum matched after vertex cut.

---

## Figures needed (checklist)
- [ ] Fig1 σ_eff bars + BDT line (have — nb17)
- [ ] Fig2 timing ladder (have — nb16/17)
- [ ] Fig3 adaptive-bin resolution vs energy (have — nb16/17)
- [ ] (opt) Fig4 containment corr scatter (have — error analysis)

## Experiments to firm up before submit
- [ ] **5–8 seeds** on the headline configs (tighter error bars) — currently 3
- [ ] optional nb18 window study (supporting, not essential for 4 pages)
- [ ] confirm BDT is the strongest fair baseline; mention LHCb rule-based if allowed

## Timeline (~6 weeks)
```
Wk1-2: finalize seeds (5-8) on 0.050 config + polish 3 figures
Wk3-4: write 4 pages
Wk5:   Felipe/Carla internal review + data-permission sign-off
Wk6:   polish + submit
```

## Blockers / must-do
1. **LHCb data-publication permission + co-authorship (Felipe/Carla)** — before anything public.
2. Verify exact ML4PS 2026 deadline on ml4physicalsciences.github.io.
3. Anonymize (double-blind) — no LHCb-identifying author info in submission.
