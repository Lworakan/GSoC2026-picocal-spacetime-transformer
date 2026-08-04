# Spec: Road to 0.0235 — aggregate design-floor campaign

Date: 2026-08-01. Status: draft for Worakan's review (brainstorming output). Research basis: three verified sweeps this date (architecture / loss / methodology) + campaign record nb12-nb48.

## 1. Goal and success criteria (approved 2026-08-01)

Target: **aggregate sigma_eff on the full minbias spectrum -> 0.0235**, the quantile floor of the metric on OUR spectrum under the assumed design curve 10%/sqrt(E) ⊕ 1% (recomputed numerically on the real test energies; sigma_eff discards the worst 32% by construction, so the aggregate floor is NOT the mean of per-bin floors). The target auto-updates when Felipe's exact ideal-resolution formula arrives (task #14): 7%/sqrt(E) would move the floor to 0.0181.

Measured state (2026-08-01): minbias stack 0.0426 (41.4% of events within |r|<=0.02) · clean best 0.0397 (42.0%) · floor 0.0235 (61.1%). Decomposition: **pileup penalty ~0.003; clean-side reconstruction gap ~0.016 — the bottleneck flipped to the clean side.** Even top bins sit at ~45-48% coverage where their floor allows ~75% — the missing 68%-mass lives at high E and does not require fighting pileup.

Milestones: M1 clean 0.0397 -> <=0.035 · M2 minbias <= clean + 0.004 at every step · M3 minbias <=0.030 this campaign, then track the floor.

Guardrails (unchanged): six-stage notebook per hypothesis; pre-registered win criterion (>0.002; <0.002 = noise); >=2 seeds; selection on val only; GPU lock check before every launch; no commits without instruction; notebooks import frozen code from scripts/.

## 2. Act 3a — sigma_eff-direct calibration (free, first)

Replace the final least-squares polyfit (optimizes MSE, not our metric) with a direct 2-parameter fit minimizing sigma_eff on val, per width-group. Precedent verified: Belle II selects models by FWHM on validation (2306.04179); ATLAS fits scale/smearing on holdout (2309.05471). Inference-only on existing nb44 quantile outputs. Criterion: report whatever it gives; adopt if val and test agree in direction.

## 3. Act 1 — nb49: clean window scan (first GPU experiment)

One variable: W ∈ {4, 6, 8}, trained and evaluated on CLEAN only, current stack (quant + width recalibration). Mechanism: containment fluctuation is the measured clean floor (nb16 corr 0.977); a wider window on clean has NO pileup cost; W was only ever scanned on minbias (nb32, pre-quant). 2 seeds x 3 windows. Win: any W beats clean-W4 by >0.002. Verdict cell must include residual-vs-containment correlation per W (did widening actually eat the containment term?) and per-bin table vs per-bin floors.

## 4. Act 1b — supervision-first architecture hypotheses (triggered by nb49 error analysis)

Research re-ranked these; the top two are supervision changes, not backbone swaps:

1. **Auxiliary containment-fraction head** (CRILIN 2606.05111: shape observables regress the contained fraction; our residual-containment corr 0.977). Label is FREE on clean events: c = sumE_window/(1000·Etrue). Multi-task: quant head + containment head; the shared encoder learns shower-shape->leakage. Trigger: nb49 still shows containment-correlated residuals at best W.
2. **Depth-conditioned per-cell compensation weights** (CALICE 2403.04632): readout becomes calib(Σ gate·(α(h)·fr + β(h)·bk)) with per-cell learned front/back weights — end-to-end software compensation. Trigger: nb49 residuals correlate with front/back ratio.
3. **Core+halo pyramid** (3x3 + 9x9 + 13x13 paths, merge before head). No published prior art for calo regression (verified gap) — novelty if it works. Trigger: wide-W wins but dilutes the core.
4. ClusTEX-style dual local/global positional encoding (2603.18172) — cheap, unfalsified (distinct from the falsified Swin bias). Backup.
5. DRN learned pooling (2003.08013, CMS DP-2024/066) — data-hungry at 30k; last resort.

## 5. Act 4 — loss and methodology round (cheap, parallel to Act 1)

Loss hypotheses (head/loss swaps on the winning config, one at a time, val-judged):
- **Quality-Driven interval loss** (Pearce 1802.07167): minimize interval width s.t. coverage >=68.3% — the literal trainable sigma_eff; known finicky, needs soft coverage + ensembling. Verified gap: NO published smooth sigma_eff surrogate exists — success here is publishable on its own.
- **Trimmed loss** (2308.02293): drop the largest per-batch losses (trim 0.25-0.5) — mirrors the metric's ignore-worst-32%.
- **Huber median + pinball dispersion jointly** (CMS 1912.06046 recipe) — small variant of current loss.
- DSCB-NLL (CMS e/gamma 2012.06888) — core/tail decoupling; medium cost, behind the others.

Methodology (evidence x cheapness, from the verified sweep):
- **EMA/SWA weight averaging** — one-line, 0.5-2% expected (2502.06761 ICML 2025; tabular 2604.15297). Add to every future run + A/B once.
- **C-Mixup** (2210.05775, NeurIPS 2022, +6.6% RMSE avg at 10^4 scale) — batch-sampler change.
- **Muon optimizer** (tabular evidence beats AdamW at small scale, 2604.15297) — behind EMA/C-Mixup.
- Free soup test across the 5 nb44 seeds (expected to fail; one eval).
- Skipped as no-evidence-at-our-scale: RMSNorm, QK-norm, stochastic depth, CRPS (adds tail weight the metric ignores).

## 6. Act 2 — minbias port + data levers

Best clean recipe (W*, aux heads, loss) -> minbias with clean-aux (proven transfer mechanism). If W*>4, pileup in the extra ring is handled by the free gate first; when Felipe's minbias-only sample arrives, the positional-matching study (task #16) supplies real-pileup per-cell labels to re-suppress it supervised (Belle II mechanism, real-pileup labels — H3's domain gap does not apply).

## 7. Sequencing

1. Act 3a (tonight, free) -> 2. nb49 W-scan (GPU ~3h) -> 3. Act 4 cheap round (EMA + C-Mixup + soup test) on the nb49 winner -> 4. Act 1b hypothesis chosen BY nb49's error analysis -> 5. Act 2 minbias port -> 6. loss-novelty round (QD/trimmed) once the pipeline is stable. Felipe inputs (ideal curve, GNN outputs, minbias-only sample) slot in whenever they arrive; meeting ~Aug 13 shows: floor math, gap decomposition, new record, this roadmap.

## 8. Out of scope (falsified, do not revisit)

Label-pressure supervision (H3/H7/DANN/distillation), time representations beyond raw+flags (H5/H9/H9b), time-gated wide window as a pileup tool (H12), equal-corpus masked pretraining (H10), pooling variants (H11a), isotonic calibration, cross-window mega-ensembles.
