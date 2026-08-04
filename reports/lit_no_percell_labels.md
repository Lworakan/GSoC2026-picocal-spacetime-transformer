# Literature landscape: reaching 0.030–0.035 per-bin without per-cell truth labels

Date: 2026-07-29. Trigger: mentor confirmed the per-cell signal-energy branches are hard to obtain. Question: what published routes remain, given event-level E_true on clean + minimum-bias samples is our only supervision?

Three parallel verified searches (weak supervision in HEP / ML pileup mitigation / self-supervised & statistical). All arXiv IDs verified against arXiv metadata. Companion file: `lit_selfsupervised_statistical.md`.

## Topic summary

ML pileup mitigation splits into three supervision families: (a) per-cell/per-particle simulation truth (PUMML 1707.08600, PUPPIML 1810.07988, ABCNet 2001.05311, PUMA 2107.02779, Belle II 2306.04179 + 2604.20518, CMS HGCAL) — the dominant family, blocked for us; (b) surrogate labels from a self-labeled subset (semi-supervised GNN 2203.15823: charged particles labeled by tracking supervise neutrals); (c) two-sample / event-level supervision (CWoLa 1708.02949, LLP 1702.00414, TOTAL 2211.02029, WOTAN ML4PS 2024, Vipr diffusion 2410.22074, sPHENIX unpaired translation 2510.23717) — directly compatible with our data. Aggregate-observation learning theory (Zhang et al. NeurIPS 2020, 2004.06316; Scott & Zhang 2006.07330) proves instance-level learning from bag-level labels is identifiable given diverse bag compositions.

## Coverage matrix vs our falsified hypotheses

| Approach family | Representative | Supervision | Already tested here? | Verdict |
|---|---|---|---|---|
| Per-cell sim truth | PUMML, Belle II GNN | per-cell fractions | oracle only (nb29) | blocked by data availability |
| Synthetic overlay labels | our H3 (nb28/30) | manufactured fractions | ❌ falsified (domain gap AUC 0.74; supervision hurt) | closed |
| Domain adaptation | DANN 1912.08001 | overlay + adversary | ❌ falsified (nb30) | closed |
| Feature distillation | Hong et al. 2103.07600 | clean-teacher pairs | ❌ falsified (nb41) | closed |
| Two-sample OT weak supervision | TOTAL 2211.02029 / WOTAN ML4PS 2024 | clean vs pileup collections, no per-cell labels | **not tested** | open — most direct match to our setup |
| Unpaired image translation | sPHENIX 2510.23717 (cycle-consistency) | unpaired signal+bkg vs bkg-only | **not tested** | open — structural match (clean vs minbias) |
| Exact per-event fraction supervision (LLP/sPlot) | 1702.00414, 1801.10158, sPlot-ML 1905.11719, theory 2004.06316/2101.07263 | per-event pileup fraction — we KNOW it exactly: (E_cluster−E_true)/E_cluster | partially (gate trained end-to-end on E_true) | open — explicit fraction-sum supervision on the gate is untested |
| Heteroscedastic / quantile heads | CMS b-jet DNN 1912.06046, Gaussian Ansatz PRL 129 082001 | event-level only (have it) | **not tested** | open — CMS reports 12–15% effective-width gain |
| Masked-cell pretraining | MPM 2401.13537, OmniLearn PRD 111 L051504 | none (unlabeled minbias stream) | **not tested** | open — cheap, uses 94-file stream |
| Timing-surrogate semi-supervision | 2203.15823 analog | quasi-labels from timing/tracks | ❌ partially (H5 pair-Δt flat) | reframe: needs track extrapolation from LHCb, ask mentor |
| Real zero-bias embedding | ATLAS overlay 2102.09495; STAR/ALICE practice | collaboration workflow, not a new branch | not available yet | **the cheaper ask to bring back to the mentor** |

## Gap identification

**Genuine gap (HIGH confidence, 3 searches, no preprint found):** no published method performs per-cell signal-fraction *regression* from event-level energy labels. Nearest neighbours are WOTAN (per-particle noise weights from unpaired collections) and E Pluribus Unum Ex Machina 2101.07263 (per-instance classifiers from aggregate labels). Our subtract-then-calibrate gate + exact per-event fractions sits exactly in this gap — publishable as a method paper if any of the open rows above moves the resolution.

## Ranked next hypotheses (scientific-method pipeline)

1. **H7 — exact-fraction supervision (LLP/sPlot route).** Error analysis: gate is currently identified only through the calibrated sum. We possess the exact per-event pileup fraction; adding a loss term tying mean gated fraction to it is stronger supervision than LLP assumes (1702.00414) and theoretically identifiable (2004.06316). Cost: one auxiliary loss, existing pipeline, CPU-safe.
2. **H8 — heteroscedastic/quantile head (CMS 1912.06046).** Same labels, second head predicts per-event resolution; correct with the predicted scale. Precedent: 12–15% effective-width improvement at CMS. Cost: small.
3. **H9 — two-sample OT loss (TOTAL/WOTAN).** Add an optimal-transport term pulling gated minbias windows toward clean-window distributions conditioned on E_true (uses Neural Conditional Reweighting 2107.08979 idea). Untested family; no pairing needed.
4. **H10 — masked-cell pretraining on the unlabeled minbias stream** (2401.13537 recipe), fine-tune the SubNet.
5. **Mentor asks (parallel, no compute):** (i) real zero-bias overlay/embedding at digitization level (ATLAS 2102.09495 — routine practice, creates per-cell truth with REAL pileup, fixes the H3 domain gap); (ii) track extrapolation onto PicoCal cells for charged surrogate labels (2203.15823 analog).

## Caveats

- Differences <0.002 in sigma_eff are noise at our n; every hypothesis needs ≥2 seeds before a verdict (campaign rule).
- Vipr (2410.22074) and sPHENIX translation are generative/heavy; keep behind H7–H9 which reuse the winning SubNet stack.
- WOTAN has no arXiv entry found — only the NeurIPS ML4PS 2024 PDF; cite carefully.
