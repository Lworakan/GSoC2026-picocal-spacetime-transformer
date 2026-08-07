# Novelty gap check — 2026-08-05

Pre-submission literature sweep for the two headline claims. Two parallel searches, ~20 queries total, all bibliographic entries verified against arXiv/PMLR/IOP pages.

## Claim 1 — Label-free emergent per-cell pileup subtraction

Readout `a*log(1+sum sigma(f_i)*E_i)+b`; gate trained only by event-level energy objective; recovers true photon fraction at r=0.92; explicit per-cell supervision (aux loss, DANN, distillation) underperforms the emergent gate.

**Verdict: novel — confidence HIGH** for the combined claim. The supervision-underperforms-emergence result (d) appears entirely unclaimed anywhere.

| Paper | Verdict | Why |
|---|---|---|
| Qiu, Han, Ju, Nachman, Wang, "Parton Labeling without Matching," EPJC 83 (2023), arXiv:2304.09208 | PARTIAL | Establishes emergent labeling from event-level regression in HEP; no calorimeter cells, no structural gate, no fraction recovery, no pileup. Cite prominently. |
| Lai et al. (CALICE), "Software Compensation ... using Machine Learning," JINST 19 (2024) P04037, arXiv:2403.04632 | PARTIAL | Closest structural prior art: NN cell weights emerge from energy-only objective; interpretation qualitative only, no truth-fraction correlation, no pileup, no supervision comparison. |
| Li et al., "Semi-supervised GNN for Pileup Noise Removal," EPJC 82 (2022), arXiv:2203.15823 | PARTIAL | Per-particle pileup weights without sim truth, but supervised by charged-particle labels (classification), not emergent from an energy objective. |
| Komiske, Metodiev, Nachman, Schwartz, PUMML, JHEP 12 (2017) 051, arXiv:1707.08600 | SAFE | Fully supervised per-pixel targets. |
| Maier, Narayanan et al., PUMA, MLST 3 (2022) 025012, arXiv:2107.02779 (+ PUMiNet arXiv:2503.02860) | SAFE | Per-particle classification labels used. |
| Lv, Miao, Xu, Wang, HistoAE, arXiv:2511.22246 (2025) | PARTIAL | Same "unsupervised recovery of physical truth" narrative, different detector/task; cite as concurrent work. |
| Maidannyk, Couderc, Malcles, Sahin, ClusTEX, arXiv:2603.18172 (2026) | SAFE | Nearest 2026 ECAL-transformer work, GEANT4-supervised, no emergent gate claim. |
| Mengel et al., PRC 108 (2023) L021901, arXiv:2303.08275 | SAFE | Jet-level interpretable subtraction, no per-constituent gate. |

Framing caveat: if pitched broadly as "emergent labels in HEP," arXiv:2304.09208 precedes that framing — pitch as the calorimeter/pileup instantiation with quantified fraction recovery plus the supervision-vs-emergence experiment.

## Claim 2 — Width-binned post-hoc calibration

Events binned by predicted interval width (q75-q25); separate linear calibration on q50 fitted per bin on validation.

**Verdict: novel in calorimetry — confidence MEDIUM-HIGH.** Exact mechanism found nowhere; each ingredient exists separately, so frame as a combination, not a new primitive.

| Paper | Verdict | Why |
|---|---|---|
| Bostrom, Johansson, "Mondrian conformal regressors," COPA 2020, PMLR 128:114-133 | PARTIAL | Bins by difficulty estimate but calibrates only intervals, never fits per-bin response/bias correction to the point prediction. |
| Pernot, arXiv:2310.11978 (2023) | PARTIAL | Uncertainty-binned per-bin scaling, but rescales the uncertainty itself, not the response. |
| CMS, b-jet energy+resolution DNN, CSBS 4 (2020), arXiv:1912.06046 | PARTIAL | Nearest HEP prior art: same q75-q25 construct, but width feeds analysis categorization, not per-width-bin energy calibration. |
| ATLAS, uncertainty-aware NN calorimeter calibration, arXiv:2412.04370 (2024) | SAFE | Uncertainty reported/interpreted, not used as conditioning variable for downstream fitted calibration. |
| CMS H->gg (e.g. arXiv:2208.12279) | SAFE | Per-event sigma_m/m used for sensitivity categorization; scale corrections binned in eta/R9, not predicted width. Must be discussed as the canonical precedent. |
| Conformal prediction as HEP calibration standard, arXiv:2512.17048 | SAFE | Interval validity only. |

Also cite as ML context: Romano, Patterson, Candes, CQR, NeurIPS 2019, arXiv:1905.03222; Kuleshov, Fenner, Ermon, ICML 2018, arXiv:1807.00263.

## Paper positioning (both claims)

- Contribution type: finding + system. Claim 1 is the headline (HIGH); claim 2 is a supporting method contribution (MEDIUM-HIGH), presented as an ablation-backed component.
- Related-work must-cites: 2304.09208, 2403.04632, 2203.15823, 1707.08600, 2107.02779, 1912.06046, PMLR 128 Bostrom, 1905.03222, 1807.00263.
- Residual risk: ML-side conformal literature moves fast; re-run this sweep once more at submission time.
