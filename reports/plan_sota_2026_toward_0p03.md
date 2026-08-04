# SOTA 2025-2026 plan toward per-bin 0.030 (error-analysis-driven)

Date: 2026-07-30. Four parallel verified searches, one per measured error axis. Baseline in flight: nb44 quantile stack (quant + clean-aux + 5 seeds + D4 TTA); nb43 quant 2-seed ens = 0.0437. Companion: lit_no_percell_labels.md (2026-07-29).

## Error axis → SOTA answer

| Error axis (our measurement) | SOTA 2025-26 answer | Key refs (verified) | Leverage |
|---|---|---|---|
| Pileup contamination — worst axis at E>17 (0.041 vs 0.028 by tertile) | Unpaired two-sample subtraction (cycle-consistency, 3 unpaired samples: mixed/bkg-only/signal-only); event-context fraction regression with buildable jet-level labels | sPHENIX 2510.23717; PUMiNet 2503.02860; ring ML 2507.16686 | HIGH but heavy (generative) / caveat: nb31 ring-rho scalar was flat |
| Time under-exploited — inner regions R0/R1 0.070/0.061 vs 0.036; raw median-centered times in our tokens | SOTA never feeds raw times: sigma_t(E)-weighted reference-time fit from high-confidence cells, then per-cell compatibility pulls Δt/sigma_t as features/soft gates; energy-dependent gate width | CMS HGCAL EPJ Web Conf 320 00046 + 2005.13324; Belle II 2306.04179 (time-since-trigger normalized + fuzzy fractions); Belle II gate ~const/E 2203.11349; DIPz ATL-DAQ-PUB-2026-002 | HIGH, cheap — pure feature/readout change |
| Containment fluctuation floor (nb16 corr 0.977); only 17k labels | Masked-cell self-supervised pretraining on unlabeled stream, then supervised fine-tune; published ~50% resolution gain vs from-scratch at 10^4-event fine-tune scale; MPM+supervised uniquely strong at low labels | Tau FM regression 2503.19165; objectives study 2606.14870; tokenizer-free MPM 2409.12589; OmniLearned 2510.24066 | HIGH — 94 minbias files = free pretraining corpus |
| Residual skew at high E (H8 quantile win just confirmed) | Per-event posterior MODE via conditional flow (mode ≠ median on skewed residuals); mixture pooling of seed ensembles instead of mean-of-means; precision-weighting alone has weak evidence | Flow calibration 2404.18992 (PRD 111 076004); DGME 2306.07235; Bayesian ensembles 2607.06776 | MEDIUM-HIGH; axis ceiling alone ~0.031-0.032 |

## Institutional finding (report to mentors)

An LHCb Zurich group (Souza de Almeida, Eschle, Bartz, Rudolph, Silva Coutinho) presented **GNN reconstruction for the Upgrade II PicoCal with learned pileup subtraction** at CPAN Days Nov 2025 (no arXiv yet; GarNet variant). Same detector, same problem. Ask Felipe to connect BEFORE we duplicate; our per-bin benchmark table + registry is the natural exchange currency.

## Ranked hypothesis queue (cost x leverage)

1. **H9 — nb45: physics-informed time features.** Replace median centering with the CMS-HGCAL recipe: per-window reference time t0 = sigma_t(E)-weighted mean over timed cells (sigma_t ∝ 1/E form from PicoCal design JINST 21 C03006), iterated once with 3-sigma outlier rejection; add per-cell pull (t−t0)/sigma_t(E) and an energy-dependent compatibility flag as token features; optionally a differentiable soft time-gate multiplying the energy gate. Cheap, CPU-safe pipeline change on SubNetFQ. Targets: inner region + E>17 contamination.
2. **H10 — nb46: masked-cell pretraining.** Tokenizer-free MPM objective (2409.12589) on ALL unlabeled minbias windows (no Etrue needed → far more events than 17k), encoder = current d=128 transformer; fine-tune per 2503.19165 protocol with the quant head. First test of the untested pretraining family.
3. **H11a — immediate, free: mixture pooling of nb44 seeds.** Pool per-seed quantile posteriors as a mixture; take mixture median/mode instead of mean-of-quantiles; post-hoc on saved outputs. H11b — flow posterior-mode head (2404.18992) if H11a shows skew is being paid for.
4. **H12 — nb47+: unpaired cycle-consistency subtraction** (sPHENIX recipe) on 9x9 windows: translate minbias window → clean-like window, feed subtracted window to the frozen regressor. Highest complexity, GAN-class training; only if H9/H10 stall before 0.032.
5. **Mentor actions (no compute):** (a) connect with the Zurich PicoCal GNN group; (b) zero-bias embedding request stands (lit_no_percell_labels.md).

## Trajectory estimate (honest)

nb44 stack ~0.043 expected → H9+H10+H11 plausibly reach 0.031-0.033 in the E>17 bins (each lever's published gain discounted for our already-strong baseline; <0.002 = noise rule holds, 2 seeds minimum). 0.030 flat in the top two bins is possible but not promised; below E≈12.5 GeV the design floor stands — per-bin targets remain 0.06/0.045/0.035/0.032/0.030/0.030.

## Guardrails

- One hypothesis per notebook, six-stage intro, anchors fixed before launch (nb44 result becomes the new anchor).
- H9 changes features only — do not co-vary architecture; H10 changes initialization only — same fine-tune config as nb44.
- GPU: SubNet classes safe at 210-900 lock; any GAN/diffusion (H12) gets a fresh crash-class assessment, default CPU.
- No test-split decision-making: config selection on val only; test reported once per notebook.
