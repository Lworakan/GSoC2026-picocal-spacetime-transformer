# Complete lever matrix — every way to work the data, with evidence status

Date: 2026-07-31. Purpose: before committing GPU time to the timing program, enumerate EVERY lever across all six layers of the problem, each with its measured status on THIS data or its literature support. Nothing gets trained unless it has (a) a measured error mode it targets, (b) a precedent or theory, (c) a pre-registered proof criterion. Anchors: nb44 stack (finishing), nb43 quant ens 0.0437; per-bin targets 0.06/0.045/0.035/0.032/0.030/0.030; 0.02 physically possible only E≳33 GeV.

Legend: ✅ tested-wins (in stack) · ❌ tested-falsified (closed) · 🔬 queued with evidence · 🧭 advisor-gated · 💤 backup only.

## A. Data levers

| Lever | Status | Evidence |
|---|---|---|
| Clean-as-auxiliary training | ✅ nb34 | best singles; gain in low-E bins |
| Synthetic overlay labels | ❌ H3 | domain gap AUC 0.74; supervision hurt |
| More minbias statistics | ❌ | Pb_modules = same events re-produced |
| Unlabeled minbias for pretraining | 🔬 H10 | MPM+supervised strongest at low labels (2606.14870); ~50% gain at 10^4 fine-tune (2503.19165) |
| Minbias-only sample, positional matching | 🧭 incoming from Felipe | real-pileup overlay → per-cell labels with REAL pileup; fixes H3's domain gap; recipe from Felipe himself (cluster position lookup) |
| Zero-bias embedding at digitization | 🧭 mentor ask (standing) | ATLAS 2102.09495 routine practice |
| Selection changes (window base, energy cut) | fixed by mentor decision | kNN25→W4, 1-100 GeV |

## B. Feature levers

| Lever | Status | Evidence |
|---|---|---|
| Raw median-centered times | current | helps only under pileup (nb16: 0.0665→0.0564) |
| σ_t(E)-weighted t0 + per-cell pulls | 🔬 H9 = nb45 READY | CMS HGCAL recipe (EPJ Web Conf 320 00046); σ_t measured from OUR clean data (1.3 MeV·ns/E ⊕ 0.26 ns, refit on full) |
| Front-back time consistency |t_f−t_b| | 🔬 H9b add-on | two-layer shower coherence vs pileup overlap; zero-cost feature |
| Time-cluster globals (n clusters, main-cluster energy fraction, Δt) | 🔬 H9c add-on | per-event contamination meter; extends H8's proven per-event-context win |
| σ_t per region × layer refinement | 🔬 nb45 v2 | sharpens pulls exactly in R0/R1 |
| Per-event reference time vs bunch clock | 🧭 ASK FELIPE | improves t0 for free if the branch exists |
| Ring-rho context scalar | ❌ nb31 | window already encodes local pileup |
| Geometry conditioning (mm coords, beam distance, region weights) | ❌ H4/nb38 | inner-region deficit is physical |

## C. Architecture levers

| Lever | Status | Evidence |
|---|---|---|
| Plain transformer d=128 L=3 | ✅ | capacity scanned (nb22); GPU-safe class |
| Pairwise Δt attention / pair-MLP | ❌ H5 + GPU crash class | flat AND hard-crashes laptop |
| GravNet / GNN multitask | ❌ nb25 | |
| Swin relative-position bias | ❌ H4 | |
| Masked-cell pretrained init (foundation-model family) | 🔬 H10 = nb46-candidate | ONLY untested architecture-adjacent family; tokenizer-free MPM 2409.12589 |
| PhyGHT-style gate variants | 💤 mine for design only | sim-supervised at source (2602.20475) |

## D. Objective / supervision levers

| Lever | Status | Evidence |
|---|---|---|
| Residual/bias target | ✅ nb12 | |
| Quantile pinball + width-binned recalibration | ✅ H8/nb43 | 0.0466→0.0445; the one 2026-session win |
| Aggregate-fraction supervision (LLP) | ❌ H7 | steers gate (corr 0.971) yet hurts — supervision-pressure result #3 |
| DANN / distillation / overlay-frac | ❌ nb30/nb41/nb28 | |
| Two-sample OT loss (TOTAL/WOTAN) | 💤 open, behind H9/H10 | family compatible but line dormant since 2024 |
| Flow posterior-mode estimator | 🔬 H11b conditional | 2404.18992; fire only if skew diagnostic shows median leaves width |
| Unpaired cycle-consistency subtraction | 💤 last resort | sPHENIX 2510.23717; GAN-class cost, new GPU crash-class assessment needed |

## E. Readout / structure levers

| Lever | Status | Evidence |
|---|---|---|
| Subtract-then-calibrate free gate | ✅ core | N1 novelty claim |
| Soft time-gate on the energy gate | 🔬 H9 = nb45 `pullgate` | learnable (κ, α) on pulls |
| Window scan (no time gate) | done | W=4 optimum; W>4 adds more pileup than tail |
| **Timing-gated wide window W=6/8** | 🧭 H12 — ASK FELIPE FIRST | the only physics route to 0.02 at E>33: recover containment, let time reject the added pileup; needs advisor sanity-check on PicoCal timing semantics |
| High-E specialist | tested nb35 | neutral; superseded by width-binned recalibration |

## F. Ensemble / inference levers

| Lever | Status | Evidence |
|---|---|---|
| Seed ensembling + D4 TTA | ✅ nb34/35/44 | |
| Mixture pooling of quantile posteriors | 🔬 H11a — free, post-hoc | DGME 2306.07235; do immediately after nb44 |
| Precision (1/σ²) weighting | fold into H11a check | published evidence weak — verify, don't assume |
| Isotonic calibration / cross-window mega-ens | ❌ nb33 | overfits val / loses to W4 |

## Execution queue (all gates respected)

1. nb44 finishes → H11a mixture pooling (free) → registry + nb20 refresh.
2. **nb45 (H9 pulls + time-gate)** — approved recipe, launches next on GPU.
3. **Advisor round in parallel** — the three timing questions + ideal-curve formula (draft in chat 2026-07-31). H12 (wide window) and the minbias-only matching study design wait for his answers.
4. H10 masked pretraining after nb45 verdict.
5. H11b / OT / cycle-consistency only on explicit trigger conditions above.

## Verification discipline (unchanged, applies to every row)

Pre-registered win criterion before launch; 2 seeds minimum; <0.002 = noise; selection on val only; mechanism diagnostic in every notebook (like fbar-corr in nb43, kt/at in nb45); negative control where meaningful; honest verdict cell whichever way it lands.
