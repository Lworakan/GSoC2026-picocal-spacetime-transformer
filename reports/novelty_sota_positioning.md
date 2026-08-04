# Novelty and SOTA positioning of the PicoCal spacetime framework

Date: 2026-07-30. Basis: two verified literature sweeps (lit_no_percell_labels.md 2026-07-29; plan_sota_2026_toward_0p03.md 2026-07-30, 40+ papers, all arXiv IDs fetched) plus the measured campaign record (nb12-nb44).

## The one-sentence claim

We present the first per-cell signal-fraction readout for calorimeter energy regression trained purely from event-level labels, and show on real pileup data that this label-free structure beats every published label-pressure mechanism we re-implemented — establishing supervision structure, not architecture, as the binding constraint under pileup.

## Novelty claims (each backed by a verified search gap)

**N1 — Per-cell SIGNAL-FRACTION regression from event-level labels (core).** Adversarially validated 2026-07-30 (12+ refutation searches): HIGH confidence **when worded as signal-fraction**, MEDIUM if worded loosely as "per-cell weights" — CALICE software compensation (classic 1202.6184; EdgeConv NN 2403.04632, verified from full text) already learns unbounded per-cell energy-correction weights from a purely event-level χ² loss. The two prior-art flanks and the mandatory distinguishing sentence:

> "Unlike CALICE software compensation [1202.6184, 2403.04632], which learns unbounded per-cell energy-correction weights from event-level labels to equalize the hadronic response of a single shower, and unlike Belle II fuzzy clustering [2306.04179], which regresses per-crystal signal fractions but requires per-crystal simulation truth, our gate is a bounded per-cell signal-fraction estimate under pileup that emerges (r = 0.92 with held-out per-cell truth) from event-level energy supervision alone."

Everything else in the family trains on per-cell/per-particle sim truth (PUMML 1707.08600, PUPPIML 1810.07988, ABCNet 2001.05311, PUMA 2107.02779, Belle II 2306.04179/2604.20518, PhyGHT 2602.20475) or operates per-collection (TOTAL 2211.02029, WOTAN ML4PS 2024). Theory frame: learning-from-aggregate-observations (2004.06316). Caveat: do not claim attention-truth emergence itself as novel (documented generically, 2406.04612) — the claim is the bounded fraction semantics + pileup setting + label-free supervision.

**N2 — Systematic falsification of label-pressure mechanisms on real data.** Six mechanisms re-implemented and measured under identical protocol, all failing where the free gate succeeds: unsupervised template decomposition (H2), synthetic-overlay fraction labels (H3, domain gap AUC 0.74), DANN 1912.08001, geometry conditioning + Swin relative-position bias (H4), pairwise Δt attention (H5), clean-teacher feature distillation 2103.07600 (H6), and exact LLP-style aggregate-fraction supervision (H7 — steers the gate to corr 0.971 yet *worsens* resolution 0.0466→0.0491). H7 is the sharpest result: even correct-on-average fraction pressure hurts a free-learned gate. No comparable head-to-head exists in the literature (weak-supervision HEP work is almost entirely classification/anomaly, not regression).

**N3 — Timing negative control.** Same architectures on clean vs pileup data show per-cell time helps only under pileup (0.0665→0.0564 minbias; no gain clean). Belle II reported time as top feature under background (2306.04179) but without the clean-data control; ours isolates the mechanism.

**N4 — DOWNGRADED to "we adopt", not "first" (adversarial check 2026-07-30).** CMS photon regression for H→γγ (1502.02702, 2015) already jointly estimates per-photon energy AND resolution under pileup (semi-parametric Crystal-Ball). Frame our contribution as adopting the quantile-loss + width-binned-recalibration formulation (1912.06046) and measuring its gain on this problem: 0.0466→0.0445 single-model, ensemble 0.0437. Cite 1502.02702 up front.

**N5 — Honest per-bin resolution accounting.** Per-bin targets derived from the design stochastic floor (10%/√E ⊕ 1%), oracle upper bounds (nb29: perfect fractions ⇒ 0.030-0.041 at E>17), and containment-fluctuation floor measurement (nb16, corr 0.977). Rare in ML4HEP papers, which typically report a single aggregate.

## SOTA status — what we can and cannot claim today

CAN claim: (a) SOTA on this detector/dataset — 16-architecture benchmark + BDT/CalibratedSum baselines, all reproducible from the model registry (nb39, 7/7 exact); no external group has published PicoCal numbers (the Zurich GNN work, CPAN Days Nov 2025, has none public). (b) First label-free per-cell readout of its class (N1).

CANNOT claim yet: superiority over sim-truth-supervised methods (Belle II-class) — untestable here by construction; frame as "matches the oracle trend at E>30 within 0.007 without labels". Must also cite and distinguish the Zurich PicoCal GNN effort once public — coordinate via Felipe first.

## Actions that harden the claims (queued)

1. nb44 (running): final stack number for the headline table.
2. H9 time-pull features (CMS-HGCAL recipe): incorporates the 2025-26 SOTA time representation — keeps the framework current rather than merely benchmarked against older baselines.
3. H10 masked-cell pretraining: tests the last untested SOTA family (2503.19165, 2606.14870); either a gain (new lever) or a seventh falsification (strengthens N2).
4. Classical-baseline completeness for reviewers: add SoftKiller-style adaptive per-event floor and PUPPI-style local-shape weighting as no-ML baselines in the comparison table (both label-free, cheap to implement on windows).
5. Ablation table for N1: gate-off vs gate-free vs gate-supervised (already measured across nb30/nb43) consolidated into one figure.

## Venue framing (IEEE-first per Worakan 2026-07-30; dates verified against official pages)

Timing fact: both 2026 IEEE options are gone (NSS/MIC 2026 Granada abstract deadline passed May 12, 2026; IEEE RT 2026 Elba already held May 2026).

| Target | What | Deadline | Note |
|---|---|---|---|
| **Primary: ANIMMA 2027, Prague** | IEEE NPSS-listed, dedicated "ML and AI for Detection and Measurement" track | **Abstracts Nov 12, 2026 — open, actionable** | Only live European IEEE-affiliated deadline |
| Backup: IEEE NSS/MIC 2027, Pasadena | Flagship IEEE NPSS conference | ~May 2027 (expected) | Full manuscript path → IEEE TNS (IF 2.4) |
| Europe flagship: NSS/MIC 2028, Athens | Same series | TBA | If Europe is a hard constraint on the flagship |

Honest caveat for the mentor discussion: the landmark ML-calorimetry papers archive in CSBS/JINST (Belle II GNN → CSBS 2023; Belle II FPGA GNN → JINST 2026), not TNS — NSS/MIC→TNS is the legitimate IEEE-branded path but secondary for this subfield. Recommended combo: ANIMMA 2027 abstract (Nov 12, 2026) + full archival paper to JINST or CSBS; ML4PS workshop remains the fast ML-community option for the N1+N2 "supervision structure beats supervision pressure" story. The negative-result density (H2-H7) is a feature — it is the evidence for N1's causal story, not filler.
