# Architecture references — ML for calorimeter energy reconstruction (2022–2026)

Literature basis for the PicoCal space-time transformer. Answers three questions the design must justify:
is a from-scratch transformer appropriate, are ~12 per-cell features enough, and what design choices are state of the art.

## Key papers

| paper | year | method | detector | input features | relevance |
|---|---|---|---|---|---|
| **ClusTEX** — Maidannyk, Couderc, Malcles, Sahin, "Reconstruction of overlapping electromagnetic showers in calorimeters using Transformers" (arXiv:2603.18172) | **2026** | single-step **graph transformer**; positional encoding separates **local** (within-graph) from **global** (detector) coordinates | ECAL-inspired | per-hit physical | most on-point: transformers for EM shower reco; best energy resolution + less shower splitting vs two-step and standard clustering; keeps di-photon (boosted pi0->gamma gamma) mass |
| **CLAS12** — (arXiv:2503.11277) | 2025 | **GravNet + Transformer + Object Condensation**; GravNet learns a detector-topology embedding used as the transformer's positional encoding | CLAS12 calorimeter | **17 per-strip**: 6 endpoint coords, energy, timing, 9 one-hot layer | embeds 150 hits x 17 -> **64-dim** before the encoder |
| **Object Condensation + GravNet** — Kieseler et al., EPJC (arXiv:2106.01832) | 2022 | end-to-end GNN doing clustering + classification + energy/position regression jointly | CMS HGCal | per-hit | canonical high-granularity reference |
| **CNN+GNN, CMS ECAL** | 2023 | CNN+GNN energy+position regression | CMS ECAL | crystal grid | improves energy AND position vs conventional reco |
| **Particle Transformer (ParT)** — Qu et al. | 2022 | transformer with **pairwise interaction bias** added pre-softmax in attention | jets | particle-level | interaction bias lets a transformer beat the GNN SOTA (ParticleNet) |

## What this means for our design

1. **~12 features is normal, not too few.** CLAS12 uses exactly 17 raw physical per-node features; the field routinely trains transformers from scratch on ~10–17. This directly refutes the "transformers need hundreds of features" worry — those hundreds are the *learned embedding width*, not raw inputs.

2. **Project raw features into a learned embedding.** SOTA models map the ~12–17 raw features into a modest learned embedding (~64-dim) before attention. We already do this (`nn.Linear(12 -> d_model)`); keeping/raising `d_model` to ~64 is aligned.

3. **Geometry-aware positional encoding is the key upgrade.** ClusTEX separates local (within-cluster) from global (detector) coordinates; CLAS12 learns the detector topology via GravNet as the PE. We currently use only local `rel/pitch` coordinates — adding a global-position channel (or a GravNet-learned PE) is the principled next step.

4. **The transformer's edge is in overlapping / boosted showers.** ClusTEX's advantage over clustering appears specifically for overlapping EM showers and boosted pi0->gamma gamma — matching our own finding that on clean single-photon 3x3 the problem is near-solved by sum-of-cells (see notebook 01 feature significance). The research question lives in the hard regime.

5. **Optional: ParT-style interaction bias** — inject pairwise cell relations as an additive attention bias, the design that let a transformer beat the GNN SOTA.

## Caveats
- The tabular-ML "do transformers beat GBDTs on small low-dim feature sets" claims could not be verified (all rate-limited). Our own result (BDT ties the transformer on R3) is consistent with the general finding that gradient-boosted trees are strong on low-dimensional tabular problems.
- ClusTEX and ParT are shower-splitting / jet-adjacent, not pure single-photon energy regression; borrow the architecture, not the exact benchmark numbers.
