# Representation and techniques for the PicoCal space-time transformer

*Week 2 research note. Goal: understand the techniques we plan to use, why they fit this data,
and what it takes for the model to be real in the LHCb PicoCal era (~2030+), not just a paper.
Written for myself — I know ML and transformers reasonably, but I am new to physics data, so I
keep my image/PCA intuition as the bridge.*

Every claim with an arXiv number below was checked against arXiv this week. The foundational ML
papers (Vaswani, Deep Sets, etc.) I cite from standard knowledge; they are universally known.

---

## 1. The right mental model: a calorimeter shower is a point cloud with physics

My instinct from images was "grid of pixels → CNN → PCA on features." That instinct is half
right and half wrong, and seeing exactly where is the whole point.

- **Wrong part:** our cluster is *not* a dense grid. It is a **variable-length, unordered set**
  of cells. From notebook 01: about half the window cells are silent, informative cells per
  cluster run p50 ≈ 71, p99 ≈ 200. A CNN on a fixed grid would waste most of its work on empty
  cells and would force a fake ordering on something that has none.
- **Right part:** each cell still has *coordinates* (x, y, depth, time) and a *value* (energy),
  exactly like a pixel has position and intensity. So the image intuition transfers — but the
  coordinates are **continuous**, not integer indices, and the object is a **set**, not a grid.

The field has a name for this: a **particle cloud** / point cloud. Treating jets and showers as
point clouds is the dominant modern approach — ParticleNet (Qu & Gouskos, arXiv:1902.08570) and
Particle Transformer (Qu, Li, Qian, arXiv:2202.03772) are the canonical examples. So the project's
"transformer over cells" is squarely on the main road, not an exotic detour.

**ML translation of what we already built:** the 6 features in `picocal.data.dataset` —
`cell_x`, `cell_y`, `log_energy`, `front_fraction`, `t_centered`, `t_valid` — are exactly the
"position + value + depth + time" of each point. `front_fraction` is the depth coordinate (the
"z" the model expects); `t_centered`/`t_valid` are the time handle. The padding mask is what lets
a set of varying size go through a fixed-shape tensor. Nothing here is wasted; it is the standard
point-cloud input.

---

## 2. Two questions every architecture for this data must answer

### (a) How to encode *continuous* coordinates so attention can use them

A plain MLP fed raw (x, y, z, t) is biased toward smooth, low-frequency functions and blurs fine
spatial structure. The standard cure is **Fourier features**: pass each coordinate through a bank
of sinusoids at geometrically spaced frequencies before the network (Tancik et al.,
arXiv:2006.10739; this is also the NeRF positional encoding). The transformer's original
sinusoidal positional encoding (Vaswani et al., 2017) is the same trick, just indexed by token
position instead of a real coordinate — so for us we swap the integer index for the real (x,y,z,t).

A relative variant, **RoPE / rotary embeddings** (Su et al., RoFormer, arXiv:2104.09864), makes
the attention score depend only on coordinate *differences*. That is attractive here because
shower physics is roughly translation-invariant in (x, y, t): a shower looks the same wherever it
lands. A multi-axis Fourier/RoPE encoding of (x, y, depth, t), with a **separate frequency scale
per axis** so picosecond-time and centimetre-space each get the right bandwidth, *is* the
"space-time positional encoding" the proposal promises.

*Image/PCA analogy:* it is like handing the network a Fourier basis of the image instead of raw
pixel indices, so a simple linear readout can represent sharp edges. Here the "edges" are the
sharp energy and time structure of the shower core.

### (b) How to be permutation-invariant (order of cells is meaningless)

The model must give the same answer if I shuffle the cell list. The ladder of standard tools:

- **Deep Sets** (Zaheer et al., arXiv:1703.06114): any permutation-invariant function is
  `ρ(Σ φ(xᵢ))` — shared per-cell MLP, then sum/pool. The minimal, robust baseline.
- **Energy Flow Networks** (Komiske et al., arXiv:1810.05165): Deep Sets with an energy weighting
  that respects the physics — the HEP-native version.
- **Set Transformer** (Lee et al., arXiv:1810.00825): self-attention over the set, with
  *inducing points* to avoid the full N² cost. Learns pairwise interactions a sum-pool cannot.
- **Perceiver** (Jaegle et al., arXiv:2103.03206): cross-attend the input set into a small fixed
  latent array, then process the latents — decouples compute from the number of cells.

A transformer is naturally permutation-equivariant already (attention has no built-in order), so
our space-time PE *adds* the geometry back in a controlled way. That is the whole design: order
should not matter, but *position* should — and the PE is how position re-enters.

---

## 3. The architecture landscape, and where our model sits

The production baseline the proposal names ("graph clustering") is a real, specific family:

- **GravNet / GarNet** (Qasim, Kieseler et al., arXiv:1902.07987): distance-weighted dynamic
  graph networks built precisely for irregular-geometry calorimeters. They learn a latent space,
  connect each cell to its neighbours in that space, and aggregate — handling sparsity and
  arbitrary geometry natively.
- **Object Condensation** (Kieseler, arXiv:2002.03605): a one-shot loss that clusters cells into
  objects and reads out each object's properties without a fixed grid or a known number of
  objects. Used end-to-end for the CMS HGCal at the HL-LHC (arXiv:2106.01832), the most
  pile-up-heavy calorimeter problem there is.
- **MLPF** (Pata, Duarte, Vlimant, Pierini, Spiropulu, arXiv:2101.08578): machine-learned
  particle flow as a graph network — the broader "reconstruct the whole event with a GNN" program.

**Where the space-time transformer fits — and a key recent paper:** a 2025 CLAS12 work
(arXiv:2503.11277) does almost exactly our idea — it uses a GravNet to learn a latent
representation that *serves as the positional encoding* for the calorimeter hits, then feeds them
to a Transformer encoder. That is strong evidence the "geometry-aware positional encoding +
transformer" direction is live and credible, and it suggests a concrete hybrid: GravNet-style or
Fourier PE for geometry, attention for the global combination. The novelty hook we own is
**timing as a coordinate** — adding the ~15 ps cell time as a fourth axis of the positional
encoding. I did not find prior calorimeter work that treats picosecond time as a spatial
coordinate in attention, which is encouraging for the project's originality (worth confirming
with the mentors and a proper literature pass before claiming it).

---

## 4. My PCA/feature-extraction instinct, answered honestly

I asked: can we make the features more powerful with abstraction, the way I use feature
extraction + PCA on small image data? The grounded answer:

- **A linear autoencoder *is* PCA**; a nonlinear one generalises it. So my instinct maps onto
  autoencoders and self-supervised representation learning.
- **But we have ~200k labelled showers.** That is plenty to train a modest set-transformer
  *supervised, end to end*. Self-supervised / contrastive pretraining (SimCLR-style) mainly pays
  off when labels are scarce or there is a *much larger* unlabelled pool to exploit, and it needs
  physically meaningful augmentations (rotations, translations, energy smearing, dropping soft
  cells) that are real work to design. Pretraining helps in HEP at scale — Particle Transformer
  pretrained on 100M jets, OmniLearn (Mikuni & Nachman, arXiv:2404.16091) as a jet foundation
  model — but that is a different regime from our 200k.
- **So the plan:** train supervised first. Keep PCA/UMAP for *visualising and sanity-checking*
  the learned embedding (exactly my comfort zone — just as a diagnostic, not the model). Treat a
  contrastive objective as an optional auxiliary loss or a robustness experiment, not the spine.
- **Physics features still matter.** Hand-engineered shower quantities — log-energy, the
  energy-weighted centroid (the W₀ centroid from notebook 01), lateral/longitudinal shower
  moments, the front/back depth fraction — are low-dimensional, interpretable, and strong. The
  best current practice is not "learned vs engineered" but **both**: feed the physics features as
  extra inputs to the learned model. We rarely lose and often gain data efficiency. There are
  also fully equivariant nets (LorentzNet, PELICAN) that bake in symmetry, but full Lorentz
  equivariance is aimed at 4-momenta/jets; for a fixed-frame calorimeter the cheaper route is
  spatial-symmetry augmentations.

---

## 5. The reality check: what "usable in ~2030" actually demands

This is the part that turns a nice result into a deployable one. The PicoCal and the LHCb
real-time system set hard limits.

- **PicoCal** (LHCb Upgrade II Framework TDR, CERN-LHCC-2021-012, CDS 2776420; PicoCal R&D,
  arXiv:2504.03088; SPIDER readout ASIC, arXiv:2512.17355): the new ECAL for Upgrade II (Run 5,
  install in LS4, ~2035+ data-taking, peak luminosity 1.5×10³⁴). It adds **~15 ps cell timing
  above 5 GeV** (≈20 ps per cluster), **two-layer front/back longitudinal segmentation**, and
  **mixed module technologies by region** (SpaCal with GAGG/tungsten in the hot inner region,
  SpaCal plastic/lead in the middle, Shashlik outer). Occupancy reaches ~30%. Timing exists
  specifically to **reject out-of-time pile-up** and assign energy to the right collision — which
  is exactly the structured-background problem I measured in notebook 02 (minimum-bias overlay
  adding energy to cells).
- **LHCb real-time flow:** 40 MHz crossings, a **full software trigger with no hardware L0**.
  **HLT1 runs on GPUs** (Allen, arXiv:1912.09161) — the first complete GPU trigger in HEP,
  ~hundreds of GPUs, microsecond-scale per event, 30 MHz → ~1 MHz. **HLT2 on CPU** does
  offline-quality reconstruction at ~1 MHz. Throughput is the currency, not single-event latency.
- **How ML is actually deployed:** small models with bounded latency. hls4ml on FPGAs for the
  L1-style sub-microsecond regime (Duarte et al., arXiv:1804.06913; and notably **transformers on
  FPGAs via hls4ml**, arXiv:2409.05207, plus radiation-hard FPGA work arXiv:2602.15751); ONNX
  Runtime and ROOT SOFIE for portable C++ inference in HLT2/offline; LHCb's long-standing
  ghost-probability NN in tracking as the canonical "ML already in production" example. Production
  models are **small and predictable**, not transformer-scale-for-its-own-sake.

**What this forces on our design (the honest bar):**

1. **Exploit what PicoCal adds — depth and ~15 ps timing — to reject out-of-time pile-up.** That
   physics payoff is what would justify the cost. A transformer that ignores timing has no
   deployment argument over GravNet.
2. **Stay small and bounded.** Plan for a model that can be quantised/distilled and exported to
   ONNX/SOFIE, with predictable latency. Linear/kernel attention (Performer, arXiv:2009.14794;
   Linear Transformers, arXiv:2006.16236) only helps at large N (thousands of cells, i.e.
   whole-event); for a single cluster of ~100 cells, **exact softmax attention with FlashAttention
   (arXiv:2205.14135) is both faster and more accurate** — so I should not over-claim the "O(N)
   kernel attention" headline until I benchmark it. The honest O(N) route for sets is inducing
   points (Set Transformer) or Perceiver latents, which also add a useful bottleneck.
3. **Determinism, memory bound, framework integration, calibration stability.** A loose PyTorch
   script is not deployable; it has to live in Gaudi/Allen via a deployable runtime and survive
   pile-up and detector aging. Accuracy is necessary, not sufficient.

---

## 6. Concrete plan for our project, ranked by effort vs payoff

1. **Per-cell Fourier/positional encoding of (x, y, depth, t) + log-energy embedding** — low
   effort, high payoff. This is the heart of the "space-time PE" and the project's novelty (time
   as a coordinate). Build it as the token embedding on top of our existing 6 features.
2. **Deep Sets / EFN baseline before any transformer** — trivial effort, essential. It is the
   number attention must beat; if attention does not beat sum-pooling, we learn that early.
3. **Set Transformer (inducing points) or Perceiver latents** — medium effort, high payoff:
   permutation invariance + learned interactions at sub-quadratic cost, the honest "O(N)".
4. **Exact softmax + FlashAttention over linear/kernel attention** at our cluster size; revisit
   linear attention only for whole-event processing.
5. **Inject physics shower features (W₀ centroid, depth fraction, log-energy, moments) as
   auxiliary inputs** — low effort, reliable data-efficiency gain.
6. **A timing ablation as a first-class experiment** (use_timing on/off) — this is both the
   scientific story and the deployment justification. The data path already carries `t_valid`, so
   the ablation is honest.
7. **Supervised end-to-end first; PCA/UMAP only to visualise the embedding; contrastive SSL only
   as an optional robustness experiment** — given 200k labels.

The thread tying it together: the model should earn its place by using **depth + timing to beat
the GravNet baseline on pile-up**, in a form small and deterministic enough to deploy. Novelty is
the timing coordinate; the deployment argument is out-of-time pile-up rejection.

---

## References (arXiv IDs verified this week unless noted)

**Our data as a point cloud / HEP architectures**
- ParticleNet — Qu & Gouskos — arXiv:1902.08570
- Particle Transformer (ParT), JetClass 100M — Qu, Li, Qian — arXiv:2202.03772
- Energy/Particle Flow Networks — Komiske, Metodiev, Thaler — arXiv:1810.05165
- GravNet / GarNet — Qasim, Kieseler, Iiyama, Pierini — arXiv:1902.07987
- Object Condensation — Kieseler — arXiv:2002.03605
- HGCal end-to-end object condensation — arXiv:2106.01832
- MLPF — Pata, Duarte, Vlimant, Pierini, Spiropulu — arXiv:2101.08578
- OmniLearn (jet foundation model) — Mikuni & Nachman — arXiv:2404.16091
- GravNet-as-PE + Transformer on calorimeter (CLAS12) — arXiv:2503.11277

**Encodings, sets, attention (ML foundations; IDs from standard knowledge, the searched ones verified)**
- Attention Is All You Need — Vaswani et al., 2017 — arXiv:1706.03762
- Fourier Features — Tancik et al. — arXiv:2006.10739 (verified)
- RoPE / RoFormer — Su et al. — arXiv:2104.09864
- Deep Sets — Zaheer et al. — arXiv:1703.06114
- Set Transformer — Lee et al. — arXiv:1810.00825
- Perceiver — Jaegle et al. — arXiv:2103.03206
- Linear Transformers — Katharopoulos et al. — arXiv:2006.16236
- Performer / FAVOR+ — Choromanski et al. — arXiv:2009.14794 (verified)
- FlashAttention — Dao et al. — arXiv:2205.14135
- Tabular: trees still beat DL on small data — Grinsztajn et al. — arXiv:2207.08815

**Deployment / LHCb real-time**
- LHCb Upgrade II Framework TDR — CERN-LHCC-2021-012 — CDS 2776420 (verified)
- PicoCal R&D — arXiv:2504.03088; SPIDER readout ASIC — arXiv:2512.17355 (verified)
- Allen GPU HLT1 — arXiv:1912.09161 (verified)
- hls4ml FPGA inference — Duarte et al. — arXiv:1804.06913 (verified)
- Transformers on FPGAs with hls4ml — arXiv:2409.05207 (verified)
- Radiation-hard FPGA ML with hls4ml — arXiv:2602.15751 (verified)

*Caveat: this is my own reading to orient the project, not a peer-reviewed survey. Before any of
this goes into the proposal or a mentor claim — especially the "timing as a coordinate is novel"
point — I should do a proper literature pass and confirm with Felipe and Carla.*
