# PicoCal Physics Primer — for ML people

This document explains every physics concept needed to work on this project,
written for someone who knows transformers but not high-energy physics.
Each section ends with the **ML translation** of the concept.

---

## 1. What an electromagnetic calorimeter (ECAL) does

A photon flying out of an LHC collision is invisible to tracking detectors
(it has no electric charge). The only way to measure it is to **destroy it**:
let it hit a dense block of material and convert all of its energy into a
signal we can count.

What happens inside the material:

1. The photon converts into an electron–positron pair (**pair production**).
2. The electron and positron radiate new photons as they decelerate in the
   material (**bremsstrahlung**, German for "braking radiation").
3. Those photons pair-produce again, and so on — a chain reaction called an
   **electromagnetic shower**. One 50 GeV photon becomes thousands of
   low-energy electrons/positrons/photons.
4. The charged shower particles excite the scintillating material, which emits
   light. Photodetectors count that light, cell by cell.

The amount of light collected in each cell is (approximately) proportional to
the energy deposited there. Summing over cells recovers the photon energy —
that is the entire job of an ECAL.

> **ML translation:** the detector *rasterizes* a single particle into a
> sparse 2D (or 3D) point cloud of `(cell position, energy)` measurements.
> Reconstruction is the inverse problem: point cloud → particle properties
> (energy `E`, impact position `(x, y)`, particle type). Our dataset stores
> exactly this point cloud per photon, and our model is a learned inverse
> function.

---

## 2. Shower shape: the two length scales

Showers have a characteristic, well-understood shape governed by two material
constants. These names appear everywhere in calorimetry papers.

### Radiation length X₀ — longitudinal (depth) scale

The mean distance over which an electron loses all but 1/e (~63%) of its
energy. A shower needs ~20–25 X₀ of material to be fully absorbed. The depth
at which the shower deposits the most energy (**shower maximum**) grows
*logarithmically* with energy:

```
t_max ≈ ln(E/E_c) + const        (t measured in units of X₀)
```

This is the key fact behind PicoCal's front/back segmentation: **a more
energetic photon showers deeper**, so the fraction of energy in the front
segment vs the back segment carries information about the true energy.
We verify this on real data in the exploration notebook (longitudinal-ratio
plot).

### Molière radius R_M — transverse (lateral) scale

The radius of a cylinder containing ~90% of the shower energy. It depends on
the material, not the photon energy. For dense scintillators it is a few cm.
This is why:

- most of the energy lands in **a handful of central cells**, with a steeply
  falling halo around the seed;
- a **5×5 module window** around the seed is enough to contain the shower;
- cell sizes are chosen to be comparable to R_M (smaller cells → better
  position resolution, more channels, more cost).

> **ML translation:** X₀ tells us depth features (front/back energy split,
> timing) encode energy. R_M tells us the token sequence per cluster is
> short and concentrated — attention will mostly need to look at a few
> high-energy tokens plus the spatial tail that distinguishes one shower
> shape from another.

---

## 3. Energy resolution — the metric that defines success

Calorimeter performance is quoted as the relative width of the reconstructed
energy distribution, parameterised as three terms added in quadrature:

```
σ(E)/E = a/√E  ⊕  b  ⊕  c/E        (⊕ means √(x² + y² + z²))
```

- **a/√E — stochastic term.** Shower development is a random branching
  process; the number of secondary particles fluctuates like a Poisson count,
  so the relative fluctuation scales as 1/√E. This is irreducible physics —
  no model can beat it.
- **b — constant term.** Energy-independent imperfections: calibration
  errors, energy leaking out the back of the detector, dead material,
  position-dependent response. At high energy this term *dominates*
  (a/√E shrinks, b does not).
- **c/E — noise term.** Electronics noise contributes a fixed amount of
  energy-equivalent noise, so its *relative* effect dies off as 1/E.

**Why this project exists:** in the evaluation task, six different models on
2D-grid data all converged to b ≈ 5.8% — evidence that the *information*
(not the model) was the bottleneck. Longitudinal leakage is invisible in 2D.
PicoCal's depth segmentation and timing measure exactly the quantities that
feed the constant term. The transformer's job is to convert that extra
information into a smaller b.

> **ML translation:** σ(E)/E is the loss landscape's floor, decomposed by
> physical cause. We always plot resolution *vs* true energy and fit this
> formula, because a single global number hides which term improved. Robust
> width estimators (68th percentile of |ΔE/E|, half-width of the [q16, q84]
> interval) are used instead of plain std because response distributions
> have non-Gaussian tails.

---

## 4. PicoCal specifics

PicoCal is the proposed ECAL for LHCb Upgrade II (≈ Run 5), designed for
~30 MHz collision rate and extreme particle density near the beamline.

Three properties matter for us:

1. **Longitudinal segmentation.** Every cell is read out in two depth
   segments: **front** and **back**. So each cell gives
   `(E_front, E_back)` — and optionally `(t_front, t_back)`. The split point
   is around shower maximum, making `E_front/E_total` a depth probe.

2. **Picosecond timing (~15 ps per cell).** Light reaches the photodetector
   at a time that depends on *where along the depth* the energy was
   deposited and on the particle's flight path. Timing therefore: (a) adds
   an independent handle on shower depth, and (b) at 30 MHz lets you tell
   apart energy from this collision vs the previous/next one (pile-up
   rejection). No current LHCb reconstruction uses timing for energy.

3. **Position-dependent granularity.** Occupancy is highest near the beam
   pipe, so the detector uses 5 module technologies with different cell
   sizes (`cell_pitch`), from highly granular SpaCal W-GAGG modules in the
   centre to coarse Shashlik 1-cell modules at the edges:

   | modType | technology | cells/module |
   |---|---|---|
   | Shashlik 1-cell | sampling, scintillator+Pb | 1 |
   | Shashlik 4-cell | sampling | 4 |
   | Shashlik 9-cell | sampling | 9 |
   | SpaCal 16-cell | Pb–polystyrene fibres | 16 |
   | SpaCal 64-cell | W–GAGG crystal fibres | 64 |

   A fixed 5×5 **module** window therefore contains a *variable number of
   cells* (25 to 1600), depending on where the photon landed.

> **ML translation:** (1) and (2) are extra input feature channels per token
> that 2D models never had. (3) kills CNNs — there is no uniform grid — and
> is the reason the input is a variable-length token set: one token per
> cell, with `(x, y, E_front, E_back, t_front, t_back, pitch, modType)`
> features, padded + masked to a fixed length. The sequence-length
> distribution per region (measured in the notebook) sets the padding
> budget.

---

## 5. Truth vs reconstruction — how the dataset is built

Simulation gives us two parallel views of every event:

- **Truth** (`sig_flux_*` branches): what Geant4 *actually simulated* — the
  photon's true energy, entry point into the ECAL, direction, identity.
  Perfect knowledge, only available in simulation. These are the **labels**.
- **Reconstruction** (`cell_*`, `*_cluster`, `total_energy` branches): what
  a real detector would *measure* — noisy, discretised cell energies and the
  output of the existing clustering algorithm. These are the **inputs** (and
  the classical baseline to beat).

The preprocessing performed by the mentors:

1. **Clustering:** group cells around a local energy maximum (the **seed**)
   into a 5×5 module window.
2. **Truth matching:** several clusters can exist per event; keep only the
   one whose seed is closest to the true photon entry point. The distance
   between the reconstructed cluster position and the truth entry point is
   stored as `sig_dr_matched` — small values mean confident matching, a long
   tail means possible mismatches (we check this distribution and may cut
   on it).

Result: **one tree entry = one matched (cluster, photon) pair** — exactly a
supervised-learning sample.

> **ML translation:** this is a pre-paired `(X, y)` dataset. `sig_dr_matched`
> is a label-quality score — analogous to noisy-label filtering. Any cut we
> apply on it changes the dataset and must be reported.

---

## 6. Branch glossary — physics meaning → ML role

Cluster-level scalars (one value per entry):

| Branch | Physics meaning | Units | ML role |
|---|---|---|---|
| `event` | original event index | – | bookkeeping / dedup checks |
| `sig_flux_eTot` | **true photon energy** | MeV (verify!) | **regression target** |
| `sig_flux_entry_x/y/z` | true ECAL entry point | mm | position target |
| `sig_flux_px/py/pz` | true momentum | MeV/c | derive incidence angle |
| `sig_dxdz_flux`, `sig_dydz_flux` | true direction slopes px/pz, py/pz | – | incidence-angle feature/target |
| `sig_flux_pdgID` | particle type (22 = photon) | – | sanity filter |
| `sig_flux_prod_vertex_x/y/z` | where the photon was created | mm | systematics checks |
| `sig_dr_matched` | distance cluster ↔ truth entry | mm | label-quality score |
| `x_cluster`, `y_cluster` | reconstructed cluster position | mm | classical position baseline |
| `total_energy` (`_front`, `_back`) | summed reconstructed energy | MeV | classical energy baseline |

Cell-level arrays (one value per cell, zero-padded; valid length in `n*`
branches where present):

| Branch | Physics meaning | Units | ML role |
|---|---|---|---|
| `cell_x`, `cell_y` | global cell position | mm | token coordinates |
| `cell_rel_x/y/dr` | position relative to seed | mm | translation-invariant coordinates |
| `cell_energies_front/back` | energy per depth segment | MeV | token features (the new physics!) |
| `energy` | front + back | MeV | token feature / padding mask |
| `cell_times_front/back` | signal time per segment | ns | token features (space-time PE) |
| `cell_pitch` | local cell size | mm | geometry-aware token feature |
| `cell_modType` | module technology | enum | categorical token feature / region split |
| `imodx`, `jmody` | module grid indices | – | region bookkeeping |
| `icell` | cell index within module | – | bookkeeping |

**Known traps**

- *Units:* the dataset spec says `sig_flux_eTot` is in MeV, but the mentor's
  example notebook multiplies it by 1000 with the comment "Convert GeV to
  MeV". One of the two is wrong — the exploration notebook settles it
  empirically by comparing against `total_energy` (which must be the same
  order of magnitude for a well-reconstructed cluster).
- *Padding:* fixed-size arrays padded with zeros. A zero-energy cell at
  (0, 0) could be either padding or a real silent cell near the detector
  centre; use the `n*` length branches where available, heuristics
  otherwise.
- *Timing placeholders:* timing branches may be empty/constant in early
  productions — check before designing features around them.

---

## 7. The reconstruction baselines we must beat

1. **Sum of cells** (`total_energy`): the dumbest estimator — add up all
   reconstructed energy in the window. Biased low (leakage, thresholds) but
   already close; classical reconstruction multiplies it by calibration
   constants.
2. **Cellular Automaton clustering**: LHCb's longest-established algorithm —
   seed finding + iterative neighbour merging with fixed rules. No ML, no
   timing.
3. **Graph Clustering** (current LHCb production): represents cells as graph
   nodes, merges connected components; 65% faster than the cellular
   automaton, same physics. This is the *primary comparison target* of the
   project.

> **ML translation:** these are the "linear probe" and "production system"
> baselines. Every metric table in the project shows them next to the
> transformer, on identical data splits, or the result doesn't count.

---

## 8. Suggested reading order

1. This file, then the exploration notebook
   (`notebooks/01_exploration_matched_clusters.ipynb`) side by side.
2. The dataset spec (`gsoc26_instruction.pdf`) for the authoritative branch
   list.
3. Fabjan & Gianotti, "Calorimetry for particle physics" — sections 2–3
   (shower physics, resolution) for the full derivations behind §2–3 here.
4. The MLPF paper (kernel attention) and ClusTEX paper (graph transformer
   for overlapping showers) for the architecture side.
