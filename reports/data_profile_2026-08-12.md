# Understanding the data, from every angle (2026-08-12)

Measured from the event caches (`.scratch/cache/*.pkl`), i.e. after the selection the pipeline actually applies: production vertex z < 100, 1 <= E_true <= 100 GeV, at least 3 cells, at least half the cells passing the grid-alignment test with the seed passing, per-cell threshold 2.49 MeV, and the 9x9 window.

## 1. Inventory

| | min-bias | clean |
|---|---|---|
| clusters after selection | 72554 | 30303 |
| E_true [GeV] p5 / median / p95 | 5.6 / 24.1 / 79.7 | 5.5 / 24.2 / 79.4 |
| cells per **cluster** median / p95 / max | 81 / 360 / 528 | 81 / 384 / 528 |

The energy spectra of the two samples match, so min-bias vs clean comparisons are not confounded by the target distribution. One cluster per event in both.

## 2. Composition per region — and a confound worth naming

min-bias:

| pitch | clusters | share | median E | E p5-p95 | cells in window | seed fraction | window/cluster energy |
|---|---|---|---|---|---|---|---|
| 15 mm | 8367 | 11.5% | 59.8 | 20.2-95.9 | 63 | **0.324** | **0.273** |
| 30 mm | 13981 | 19.3% | 45.1 | 11.8-88.3 | 69 | 0.292 | 0.621 |
| 40 mm | 20876 | 28.8% | 26.1 | 6.9-57.3 | 64 | 0.338 | 0.886 |
| 60 mm | 24943 | 34.4% | 16.8 | 4.4-34.6 | 34 | 0.507 | 1.000 |
| 120 mm | 4387 | 6.0% | 12.5 | 3.2-21.6 | 9 | 0.743 | 1.000 |

clean:

| pitch | clusters | median E | cells in window | seed fraction | window/cluster energy |
|---|---|---|---|---|---|
| 15 mm | 3973 | 55.8 | 54 | **0.650** | **0.992** |
| 30 mm | 5744 | 43.6 | 44 | 0.699 | 0.997 |
| 40 mm | 8550 | 26.0 | 45 | 0.632 | 0.998 |
| 60 mm | 10280 | 16.8 | 24 | 0.708 | 1.000 |
| 120 mm | 1756 | 12.7 | 8 | 0.857 | 1.000 |

**The confound**: median energy falls by a factor 5 from the innermost to the outermost region (59.8 -> 12.5 GeV), because the geometry puts high-energy photons at small radius. So the "low-energy tercile" of the 15 mm region spans roughly 20-47 GeV, which is *above* the entire spectrum of the 120 mm region. Comparing terciles across regions compares different absolute energies. It cuts the right way for us: at the same absolute energy the inner regions are still worse, so their deficit is physical and not a binning artefact — but any per-region table should say which energies it refers to.

## 3. How much of the cluster we actually see

`window/cluster energy` is the striking column. In clean data the 9x9 window captures **99.2-100%** of the cluster in every region. Under pileup the same window captures only **27% at 15 mm** and 62% at 30 mm, while 60 and 120 mm remain at 100%.

Two consequences, and the first one corrects an earlier conclusion of ours:

1. **Photon containment is not the problem.** Since clean clusters are 99% contained in the window, the photon's own energy barely leaks out at W=4. The earlier "containment-dominated" reading — which came from a fit whose a and c were pinned on their bounds — should not be trusted. What is low in min-bias is the fraction of the *cluster* we see, and the cluster grows because pileup lights up extra cells (63 cells in window at 15 mm versus 54 in clean; 34 versus 24 at 60 mm).
2. **`total_energy` worked because it is a pileup-density measure, not a containment measure.** That reframes the `--extra` win: the model gained an estimate of how much pileup surrounds the window, not of how much photon it was missing.

## 4. How badly pileup contaminates the signal

- Photon energy as a fraction of cluster energy, min-bias: median **0.539**, p5 **0.101**, p95 0.956. So typically about half the cluster is not our photon, and in the worst 5% of events the photon is a tenth of it. In clean the same ratio is 1.049 (the 5% excess is calorimeter response, not contamination).
- Seed fraction inside the window collapses from 0.65-0.86 (clean) to 0.29-0.34 (min-bias) in the three inner regions. The window energy stops being dominated by the shower core.

That is the quantitative statement of the problem: at 15 mm, two thirds of what is in our window is not the photon.

## 5. Timing, measured on our own data

Per-cell time resolution against cell energy (clean, energy-weighted reference from the top-decile cells):

| cell energy | sigma_t clean | sigma_t min-bias |
|---|---|---|
| 2-10 MeV | 0.756 ns | 1.275 ns |
| 30-100 MeV | 0.435 | 0.804 |
| 100-300 MeV | 0.263 | 0.594 |
| 300-1000 MeV | 0.156 | 0.401 |
| > 3000 MeV | **0.038** | 0.132 |

A factor 20 across the energy range, so a cell's time is only meaningful when the cell is energetic; and every bin degrades 1.7-2x under pileup. In-time pileup from vertex spread differs by only ~0.2 ns, below the resolution of the low-energy cells where pileup lives. This is why an explicit time gate and a resolution-weighted time pull both added nothing — while the plain timing features are nevertheless worth 20% overall and 24-39% in the weak bins (`--no-time` ablation).

## 6. Longitudinal structure

`log((E_front+1)/(E_back+1))` per cell, by ring:

| ring | clean | min-bias | shift |
|---|---|---|---|
| 0 (seed) | -0.556 | -0.524 | +0.03 |
| 1 | -1.943 | -1.295 | +0.65 |
| 2 | -1.996 | -0.923 | +1.07 |
| 3 | -1.879 | -0.620 | +1.26 |
| 4 | -1.756 | -0.444 | +1.31 |

Genuine shower cells are back-heavy and pileup cells are not, a ~0.6 sigma per-cell separation. The seed agrees between samples, confirming the seed is signal-dominated in both. As a trained feature this gave nothing (`--depth`: +0.0005 aggregate), consistent with 0.6 sigma translating to rho^2 = 0.08.

## 7. Spatial correlation of the noise

Residual covariance across rings (min-bias, 15+30 mm), correlation matrix:

```
        r0     r1     r2     r3     r4
r0    1.00  -0.14   0.10   0.07   0.04
r1   -0.14   1.00   0.41   0.27   0.22
r2    0.10   0.41   1.00   0.60   0.45
r3    0.07   0.27   0.60   1.00   0.63
r4    0.04   0.22   0.45   0.63   1.00
```

Outer rings are strongly correlated with each other (0.45-0.63) and with ring 1 (0.22-0.41): pileup is a spatially coherent perturbation, not independent cell noise. A GLS estimator w = C^-1 f exploits this and beats the profile-weighted sum, 0.1648 -> 0.1484 at ring level, with **negative** weights on the outer rings. Learned end-to-end (`--gate signed`) it gave nothing, presumably because the network's nonlinear, pattern-conditioned gate already exceeds any fixed linear estimator.

## 8. Truth information available

Used as targets: `sig_flux_eTot` (energy), and `sig_flux_entry_x/y`, `sig_flux_timing` (auxiliary, tried, no gain). Available and unused: `sig_flux_entry_z`, `sig_flux_prod_vertex_x/y`, `sig_flux_pdgID`, `sig_dr_matched`, `sig_dxdz_flux`, `sig_dydz_flux` (incidence angles — plausible auxiliary target, since a slanted shower leaks differently). None of these may ever be an input.

## 9. Data-quality notes

- Timing is stored as 0 or non-finite when a cell has no valid measurement; the pipeline maps that to a validity flag. Fraction of valid front times rises with cell energy.
- The grid-alignment test rejects whole events when fewer than half the cells sit on the seed's lattice (mixed-pitch or misaligned clusters). This silently removes the hardest boundary cases — worth quantifying before publishing, since it is a selection the reader cannot see.
- Cell energies are MeV, truth energies GeV. Everything in the pipeline respects this, but any new feature must.
- Splits come from `run_experiments.split()` on the event index and are therefore event-disjoint; the clean sample is used only as auxiliary *training* data, never in validation or test.

## 10. What this suggests next, and why it is different from what failed

The cells **outside** the 9x9 window are a pileup sample from the same event with essentially no photon in them — clean data says the photon is 99% contained inside the window, so whatever lies outside is pileup almost by construction. At 15 mm that is 73% of the cluster energy, measured event by event.

That is a far better pileup estimator than the ring-based rho we tried and which failed: ring 3-4 sit *inside* the window and still carry photon energy (clean seed fraction is 0.65, so the rings hold about a third of the shower), so subtracting a density estimated there removes signal along with pileup. An outside-window estimate does not have that defect.

Concretely: rho_event = (total_energy - window_energy) / (cells outside x cell area), then a per-cell corrected energy E_i - rho_event * area_i, given to the model as features alongside the raw energies. This is the only untried lever that is both physics-clean and aimed at the dominant error source, and it follows directly from the profile above rather than from analogy.
