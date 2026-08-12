# Two lists: what Felipe asked for, and what it takes to put every bin under 0.05

Numbers are the current best model `SubNetW4CleanAuxExDnQdEma`, 5 seeds, test split, from `scripts/score_regions.py --bin all`.

## A. Everything owed to Felipe / the mentors

### Done

| item | where | note |
|---|---|---|
| Resolution per calorimeter region, best model, min-bias vs clean, one figure per cell pitch | `reports/figures/regions.png`, `regions_region_*.png` | `plot_resolution.py --by-region`, per-bin errors 0.96*sigma/sqrt(n) |
| Inference throughput in clusters/second | `reports/benchmark_inference.csv` | 4677/s single pass, 935/s 5-seed, 117/s 5-seed+TTA (idle CPU; 332/s if measured while training) |
| Throughput across model types | same file | analytic sum 5.6e6/s, BDT 16554/s, transformer 4677/s |
| Confirmation that timing is used, and what it is worth | `meeting_2026-08-13.md` section 5 | `--no-time` ablation: **19% aggregate, 29% at 15mm low-E, 39% at 30mm low-E** |
| Where the model performs worst, and why | `region_deep_dive_2026-08-08.md` | low-energy photons in 15/30mm; noise term containment-dominated, not pileup-dominated |
| Resolution parametrisation per region | `scripts/fit_resolution.py` | a, b, c with errors and chi2/ndf; 15mm fit is poor (chi2/ndf 23) — flagged, not hidden |
| Everything script-based, reproducible, `--help` on every script | `scripts/` | training resumes from checkpoints and skips finished seeds |
| Meeting package | `reports/meeting_2026-08-13.md` | consolidated, with the corrections |

### Pending, blocked on Felipe

| item | why it matters |
|---|---|
| Design a/b/c resolution values per region | lets us say how far we are from the detector limit instead of comparing to an assumed curve |
| GNN and standard-reconstruction outputs as CSVs | `plot_resolution.py` overlays any number of files; needed to position our result against the baseline he has |
| The minimum-bias-only sample | also the only true holdout: 20+ configurations have now been compared on the same test split |
| Guidance on the fit form for the inner regions | the standard three-term form fails at 15mm (chi2/ndf 23) |
| A lab GPU node | one CUDA job at a time locally (several power the laptop off); throughput is worth quoting on standard hardware |

### Pending on our side

| item | state |
|---|---|
| Commit and push everything | waiting on approval; two days of work is local only |
| `signed` gate result (4 seeds: 0,1 local + 2,3 cloud) | training |
| `depth` result (2 seeds) | training |
| `--no-time` ablation of the new best model | training |
| Conference shortlist | `novelty_sota_positioning.md` exists, needs a decision |
| Add arXiv:2603.18172 to the novelty check | closest published work, currently missing |

## B. Every bin against the 0.05 target

Energy bins are terciles **within** each region, so their edges differ per region.

| region | low-E | mid-E | high-E |
|---|---|---|---|
| 15mm | **0.1683** | 0.0451 | 0.0295 |
| 30mm | **0.0948** | 0.0382 | 0.0281 |
| 40mm | **0.0605** | 0.0308 | 0.0232 |
| 60mm | 0.0503 | 0.0300 | 0.0224 |
| 120mm | **0.0551** | 0.0333 | 0.0285 |

**Ten of fifteen bins are already under 0.05**, and 60mm low-E is 0.0503 +- 0.0014, i.e. at the target within its error. Four bins miss:

| bin | now | need | factor | verdict |
|---|---|---|---|---|
| 120mm low-E | 0.0551 | 0.0500 | **1.10** | very likely reachable |
| 40mm low-E | 0.0605 | 0.0500 | **1.21** | likely reachable |
| 30mm low-E | 0.0948 | 0.0500 | **1.90** | at the edge of what the literature supports |
| 15mm low-E | 0.1683 | 0.0500 | **3.37** | not reachable with these inputs |

So "every bin under 0.05" is really a question about **four bins**, and two of them need only 10-20%.

### What closes each gap

**1. Energy-binned calibration (all four bins, expect 5-15%, cheap).** Every low-E bin carries a positive bias: +0.0132 at 15mm, +0.0085 at 30mm, +0.0052 at 120mm, +0.0024 at 60mm, +0.0016 at 40mm. That is classic regression compression toward the mean. Our calibration is binned by predicted interval width only; sigma varies by 3x *inside* a single tercile, so a calibration binned in predicted energy as well should flatten the bias and tighten the width. This alone may finish 120mm and 40mm.

**2. Containment estimation (30mm and 15mm, expect 20-40%).** The noise term is containment-dominated: theory says pileup noise scales as sigma*sqrt(area) (arXiv:0912.4926), which predicts b*0.56 when going 9x9 -> 5x5, but we measured b getting 14-32% *worse*; and nb16 found the residual correlates with containment at 0.977. Consistent with this, the containment block (`--extra`: cluster total energy, window/total ratio, reco shower position) is the only change that has moved the aggregate (-0.0020, 3.5 sigma). Not yet tried, in order of expected value:
   - radial profile of the **full cluster** outside the window (energy in annuli at r > 4, 6, 8 pitch units). We currently pass only the scalar total; the shape of what we are missing is thrown away.
   - distance from the seed to the nearest region boundary and to the detector edge, where containment breaks differently.
   - a dedicated containment head trained on the clean sample, where containment is measurable without pileup, then applied to min-bias.

**3. Spatial covariance weighting (10-15%, has a proof, in training now).** GLS gives w = C^-1 f as the minimum-variance linear estimator. Measured at ring level: sigma_eff 0.1648 -> 0.1484 (-10%), and the optimal weights are **negative on the outer rings** — outer-ring energy is a pileup monitor to be subtracted. A sigmoid gate cannot express a negative weight, so our architecture could not represent the optimal estimator; `--gate signed` (range -0.5 to 1.5) fixes that. Published ceiling for reweighting inside a fixed window is 10-15% (Naylor 1998 MNRAS 296:339; CMS multifit arXiv:2006.14359; matched filter arXiv:2204.13780), consistent with what we measured. Nobody has published spatial cell-cell covariance weighting for calorimeter clusters, so this is also the clearest novelty in the project.

**4. Adaptive per-cell thresholding (10-25%, untried).** SoftKiller (arXiv:1407.0408) removes particles below an event-by-event threshold chosen so the median pileup density vanishes: 20% on jet pT, 30% on mass. Our threshold is a fixed 2.49 MeV for every event and every region. An event- and region-adaptive threshold is a small change to `make_windows`. ML variants exist (arXiv:2509.11291).

**5. Finer energy binning (definitional, and legitimate).** The 15mm low tercile spans 21-95 GeV and its own lower half is 0.238 while its upper half is 0.075. Resolution is conventionally quoted **as a function of energy**, not in three fat bins; the fit says 15mm crosses 0.05 at 53 GeV. Reporting sigma(E) in narrower bins is not moving the goalposts, it is the standard presentation — but it should be agreed with Felipe rather than adopted quietly.

### Honest bottom line

- 120mm low-E and 40mm low-E: **expect to reach 0.05** with items 1-2.
- 30mm low-E: **plausible** at 1.9x, needing items 1+2+3 to stack. This is the interesting fight.
- 15mm low-E: **3.4x is not supported by any evidence we have.** The only published routes to factor-two-class gains are shower attribution (ClusTEX arXiv:2603.18172: 0.80 vs 6.39 GeV on overlapping photons) and adaptive thresholding, and neither claims 3x. Reaching 0.05 there would need either finer energy binning (item 5) or new detector information.
- What is already true and worth stating plainly: **every mid- and high-energy bin in every region is under 0.05**, and timing is responsible for 19-39% of that.
