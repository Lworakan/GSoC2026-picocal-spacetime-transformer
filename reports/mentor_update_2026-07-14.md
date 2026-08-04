# Weekly update — 2026-07-14 → 15

Hi Felipe, Carla — here's where I got to this week. I moved everything onto the single-photon + minimum-bias sample like you suggested, and it changed the picture in a useful way. The headline this week: **per-cell timing gives a real, significant improvement** — the "spacetime" part of the project finally doing work.

## Your three suggestions — done

1. **Adaptive binning.** I switched the resolution-vs-energy plots to quantile bins (`np.quantile(E_true, linspace(0,1,n+1))`). The true-energy spectrum really is far from flat, so the fixed bins were noisy at high energy. Bin edges I get on the test set: ~[1.2, 9.1, 14.3, 19.2, 24.6, 32.2, 43.2, 62.6, 100] GeV.

2. **Moved to the minimum-bias sample.** I unpacked the 94-file min-bias set (~90k clusters after the 1–100 GeV cut, 5×5 / kNN-25 cells). All results below are on that sample.

3. **Two regression targets, same architecture.** I trained the identical network on (a) true energy directly and (b) the energy bias / correction on top of the calibrated sum. The only code difference is a `base +` term.

## What we found

**The transformer clearly beats the BDT on the realistic sample.** σ_eff (lower = better):

| model | σ_eff (min-bias) |
|---|---|
| Transformer (best) | **0.065** |
| BDT | 0.125 |
| calibrated sum | 0.184 |
| raw total energy | 0.358 |

So ~1.9× better than the BDT and ~2.8× better than the calibrated sum.

**The margin over the BDT grew with pileup.** On the clean single-photon sample the transformer only edged the BDT (0.036 vs 0.039, ~7%). On min-bias it's ~2×. That matches the intuition you gave — once each cell mixes photon + background energy, the model that can *weight cells* pulls away from anything that just sums them.

**The target ranking changed under pileup, exactly as you warned.** On clean signal the bias target clearly beat the direct target (0.036 vs 0.049). On min-bias the two are basically tied (0.072 vs 0.071) — because the calibrated sum that the bias target anchors on is itself degraded by pileup. What *did* matter on min-bias was the **pooling**: energy-weighted (EFN, IRC-safe) sum-pooling beat plain mean-pooling. So under pileup, "how you aggregate" matters more than "what you regress toward."

## Architecture (the main build this week)

I implemented a Particle-Transformer-style model: standard multi-head attention plus a learned **pairwise-interaction bias U** added before the softmax, `SoftMax(QKᵀ/√d + U)V`, with U built from cell-pair features (Δx, Δy, ΔR, log energy product). On top: energy-weighted EFN pooling + the residual/correction target (DeepSC-style).

Ablation (PairT with U vs the same model with U=0):
- PairT (with U): 0.0665 ± 0.0016
- U = 0: 0.0683 ± 0.0023

So the pairwise matrix **trends better but it's within noise at 3 seeds** (~1σ) — not a decisive win for U on its own.

## Timing — the spacetime transformer (this week's key result)

The space-only architectures all plateau at ~0.065, so I added the variable we'd never used: **per-cell timing** (`cell_times_front/back`). Signal cells are tightly in-time; background is out-of-time or untimed (I mask the ~69% sentinel cells and reference Δt to the seed cell). Ablation ladder on min-bias (3 seeds):

| config | σ_eff |
|---|---|
| space (no timing) | 0.0665 ± 0.0016 |
| **+ per-cell time in tokens** | **0.0564 ± 0.0007** |
| + Δt in the pairwise bias U | 0.0589 ± 0.0010 |

**Timing cuts σ_eff by ~15% (0.0665 → 0.0564), ~6σ significant** — the first thing that clearly beats the space-only floor, and it helps in *every* energy bin (most at low energy, where pileup is worst). Interesting detail: putting time in the tokens beats putting Δt in the attention bias — the model uses raw per-cell time better than a pairwise |Δt| constraint. This is exactly what PicoCal's fast timing is designed for (pileup mitigation), now shown in the reconstruction.

## The honest part

Per adaptive energy bin, the **best bin (22–37 GeV, where pileup barely matters) floors at ~0.047**, and timing only nudged it there (0.051 → 0.047). That's the detector's near-intrinsic resolution in this sample. So there's a **floor at ~0.047 that no architecture breaks** — very-low targets like 0.02 sit ~2.5× below even the best bin, i.e. below the intrinsic/stochastic resolution, not a modeling gap. The result I'd stand behind: **a research-grounded spacetime transformer at σ_eff ≈ 0.056 on min-bias, ~2.2× better than the BDT, with timing giving a validated gain.**

## Next

- Fold the timing result into the write-up / plots; repopulate the multi-seed ablation for the record.
- Feature-importance clean vs pileup, to show *why* we're at the floor (which features the model leans on once background is present).
- Open questions for you: (1) do we have **matched** with/without-pileup events (same photon, background on/off) to measure pileup degradation per-event? (2) was the very-low resolution target meant **per high-energy bin** or a **different sample / timing resolution** than this simulation?

Thanks — happy to walk through any of the plots.
