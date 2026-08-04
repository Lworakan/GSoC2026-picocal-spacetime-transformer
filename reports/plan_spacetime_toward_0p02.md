# Plan — pushing toward σ_eff 0.02 with the *spacetime* transformer (2026-07-14)

## Where we are
Space-only architectures (energy + geometry, kNN-25) all plateau at **σ_eff ≈ 0.065** on min-bias
(nb13/nb15). Adding the ParT pairwise matrix U trended better but within noise. The 0.065 looks like a
**pileup-limited floor for space-only features.**

## The unused lever: per-cell timing
We have never used timing. But:
- The data carries `cell_times_front` / `cell_times_back` per cell.
- **This is exactly what PicoCal's fast timing is *for*.** LHCb design studies: with ~10–20 ps cluster
  timing, "it will be possible to exploit the time separation of the primary pp collisions and
  effectively mitigate the pileup" (LHCb PicoCal, JINST 21 (2026) C03006; arXiv:2203.07286 on precision
  timing for calorimetry). ML-with-timing for pileup rejection is an active, validated direction.
- Our data confirms the physics: the signal (highest-energy) cell is tightly **in-time** (IQR ~1.9),
  while background cells are out-of-time or untimed. Timing *is* a discriminant here.
- The 0.065 floor is **low-energy dominated** (sum baseline: 0.35 at 1–9 GeV → 0.12 at 62–100 GeV).
  Low energy is where pileup contamination is worst — and where timing helps most.

**Thesis:** the space-only floor is set by not being able to tell photon cells from pileup cells. Timing
is the physical variable that tells them apart. This is the "**spacetime**" in the project title, and we
have not used it yet.

## Honest expectation on 0.02
Timing makes min-bias look more like clean signal by rejecting out-of-time background. So the realistic
ceiling is the **clean-signal floor (~0.036 aggregate)**, not 0.02 — you cannot beat the no-pileup
resolution. BUT resolution is strongly energy-dependent, and in the **high-energy bins** (constant-term
regime) per-bin σ_eff is already ~0.03; timing cleaning could plausibly reach **0.02–0.03 in the top
bins**. So the honest framing: chase 0.02 **per high-energy bin**, drive the aggregate from 0.065 toward
the ~0.04 clean floor, and report per adaptive-energy-bin, not one aggregate number.

## Concrete steps

### nb16 — Spacetime transformer (main build)
1. **Extend the tokenizer** (pipeline change in `scripts/run_experiments.py`, `tokens()`): read
   `cell_times_front/back`, add per-cell features `[t_front, t_back, has_valid_time]` with sentinels
   masked/clipped and time referenced to the seed-cell time (Δt = t_cell − t_seed). Tokens grow 12 → ~15.
2. **Timing in the pairwise bias U** (timing-aware ParT): add `Δt_ij = |t_i − t_j|` to the cell-pair
   features feeding U, so in-time cells attend together and out-of-time cells are suppressed pre-softmax.
3. **Ablation ladder** (isolate each contribution): space-only (nb15 baseline) → + per-cell time →
   + Δt in U. 3–5 seeds, on the cached min-bias sample.
4. **Report per adaptive energy bin**, space-only vs spacetime — show where timing buys the most.

### Secondary levers (only if timing under-delivers)
- **Per-cell pileup classification auxiliary head** (object-condensation style): predict signal-vs-pileup
  per cell and let the pooling down-weight pileup. Uses timing + energy as supervision proxy.
- **In-time cell pre-selection**: keep cells within a timing window of the seed before kNN.
- **Loss**: Huber / quantile on log-energy instead of plain MSE, to stop tails inflating σ_eff.

## Risks / unknowns
- Sentinel handling: many low-E cells have no valid time; the `has_valid_time` flag + masking is essential
  or the huge sentinels wreck standardization.
- Timing resolution in the sample may be idealized vs the real ~20 ps; gains here are an upper bound.
- Pipeline change touches `tokens()` — needs a smoke test that the 12-feature path still reproduces
  nb13/nb15 before trusting the +timing numbers.

## Deliverable
Best spacetime architecture + honest per-bin resolution vs the space-only floor. If 0.02 lands only in
high-E bins, that is the correct, defensible result — and it directly motivates *why the detector has
timing*, which is a strong story for the mentors.
