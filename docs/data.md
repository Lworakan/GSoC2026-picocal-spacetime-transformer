# Data

PicoCal Geant4 simulation is provided by the mentors and is **not** committed.
Everything under `data/` is git-ignored.

## Getting the data

The dataset is distributed by the project mentors as a CERNBox archive
(link provided privately to participants). Extract it with:

```bash
mkdir -p data/full
tar -xf download.tar -C data/full
```

This yields 100 files `matched_<i>_<j>.root` (~2.2 MB each, ~246 MB total),
together holding **199,538 matched clusters** in the `clusters_matched` tree.

## What one entry is

One reconstructed calorimeter cluster (a 5×5 *module* window around the seed)
matched to one true photon. Cell-level quantities are **jagged arrays** (one
value per cell in the window); cluster-level and truth quantities are scalars.
The authoritative branch list is in `gsoc26_instruction.pdf`; the branch →
physics → ML-role glossary is in
[`physics-primer.md`](physics-primer.md) §6.

## Empirical conventions (measured, not documented)

Verified in `notebooks/01_exploration_matched_clusters.ipynb`:

- `sig_flux_eTot` is in **GeV**; reconstructed energies are in raw
  uncalibrated units with gain ≈ 970 per GeV (for isolated photons).
- The `n*` branches equal the raw array length — there is **no zero padding**;
  `energy == 0` cells are real silent window cells and carry no information.
- `cell_times_*` contain uninitialised garbage; valid only where the cell has
  energy and |t| < 10⁴ ns (see `picocal.data.clean_cell_times`).
- The same `event` index appears for several photons of the same event
  (up to ~13). **Split train/val/test by `(file, event)`**, never by entry.
- ~67% of photons share their matched cluster with a sibling photon —
  responses for non-isolated photons include overlapping shower energy.

## Minbias comparison sample (Drive)

A small paired sample shared by the mentors on Google Drive (link provided privately):

```
data/gsoc_drive/with_minimum_bias/      5 files,  4,913 clusters  (2024-04 production)
data/gsoc_drive/without_minimum_bias/   5 files,  9,861 clusters  (2024-07 production)
```

Same `clusters_matched` tree but an **older format** than the 100-file set:

- only the ~9 cluster-member cells are stored per entry (no full 5×5 window, so almost no
  silent cells), and there are no seed branches;
- extra branches: `cell_energies_{front,back}_orig`, `total_energy_corr`,
  `x/y_cluster_corr`, `z_cluster`. `_orig` minus the plain cell energies is a per-cell
  record of removed (overlay) energy;
- the raw gain is production-specific (~2030 and ~1890 raw/GeV here vs ~970 in the
  100-file set) — calibrate each sample separately.

Explored in `notebooks/02_minbias_comparison.ipynb`.

## Dataset stages (proposal plan)

1. **Stage 1** — single-photon clusters (this dataset). Baseline floor.
2. **Stage 2** — + minimum-bias background (realistic occupancy at 30 MHz).
3. **Stage 3** — region boundaries with mixed cell sizes.
4. **Stage 4** — multiple regions + two longitudinal layers (full complexity).
