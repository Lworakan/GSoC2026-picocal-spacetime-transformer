# Data

PicoCal Geant4 simulation is provided by the mentors and is **not** committed.
Place files under `data/` (git-ignored).

## Dataset stages

1. **Stage 1** — single-photon, uniform 40 mm cells, no background. Baseline floor.
2. **Stage 2** — + minimum-bias background (realistic occupancy at 30 MHz).
3. **Stage 3** — region boundaries with mixed cell sizes.
4. **Stage 4** — multiple regions + two longitudinal layers (full complexity).

## Expected per-cell fields

`x, y, z` (position), `energy`, `t` (~15 ps timing). The loader normalises units
and builds a k-NN graph per event.
