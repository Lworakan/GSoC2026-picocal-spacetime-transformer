# PicoCal Space-Time Transformer

> **Rebuild status:** this repository is being relearned and rebuilt one guided
> lesson at a time. Older generated notebooks are **unverified reference material**
> until reproduced and interpreted. Start with
> [`notebooks/01_Data understanding _PyTorch tokens.ipynb`](notebooks/01_Data%20understanding%20_PyTorch%20tokens.ipynb)
> (data understanding → PyTorch token dataset → transformer-ready features).

Calorimeter energy reconstruction for the **LHCb PicoCal** using a space-time
kernel transformer, benchmarked against LHCb's production clustering algorithms.

> Google Summer of Code 2026 — CERN-HSF
> ([program page](https://hepsoftwarefoundation.org/activities/gsoc.html) ·
> [2026 projects](https://hepsoftwarefoundation.org/gsoc/2026/summary.html)).
> This repository is developed openly and documented for handoff to the LHCb group.

---

## What this project does

PicoCal records both **longitudinal depth** and **~15 ps timing** per cell. This
project tests whether a transformer that treats timing as a spatial coordinate
(space-time positional encoding) improves photon energy resolution over LHCb's
rule-based and graph-based reconstruction.

The work is structured **pipeline-first**: a complete data → train → evaluate
loop runs end-to-end on the simplest dataset before any modelling work begins,
so every later change slots into a working, reproducible pipeline.

| Model | Role | Status |
|---|---|---|
| Cellular Automaton | Simplest LHCb baseline, establishes the pipeline | planned |
| Graph Clustering | Current LHCb production algorithm — target to beat | planned |
| Space-Time Transformer | Proposed: O(N) kernel attention + space-time PE | planned |
| GravNet | Optional GNN comparison | optional |
| ClusTEX-style | Optional graph transformer | optional |

## Pipeline

```
Geant4 → ROOT files → uproot + PyG → exploration → preprocessing
       → baselines / transformer → evaluation (resolution, latency)
```

## Repository layout

```
src/picocal/
  data/         data loading (uproot + PyG), k-NN graph construction
  models/       cellular automaton, graph clustering, space-time transformer
  evaluation/   physics + computational metrics
configs/        experiment configs per dataset stage / model
notebooks/      data exploration and result analysis
scripts/        train / evaluate entry points
tests/          unit tests (run in CI)
docs/           design notes and usage
```

## Quick start

```bash
# 1. Clone
git clone https://github.com/Lworakan/GSoC2026-picocal-spacetime-transformer.git
cd GSoC2026-picocal-spacetime-transformer

# 2. Environment (choose one)
conda env create -f environment.yml && conda activate picocal
# or
pip install -e ".[dev]"

# 3. Run the tests
pytest

# 4. Train a baseline on Stage 1 data
python scripts/train.py --config configs/stage1.yaml
```

## Data Explorer (interactive)

A local web app to explore the matched-cluster dataset as a **real sensor** — the
detector face, per-cell energies at true module pitch, every field with its
meaning, and the physics/formulas — **bilingual TH/ENG**. It reads the ROOT files
live (all 110 files) and needs no build step.

```bash
python scripts/run_explorer.py
# or
uvicorn picocal_explorer.app:app --app-dir src
```

Then open <http://127.0.0.1:8000>. Eight views: fundamentals, overview,
detector/sensor, 3D event, shower depth, data dictionary, guided tour, and
exploration & formulas.

## Reproducibility

Following the HSF reproducibility guidance, the repo ships with a `Dockerfile`
and a pinned environment so results can be reproduced exactly:

```bash
docker build -t picocal .
docker run --rm -it picocal pytest
```

Each result in the final report is tied to a tagged release and a config file.

## Data

The PicoCal Geant4 simulation is provided by the mentors and is **not** committed
to this repository. Place ROOT files under `data/` (git-ignored); the explorer and
notebook read them from `data/full/`.

## Documentation

- `notebooks/01_Data understanding _PyTorch tokens.ipynb` — data understanding → PyTorch token dataset → transformer-ready features
- Interactive data explorer — see the section above
- `notebooks/` — data exploration and result analysis

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project follows the
[HSF Code of Conduct](CODE_OF_CONDUCT.md).

## License

See [LICENSE](LICENSE). **Confirm the license choice with your mentor** before
the first public release — LHCb / CERN projects sometimes require a specific one.

## Citation

If you use this code, please cite it via [CITATION.cff](CITATION.cff).

## Acknowledgements

Developed during Google Summer of Code 2026 under CERN-HSF, with mentorship from
the LHCb calorimeter group.
