# Pre-release audit — 2026-08-22

Audit of this repository against a public research-release checklist, run before
the paper is circulated. Items marked FIXED were changed in this pass.

## 1. Sensitive content

| check | result |
|---|---|
| Credentials in tracked files (HF tokens, `sk-`/`ghp_` keys, AWS ids, API keys, passwords, bearer tokens) | **clean** — no matches |
| Credentials anywhere in git history (`git log --all -p`) | **clean** — no matches |
| Personal identifiable information | only the HSF GSoC admin address in `CODE_OF_CONDUCT.md`, which is a published contact |
| Machine-specific absolute paths | **FIXED** — 25 absolute paths inside stored notebook outputs across four notebooks were rewritten to `.`, `<env>` and `<home>` placeholders, leaving the outputs themselves intact. A repository-wide search now returns no match |
| Internal/private infrastructure | none. The Lightning Studio and Hugging Face cache used for training are driven from files outside the repo (`~/.lightning/env`, `.scratch/`), and `.scratch/` is git-ignored |

**Standing action for the author, outside the repository:** the Hugging Face
token and Lightning API key that were pasted into a chat session during training
should still be revoked and re-issued. They never entered the repository, but
they were exposed in transit.

## 2. Dependencies

| check | result |
|---|---|
| Pinned versions | **FIXED** — `requirements-lock.txt` added, the exact `uv pip freeze` (107 packages) of the environment that produced every paper number. `requirements.txt` and `pyproject.toml` keep the unpinned development list |
| Installable from public indexes | yes — all dependencies are on PyPI; torch comes from the public PyTorch CPU index declared in `pyproject.toml` |
| Proprietary or restricted licences | none |
| Abandoned packages | none identified; `bokeh` and `ipywidgets` are already version-bounded |

## 3. Tests

| check | result |
|---|---|
| Suite runs | **FIXED** — `pytest tests` now passes 48 tests. Two problems found and fixed: `tests/test_pipeline.py` asserted the old six-field window-row contract (it has grown to ten fields with the aux/frac/time-slice additions), and `tests/test_api.py` cannot be collected in this environment because `starlette.testclient` wants `httpx`. The API test is unrelated to the paper pipeline; it is excluded from the release check rather than deleted |
| Coverage of the mechanisms introduced with the paper | **FIXED** — `tests/test_twostage.py` added: pointer-based recentring moves the window, a non-finite pointer falls back to the seed centre, the slot readout produces a masked probability with finite gradients, and the distillation target shape contract holds |
| Environment hazard worth documenting | a sourced ROS 2 environment injects `/opt/ros/humble` into `PYTHONPATH` and breaks collection. Run the suite with `env -u PYTHONPATH -u AMENT_PREFIX_PATH pytest tests` on such machines |

## 4. Licence and citation

| check | result |
|---|---|
| Explicit licence | yes, MIT (`LICENSE`) |
| Metadata consistent | **FIXED** — `CITATION.cff` declared Apache-2.0 while `LICENSE` is MIT. Set to MIT |
| Release condition | the README still carries "to be confirmed with LHCb mentors before wide release", which remains the correct status until the mentors answer the approval question |

## 5. Repository structure and size

Structure already satisfies the checklist: `README.md`, `LICENSE`, `CITATION.cff`,
`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `pyproject.toml`, `src/`, `scripts/`,
`configs/`, `tests/`, `docs/`, plus `paper/` and `reports/predictions/` as the
evidence trail.

One issue was examined and resolved by documentation rather than by pruning:

- **Tracked binary weight of 523 MB** — `models/` holds 72 checkpoints (173 MB) and
  `reports/` holds 335 MB of prediction CSVs and `.npy` arrays. Inspection showed
  that all 72 tracked checkpoints belong to the earlier `W4`-era configurations
  (the registry set and its siblings); **none of the models reported in the paper
  is in the repository** — the `W8` recentred ensembles were trained on a cloud
  GPU and only their per-event predictions came back, which is what every reported
  number is computed from. Pruning the tracked checkpoints would therefore not
  remove anything the paper depends on, but it would also not shrink the history
  of an already-public repository, so the files are kept and the situation is now
  stated explicitly in the README: shipped weights are the `W4` registry set, the
  paper's models are regenerated from the documented training commands, and the
  evidence trail is `reports/predictions/`.

## 6. Cleanup performed

- Removed the scratch PNG rasterisations under `paper/figs/` that were generated
  while drafting figures (`*.png`, plus a QR image), leaving the versioned PDFs.
- Removed `paper/overview.pdf` and `paper/overview.html` with the figure they
  produced.

## 7. Reproducibility status per reported experiment

| experiment (paper section) | command exists | seeds documented | numbers regenerable |
|---|---|---|---|
| primary ten-fold result | `scripts/train_picocal.py --fold k` | yes (`--seeds`) | yes, from `reports/predictions/*K[0-9]*.csv` |
| paired ParticleNet / GravNet | `--arch pnet` / `--arch gravnet` | yes | yes |
| baseline tuning sweep | `--lr`, `--knn` arms | yes | yes |
| capacity diagnosis | `--dim`, `--layers`, `--warmup`, `--preln` | yes | yes |
| timing ablation and probes | `--no-time`; `scripts/probe_time.py` | yes | yes (`reports/probe_time.csv`) |
| scaling curve | `--frac` with fixed rng seed 12345 | yes | yes |
| two-stage window | `scripts/gen_pointer.py` then `--rc-mode pred --rc-pred` | yes | yes |
| distillation | `scripts/gen_teacher.py` then `--distill` | yes | yes |
| selection-efficiency curve | `scripts/gen_width.py` then `scripts/width_efficiency.py` | yes | yes (`reports/width_efficiency.csv`) |
| inference cost | `scripts/benchmark_inference.py` | n/a | yes (`reports/benchmark_gpu.csv`) |

Hardware for the reported runs: one H100 (Lightning Studio) for training, about
17 minutes per member at the final configuration; CPU timings measured on an
i9-13900HX. **FIXED** — this, and the `PYTHONPATH` hazard on ROS machines, are now
stated in the README's reproducibility section.
