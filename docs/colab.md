# Running the training on Colab (or any cloud GPU)

The laptop can only run **one** CUDA job at a time — several at once power the machine
off — so the experiment queue is serial and slow. Colab lifts that: each Colab session is
an independent GPU, so several experiments can run in parallel.

The dataset does **not** need to be uploaded. `scripts/train_picocal.py` caches the parsed
events, and only the cache is needed to train:

| file | size |
|---|---|
| `.scratch/cache/minbias_94.pkl` | 246 MB |
| `.scratch/cache/clean-aux_100.pkl` | 108 MB |

354 MB total, versus 1.6 GB of ROOT files. Put those two files in Google Drive once
(e.g. `MyDrive/picocal/cache/`) and every session reuses them.

## Cell 1 — repo, dependencies, Drive

```python
!git clone https://github.com/Lworakan/GSoC2026-picocal-spacetime-transformer.git repo
%cd repo
!pip -q install uproot awkward          # imported by picocal_data even when the cache is used
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p .scratch/cache
!cp /content/drive/MyDrive/picocal/cache/*.pkl .scratch/cache/
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Cell 2 — train, writing everything to Drive so a disconnect loses nothing

```python
OUT = '/content/drive/MyDrive/picocal/out'
!mkdir -p {OUT}/preds {OUT}/models {OUT}/ckpt

!python -u scripts/train_picocal.py --sample minbias --cleanaux \
    --extra --dens --seeds 0 1 --device cuda \
    --out {OUT}/preds --models-dir {OUT}/models --ckpt-dir {OUT}/ckpt
```

`--ckpt-dir` on Drive is the important part: the script checkpoints every epoch and resumes
from the checkpoint, so if Colab disconnects you re-run the same cell and it continues.
Seeds already present in the output CSV are skipped, so re-running is always safe.

## Cell 3 — score the result

```python
!python scripts/score_regions.py {OUT}/preds/minbias__SubNetW4CleanAuxExDnQdEma.csv \
    --baseline reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv --bin low
!python scripts/fit_resolution.py {OUT}/preds/minbias__SubNetW4CleanAuxExDnQdEma.csv --target 0.05
```

Then copy the CSV back into `reports/predictions/` in the repo and commit it — the CSVs are
a few MB, the model weights are 2.4 MB each.

## Running several experiments at once

Open one Colab session per experiment and change only the flags in Cell 2. Give each its own
`--ckpt-dir` so two sessions never write the same checkpoint:

| experiment | flags | ckpt dir |
|---|---|---|
| smaller window | `--extra --dens --window 2` | `{OUT}/ckpt_w2` |
| ring-rho subtraction | `--extra --dens --rho` | `{OUT}/ckpt_rho` |
| resolution-weighted time pull | `--extra --dens --tpull` | `{OUT}/ckpt_tp` |
| time pull + smaller window | `--extra --dens --tpull --window 2` | `{OUT}/ckpt_tpw2` |

Two cautions. Free Colab hands out a T4, which is slower than the laptop's 4080 for this
model — the gain is parallelism, not per-job speed; Pro (L4/A100) is faster per job as well.
And a free session is reclaimed after a few hours of use or on idling, which is exactly why
the checkpoint directory must live on Drive.

## What this cannot do

There is no Colab MCP connector, so the cells above have to be run by hand — the notebook
cannot be driven from here. If a persistent machine is preferred over Colab, the same
commands work unchanged on any Linux box with a CUDA GPU after `uv sync`; that is the
argument for asking the mentors for the lab node.
