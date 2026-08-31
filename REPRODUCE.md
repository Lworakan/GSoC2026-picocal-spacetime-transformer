# Reproducing every number and figure in the paper

The intent is that someone who has never seen this repository can regenerate any figure or
table from the shipped prediction files with one command, and can retrain from raw data if
they have the sample. Nothing here is bespoke to the machine it was developed on.

## Environment

Developed and measured on Python 3.11.15 with torch 2.13.0, numpy 2.4.6, pandas 3.0.3 and
scikit-learn 1.9.0. Training was run on a single A100 40\,GB; everything in the "figures and
tables" section below runs on a laptop CPU in minutes.

```bash
uv venv && uv pip install -r requirements.txt
```

Results in the paper were produced on CPU and on a single GPU. No multi-GPU path is used
anywhere, and no result depends on one.

## The split, stated exactly

Every number in the paper comes from one of two protocols and they are never mixed. Both
are defined in one function, `split()` in `scripts/run_experiments.py`, and both derive from
the same seeded permutation, so they are comparable to each other:

```python
idx = np.random.default_rng(seed=0).permutation(n)
```

- **Development protocol.** A fixed 70/15/15 split of that permutation: `idx[:0.70n]`,
  `idx[0.70n:0.85n]`, `idx[0.85n:]`. On the minimum-bias sample this is 10,877 test events.
  Used for exploration and labelled as such wherever it appears.
- **Primary protocol.** Ten-fold cross-validation over the same permutation, five members
  per fold. Every one of the 72,533 events is predicted by models that never saw it. Fold 0's
  test block is the same events the fixed split tests, by construction.

A development-split result is never compared against a cross-validated one. Two claims in
this project failed exactly that way before the rule was adopted, and both are reported in
the paper.

## Metric

`resolution()` in `scripts/run_experiments.py`. `sigma_eff` is the half-width of the
narrowest interval of relative residuals `(E_pred - E_true)/E_true` containing 68.3% of
events -- not a Gaussian fit, and not an RMS. This is the same definition the graph-network
work on this detector reports, which is what makes the two comparable.

Significance is a paired bootstrap over 400 resamples shared between the two estimators
being compared, with a decision bar of 380 of 400.

## Figures and tables, one command each

All of these read the prediction CSVs in `reports/predictions/` and need no GPU and no raw
data.

```bash
# per-region resolution against transverse energy, and against the design specification
uv run scripts/paper_figs/design_and_bins.py
#   -> paper/figs/resolution_vs_et.pdf, design_gap.pdf, perbin_bars.pdf

# resolution against transverse energy in the format the GNN work on this detector uses
uv run scripts/paper_figs/vs_gnn.py            # -> paper/figs/vs_gnn.pdf

# a/sqrt(E) + b/E + c fit per region, with the crossing energy for a chosen target
uv run scripts/fit_resolution.py reports/predictions/minbias__SubNetW8CleanAuxExDnGs50RcOvV2CrQdEma.csv

# the cross-validated tables and per-region curves
uv run scripts/paper_figs/cv_tables_and_figures.py

# window scan, timing ablation, throughput, scaling fit, containment and mis-seeding
uv run scripts/paper_figs/window_scan.py
uv run scripts/paper_figs/timing_ablation.py
uv run scripts/paper_figs/throughput_figure.py
uv run scripts/paper_figs/scaling_fit.py
uv run scripts/paper_figs/containment_misseed_perbin.py

# detector and event renderings
uv run scripts/paper_figs/regions_3d.py
uv run scripts/paper_figs/event_3d.py
uv run scripts/paper_figs/photon_vs_pileup_3d.py
uv run scripts/paper_figs/input_example.py

# the paired comparison against the gradient-boosted reference, on shared events
uv run scripts/paired_baseline.py

# the comparison against the graph-network work, anchored on a shared 3x3 sum
uv run scripts/gnn_compare.py --region 30mm
```

## Retraining

Three commands reproduce the primary members. Each writes a prediction CSV into
`reports/predictions/` named after the configuration, which is how every figure above finds
its inputs.

```bash
# the cross-validated primary configuration, one fold at a time
uv run scripts/train_picocal.py --sample minbias --window 8 --extra --dens --recenter \
    --cleanaux --fold K --nfold 10 --seeds 0

# the development-split configuration reported in Sec. 8
uv run scripts/train_picocal.py --sample minbias --window 8 --extra --dens --recenter \
    --cleanaux --cellreg --gatesup 5.0 --overlay .scratch/cache/overlay_v2.pkl --seeds 0 1 2

# the ten-fold run of that configuration
bash scripts/cv_final.sh
```

The per-cell overlay labels are generated, not shipped:

```bash
uv run scripts/make_overlay.py --regions 0 1 2 3 4 --per-event 3 --seed 3 \
    --out .scratch/cache/overlay_v2.pkl
```

## Data

The simulated sample is an LHCb production and is not ours to redistribute. `docs/data.md`
records the file layout, the branch names actually present, and four places where the real
files differ from the documentation we were given. Until the sample or a generator
configuration can be released, the figures above are reproducible from the shipped
predictions and the training commands are reproducible only inside the collaboration --
which is stated here rather than left for a reader to discover.

## What is not reproducible from this repository

Named so that nobody wastes time looking:

- The GNNMP curve in the comparison. Those values were measured off a figure in a
  conference talk and are marked as such; they are not ours to publish and the comparison
  script keeps them in one clearly-labelled constant.
- The luminosity of the minimum-bias sample. It is not recorded in the files we have.
- `scripts/oracle_ceiling.py` prints numbers that are not yet valid; its header lists the
  three things that have to be fixed first.
