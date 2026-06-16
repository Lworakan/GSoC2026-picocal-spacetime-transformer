# Notebooks

- `01_exploration_matched_clusters.ipynb` — first look at the matched-cluster data: GeV units and
  calibration, jagged cell tokens, timing cleaning, 67% cluster sharing, the calibrated-sum
  baseline (Week 1).
- `02_minbias_comparison.ipynb` — with vs without minimum-bias overlay on the mentors' paired
  sample: what "noise" is, the per-cell record of removed energy, and why timing can't tag it on
  this sample (Week 1 addendum).
- `03_poc_spacetime_transformer.ipynb` — proof of concept on the GPU: a space-time transformer
  (Fourier positional encoding of x, y, depth, time) vs a DeepSets baseline vs the calibrated sum,
  with a timing ablation and a PCA of the learned embedding (Week 2).
- `04_feature_engineering_plan.ipynb` — plan for pulling more signal out of the raw cells: the
  unused channels, the feature tiers, and the A/B test that decides whether they help (Week 2).

Conventions:

- Notebooks are built with a generator script and run with `jupyter nbconvert --execute`, then
  committed **with their outputs** so the figures and findings render on GitHub.
- No machine-specific paths (`/home/...`, `/tmp/...`) in committed outputs.
- Move reusable code into `src/picocal/` so it can be unit-tested; notebooks call the package.
