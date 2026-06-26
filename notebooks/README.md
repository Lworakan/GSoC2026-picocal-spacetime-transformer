# Guided notebooks

- `01_Data understanding _PyTorch tokens.ipynb` — the main lesson. Understand the
  matched-cluster ROOT data (branches, scalar vs jagged, the event/entry/cluster/cell
  hierarchy, units), reproduce the mentor's exploration on real `data/full`, then build
  a PyTorch token dataset and transformer-ready features (normalized, space-time,
  `log` energy target, padding mask, cleaned cell timing).

Earlier AI-generated notebooks were removed from the working tree; they remain in
git history if an audit is needed.

## Conventions

- Work through one notebook at a time; run cells top-to-bottom from a fresh kernel.
- No machine-specific paths.
- Every conclusion is verified against the data, the mentor, or a primary source.
- Reusable code moves into `src/` only after it is understood and has tests.
