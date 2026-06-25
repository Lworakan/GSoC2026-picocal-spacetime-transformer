# PicoCal Data Explorer — Design Spec

Date: 2026-06-24
Status: approved design, pending spec review

## 1. Purpose

A local web application that lets a person with **zero high-energy-physics background**
understand the PicoCal matched-cluster dataset end to end: what a photon does when it
hits the calorimeter, how that becomes the numbers in the ROOT files, what every column
means and when it is produced, and why the project uses a Transformer to predict true
energy. It replaces ad-hoc Jupyter exploration with one interactive, sensor-faithful
tool plus a written physics primer.

The tool must reference **both** the instruction PDF and the **real ROOT files**, and
must be honest about every place the two disagree.

## 2. Goals / Non-goals

Goals:
- Browse **all 110 ROOT files** (`data/full` + `data/gsoc_drive/*`), every event, on demand.
- Show the data as a **real sensor**: modules drawn as squares at true pitch, cells
  colored by energy, in their true location on the full ECAL face.
- Resolve the event/entry confusion visually and in prose.
- A complete **data dictionary** of every field (event / truth / cluster / cell),
  including instruction-only fields, with unit, formula, presence-in-ROOT, and
  real-vs-instruction discrepancy notes.
- A **physics primer** (`docs/physics-primer.md`) that teaches from basics to the real
  formulas and the simulation model, in the user's student voice.

Non-goals:
- No model training, no inference, no GPU work.
- No editing of the ROOT files.
- No npm / build toolchain (vanilla JS + CDN libraries only).

## 3. Architecture

- **Backend**: FastAPI + uvicorn. Reuses existing `uproot` / `awkward` / `numpy`.
  Reads ROOT live with lazy per-event reads (`entry_start`/`entry_stop`, as in the
  mentor notebook). `functools.lru_cache` for per-file overview and per-event detail so
  repeat loads are instant. Serves the static frontend and a small JSON API.
- **Frontend**: one static page served by the backend. **D3.js via CDN** for rendering,
  **KaTeX via CDN** for math/formula rendering. Heavy ROOT I/O stays server-side; only
  small JSON per event/distribution crosses the wire.
- **Bilingual (TH/ENG)**: a language toggle in the header, **default Thai**. Every
  human-facing string (labels, buttons, explainers, dictionary text, tour, discrepancy
  notes) carries both languages; toggling re-renders text with no refetch. Formulas
  (KaTeX) and numbers are language-neutral.
- **Data flow**: browser fetch → JSON → D3 render. No data baked into the page.
- **Run**: `uv run uvicorn picocal_explorer.app:app` → `http://localhost:8000`.

## 4. Backend modules (`src/picocal_explorer/`)

- `app.py` — FastAPI app, route definitions, static mount, startup geometry load.
- `data.py` — ROOT reading + caching; assembles per-file overview and per-event detail.
- `geometry.py` — derives `cell_pitch` / `cell_modType` from per-module cell geometry;
  builds the global module map (data-driven full ECAL face); region classification.
- `dictionary.py` — curated field metadata + discrepancy notes; merged with live
  branch introspection. Human-facing text is bilingual `{th, en}`.
- `analysis.py` — dataset-level distributions for the exploration view (truth-energy
  spectrum, cell multiplicity, ΔE/E response, cluster↔truth dr, seed↔truth distance,
  efficiency-vs-radius), per-file and aggregated across files.
- `explainers.py` — curated teaching content (the "what / why / formula / ML analogy"
  for each exploration plot and each formula), served as structured JSON; reused by both
  the embedded panels and `docs/physics-primer.md`. All human-facing strings are
  bilingual `{th, en}`.

### API endpoints

- `GET /api/files` → `[{name, dataset, n_entries}]`
- `GET /api/files/{name}/overview` →
  `{n_events, entries: [{tree_entry, event, sig_flux_eTot, pdgID, sig_dr_matched, x_cluster, y_cluster, multiplicity}]}`
- `GET /api/files/{name}/event/{event_id}` →
  ```
  {
    event,
    truth_photons: [{tree_entry, energy_gev, entry_x, entry_y, entry_z,
                     px, py, pz, dxdz, dydz, dr_matched, pdgID}],
    clusters: [{x_cluster, y_cluster, total_energy, total_energy_front,
                total_energy_back, n_cells, seed_index,
                cells: [{imodx, jmody, icell, x, y, front, back, energy,
                         t_front, t_back, pitch_derived, modtype_derived,
                         rel_x, rel_y, rel_dr, is_seed}]}],
    window_modules: [{imodx, jmody, x, y, pitch, modtype, n_cells}]
  }
  ```
- `GET /api/geometry` → global module map + region summary (data-driven full ECAL face).
- `GET /api/schema` → live branch list (name, typename, scalar/jagged, sample range)
  joined with dictionary metadata.
- `GET /api/dictionary` → curated dictionary entries (all fields, incl. instruction-only).
- `GET /api/files/{name}/distributions` (optional `?aggregate=true` across files) →
  histograms/series for the exploration view: truth-energy spectrum (linear + log),
  cell multiplicity, ΔE/E response, cluster↔truth dr, seed↔truth distance,
  efficiency-vs-radius.
- `GET /api/explainers` → curated teaching content: per-plot and per-formula
  `{id, title, formula_latex, what, why, ml_analogy}` for embedded panels.

### Caching / startup

- Per-file overview and per-event detail: `lru_cache`.
- Global module map: built by scanning `data/full` once, cached to
  `data/.geometry_cache.json` so subsequent startups are fast. Cache keyed by file set.

## 5. Frontend views (single page, tabbed)

1. **Dataset overview** — pick a file; D3 scatter of all matched pairs (event vs
   `sig_flux_eTot`), colored by multiplicity; click a point → open that event.
2. **Detector / sensor view** *(primary)* — two linked panels:
   - *Full ECAL face*: data-driven module map (±~3800 mm, beam hole, 6 granularity
     regions) drawn faintly, with the current 5×5 module window highlighted **in its
     real location**.
   - *Zoomed window*: each module a square at true (derived) pitch; cells as squares
     colored by energy; overlays truth entry (×), reco cluster (+), seed cell (★).
     Toggles: front / back / total energy, linear/log color, timing on/off.
     Hover a cell → panel with every per-cell number + plain-language "what / when".
3. **Shower depth view** — front-vs-back energy per cell (longitudinal development),
   energy-sharing, timing.
4. **Data dictionary** — searchable table of every field: level, type, unit, formula,
   **present-in-ROOT vs instruction-only**, discrepancy notes.
5. **Guided tour / story** — step-through: photon born → flies → enters ECAL face →
   showers → deposits in front+back cells across modules → reconstruction groups a
   cluster → truth-match keeps the closest → becomes one *entry*. Each step links to the
   relevant view and the primer.
6. **Exploration & Formulas** *(interactive EDA + teaching)* — reproduces the mentor /
   `no_minbias` notebook's exploration plots **live across the data** (truth-energy
   spectrum with linear/log toggle, cell-multiplicity = sequence length, ΔE/E response,
   cluster↔truth dr, seed↔truth distance, efficiency-vs-radius). Each plot has a side
   panel: **what it computes · why we make it (the model-design decision it drives) ·
   the formula (KaTeX) · the ML analogy**, plus a top note on *why EDA matters even with
   a guideline* (it caught the four discrepancies). Surfaces the through-line:
   predict `log E` ⟺ relative error ⟺ resolution `σ_E/E`.

Frontend files: `static/index.html`, `static/app.js`, `static/styles.css`.

## 6. Derived fields (labeled "derived — not stored in ROOT")

For cells sharing one `(imodx, jmody)` module:
- `cell_pitch` = median nearest-neighbour distance between cells in that module.
- `cell_modtype` = mapped from cell count per module:
  `1→Shashlik-1`, `4→Shashlik-4`, `9→Shashlik-9`, `16→SpaCal-16-PbPoly`,
  `64→SpaCal-64-WGAGG`; counts that fall between canonical values (window-edge
  truncation) are labeled `partial/edge` and pitch is still taken from spacing.
- `cell_rel_x = cell_x − x_seed`, `cell_rel_y = cell_y − y_seed`,
  `cell_rel_dr = sqrt(rel_x² + rel_y²)`. Seed = max-energy cell in the cluster.

## 7. Data dictionary content (real + instruction), with the four discrepancies

Fields present in ROOT (verified via `scripts/inspect_full_data.py`):
- Truth: `sig_flux_entry_x/y/z`, `sig_flux_px/py/pz`, `sig_flux_prod_vertex_x/y/z`,
  `sig_flux_timing`, `sig_flux_eTot`, `sig_flux_pdgID`, `sig_dr_matched`,
  `sig_dxdz_flux`, `sig_dydz_flux`.
- Cluster/seed: `x_cluster`, `y_cluster`, `total_energy`, `total_energy_front`,
  `total_energy_back`, `seed_icell_x`, `seed_icell_y`, `event`.
- Cells (jagged): `imodx`, `jmody`, `icell`, `cell_x`, `cell_y`,
  `cell_energies_front`, `cell_energies_back`, `energy`, `cell_times_front`,
  `cell_times_back`, plus seed-window arrays `seed_cell_x`, `seed_cell_y` (3×3).
- Counters present: `n*` (`nimodx`, `nenergy`, `ncell_x`, …) — uproot uses these to
  produce correctly-sized jagged arrays.

Truly absent from ROOT (must derive): `cell_pitch`, `cell_modType`,
`cell_rel_x/y`, `cell_rel_dr`.

The four discrepancies, surfaced in the dictionary and tour:
1. **event ↔ entry**: instruction says "1 event = 1 matched cluster"; real files store
   **multiple entries per event** (same cluster, several truth photons).
2. **Storage**: instruction describes fixed-size zero-padded arrays. Reality: `n*`
   counters exist, but cell branches read as **true variable-length jagged arrays**
   (`ak.num()` = real count, no padding to strip); cell count ranges ≈6→528.
3. **Units (confirmed)**: instruction says `sig_flux_eTot` is MeV; it is **GeV**
   (max eTot 203.1 vs max `sig_flux_pz` ≈ 203,000 MeV; cells are MeV). Backend reports
   `total_energy / (sig_flux_eTot·1000)` to show the reconciliation.
4. **Derived-not-stored fields**: `cell_pitch`, `cell_modType`, `cell_rel_*` are computed
   and labeled "derived." (`sig_flux_prod_vertex_*` etc. are present and shown as real.)

## 8. Physics primer outline (`docs/physics-primer.md`)

Student voice, ML analogies, story-first. Real formulas, each explained term by term:
1. Particles 101 — photon vs electron, energy & units (MeV/GeV), PDG codes (22 = photon).
2. Calorimeter + EM shower — radiation length `X0`, Molière radius `R_M`; longitudinal
   profile `dE/dt ∝ t^a e^{-bt}`; shower max `t_max ≈ ln(E/E_c) + C`.
3. Why energy fluctuates — sampling calorimeter; resolution
   `σ_E/E = a/√E ⊕ b ⊕ c/E` (stochastic ⊕ constant ⊕ noise), each term explained.
4. Front/back longitudinal segmentation and why timing helps.
5. Geant4 simulation as a **forward stochastic process**; why truth energy is known in
   simulation but not in real data.
6. **Why we cannot invert the simulation** — many-to-one + randomness ⇒ no closed-form
   inverse; we learn a statistical estimator instead.
7. **Why a Transformer** — variable-length set of cells (variable granularity),
   permutation invariance, position+time as "spacetime" tokens, attention over cells;
   vs CNN-on-fixed-grid limits.
8. Decoding the "log energy spectrum" formula used in the notebooks, from scratch.
9. **Purpose of each exploration plot** — the EDA→model-decision map (cell multiplicity =
   sequence length; truth-energy spectrum → log-target decision; ΔE/E → bias/resolution;
   dr → label-noise check; seed → baseline; efficiency-vs-radius → window choice) and the
   **through-line** `log E` ⟺ relative error ⟺ `σ_E/E`. Includes *why EDA matters even
   with a guideline* (it surfaced the four real-vs-instruction discrepancies). This is the
   written twin of frontend view 6; both pull from `explainers.py`.

## 9. Project layout

```
src/picocal_explorer/
  __init__.py
  app.py
  data.py
  geometry.py
  dictionary.py
  analysis.py
  explainers.py
static/
  index.html
  app.js
  styles.css
docs/
  physics-primer.md
tests/
  test_geometry.py
  test_data.py
  test_api.py
```

## 10. Dependencies & run

- Add `fastapi`, `uvicorn[standard]` to `pyproject.toml` / `requirements.txt`.
- Already present: `uproot`, `awkward`, `numpy`, `pandas`.
- Frontend libs via CDN (no install): D3.js, KaTeX. No new Python deps for the
  exploration/explainer content.
- Run: `uv run uvicorn picocal_explorer.app:app` (optionally a `scripts/run_explorer.py`
  wrapper that also opens the browser).

## 11. Testing

- `test_geometry.py` — module with N cells → expected pitch bucket + modtype label;
  rel-coord math.
- `test_data.py` — event-detail assembly on a real file (event 4 → 3 truth photons,
  1 distinct cluster); overview shapes.
- `test_analysis.py` — distribution builders return expected shapes/bins on a real file;
  ΔE/E uses the correct GeV→MeV reconciliation.
- `test_api.py` — endpoints (incl. `/distributions`, `/explainers`) return the documented
  JSON shapes (FastAPI TestClient, no network).

## 12. Conventions

- No code comments / docstrings in generated code (user preference).
- Student voice in prose; HEP via ML analogies; story over results.
- UI is bilingual TH/ENG (default Thai). `docs/physics-primer.md` is Thai-primary (English
  technical terms kept); the web explainers are its fully bilingual interactive twin.
- No autonomous commits — content shown before any commit/push.
- CPU-only; no GPU.

## 13. Build order

1. Backend `geometry.py` + `data.py` + `dictionary.py` + `analysis.py` + `explainers.py`
   (+ tests).
2. `app.py` endpoints incl. `/distributions` and `/explainers` (+ API tests).
3. Frontend: overview → sensor view → shower view → dictionary → tour →
   exploration & formulas.
4. `docs/physics-primer.md` (incl. the exploration-purpose section + through-line).
5. README run section; dependency additions.
