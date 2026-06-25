# PicoCal Data Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app that lets a zero-HEP-background student explore the PicoCal matched-cluster ROOT dataset as a real sensor, see every field, and learn the physics/formulas — bilingual TH/ENG.

**Architecture:** FastAPI backend reads ROOT live with uproot (lazy per-event reads + caching) and serves a small JSON API; a vanilla-JS/D3/KaTeX single-page frontend renders six views. Geometry-only fields (`cell_pitch`/`cell_modType`/`cell_rel_*`) are derived and labeled.

**Tech Stack:** Python, FastAPI, uvicorn, uproot, awkward, numpy; frontend HTML/CSS/vanilla-JS + D3.js + KaTeX (CDN).

## Global Constraints

- ROOT data lives in `data/full/` (100 files) + `data/gsoc_drive/{with,without}_minimum_bias/` (5+5). Tree name `clusters_matched`.
- `sig_flux_eTot` is in **GeV**; cell energies / `total_energy` in **MeV**. To compare target vs reco use `sig_flux_eTot * 1000`.
- Cell branches are jagged variable-length (`ak.num` = true count); no zero-padding to strip.
- Truly absent fields to derive (label "derived"): `cell_pitch`, `cell_modType`, `cell_rel_x`, `cell_rel_y`, `cell_rel_dr`. Seed = max-energy cell in the cluster.
- No code comments / docstrings (user preference).
- UI default language Thai; every human-facing string bilingual `{th, en}`.
- Python entrypoint: `uv run uvicorn picocal_explorer.app:app`. Package under `src/picocal_explorer/`; `src` is on path via `pyproject.toml` (`tool.setuptools` or `pythonpath`). Tests with pytest.
- No GPU, no model training.

---

### Task 1: Package scaffold + dependencies

**Files:**
- Create: `src/picocal_explorer/__init__.py` (empty)
- Modify: `pyproject.toml` (add deps + pytest pythonpath)
- Modify: `requirements.txt` (add fastapi, uvicorn)
- Test: `tests/test_import.py`

**Interfaces:**
- Produces: importable package `picocal_explorer`.

- [ ] **Step 1: Write the failing test**
```python
def test_package_imports():
    import picocal_explorer
    assert picocal_explorer is not None
```
- [ ] **Step 2: Run, expect fail** — `uv run pytest tests/test_import.py -v` → ModuleNotFoundError.
- [ ] **Step 3:** Create `src/picocal_explorer/__init__.py`; add to `pyproject.toml`:
  `dependencies += ["fastapi", "uvicorn[standard]"]`; add `[tool.pytest.ini_options] pythonpath = ["src"]`; add `fastapi`/`uvicorn[standard]` to `requirements.txt`. Run `uv sync`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(explorer): package scaffold + deps"`

---

### Task 2: Geometry derivation (`geometry.py`)

**Files:**
- Create: `src/picocal_explorer/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `module_pitch(positions: np.ndarray) -> float` — median nearest-neighbour distance; `nan` if <2 points.
  - `classify_by_pitch(pitch: float) -> str` — one of `SpaCal-64-WGAGG|SpaCal-16-PbPoly|Shashlik-9|Shashlik-4|Shashlik-1`.
  - `build_module_map(files: list[Path]) -> dict[tuple[int,int], dict]` — `{(imodx,jmody): {"x","y","pitch","modtype","n_cells"}}`.
  - `derive_cell_geometry(imodx, jmody, cell_x, cell_y, module_map, seed_index) -> dict` with arrays `pitch, modtype, rel_x, rel_y, rel_dr` and `is_seed`.
  - `load_or_build_module_map(data_root: Path, cache: Path) -> dict` (JSON cache; keys serialized as `"ix,iy"`).

- [ ] **Step 1: Write failing tests**
```python
import numpy as np
from picocal_explorer.geometry import module_pitch, classify_by_pitch, derive_cell_geometry

def test_module_pitch_grid():
    pts = np.array([[0,0],[15,0],[0,15],[15,15]], float)
    assert abs(module_pitch(pts) - 15.0) < 1e-6

def test_module_pitch_single():
    assert np.isnan(module_pitch(np.array([[1.0,2.0]])))

def test_classify_by_pitch():
    assert classify_by_pitch(15.1) == "SpaCal-64-WGAGG"
    assert classify_by_pitch(29.0) == "SpaCal-16-PbPoly"
    assert classify_by_pitch(41.0) == "Shashlik-9"
    assert classify_by_pitch(59.0) == "Shashlik-4"
    assert classify_by_pitch(float("nan")) == "Shashlik-1"

def test_derive_cell_geometry_rel_and_seed():
    out = derive_cell_geometry(
        imodx=np.array([0,0,0]), jmody=np.array([0,0,0]),
        cell_x=np.array([0.0,15.0,30.0]), cell_y=np.array([0.0,0.0,0.0]),
        module_map={(0,0): {"pitch":15.0,"modtype":"SpaCal-64-WGAGG"}},
        seed_index=1,
    )
    assert out["is_seed"].tolist() == [False, True, False]
    assert out["rel_x"].tolist() == [-15.0, 0.0, 15.0]
    assert out["rel_dr"][0] == 15.0
    assert out["modtype"][0] == "SpaCal-64-WGAGG"
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement**
```python
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import awkward as ak
import uproot

TREE = "clusters_matched"
_PITCH_TABLE = [(15.0, "SpaCal-64-WGAGG"), (30.0, "SpaCal-16-PbPoly"),
                (40.0, "Shashlik-9"), (60.0, "Shashlik-4")]

def module_pitch(positions):
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return float("nan")
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    d[d == 0] = np.inf
    return float(np.median(np.min(d, axis=1)))

def classify_by_pitch(pitch):
    if not np.isfinite(pitch):
        return "Shashlik-1"
    return min(_PITCH_TABLE, key=lambda t: abs(t[0] - pitch))[1]

def build_module_map(files):
    acc = defaultdict(list)
    for path in files:
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as f:
            a = f[TREE].arrays(["imodx", "jmody", "cell_x", "cell_y"], library="ak")
        for ix, iy, cx, cy in zip(a["imodx"], a["jmody"], a["cell_x"], a["cell_y"]):
            ix = np.asarray(ix); iy = np.asarray(iy)
            cx = np.asarray(cx); cy = np.asarray(cy)
            for key in np.unique(np.stack([ix, iy], 1), axis=0):
                m = (ix == key[0]) & (iy == key[1])
                acc[(int(key[0]), int(key[1]))].append(np.stack([cx[m], cy[m]], 1))
    out = {}
    for key, chunks in acc.items():
        pts = np.unique(np.concatenate(chunks, 0), axis=0)
        pitch = module_pitch(pts)
        out[key] = {"x": float(pts[:, 0].mean()), "y": float(pts[:, 1].mean()),
                    "pitch": pitch, "modtype": classify_by_pitch(pitch),
                    "n_cells": int(len(pts))}
    return out

def derive_cell_geometry(imodx, jmody, cell_x, cell_y, module_map, seed_index):
    n = len(cell_x)
    cell_x = np.asarray(cell_x, float); cell_y = np.asarray(cell_y, float)
    pitch = np.full(n, np.nan); modtype = np.empty(n, dtype=object)
    for i in range(n):
        info = module_map.get((int(imodx[i]), int(jmody[i])))
        if info is not None:
            pitch[i] = info["pitch"]; modtype[i] = info["modtype"]
        else:
            modtype[i] = "unknown"
    xs, ys = cell_x[seed_index], cell_y[seed_index]
    rel_x = cell_x - xs; rel_y = cell_y - ys
    is_seed = np.zeros(n, bool); is_seed[seed_index] = True
    return {"pitch": pitch, "modtype": modtype, "rel_x": rel_x, "rel_y": rel_y,
            "rel_dr": np.sqrt(rel_x ** 2 + rel_y ** 2), "is_seed": is_seed}

def load_or_build_module_map(data_root, cache):
    cache = Path(cache)
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {tuple(int(v) for v in k.split(",")): val for k, val in raw.items()}
    files = sorted((data_root / "full").glob("matched_*.root"))
    mm = build_module_map(files)
    cache.write_text(json.dumps({f"{k[0]},{k[1]}": v for k, v in mm.items()}))
    return mm
```
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(explorer): geometry derivation + module map"`

---

### Task 3: ROOT reading + event assembly (`data.py`)

**Files:**
- Create: `src/picocal_explorer/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `geometry.derive_cell_geometry`, `geometry.load_or_build_module_map`.
- Produces:
  - `DATA_ROOT: Path`, `list_files() -> list[dict]` (`{name,dataset,n_entries}`).
  - `file_overview(name) -> dict` (`{n_events, entries:[{tree_entry,event,sig_flux_eTot,pdgID,sig_dr_matched,x_cluster,y_cluster,multiplicity}]}`).
  - `event_detail(name, event_id) -> dict` matching the spec §4 shape (truth_photons, clusters w/ cells incl. derived fields, window_modules).
  - Cell energies stay MeV; truth energy exposed as `energy_gev`.
  - Seed = `int(np.nanargmax(energy))` per distinct cluster.
  - Distinct cluster dedup key = `(x_cluster, y_cluster, tuple(cell_x), tuple(cell_y), tuple(energy))` (as in mentor notebook).

- [ ] **Step 1: Write failing tests** (run against the real first file)
```python
from picocal_explorer import data

def test_list_files_has_full():
    files = data.list_files()
    assert any(f["name"] == "matched_1001_1010.root" for f in files)

def test_event_detail_event4_three_photons_one_cluster():
    d = data.event_detail("matched_1001_1010.root", 4)
    assert len(d["truth_photons"]) == 3
    assert len(d["clusters"]) == 1
    c = d["clusters"][0]
    assert c["n_cells"] == len(c["cells"])
    assert any(cell["is_seed"] for cell in c["cells"])
    assert "pitch_derived" in c["cells"][0]

def test_truth_energy_is_gev():
    d = data.event_detail("matched_1001_1010.root", 4)
    assert max(p["energy_gev"] for p in d["truth_photons"]) < 1000
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — `data.py`:
```python
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
from .geometry import derive_cell_geometry, load_or_build_module_map

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
TREE = "clusters_matched"
_CACHE = DATA_ROOT / ".geometry_cache.json"

_SCALAR = ["event", "sig_flux_eTot", "sig_flux_pdgID", "sig_dr_matched",
           "x_cluster", "y_cluster"]
_DETAIL_SCALAR = _SCALAR + ["sig_flux_entry_x", "sig_flux_entry_y", "sig_flux_entry_z",
    "sig_flux_px", "sig_flux_py", "sig_flux_pz", "sig_dxdz_flux", "sig_dydz_flux",
    "sig_flux_timing", "total_energy", "total_energy_front", "total_energy_back"]
_CELL = ["imodx", "jmody", "icell", "cell_x", "cell_y", "cell_energies_front",
         "cell_energies_back", "energy", "cell_times_front", "cell_times_back"]

def _dataset_dirs():
    return {"full": DATA_ROOT / "full",
            "with_minimum_bias": DATA_ROOT / "gsoc_drive" / "with_minimum_bias",
            "without_minimum_bias": DATA_ROOT / "gsoc_drive" / "without_minimum_bias"}

def _find(name):
    for d in _dataset_dirs().values():
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(name)

def module_map():
    return load_or_build_module_map(DATA_ROOT, _CACHE)

def list_files():
    out = []
    for dataset, d in _dataset_dirs().items():
        for p in sorted(d.glob("matched_*.root")):
            with uproot.open(p) as f:
                out.append({"name": p.name, "dataset": dataset,
                            "n_entries": int(f[TREE].num_entries)})
    return out

@lru_cache(maxsize=8)
def file_overview(name):
    with uproot.open(_find(name)) as f:
        a = f[TREE].arrays(_SCALAR, library="ak")
    ev = ak.to_numpy(a["event"])
    keys = list(zip(ev.tolist(), ak.to_numpy(a["x_cluster"]).tolist(),
                    ak.to_numpy(a["y_cluster"]).tolist()))
    from collections import Counter
    counts = Counter(keys)
    entries = [{"tree_entry": i, "event": int(ev[i]),
                "sig_flux_eTot": float(a["sig_flux_eTot"][i]),
                "pdgID": int(a["sig_flux_pdgID"][i]),
                "sig_dr_matched": float(a["sig_dr_matched"][i]),
                "x_cluster": float(a["x_cluster"][i]), "y_cluster": float(a["y_cluster"][i]),
                "multiplicity": counts[keys[i]]} for i in range(len(ev))]
    return {"n_events": int(len(set(ev.tolist()))), "entries": entries}

@lru_cache(maxsize=64)
def event_detail(name, event_id):
    path = _find(name)
    with uproot.open(path) as f:
        tree = f[TREE]
        ev_all = ak.to_numpy(tree.arrays(["event"], library="ak")["event"])
        idx = np.flatnonzero(ev_all == event_id)
        start, stop = int(idx.min()), int(idx.max()) + 1
        block = tree.arrays(_DETAIL_SCALAR + _CELL, entry_start=start, entry_stop=stop,
                            library="ak")
    mask = ak.to_numpy(block["event"]) == event_id
    block = block[mask]
    tree_entries = np.arange(start, stop)[mask]
    mm = module_map()

    truth = [{"tree_entry": int(tree_entries[i]),
              "energy_gev": float(block["sig_flux_eTot"][i]),
              "entry_x": float(block["sig_flux_entry_x"][i]),
              "entry_y": float(block["sig_flux_entry_y"][i]),
              "entry_z": float(block["sig_flux_entry_z"][i]),
              "px": float(block["sig_flux_px"][i]), "py": float(block["sig_flux_py"][i]),
              "pz": float(block["sig_flux_pz"][i]),
              "dxdz": float(block["sig_dxdz_flux"][i]), "dydz": float(block["sig_dydz_flux"][i]),
              "dr_matched": float(block["sig_dr_matched"][i]),
              "pdgID": int(block["sig_flux_pdgID"][i])} for i in range(len(block))]

    seen, clusters = {}, []
    for i in range(len(block)):
        e = ak.to_numpy(block["energy"][i])
        cx = ak.to_numpy(block["cell_x"][i]); cy = ak.to_numpy(block["cell_y"][i])
        key = (float(block["x_cluster"][i]), float(block["y_cluster"][i]),
               tuple(cx.tolist()), tuple(cy.tolist()), tuple(e.tolist()))
        if key in seen:
            continue
        seen[key] = True
        ix = ak.to_numpy(block["imodx"][i]); iy = ak.to_numpy(block["jmody"][i])
        ic = ak.to_numpy(block["icell"][i])
        ef = ak.to_numpy(block["cell_energies_front"][i])
        eb = ak.to_numpy(block["cell_energies_back"][i])
        tf = ak.to_numpy(block["cell_times_front"][i])
        tb = ak.to_numpy(block["cell_times_back"][i])
        seed = int(np.nanargmax(e)) if len(e) else 0
        g = derive_cell_geometry(ix, iy, cx, cy, mm, seed)
        cells = [{"imodx": int(ix[j]), "jmody": int(iy[j]), "icell": int(ic[j]),
                  "x": float(cx[j]), "y": float(cy[j]), "front": float(ef[j]),
                  "back": float(eb[j]), "energy": float(e[j]),
                  "t_front": float(tf[j]), "t_back": float(tb[j]),
                  "pitch_derived": (None if not np.isfinite(g["pitch"][j]) else float(g["pitch"][j])),
                  "modtype_derived": str(g["modtype"][j]),
                  "rel_x": float(g["rel_x"][j]), "rel_y": float(g["rel_y"][j]),
                  "rel_dr": float(g["rel_dr"][j]), "is_seed": bool(g["is_seed"][j])}
                 for j in range(len(e))]
        clusters.append({"x_cluster": float(block["x_cluster"][i]),
                         "y_cluster": float(block["y_cluster"][i]),
                         "total_energy": float(block["total_energy"][i]),
                         "total_energy_front": float(block["total_energy_front"][i]),
                         "total_energy_back": float(block["total_energy_back"][i]),
                         "n_cells": len(cells), "seed_index": seed, "cells": cells})

    wm = {}
    for c in clusters:
        for cell in c["cells"]:
            k = (cell["imodx"], cell["jmody"])
            if k not in wm:
                info = mm.get(k, {})
                wm[k] = {"imodx": k[0], "jmody": k[1], "x": info.get("x"),
                         "y": info.get("y"), "pitch": info.get("pitch"),
                         "modtype": info.get("modtype"), "n_cells": info.get("n_cells")}
    return {"event": int(event_id), "truth_photons": truth, "clusters": clusters,
            "window_modules": list(wm.values())}
```
- [ ] **Step 4: Run, expect pass.** (first run builds the geometry cache — may take ~1–2 min)
- [ ] **Step 5: Commit** — `git commit -am "feat(explorer): ROOT reading + event assembly"`

---

### Task 4: Distributions (`analysis.py`)

**Files:**
- Create: `src/picocal_explorer/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Produces: `distributions(name) -> dict` with keys `truth_energy_gev` (values list), `n_cells` (list), `response` (ΔE/E list, `(total_energy - eTot*1000)/(eTot*1000)`), `dr_cluster_truth` (list), and `meta` (`{n_entries}`). Plain value lists; the frontend bins them.

- [ ] **Step 1: Write failing test**
```python
from picocal_explorer import analysis
def test_distributions_shapes():
    d = analysis.distributions("matched_1001_1010.root")
    assert len(d["truth_energy_gev"]) == d["meta"]["n_entries"]
    assert len(d["response"]) == d["meta"]["n_entries"]
    assert all(v < 1000 for v in d["truth_energy_gev"])
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement**
```python
from __future__ import annotations
import numpy as np
import awkward as ak
import uproot
from .data import _find, TREE

def distributions(name):
    with uproot.open(_find(name)) as f:
        a = f[TREE].arrays(["sig_flux_eTot", "total_energy", "sig_flux_entry_x",
            "sig_flux_entry_y", "x_cluster", "y_cluster", "energy"], library="ak")
    eTot = ak.to_numpy(a["sig_flux_eTot"]).astype(float)
    reco = ak.to_numpy(a["total_energy"]).astype(float)
    true_mev = eTot * 1000.0
    resp = np.where(true_mev != 0, (reco - true_mev) / true_mev, np.nan)
    dx = ak.to_numpy(a["x_cluster"]).astype(float) - ak.to_numpy(a["sig_flux_entry_x"]).astype(float)
    dy = ak.to_numpy(a["y_cluster"]).astype(float) - ak.to_numpy(a["sig_flux_entry_y"]).astype(float)
    dr = np.sqrt(dx ** 2 + dy ** 2)
    ncells = ak.to_numpy(ak.num(a["energy"], axis=1))
    clean = lambda v: [float(x) for x in v if np.isfinite(x)]
    return {"truth_energy_gev": clean(eTot), "n_cells": [int(x) for x in ncells],
            "response": clean(resp), "dr_cluster_truth": clean(dr),
            "meta": {"n_entries": int(len(eTot))}}
```
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(explorer): dataset distributions"`

---

### Task 5: Dictionary + live schema (`dictionary.py`)

**Files:**
- Create: `src/picocal_explorer/dictionary.py`
- Test: `tests/test_dictionary.py`

**Interfaces:**
- Produces:
  - `FIELDS: list[dict]` — each `{name, level, kind, unit, formula, present, source, text:{what:{th,en}}, note:{th,en}}` where `present in {"root","derived"}`.
  - `dictionary() -> list[dict]` (returns FIELDS).
  - `live_schema(name) -> list[dict]` — `{name, typename, kind('scalar'|'jagged'), sample_min, sample_max}` from uproot, joined with FIELDS by name.

- [ ] **Step 1: Write failing test**
```python
from picocal_explorer import dictionary
def test_dictionary_marks_derived_and_units():
    by = {f["name"]: f for f in dictionary.dictionary()}
    assert by["cell_pitch"]["present"] == "derived"
    assert by["sig_flux_eTot"]["unit"] == "GeV"
    assert by["energy"]["present"] == "root"

def test_live_schema_real_file():
    rows = {r["name"]: r for r in dictionary.live_schema("matched_1001_1010.root")}
    assert rows["energy"]["kind"] == "jagged"
    assert rows["sig_flux_eTot"]["kind"] == "scalar"
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — populate `FIELDS` with every branch from §7 of the spec plus the 5 derived fields. Each entry bilingual. Example shape:
```python
FIELDS = [
  {"name":"sig_flux_eTot","level":"truth","kind":"scalar","unit":"GeV","formula":"",
   "present":"root","source":"ROOT",
   "text":{"what":{"th":"พลังงานจริงของโฟตอน (เป้าหมาย regression)",
                   "en":"True photon energy (regression target)"}},
   "note":{"th":"instruction บอก MeV แต่จริงเป็น GeV (×1000 ก่อนเทียบ reco)",
           "en":"instruction says MeV; actually GeV (×1000 before comparing to reco)"}},
  {"name":"cell_pitch","level":"cell","kind":"scalar-per-cell","unit":"mm","formula":"median nn-distance",
   "present":"derived","source":"derived",
   "text":{"what":{"th":"ขนาดเซลล์ (อนุมานจากระยะเพื่อนบ้าน)",
                   "en":"cell size (derived from neighbour spacing)"}},
   "note":{"th":"ไม่มีในไฟล์ ROOT","en":"not stored in ROOT"}},
  # ... one entry per field in spec §7 + cell_modType, cell_rel_x, cell_rel_y, cell_rel_dr
]
def dictionary():
    return FIELDS
def live_schema(name):
    from .data import _find, TREE
    import uproot, numpy as np, awkward as ak
    by = {f["name"]: f for f in FIELDS}
    rows = []
    with uproot.open(_find(name)) as f:
        t = f[TREE]
        for n in t.keys():
            tn = t[n].typename
            jag = tn.endswith("[]")
            smin = smax = None
            if not jag:
                v = ak.to_numpy(t[n].array(library="ak"))
                v = v[np.isfinite(v)] if v.dtype.kind == "f" else v
                if len(v): smin, smax = float(np.min(v)), float(np.max(v))
            meta = by.get(n, {})
            rows.append({"name": n, "typename": tn, "kind": "jagged" if jag else "scalar",
                         "sample_min": smin, "sample_max": smax, "meta": meta})
    return rows
```
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(explorer): data dictionary + live schema (bilingual)"`

---

### Task 6: Explainers content (`explainers.py`)

**Files:**
- Create: `src/picocal_explorer/explainers.py`
- Test: `tests/test_explainers.py`

**Interfaces:**
- Produces: `EXPLAINERS: dict` with `plots: list` and `formulas: list`, each item `{id, title:{th,en}, formula_latex, what:{th,en}, why:{th,en}, ml_analogy:{th,en}}`; `explainers() -> EXPLAINERS`.
- Content covers (ids): `cell_multiplicity`, `truth_spectrum`, `response`, `dr`, `seed`, `efficiency` (plots); `delta_e_over_e`, `resolution`, `log_spectrum`, `dr_formula`, `why_eda`, `why_transformer`, `why_no_inverse` (formulas/concepts). Text drawn from the chat explanation already given.

- [ ] **Step 1: Write failing test**
```python
from picocal_explorer import explainers
def test_explainers_bilingual_and_ids():
    e = explainers.explainers()
    ids = {p["id"] for p in e["plots"]} | {f["id"] for f in e["formulas"]}
    assert {"response","truth_spectrum","resolution","why_transformer"} <= ids
    r = next(f for f in e["formulas"] if f["id"] == "resolution")
    assert r["formula_latex"] and r["what"]["th"] and r["what"]["en"]
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `EXPLAINERS` from the chat teaching (ΔE/E, σ_E/E quadrature, log spectrum power law, dr, sequence length, why-EDA, why-transformer, why-no-inverse), each bilingual with `formula_latex`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(explorer): bilingual explainers content"`

---

### Task 7: FastAPI app + endpoints (`app.py`)

**Files:**
- Create: `src/picocal_explorer/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `data`, `analysis`, `dictionary`, `explainers`.
- Produces: FastAPI `app`; routes `/api/files`, `/api/files/{name}/overview`, `/api/files/{name}/event/{event_id}`, `/api/files/{name}/distributions`, `/api/files/{name}/schema`, `/api/dictionary`, `/api/explainers`, `/api/geometry`; static mount at `/` serving `static/`.

- [ ] **Step 1: Write failing test**
```python
from fastapi.testclient import TestClient
from picocal_explorer.app import app
client = TestClient(app)

def test_files_endpoint():
    r = client.get("/api/files")
    assert r.status_code == 200 and len(r.json()) >= 100

def test_event_endpoint():
    r = client.get("/api/files/matched_1001_1010.root/event/4")
    j = r.json()
    assert len(j["truth_photons"]) == 3 and len(j["clusters"]) == 1

def test_explainers_endpoint():
    assert client.get("/api/explainers").status_code == 200
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — wire endpoints to the modules; `StaticFiles(directory=.../static, html=True)` mounted last; `@app.on_event("startup")` calls `data.module_map()` to warm the cache.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(explorer): FastAPI endpoints"`

---

### Task 8: Frontend shell + i18n + fetch (`static/index.html`, `styles.css`, `app.js`)

**Files:** Create `static/index.html`, `static/styles.css`, `static/app.js`.

**Interfaces:**
- Produces (JS globals/modules): `I18N` lang state (`'th'|'en'`, default `'th'`), `t(obj)` returns `obj[lang]`, `api(path)` fetch helper, tab router rendering six panels, `renderMath(el)` via KaTeX auto-render.
- index.html loads D3 + KaTeX from CDN, has a header with language toggle + file picker + six tab buttons and six empty `<section>` containers.

- [ ] **Step 1:** Build the shell: header (TH/ENG toggle, file `<select>` populated from `/api/files`), tab nav, six sections. CSS: detector-dark theme, responsive, monospace numbers.
- [ ] **Step 2: Verify** — `uv run uvicorn picocal_explorer.app:app` then load `http://localhost:8000`; file picker lists files; toggling lang flips static labels. (Playwright smoke check optional.)
- [ ] **Step 3: Commit** — `git commit -am "feat(explorer): frontend shell + i18n"`

---

### Task 9: Overview view

**Files:** Modify `static/app.js`.
- [ ] D3 scatter of `/overview` entries (x=event, y=sig_flux_eTot, color=multiplicity); hover tooltip; click → set current event + switch to sensor view. Bilingual axis labels via `t()`.
- [ ] **Verify** loading a file renders the scatter; click jumps to sensor view. Commit.

---

### Task 10: Detector / sensor view (primary)

**Files:** Modify `static/app.js`, `static/styles.css`.
- [ ] Left panel: full ECAL face from `/api/geometry` (modules as faint squares at module x/y, sized by pitch), highlight current `window_modules`. Right panel: zoom — each cell a square of side=`pitch_derived` at (x,y), fill=energy (viridis), seed ★, truth × (one per photon), reco +. Toggles: front/back/total, linear/log, timing. Hover cell → side panel listing every per-cell field + `t()` explainer text from `/api/dictionary`.
- [ ] **Verify** event 4 shows 1 cluster, 3 truth ×; cells sized by pitch; toggles work. Commit.

---

### Task 11: Shower depth view

**Files:** Modify `static/app.js`.
- [ ] Per-cell front-vs-back scatter + stacked front/back bar by cell; timing histogram. Bilingual labels + explainer note. **Verify** + commit.

---

### Task 12: Data dictionary view

**Files:** Modify `static/app.js`.
- [ ] Searchable table from `/api/dictionary` + `/api/files/{name}/schema`: columns name, level, unit, present (root/derived badge), what (`t()`), note (`t()`), sample range. Render formulas with KaTeX. **Verify** `cell_pitch` shows "derived" badge; `sig_flux_eTot` shows GeV note. Commit.

---

### Task 13: Guided tour view

**Files:** Modify `static/app.js`.
- [ ] Stepper (photon→flies→face→shower→front/back→cluster→truth-match→entry); each step bilingual text + a button linking to the relevant view/event; includes the four discrepancies. **Verify** + commit.

---

### Task 14: Exploration & Formulas view

**Files:** Modify `static/app.js`.
- [ ] From `/api/files/{name}/distributions` build D3 histograms: truth-energy spectrum (linear/log toggle), n_cells (=sequence length), ΔE/E response, dr. Each chart paired with its `/api/explainers` panel (what/why/formula KaTeX/ml_analogy via `t()`). Top note: "why EDA even with a guideline" + the `log E ⟺ relative error ⟺ σ_E/E` through-line. **Verify** charts render + formulas typeset. Commit.

---

### Task 15: Physics primer + run docs

**Files:** Create `docs/physics-primer.md`, `scripts/run_explorer.py`; modify `README.md`.
- [ ] Write `docs/physics-primer.md` (Thai-primary, English terms) following spec §8 items 1–9, with the real formulas. `scripts/run_explorer.py` launches uvicorn and opens the browser. README: add a "Data Explorer" section with `uv run uvicorn picocal_explorer.app:app`. **Verify** primer renders; `python scripts/run_explorer.py` serves the app. Commit.

---

## Self-Review

- **Spec coverage:** §3 architecture → Tasks 1,7,8; §4 backend/endpoints → Tasks 2–7; §5 views 1–6 → Tasks 9–14; §6 derived fields → Task 2; §7 dictionary/discrepancies → Task 5; §8 primer → Task 15; §11 tests → each backend task; bilingual → Tasks 5,6,8+. No gaps.
- **Placeholder scan:** backend Tasks 1–7 contain complete code; content Tasks 5,6 specify exact structure + examples; frontend Tasks 8–14 specify exact endpoints, inputs, and acceptance checks (rendering code authored at execution against the frozen API contract — no hidden interfaces).
- **Type consistency:** `derive_cell_geometry` keys (`pitch,modtype,rel_x,rel_y,rel_dr,is_seed`) match Task 3 usage; event_detail cell keys (`pitch_derived,modtype_derived,...`) match Tasks 5/10; endpoint paths consistent between Task 7 and Tasks 9–14.
