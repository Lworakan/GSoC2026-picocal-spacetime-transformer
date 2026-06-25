from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .geometry import derive_cell_geometry, load_or_build_module_map

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
TREE = "clusters_matched"
_CACHE = DATA_ROOT / ".geometry_cache.json"

_SCALAR = [
    "event",
    "sig_flux_eTot",
    "sig_flux_pdgID",
    "sig_dr_matched",
    "x_cluster",
    "y_cluster",
]
_DETAIL_SCALAR = _SCALAR + [
    "sig_flux_entry_x",
    "sig_flux_entry_y",
    "sig_flux_entry_z",
    "sig_flux_prod_vertex_x",
    "sig_flux_prod_vertex_y",
    "sig_flux_prod_vertex_z",
    "sig_flux_px",
    "sig_flux_py",
    "sig_flux_pz",
    "sig_dxdz_flux",
    "sig_dydz_flux",
    "sig_flux_timing",
    "total_energy",
    "total_energy_front",
    "total_energy_back",
]
_CELL = [
    "imodx",
    "jmody",
    "icell",
    "cell_x",
    "cell_y",
    "cell_energies_front",
    "cell_energies_back",
    "energy",
    "cell_times_front",
    "cell_times_back",
]


def _safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_safe(v) for v in o]
    return o


def _dataset_dirs():
    return {
        "full": DATA_ROOT / "full",
        "with_minimum_bias": DATA_ROOT / "gsoc_drive" / "with_minimum_bias",
        "without_minimum_bias": DATA_ROOT / "gsoc_drive" / "without_minimum_bias",
    }


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
                out.append(
                    {"name": p.name, "dataset": dataset, "n_entries": int(f[TREE].num_entries)}
                )
    return out


@lru_cache(maxsize=8)
def file_overview(name):
    with uproot.open(_find(name)) as f:
        a = f[TREE].arrays(_SCALAR, library="ak")
    ev = ak.to_numpy(a["event"])
    xc = ak.to_numpy(a["x_cluster"])
    yc = ak.to_numpy(a["y_cluster"])
    keys = list(zip(ev.tolist(), xc.tolist(), yc.tolist()))
    counts = Counter(keys)
    eTot = ak.to_numpy(a["sig_flux_eTot"])
    pdg = ak.to_numpy(a["sig_flux_pdgID"])
    dr = ak.to_numpy(a["sig_dr_matched"])
    entries = [
        {
            "tree_entry": i,
            "event": int(ev[i]),
            "sig_flux_eTot": float(eTot[i]),
            "pdgID": int(pdg[i]),
            "sig_dr_matched": float(dr[i]),
            "x_cluster": float(xc[i]),
            "y_cluster": float(yc[i]),
            "multiplicity": counts[keys[i]],
        }
        for i in range(len(ev))
    ]
    return _safe({"n_events": int(len(set(ev.tolist()))), "entries": entries})


@lru_cache(maxsize=64)
def event_detail(name, event_id):
    path = _find(name)
    with uproot.open(path) as f:
        tree = f[TREE]
        ev_all = ak.to_numpy(tree.arrays(["event"], library="ak")["event"])
        idx = np.flatnonzero(ev_all == event_id)
        if len(idx) == 0:
            raise KeyError(event_id)
        start, stop = int(idx.min()), int(idx.max()) + 1
        block = tree.arrays(
            _DETAIL_SCALAR + _CELL, entry_start=start, entry_stop=stop, library="ak"
        )
    mask = ak.to_numpy(block["event"]) == event_id
    block = block[mask]
    tree_entries = np.arange(start, stop)[mask]
    mm = module_map()

    truth = [
        {
            "tree_entry": int(tree_entries[i]),
            "energy_gev": float(block["sig_flux_eTot"][i]),
            "entry_x": float(block["sig_flux_entry_x"][i]),
            "entry_y": float(block["sig_flux_entry_y"][i]),
            "entry_z": float(block["sig_flux_entry_z"][i]),
            "prod_x": float(block["sig_flux_prod_vertex_x"][i]),
            "prod_y": float(block["sig_flux_prod_vertex_y"][i]),
            "prod_z": float(block["sig_flux_prod_vertex_z"][i]),
            "px": float(block["sig_flux_px"][i]),
            "py": float(block["sig_flux_py"][i]),
            "pz": float(block["sig_flux_pz"][i]),
            "dxdz": float(block["sig_dxdz_flux"][i]),
            "dydz": float(block["sig_dydz_flux"][i]),
            "timing": float(block["sig_flux_timing"][i]),
            "dr_matched": float(block["sig_dr_matched"][i]),
            "pdgID": int(block["sig_flux_pdgID"][i]),
        }
        for i in range(len(block))
    ]

    seen = {}
    clusters = []
    for i in range(len(block)):
        e = ak.to_numpy(block["energy"][i])
        cx = ak.to_numpy(block["cell_x"][i])
        cy = ak.to_numpy(block["cell_y"][i])
        key = (
            float(block["x_cluster"][i]),
            float(block["y_cluster"][i]),
            tuple(cx.tolist()),
            tuple(cy.tolist()),
            tuple(e.tolist()),
        )
        if key in seen:
            continue
        seen[key] = True
        ix = ak.to_numpy(block["imodx"][i])
        iy = ak.to_numpy(block["jmody"][i])
        ic = ak.to_numpy(block["icell"][i])
        ef = ak.to_numpy(block["cell_energies_front"][i])
        eb = ak.to_numpy(block["cell_energies_back"][i])
        tf = ak.to_numpy(block["cell_times_front"][i])
        tb = ak.to_numpy(block["cell_times_back"][i])
        seed = int(np.nanargmax(e)) if len(e) else 0
        g = derive_cell_geometry(ix, iy, cx, cy, mm, seed)
        cells = [
            {
                "imodx": int(ix[j]),
                "jmody": int(iy[j]),
                "icell": int(ic[j]),
                "x": float(cx[j]),
                "y": float(cy[j]),
                "front": float(ef[j]),
                "back": float(eb[j]),
                "energy": float(e[j]),
                "t_front": float(tf[j]),
                "t_back": float(tb[j]),
                "pitch_derived": None if not np.isfinite(g["pitch"][j]) else float(g["pitch"][j]),
                "modtype_derived": str(g["modtype"][j]),
                "rel_x": float(g["rel_x"][j]),
                "rel_y": float(g["rel_y"][j]),
                "rel_dr": float(g["rel_dr"][j]),
                "is_seed": bool(g["is_seed"][j]),
            }
            for j in range(len(e))
        ]
        clusters.append(
            {
                "x_cluster": float(block["x_cluster"][i]),
                "y_cluster": float(block["y_cluster"][i]),
                "total_energy": float(block["total_energy"][i]),
                "total_energy_front": float(block["total_energy_front"][i]),
                "total_energy_back": float(block["total_energy_back"][i]),
                "n_cells": len(cells),
                "seed_index": seed,
                "cells": cells,
            }
        )

    wm = {}
    for c in clusters:
        for cell in c["cells"]:
            k = (cell["imodx"], cell["jmody"])
            if k not in wm:
                info = mm.get(k, {})
                wm[k] = {
                    "imodx": k[0],
                    "jmody": k[1],
                    "x": info.get("x"),
                    "y": info.get("y"),
                    "pitch": info.get("pitch"),
                    "modtype": info.get("modtype"),
                    "n_cells": info.get("n_cells"),
                }
    return _safe(
        {
            "event": int(event_id),
            "truth_photons": truth,
            "clusters": clusters,
            "window_modules": list(wm.values()),
        }
    )
