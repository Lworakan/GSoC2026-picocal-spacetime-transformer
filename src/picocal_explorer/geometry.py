from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

TREE = "clusters_matched"

_PITCH_TABLE = [
    (15.0, "SpaCal-W (1.5 cm)"),
    (30.0, "SpaCal-W (3 cm)"),
    (40.0, "Shashlik (4 cm)"),
    (60.0, "SpaCal-Pb (6 cm)"),
    (120.0, "Shashlik (12 cm)"),
]


def module_pitch(positions):
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return float("nan")
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    d[d == 0] = np.inf
    return float(np.median(np.min(d, axis=1)))


def classify_by_pitch(pitch):
    if not np.isfinite(pitch):
        return "unknown"
    return min(_PITCH_TABLE, key=lambda t: abs(t[0] - pitch))[1]


def build_module_map(files):
    frames = []
    for path in files:
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as f:
            a = f[TREE].arrays(["imodx", "jmody", "cell_x", "cell_y"], library="ak")
        frames.append(
            pd.DataFrame(
                {
                    "ix": ak.to_numpy(ak.flatten(a["imodx"])).astype(np.int64),
                    "iy": ak.to_numpy(ak.flatten(a["jmody"])).astype(np.int64),
                    "x": ak.to_numpy(ak.flatten(a["cell_x"])).astype(float),
                    "y": ak.to_numpy(ak.flatten(a["cell_y"])).astype(float),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["ix", "iy", "x", "y"])
    out = {}
    for (a, b), g in df.groupby(["ix", "iy"], sort=False):
        pts = g[["x", "y"]].to_numpy()
        out[(int(a), int(b))] = {
            "x": float(pts[:, 0].mean()),
            "y": float(pts[:, 1].mean()),
            "pitch": module_pitch(pts),
            "n_cells": int(len(pts)),
        }
    keys = list(out.keys())
    centers = np.array([[out[k]["x"], out[k]["y"]] for k in keys])
    for i, k in enumerate(keys):
        if not np.isfinite(out[k]["pitch"]) and len(centers) > 1:
            d = np.sqrt(((centers - centers[i]) ** 2).sum(1))
            d[i] = np.inf
            out[k]["pitch"] = float(np.min(d))
    for k in keys:
        out[k]["modtype"] = classify_by_pitch(out[k]["pitch"])
    return out


def derive_cell_geometry(imodx, jmody, cell_x, cell_y, module_map, seed_index):
    cell_x = np.asarray(cell_x, dtype=float)
    cell_y = np.asarray(cell_y, dtype=float)
    n = len(cell_x)
    if n == 0:
        empty = np.array([])
        return {
            "pitch": empty,
            "modtype": np.array([], dtype=object),
            "rel_x": empty,
            "rel_y": empty,
            "rel_dr": empty,
            "is_seed": np.array([], dtype=bool),
        }
    pitch = np.full(n, np.nan)
    modtype = np.empty(n, dtype=object)
    for i in range(n):
        info = module_map.get((int(imodx[i]), int(jmody[i])))
        if info is not None:
            pitch[i] = info["pitch"]
            modtype[i] = info["modtype"]
        else:
            modtype[i] = "unknown"
    xs, ys = cell_x[seed_index], cell_y[seed_index]
    rel_x = cell_x - xs
    rel_y = cell_y - ys
    is_seed = np.zeros(n, dtype=bool)
    is_seed[seed_index] = True
    return {
        "pitch": pitch,
        "modtype": modtype,
        "rel_x": rel_x,
        "rel_y": rel_y,
        "rel_dr": np.sqrt(rel_x ** 2 + rel_y ** 2),
        "is_seed": is_seed,
    }


def load_or_build_module_map(data_root, cache):
    cache = Path(cache)
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {tuple(int(v) for v in k.split(",")): val for k, val in raw.items()}
    files = sorted((Path(data_root) / "full").glob("matched_*.root"))[:15]
    mm = build_module_map(files)
    cache.write_text(json.dumps({f"{k[0]},{k[1]}": v for k, v in mm.items()}))
    return mm
