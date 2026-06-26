from __future__ import annotations

import awkward as ak
import numpy as np
import uproot

from .data import TREE, _find


def distributions(name):
    with uproot.open(_find(name)) as f:
        a = f[TREE].arrays(
            [
                "sig_flux_eTot",
                "total_energy",
                "sig_flux_entry_x",
                "sig_flux_entry_y",
                "x_cluster",
                "y_cluster",
                "energy",
            ],
            library="ak",
        )
    eTot = ak.to_numpy(a["sig_flux_eTot"]).astype(float)
    reco = ak.to_numpy(a["total_energy"]).astype(float)
    true_mev = eTot * 1000.0
    resp = np.where(true_mev != 0, (reco - true_mev) / true_mev, np.nan)
    dx = ak.to_numpy(a["x_cluster"]).astype(float) - ak.to_numpy(a["sig_flux_entry_x"]).astype(float)
    dy = ak.to_numpy(a["y_cluster"]).astype(float) - ak.to_numpy(a["sig_flux_entry_y"]).astype(float)
    dr = np.sqrt(dx ** 2 + dy ** 2)
    ncells = ak.to_numpy(ak.num(a["energy"], axis=1))

    def clean(v):
        return [float(x) for x in v if np.isfinite(x)]

    return {
        "truth_energy_gev": clean(eTot),
        "n_cells": [int(x) for x in ncells],
        "response": clean(resp),
        "dr_cluster_truth": clean(dr),
        "meta": {"n_entries": int(len(eTot))},
    }
