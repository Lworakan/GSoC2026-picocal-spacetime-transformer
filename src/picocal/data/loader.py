"""Loader for PicoCal Run5-like matched-cluster ROOT files.

Each entry of the ``clusters_matched`` tree is one reconstructed calorimeter
cluster (a 5x5 module window around the seed) matched to one true photon.

Empirical facts about this production (verified on data, 2026-06; see
``notebooks/01_exploration_matched_clusters.ipynb``):

- Cell arrays are stored *jagged* (variable length), and the accompanying
  ``n*`` branches always equal the raw array length — they do NOT mark
  zero-padding. Cells with ``energy == 0`` are real window cells with no
  recorded deposit, not array padding.
- ``sig_flux_eTot`` is in **GeV** while reconstructed energies are in MeV
  (the dataset spec sheet says MeV for both; the data says otherwise).
- ``cell_times_*`` contain uninitialised garbage (|t| up to ~1e38 ns) in
  cells without a deposit; times are only meaningful where energy > 0.
"""

from __future__ import annotations

import glob
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

TREE_NAME = "clusters_matched"

# True-photon (label) branches.
BRANCHES_TRUTH = [
    "sig_flux_eTot",
    "sig_flux_entry_x",
    "sig_flux_entry_y",
    "sig_flux_entry_z",
    "sig_flux_px",
    "sig_flux_py",
    "sig_flux_pz",
    "sig_flux_pdgID",
    "sig_flux_timing",
    "sig_dxdz_flux",
    "sig_dydz_flux",
    "sig_dr_matched",
]

# Reconstructed cluster-level scalar branches.
BRANCHES_CLUSTER = [
    "event",
    "x_cluster",
    "y_cluster",
    "total_energy",
    "total_energy_front",
    "total_energy_back",
    "seed_icell_x",
    "seed_icell_y",
]

# Per-cell (token) branches.
BRANCHES_CELL = [
    "imodx",
    "jmody",
    "icell",
    "cell_x",
    "cell_y",
    "energy",
    "cell_energies_front",
    "cell_energies_back",
    "cell_times_front",
    "cell_times_back",
]

DEFAULT_BRANCHES = BRANCHES_TRUTH + BRANCHES_CLUSTER + BRANCHES_CELL

# Times outside this window (ns) are treated as uninitialised garbage.
TIME_SANITY_NS = 1e4


def load_clusters(
    path: str | Path | list[str | Path],
    branches: list[str] | None = None,
    tree: str = TREE_NAME,
    max_files: int | None = None,
) -> ak.Array:
    """Load matched clusters from one or more ROOT files into awkward arrays.

    ``path`` may be a single file, a directory (all ``matched_*.root`` inside),
    a glob pattern, or a list of files. Branches missing from the files are
    silently dropped so older/newer productions stay loadable.
    """
    files = _resolve_files(path)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No ROOT files found for {path!r}")

    if branches is None:
        branches = DEFAULT_BRANCHES

    with uproot.open(files[0]) as f:
        available = set(f[tree].keys())
    kept = [b for b in branches if b in available]

    parts = []
    for fname in files:
        with uproot.open(fname) as f:
            parts.append(f[tree].arrays(kept, library="ak"))
    return parts[0] if len(parts) == 1 else ak.concatenate(parts)


def _resolve_files(path: str | Path | list[str | Path]) -> list[str]:
    if isinstance(path, (list, tuple)):
        return [str(p) for p in path]
    p = Path(path)
    if p.is_dir():
        return sorted(str(q) for q in p.glob("matched_*.root"))
    if any(ch in str(path) for ch in "*?["):
        return sorted(glob.glob(str(path)))
    return [str(p)]


def valid_cell_mask(arrays: ak.Array, energy_branch: str = "energy") -> ak.Array:
    """Jagged boolean mask selecting cells that carry information.

    In this production a cell is informative iff it recorded energy: the
    ``n*`` branches equal the raw array length, so they cannot be used to
    strip anything. Zero-energy cells are kept out of the token set because
    every one of their stored quantities (including times) is meaningless.
    """
    return arrays[energy_branch] != 0


def clean_cell_times(times: ak.Array, mask: ak.Array) -> ak.Array:
    """Replace garbage / no-deposit time entries with ``np.nan``.

    A time is kept only if its cell passes ``mask`` and the value lies within
    ``±TIME_SANITY_NS`` — uninitialised entries show |t| up to ~1e38 ns.
    """
    ok = mask & (abs(times) < TIME_SANITY_NS)
    return ak.where(ok, times, np.nan)


def cluster_to_tokens(arrays: ak.Array) -> dict[str, ak.Array]:
    """Build the per-cell token features for every cluster.

    Returns a dict of jagged arrays, one entry per *informative* cell
    (energy > 0), aligned across keys. This is the precursor of the
    transformer input; padding/stacking to fixed length happens in the
    PyTorch collate step, not here.
    """
    mask = valid_cell_mask(arrays)
    tokens = {
        "x": arrays["cell_x"][mask],
        "y": arrays["cell_y"][mask],
        "e": arrays["energy"][mask],
        "e_front": arrays["cell_energies_front"][mask],
        "e_back": arrays["cell_energies_back"][mask],
        "t_front": clean_cell_times(arrays["cell_times_front"], mask)[mask],
        "t_back": clean_cell_times(arrays["cell_times_back"], mask)[mask],
    }
    tokens["n_cells"] = ak.num(tokens["e"], axis=1)
    return tokens


def truth_dataframe(arrays: ak.Array):
    """Cluster-level scalars as a pandas DataFrame for quick EDA.

    Adds derived columns:
    - ``e_true_mev``: true energy converted GeV -> MeV (unit fix),
    - ``n_cells`` / ``n_cells_nonzero``: window size and informative cells.
    """
    import pandas as pd

    scalars = [b for b in BRANCHES_TRUTH + BRANCHES_CLUSTER if b in arrays.fields]
    df = pd.DataFrame({b: ak.to_numpy(arrays[b]) for b in scalars})
    if "sig_flux_eTot" in df:
        df["e_true_mev"] = df["sig_flux_eTot"] * 1000.0
    if "energy" in arrays.fields:
        df["n_cells"] = ak.to_numpy(ak.num(arrays["energy"], axis=1))
        df["n_cells_nonzero"] = ak.to_numpy(ak.sum(arrays["energy"] != 0, axis=1))
    return df
