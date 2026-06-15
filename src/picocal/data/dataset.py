"""Batching layer: matched clusters -> fixed-shape padded tensors with a mask.

`loader.cluster_to_tokens` gives one variable-length set of cells per cluster.
A transformer needs those padded to a common length per batch, with a mask that
tells attention which positions are padding. This module builds the per-cell
feature matrix, splits clusters without leaking sibling photons across
train/val/test, standardises features using train statistics only, and collates
clusters into batch tensors.

Design notes (see docs/research-log/2026-06-week2.md):

- Clusters are stored CSR-style: one flat ``[N_total_cells, F]`` feature array
  plus an ``offsets`` array, so a cluster is a view, not a Python-level copy.
- Splits are keyed on the ``(file, event)`` pair. The same ``event`` index
  reappears for several photons of one event and also across files, so a split
  on ``event`` alone would put siblings on both sides. Whole ``(file, event)``
  groups go to exactly one split.
- Times are NaN where the cell time was garbage (see ``clean_cell_times``). The
  network never sees a NaN: the time feature is median-centred and filled with
  0, and a separate ``t_valid`` flag carries the "time was missing" information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np

from picocal.data.loader import _resolve_files, cluster_to_tokens, load_clusters

# Per-cell feature columns. Order is the contract with the model's input layer.
FEATURE_NAMES = [
    "cell_x",        # 0 transverse position [mm]
    "cell_y",        # 1 transverse position [mm]
    "log_energy",    # 2 log of cell energy [raw units]
    "front_fraction",  # 3 e_front / (e_front + e_back), longitudinal/depth coordinate
    "t_centered",    # 4 t_front minus the per-cluster median time [ns]
    "t_valid",       # 5 1.0 if the cell time was valid, else 0.0
]
N_FEATURES = len(FEATURE_NAMES)

# Columns that get standardised (mean 0 / std 1) using train statistics.
# front_fraction is already in [0, 1] and t_valid is a flag, so both pass through.
_STANDARDISE_COLS = (0, 1, 2, 4)


@dataclass
class ClusterFeatures:
    """CSR-style per-cell features plus per-cluster scalars.

    ``flat[offsets[i]:offsets[i + 1]]`` are the cells of cluster ``i``.
    """

    flat: np.ndarray            # [N_total_cells, N_FEATURES] float32
    offsets: np.ndarray         # [N_clusters + 1] int64
    target_gev: np.ndarray      # [N_clusters] float32, true photon energy [GeV]
    file_id: np.ndarray         # [N_clusters] int64
    event: np.ndarray           # [N_clusters] int64
    group: np.ndarray           # [N_clusters] int64, factorised (file, event)
    sig_dr_matched: np.ndarray  # [N_clusters] float32
    entry_x: np.ndarray         # [N_clusters] float32, true entry point [mm]
    entry_y: np.ndarray         # [N_clusters] float32
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def n_cells(self) -> np.ndarray:
        """Cells per cluster."""
        return np.diff(self.offsets)


def _scalar(arrays: ak.Array, name: str, n: int, default: float = 0.0) -> np.ndarray:
    """Per-cluster scalar as float, tolerant of branches missing in a production."""
    if name in arrays.fields:
        return ak.to_numpy(arrays[name]).astype(np.float64)
    return np.full(n, default, dtype=np.float64)


def _features_from_arrays(arrays: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """Build the flat feature matrix and offsets for one loaded array."""
    tokens = cluster_to_tokens(arrays)  # jagged, silent cells already dropped
    counts = ak.to_numpy(tokens["n_cells"]).astype(np.int64)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    def flat(name: str) -> np.ndarray:
        return ak.to_numpy(ak.flatten(tokens[name])).astype(np.float64)

    x = flat("x")
    y = flat("y")
    e = flat("e")
    e_front = flat("e_front")
    e_back = flat("e_back")
    t_front = flat("t_front")  # NaN where the time was garbage

    denom = e_front + e_back
    front_fraction = np.divide(
        e_front, denom, out=np.full_like(e_front, 0.5), where=denom > 0
    )

    # Median-centre the time within each cluster, ignoring NaN, then fill NaN.
    t_centered = np.zeros_like(t_front)
    t_valid = np.isfinite(t_front).astype(np.float64)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        seg = t_front[lo:hi]
        finite = np.isfinite(seg)
        if finite.any():
            med = np.median(seg[finite])
            centered = seg - med
            t_centered[lo:hi] = np.where(finite, centered, 0.0)

    feat = np.empty((len(x), N_FEATURES), dtype=np.float64)
    feat[:, 0] = x
    feat[:, 1] = y
    feat[:, 2] = np.log(e)  # tokens already drop energy == 0 cells
    feat[:, 3] = front_fraction
    feat[:, 4] = t_centered
    feat[:, 5] = t_valid
    return feat.astype(np.float32), offsets


def build_cluster_features(
    path: str | Path | list[str | Path],
    *,
    max_files: int | None = None,
) -> ClusterFeatures:
    """Load matched clusters file by file and build the batching-ready features.

    Files are loaded one at a time (not concatenated) so each cluster keeps a
    ``file_id``; the ``(file, event)`` pair is what the split is keyed on.
    """
    files = _resolve_files(path)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No ROOT files found for {path!r}")

    feats: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    targets, file_ids, events, drs, ex, ey = [], [], [], [], [], []
    for fid, fname in enumerate(files):
        arrays = load_clusters(fname)
        n = len(arrays)
        feat, offs = _features_from_arrays(arrays)
        feats.append(feat)
        counts.append(np.diff(offs))
        targets.append(_scalar(arrays, "sig_flux_eTot", n))
        events.append(_scalar(arrays, "event", n))
        drs.append(_scalar(arrays, "sig_dr_matched", n, default=np.inf))
        ex.append(_scalar(arrays, "sig_flux_entry_x", n))
        ey.append(_scalar(arrays, "sig_flux_entry_y", n))
        file_ids.append(np.full(n, fid, dtype=np.int64))

    flat = np.concatenate(feats, axis=0)
    all_counts = np.concatenate(counts)
    offsets = np.empty(len(all_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(all_counts, out=offsets[1:])

    file_id = np.concatenate(file_ids)
    event = np.concatenate(events).astype(np.int64)
    pairs = np.stack([file_id, event], axis=1)
    _, group = np.unique(pairs, axis=0, return_inverse=True)

    return ClusterFeatures(
        flat=flat,
        offsets=offsets,
        target_gev=np.concatenate(targets).astype(np.float32),
        file_id=file_id,
        event=event,
        group=group.astype(np.int64),
        sig_dr_matched=np.concatenate(drs).astype(np.float32),
        entry_x=np.concatenate(ex).astype(np.float32),
        entry_y=np.concatenate(ey).astype(np.float32),
    )


def make_event_splits(
    group: np.ndarray,
    fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split cluster indices into train/val/test by whole ``(file, event)`` group.

    A group never spans two splits, so sibling photons stay together. Seeded, so
    the same group array and seed reproduce the same partition.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"fracs must sum to 1, got {fracs} (sum {sum(fracs)})")
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uniq)
    n = len(shuffled)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    chunks = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    return {
        name: np.nonzero(np.isin(group, g))[0].astype(np.int64)
        for name, g in chunks.items()
    }


class FeatureScaler:
    """Standardise selected feature columns using train-split statistics only.

    Fit on train clusters, then ``transform`` is applied to every split, so val
    and test never contribute to the mean/std (no leakage).
    """

    def __init__(self, cols: tuple[int, ...] = _STANDARDISE_COLS) -> None:
        self.cols = cols
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, features: ClusterFeatures, train_idx: np.ndarray) -> FeatureScaler:
        rows = _rows_for_clusters(features.offsets, train_idx)
        sub = features.flat[rows][:, self.cols]
        self.mean = sub.mean(axis=0)
        std = sub.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)  # guard constant columns
        return self

    def transform(self, flat: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler.transform called before fit")
        out = flat.copy()
        out[:, self.cols] = (out[:, self.cols] - self.mean) / self.std
        return out

    def to_dict(self) -> dict:
        return {
            "cols": list(self.cols),
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
        }


def _rows_for_clusters(offsets: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Flat cell-row indices belonging to the given cluster indices."""
    return np.concatenate(
        [np.arange(offsets[i], offsets[i + 1]) for i in idx]
    ).astype(np.int64) if len(idx) else np.empty(0, dtype=np.int64)


def add_isolation_flag(
    features: ClusterFeatures, radius_mm: float = 200.0
) -> np.ndarray:
    """True for clusters whose photon has no sibling within ``radius_mm``.

    Distance is between true entry points of photons sharing a ``(file, event)``
    group. A lone photon in its group is isolated. Built as a carried label;
    selecting on it for training is a later, mentor-dependent decision.
    """
    n = len(features)
    isolated = np.ones(n, dtype=bool)
    order = np.argsort(features.group, kind="stable")
    g = features.group[order]
    boundaries = np.nonzero(np.diff(g))[0] + 1
    for members in np.split(order, boundaries):
        if len(members) <= 1:
            continue
        xs = features.entry_x[members]
        ys = features.entry_y[members]
        d = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
        np.fill_diagonal(d, np.inf)
        isolated[members] = d.min(axis=1) > radius_mm
    return isolated


# --- torch-dependent part: Dataset + collate ------------------------------

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except Exception:  # torch not installed in lint-only environments
    _HAS_TORCH = False
    Dataset = object  # type: ignore[assignment,misc]


class ClusterDataset(Dataset):
    """One cluster per item: ``cells [n, F]`` plus the regression target.

    ``target='gev'`` returns the energy in GeV (matches ``picocal.evaluation``);
    ``target='log'`` returns natural-log GeV. ``e_true_gev`` is always carried so
    metrics can be computed in physical units regardless of the training target.
    Tensors are built on CPU; move batches to the GPU in the training loop.
    """

    def __init__(
        self,
        features: ClusterFeatures,
        indices: np.ndarray,
        *,
        target: str = "gev",
        flat: np.ndarray | None = None,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("ClusterDataset requires torch")
        if target not in ("gev", "log"):
            raise ValueError("target must be 'gev' or 'log'")
        self.features = features
        self.flat = features.flat if flat is None else flat
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target = target

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        c = int(self.indices[i])
        lo, hi = self.features.offsets[c], self.features.offsets[c + 1]
        cells = torch.from_numpy(np.ascontiguousarray(self.flat[lo:hi]))
        e_gev = float(self.features.target_gev[c])
        y = e_gev if self.target == "gev" else float(np.log(e_gev))
        return {
            "cells": cells,                                   # [n, F] float32
            "n_cells": int(hi - lo),
            "target": torch.tensor(y, dtype=torch.float32),
            "e_true_gev": torch.tensor(e_gev, dtype=torch.float32),
        }


def collate_clusters(batch: list[dict]) -> dict:
    """Pad a list of clusters to the batch-max length and build the mask.

    ``key_padding_mask`` follows the PyTorch ``src_key_padding_mask`` convention:
    True marks a padding position to be ignored by attention.
    """
    if not _HAS_TORCH:
        raise RuntimeError("collate_clusters requires torch")
    b = len(batch)
    lengths = torch.tensor([item["n_cells"] for item in batch], dtype=torch.long)
    lmax = int(lengths.max())
    f = batch[0]["cells"].shape[1]

    cells = torch.zeros(b, lmax, f, dtype=torch.float32)
    key_padding_mask = torch.ones(b, lmax, dtype=torch.bool)  # True = padding
    for i, item in enumerate(batch):
        n = item["n_cells"]
        cells[i, :n] = item["cells"]
        key_padding_mask[i, :n] = False

    return {
        "cells": cells,                                       # [B, Lmax, F]
        "key_padding_mask": key_padding_mask,                 # [B, Lmax] bool
        "n_cells": lengths,                                   # [B]
        "target": torch.stack([item["target"] for item in batch]),       # [B]
        "e_true_gev": torch.stack([item["e_true_gev"] for item in batch]),  # [B]
    }
