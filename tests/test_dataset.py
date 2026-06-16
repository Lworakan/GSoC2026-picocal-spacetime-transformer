"""Unit tests for picocal.data.dataset on synthetic arrays.

The numpy pieces (feature build, splits, scaler, isolation) run everywhere.
The torch pieces (Dataset, collate) are skipped when torch is absent, matching
the integration-test pattern in test_loader.py.
"""

from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from picocal.data import (
    FEATURE_NAMES,
    N_FEATURES,
    ClusterFeatures,
    FeatureScaler,
    add_isolation_flag,
    make_event_splits,
)
from picocal.data.dataset import _features_from_arrays

GARBAGE_T = -6.8e37

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@pytest.fixture
def synthetic():
    """Two clusters mirroring the loader fixture (one silent, one garbage time)."""
    return ak.Array(
        {
            "event": [10, 11],
            "sig_flux_eTot": [5.0, 50.0],
            "sig_flux_entry_x": [100.0, -200.0],
            "sig_flux_entry_y": [50.0, 75.0],
            "sig_dr_matched": [12.0, 30.0],
            "cell_x": [[100.0, 110.0, 120.0], [-200.0, -210.0]],
            "cell_y": [[50.0, 50.0, 60.0], [75.0, 75.0]],
            "energy": [[25000.0, 1000.0, 0.0], [250000.0, 10000.0]],
            "cell_energies_front": [[20000.0, 0.0, 0.0], [170000.0, 10000.0]],
            "cell_energies_back": [[5000.0, 1000.0, 0.0], [80000.0, 0.0]],
            "cell_times_front": [[42.1, 42.3, GARBAGE_T], [42.0, GARBAGE_T]],
            "cell_times_back": [[42.5, GARBAGE_T, GARBAGE_T], [42.4, 42.6]],
        }
    )


def test_offsets_match_informative_cells(synthetic):
    feat, offsets = _features_from_arrays(synthetic)
    assert list(np.diff(offsets)) == [2, 2]  # silent cell dropped from cluster 0
    assert feat.shape == (4, N_FEATURES)


def test_feature_values(synthetic):
    feat, offsets = _features_from_arrays(synthetic)
    cols = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # log energy of the four kept cells
    assert feat[:, cols["log_energy"]] == pytest.approx(
        np.log([25000.0, 1000.0, 250000.0, 10000.0]).astype(np.float32), rel=1e-5
    )
    # front fraction: 0.8, 0.0 (cluster 0); 0.68, 1.0 (cluster 1)
    assert feat[:, cols["front_fraction"]] == pytest.approx([0.8, 0.0, 0.68, 1.0], abs=1e-4)
    # time centred on the per-cluster median: [-0.1, +0.1] then [0.0, 0.0]
    assert feat[:, cols["t_centered"]] == pytest.approx([-0.1, 0.1, 0.0, 0.0], abs=1e-4)
    # validity flag: garbage time in cluster 1, cell 1 -> 0
    assert list(feat[:, cols["t_valid"]]) == [1.0, 1.0, 1.0, 0.0]


def test_no_nan_anywhere(synthetic):
    feat, _ = _features_from_arrays(synthetic)
    assert np.isfinite(feat).all()


def _toy_features(n_groups=10, per_group=5, seed=1):
    """A ClusterFeatures with a known group structure for split/scaler tests."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), per_group).astype(np.int64)
    n = len(group)
    cells_per = rng.integers(1, 4, size=n)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(cells_per, out=offsets[1:])
    flat = rng.normal(size=(int(offsets[-1]), N_FEATURES)).astype(np.float32)
    return ClusterFeatures(
        flat=flat,
        offsets=offsets,
        target_gev=rng.uniform(1, 200, n).astype(np.float32),
        file_id=np.zeros(n, dtype=np.int64),
        event=group.copy(),
        group=group,
        sig_dr_matched=np.zeros(n, dtype=np.float32),
        entry_x=rng.normal(size=n).astype(np.float32),
        entry_y=rng.normal(size=n).astype(np.float32),
    )


def test_splits_no_group_leakage_and_deterministic():
    feats = _toy_features()
    s1 = make_event_splits(feats.group, seed=7)
    s2 = make_event_splits(feats.group, seed=7)
    # deterministic
    for k in ("train", "val", "test"):
        assert np.array_equal(s1[k], s2[k])
    # partition covers every cluster exactly once
    allidx = np.concatenate([s1["train"], s1["val"], s1["test"]])
    assert sorted(allidx.tolist()) == list(range(len(feats)))
    # no group appears in more than one split
    split_of_group = {}
    for name, idx in s1.items():
        for g in np.unique(feats.group[idx]):
            assert g not in split_of_group
            split_of_group[g] = name


def test_scaler_uses_train_stats_only():
    feats = _toy_features()
    splits = make_event_splits(feats.group, seed=3)
    scaler = FeatureScaler().fit(feats, splits["train"])
    scaled = scaler.transform(feats.flat)
    # rows of the train clusters, standardised columns -> mean 0, std 1
    train_rows = np.concatenate(
        [np.arange(feats.offsets[i], feats.offsets[i + 1]) for i in splits["train"]]
    )
    sub = scaled[train_rows][:, scaler.cols]
    assert sub.mean(axis=0) == pytest.approx(np.zeros(len(scaler.cols)), abs=1e-5)
    assert sub.std(axis=0) == pytest.approx(np.ones(len(scaler.cols)), abs=1e-5)
    # pass-through columns are untouched
    assert np.array_equal(scaled[:, 3], feats.flat[:, 3])
    assert np.array_equal(scaled[:, 5], feats.flat[:, 5])


def test_isolation_flag():
    # group 0: two photons 50 mm apart -> both not isolated
    # group 1: one photon alone -> isolated
    # group 2: two photons 500 mm apart -> both isolated
    n = 5
    feats = ClusterFeatures(
        flat=np.zeros((n, N_FEATURES), dtype=np.float32),
        offsets=np.arange(n + 1, dtype=np.int64),
        target_gev=np.ones(n, dtype=np.float32),
        file_id=np.zeros(n, dtype=np.int64),
        event=np.array([0, 0, 1, 2, 2]),
        group=np.array([0, 0, 1, 2, 2]),
        sig_dr_matched=np.zeros(n, dtype=np.float32),
        entry_x=np.array([0.0, 50.0, 0.0, 0.0, 500.0], dtype=np.float32),
        entry_y=np.zeros(n, dtype=np.float32),
    )
    iso = add_isolation_flag(feats, radius_mm=200.0)
    assert list(iso) == [False, False, True, True, True]


# --- torch-dependent --------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_collate_shapes_and_mask():
    from picocal.data import ClusterDataset, collate_clusters

    feats = _toy_features(n_groups=6, per_group=4)
    splits = make_event_splits(feats.group, seed=0)
    ds = ClusterDataset(feats, splits["train"], target="gev")
    batch = [ds[i] for i in range(min(4, len(ds)))]
    out = collate_clusters(batch)

    b = len(batch)
    lmax = max(item["n_cells"] for item in batch)
    assert out["cells"].shape == (b, lmax, N_FEATURES)
    assert out["key_padding_mask"].shape == (b, lmax)
    # mask True exactly at padding positions
    for i, item in enumerate(batch):
        n = item["n_cells"]
        assert bool(out["key_padding_mask"][i, :n].any()) is False
        assert bool(out["key_padding_mask"][i, n:].all()) is True
        # unpadded rows equal the per-item tensor
        assert torch.allclose(out["cells"][i, :n], item["cells"])
    assert out["target"].shape == (b,)
    assert out["e_true_gev"].shape == (b,)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_dataset_log_target():
    from picocal.data import ClusterDataset

    feats = _toy_features(n_groups=4, per_group=3)
    ds = ClusterDataset(feats, np.arange(len(feats)), target="log")
    item = ds[0]
    e_gev = float(item["e_true_gev"])
    assert float(item["target"]) == pytest.approx(float(np.log(e_gev)), rel=1e-5)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "full"


@pytest.mark.skipif(not _HAS_TORCH or not DATA_DIR.exists(), reason="torch or data absent")
def test_integration_one_file():
    from torch.utils.data import DataLoader

    from picocal.data import ClusterDataset, build_cluster_features, collate_clusters

    feats = build_cluster_features(DATA_DIR, max_files=1)
    assert len(feats) > 0
    splits = make_event_splits(feats.group, seed=0)
    scaler = FeatureScaler().fit(feats, splits["train"])
    flat = scaler.transform(feats.flat)
    ds = ClusterDataset(feats, splits["train"], target="gev", flat=flat)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_clusters)
    batch = next(iter(loader))
    assert batch["cells"].shape[2] == N_FEATURES
    assert np.isfinite(batch["cells"].numpy()).all()
