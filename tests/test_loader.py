"""Unit tests for picocal.data.loader on synthetic awkward arrays.

No ROOT file is needed: the mask / time-cleaning / tokenization logic is
exercised on hand-built jagged arrays mirroring the real schema. One
integration test runs only when local data is present.
"""

from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from picocal.data import (
    clean_cell_times,
    cluster_to_tokens,
    load_clusters,
    truth_dataframe,
    valid_cell_mask,
)

GARBAGE_T = -6.8e37  # uninitialised time value as seen in the real files


@pytest.fixture
def synthetic():
    """Two clusters: 3 cells (one silent) and 2 cells (one garbage time)."""
    return ak.Array(
        {
            "event": [10, 11],
            "sig_flux_eTot": [5.0, 50.0],  # GeV
            "sig_flux_entry_x": [100.0, -200.0],
            "sig_flux_entry_y": [50.0, 75.0],
            "sig_flux_pdgID": [22, 22],
            "sig_dr_matched": [12.0, 30.0],
            "x_cluster": [101.0, -199.0],
            "y_cluster": [51.0, 74.0],
            "total_energy": [26000.0, 260000.0],
            "total_energy_front": [20000.0, 180000.0],
            "total_energy_back": [6000.0, 80000.0],
            "cell_x": [[100.0, 110.0, 120.0], [-200.0, -210.0]],
            "cell_y": [[50.0, 50.0, 60.0], [75.0, 75.0]],
            "energy": [[25000.0, 1000.0, 0.0], [250000.0, 10000.0]],
            "cell_energies_front": [[20000.0, 0.0, 0.0], [170000.0, 10000.0]],
            "cell_energies_back": [[5000.0, 1000.0, 0.0], [80000.0, 0.0]],
            "cell_times_front": [[42.1, 42.3, GARBAGE_T], [42.0, GARBAGE_T]],
            "cell_times_back": [[42.5, GARBAGE_T, GARBAGE_T], [42.4, 42.6]],
        }
    )


def test_valid_cell_mask_drops_silent_cells(synthetic):
    mask = valid_cell_mask(synthetic)
    assert ak.to_list(mask) == [[True, True, False], [True, True]]


def test_clean_cell_times_nans_garbage_and_silent(synthetic):
    mask = valid_cell_mask(synthetic)
    t = clean_cell_times(synthetic["cell_times_front"], mask)
    t0, t1 = ak.to_list(t)
    assert t0[0] == pytest.approx(42.1)
    assert np.isnan(t0[2])  # silent cell
    assert np.isnan(t1[1])  # garbage value in an energetic... no, masked cell
    # a sane time in a cell with energy survives
    assert t1[0] == pytest.approx(42.0)


def test_cluster_to_tokens_alignment(synthetic):
    tokens = cluster_to_tokens(synthetic)
    assert ak.to_list(tokens["n_cells"]) == [2, 2]
    # all token arrays have identical jagged structure
    for key in ("x", "y", "e", "e_front", "e_back", "t_front", "t_back"):
        assert ak.to_list(ak.num(tokens[key], axis=1)) == [2, 2]
    # energies of the kept cells of cluster 0
    assert ak.to_list(tokens["e"][0]) == [25000.0, 1000.0]
    # garbage time of a kept cell becomes nan, sane one survives
    assert np.isnan(ak.to_list(tokens["t_back"][0])[1])
    assert ak.to_list(tokens["t_front"][0]) == pytest.approx([42.1, 42.3])


def test_truth_dataframe_unit_conversion(synthetic):
    df = truth_dataframe(synthetic)
    assert df["e_true_mev"].tolist() == [5000.0, 50000.0]
    assert df["n_cells"].tolist() == [3, 2]
    assert df["n_cells_nonzero"].tolist() == [2, 2]


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "full"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="local dataset not present")
def test_load_clusters_integration():
    arrays = load_clusters(DATA_DIR, max_files=1)
    assert len(arrays) > 0
    assert "sig_flux_eTot" in arrays.fields
    mask = valid_cell_mask(arrays)
    # every cluster keeps at least one informative cell
    assert ak.min(ak.sum(mask, axis=1)) >= 1
