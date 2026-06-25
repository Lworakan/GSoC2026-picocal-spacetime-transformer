import numpy as np

from picocal_explorer.geometry import classify_by_pitch, derive_cell_geometry, module_pitch


def test_module_pitch_grid():
    pts = np.array([[0, 0], [15, 0], [0, 15], [15, 15]], float)
    assert abs(module_pitch(pts) - 15.0) < 1e-6


def test_module_pitch_single():
    assert np.isnan(module_pitch(np.array([[1.0, 2.0]])))


def test_classify_by_pitch():
    assert classify_by_pitch(15.1) == "SpaCal-W (1.5 cm)"
    assert classify_by_pitch(29.0) == "SpaCal-W (3 cm)"
    assert classify_by_pitch(41.0) == "Shashlik (4 cm)"
    assert classify_by_pitch(59.0) == "SpaCal-Pb (6 cm)"
    assert classify_by_pitch(118.0) == "Shashlik (12 cm)"
    assert classify_by_pitch(float("nan")) == "unknown"


def test_derive_cell_geometry_rel_and_seed():
    out = derive_cell_geometry(
        imodx=np.array([0, 0, 0]),
        jmody=np.array([0, 0, 0]),
        cell_x=np.array([0.0, 15.0, 30.0]),
        cell_y=np.array([0.0, 0.0, 0.0]),
        module_map={(0, 0): {"pitch": 15.0, "modtype": "SpaCal-W (1.5 cm)"}},
        seed_index=1,
    )
    assert out["is_seed"].tolist() == [False, True, False]
    assert out["rel_x"].tolist() == [-15.0, 0.0, 15.0]
    assert out["rel_dr"][0] == 15.0
    assert out["modtype"][0] == "SpaCal-W (1.5 cm)"
