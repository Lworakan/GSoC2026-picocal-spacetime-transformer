import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from run_experiments import resolution, split, PITCH, EPS
from picocal_models import (SubNetFQ, QUANTILES, pinball_loss, qd_pinball_loss,
                            width_binned_calibration, linear_calibration,
                            save_model, load_model, CFG, NG)

DATA_MB = sorted((REPO / 'data' / 'minimum_bias').glob('*.root'))
CHAMPION_CSV = REPO / 'reports' / 'predictions' / 'minbias__SubNetW4CleanAuxQdEma.csv'


def test_resolution_recovers_gaussian_width():
    rng = np.random.default_rng(0)
    true = rng.uniform(5, 80, 20000)
    pred = true * (1 + rng.normal(0, 0.05, true.size))
    r = resolution(pred, true)
    assert abs(r['sigma_eff'] - 0.05) < 0.003
    assert abs(r['bias']) < 0.003


def test_resolution_is_deterministic():
    true = np.linspace(1, 100, 500)
    pred = true * 1.01
    assert resolution(pred, true) == resolution(pred, true)


def test_split_deterministic_disjoint_complete():
    a1, b1, t1 = split(1000)
    a2, b2, t2 = split(1000)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and np.array_equal(t1, t2)
    allidx = np.concatenate([a1, b1, t1])
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000
    assert len(a1) == 700 and len(b1) == 150 and len(t1) == 150


def _toy_batch(n=4, L=81, in_dim=16, ng=NG, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, in_dim, generator=g)
    m = torch.zeros(n, L, dtype=torch.bool)
    for i in range(n):
        m[i, : 10 + 5 * i] = True
    x[~m] = 0.0
    glob = torch.randn(n, ng, generator=g)
    e = torch.rand(n, L, generator=g) * 1000 * m.float()
    return x, m, glob, e


def test_subnetfq_forward_shape_and_finite():
    x, m, g, e = _toy_batch()
    model = SubNetFQ(16, 1.0, 0.0)
    q = model(x, m, g, e)
    assert q.shape == (4, 3)
    assert torch.isfinite(q).all()


def test_subnetfq_gate_off_mode():
    x, m, g, e = _toy_batch()
    model = SubNetFQ(16, 1.0, 0.0, gate='off')
    q = model(x, m, g, e)
    assert q.shape == (4, 3) and torch.isfinite(q).all()


def test_losses_finite_and_qd_dominates_pinball():
    qs = torch.tensor(QUANTILES)
    torch.manual_seed(1)
    q = torch.randn(64, 3)
    y = torch.randn(64, 1)
    pin = pinball_loss(q, y, qs)
    qd = qd_pinball_loss(q, y, qs)
    assert torch.isfinite(pin) and torch.isfinite(qd)
    assert qd.item() >= pin.item()


def test_width_binned_calibration_recovers_linear_map():
    rng = np.random.default_rng(2)
    y = rng.uniform(1.0, 4.0, 3000)
    q50 = 0.5 * y + 0.2
    width = rng.uniform(0.1, 0.3, y.size)
    qv = np.stack([q50 - width, q50, q50 + width], 1)
    pe = width_binned_calibration(qv, qv, y)
    assert np.abs(np.log(pe) - y).max() < 1e-6


def test_linear_calibration_matches_polyfit():
    rng = np.random.default_rng(3)
    y = rng.uniform(1.0, 4.0, 500)
    q = np.stack([y - 0.1, 2.0 * y + 1.0, y + 0.1], 1)
    pe = linear_calibration(q, q, y)
    assert np.abs(np.log(pe) - y).max() < 1e-9


def test_save_load_roundtrip_identical(tmp_path):
    x, m, g, e = _toy_batch()
    model = SubNetFQ(16, 1.2, -0.3)
    model.eval()
    p = tmp_path / 'model.pt'
    save_model(p, model, dict(in_dim=16, la0=1.2, lb0=-0.3, ng=NG))
    loaded, st = load_model(p)
    with torch.no_grad():
        assert torch.allclose(model(x, m, g, e), loaded(x, m, g, e))
    assert st['in_dim'] == 16


@pytest.mark.skipif(not CHAMPION_CSV.exists(), reason='champion CSV not present')
def test_champion_regression_canary():
    df = pd.read_csv(CHAMPION_CSV)
    df = df[df['split'] == 'test']
    seeds = sorted(df['seed'].unique())
    assert len(seeds) == 5
    t = df[df['seed'] == seeds[0]]['true_energy'].to_numpy()
    pred = np.mean([df[df['seed'] == s]['pred_energy'].to_numpy() for s in seeds], axis=0)
    sig = resolution(pred, t)['sigma_eff']
    assert abs(sig - 0.0409) < 0.0002, f'champion sigma_eff drifted: {sig}'
    for s in seeds:
        assert np.allclose(df[df['seed'] == s]['true_energy'].to_numpy(), t)


@pytest.mark.skipif(len(DATA_MB) == 0, reason='ROOT data not present')
def test_build_grid_and_prep_contract():
    from picocal_data import build_grid, make_windows, splits_for, prep, THRESH
    ev = build_grid(DATA_MB[:1])
    assert len(ev) > 50
    e0 = ev[0]
    for key in ('di', 'dj', 'x', 'y', 'e', 'fr', 'bk', 'tf', 'tb', 'ps', 'reg', 'Etrue', 'ET'):
        assert key in e0, key
    assert 0 <= e0['reg'] < len(PITCH)
    assert np.isfinite(e0['ET']) and e0['ET'] > 0
    rows, keep = make_windows(4, ev)
    # (tok, sum_e, max_e, Etrue, region, ET, extra_globals, aux_targets, cell_frac, time_slice)
    assert len(rows[0]) == 10
    tok = rows[0][0]
    assert tok.shape[1] == 16
    assert (np.expm1(tok[:, 0]) >= THRESH - 1e-3).all()
    D = prep(4, ev, None, ng=5)
    for key in ('X', 'M', 'G', 'y', 'Et', 'ET', 'reg', 'Eraw', 'ktr', 'kva', 'kte',
                'ctr', 'IN_DIM', 'la0', 'lb0', 'mean', 'std'):
        assert key in D, key
    assert D['IN_DIM'] == 16
    assert D['X'].shape[1] == 81
    assert np.isfinite(D['la0']) and np.isfinite(D['lb0'])
    assert (D['Eraw'][~D['M']] == 0).all()
    assert len(set(D['ktr']) & set(D['kte'])) == 0


@pytest.mark.skipif(len(DATA_MB) == 0, reason='ROOT data not present')
def test_prep_is_deterministic():
    from picocal_data import build_grid, prep
    ev = build_grid(DATA_MB[:1])
    d1 = prep(4, ev, None, ng=5)
    d2 = prep(4, ev, None, ng=5)
    assert np.array_equal(d1['ktr'], d2['ktr'])
    assert np.allclose(d1['X'], d2['X'])
    assert np.allclose(d1['mean'], d2['mean'])
