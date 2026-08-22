import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def synthetic_event(px=0.0, py=0.0, seed=0):
    rng = np.random.default_rng(seed)
    di, dj = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7))
    di, dj = di.ravel().astype(np.int64), dj.ravel().astype(np.int64)
    r2 = (di - 2.0) ** 2 + (dj + 1.0) ** 2
    e = (500.0 * np.exp(-r2 / 4.0) + rng.uniform(5.0, 40.0, len(di))).astype(np.float32)
    n = len(di)
    return dict(di=di, dj=dj, x=(di * 15.0).astype(np.float32), y=(dj * 15.0).astype(np.float32),
                e=e, fr=(0.6 * e).astype(np.float32), bk=(0.4 * e).astype(np.float32),
                tf=rng.normal(0, 2, n).astype(np.float32),
                tb=rng.normal(0, 2, n).astype(np.float32),
                pc=np.full(n, 15.0, np.float32), ps=15.0, reg=0,
                Etrue=40.0, ET=20.0, tot=float(e.sum()), totf=float(e.sum() * 0.6),
                totb=float(e.sum() * 0.4), xc=30.0, yc=-15.0, xs=0.0, ys=0.0, ncl=1.0,
                ax=30.0, ay=-15.0, at=0.0, px=px, py=py)


def window_centre(rows_row, W):
    tok = rows_row[0]
    di, dj = tok[:, 3], tok[:, 4]
    return float(di.max() + di.min()) / 2.0, float(dj.max() + dj.min()) / 2.0


def test_pred_recentring_moves_the_window():
    from picocal_data import make_windows
    W = 4
    base, _ = make_windows(W, [synthetic_event()], recenter=False)
    pred, _ = make_windows(W, [synthetic_event(px=2, py=-1)], recenter=True, rc_mode='pred')
    # the pointer offset (2, -1) must land the window centre on that cell, so the
    # recentred window covers a different set of cells than the seed-centred one
    assert base[0][0].shape[0] == pred[0][0].shape[0] == (2 * W + 1) ** 2
    assert not np.allclose(np.sort(base[0][0][:, 0]), np.sort(pred[0][0][:, 0]))


def test_pred_recentring_ignores_nan_pointer():
    from picocal_data import make_windows
    seed_centred, _ = make_windows(4, [synthetic_event()], recenter=False)
    nan_pointer, _ = make_windows(4, [synthetic_event(px=np.nan, py=np.nan)],
                                  recenter=True, rc_mode='pred')
    assert np.allclose(np.sort(seed_centred[0][0][:, 0]), np.sort(nan_pointer[0][0][:, 0]))


def test_slot_readout_shapes_and_gradients():
    torch = pytest.importorskip('torch')
    from picocal_models import SubNetFQ
    B, L, D, G = 3, 25, 12, 6
    model = SubNetFQ(D, la0=1.0, lb0=0.0, ng=G, slots=4)
    x = torch.randn(B, L, D)
    m = torch.ones(B, L, dtype=torch.bool)
    m[0, -5:] = False
    out = model(x, m, torch.randn(B, G), torch.rand(B, L) * 100.0)
    assert out.shape == (B, 3)
    assert torch.isfinite(out).all()
    # the slot responsibility is a masked probability, so it stays inside [0, 1]
    w = model.last_w.detach()
    assert w.shape == (B, L)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
    assert float(w[0, -5:].abs().max()) == 0.0
    out.sum().backward()
    assert model.slot_q.grad is not None and torch.isfinite(model.slot_q.grad).all()


def test_distillation_target_shape_contract():
    torch = pytest.importorskip('torch')
    from picocal_models import SubNetFQ
    model = SubNetFQ(8, la0=1.0, lb0=0.0, ng=4)
    x, m = torch.randn(2, 9, 8), torch.ones(2, 9, dtype=torch.bool)
    q = model(x, m, torch.randn(2, 4), torch.rand(2, 9) * 10.0)
    # --distill compares the raw quantile head against an (N, 3) teacher
    assert q.shape[1] == 3
    assert torch.nn.functional.huber_loss(q, torch.zeros_like(q), delta=0.1).ndim == 0
