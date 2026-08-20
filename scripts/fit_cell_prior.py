import argparse
import pickle
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from picocal_data import cell_prior_features, THRESH

EXAMPLES = """\
example:
  uv run scripts/fit_cell_prior.py --overlay .scratch/cache/overlay.pkl --out .scratch/cell_prior.pkl

Why this exists.

The network's emergent gate correlates only 0.211 with the true per-cell photon fraction, while a
plain gradient-boosted model on eight hand-made observables reaches 0.93-0.95 on the same cells
(scripts/cell_info_ceiling.py). The information is in the inputs; the network is not using it.

This fits that estimator once and saves it, so it can be either
  --prior-feat   appended as a per-cell input the network cannot ignore, or
  --prior-teach  used as the gate target on EVERY event, including real min-bias.

The second is the important one. Truth-based gate supervision failed because the truth only
exists on synthetic events, so the gate on real data was never touched (reports/gatesup_2026-08-16.md).
This estimator's output is a function of observables alone, so it transfers to real events -- the
synthetic sample is used only to learn the mapping, never at inference.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Fit a per-cell photon-fraction estimator on the paired overlay sample.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--overlay', default='.scratch/cache/overlay.pkl')
    ap.add_argument('--out', default='.scratch/cell_prior.pkl')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--depth', type=int, default=6)
    return ap.parse_args()


def main():
    a = parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor

    EV = pickle.load(open(a.overlay, 'rb'))
    Xs, ys, gs = [], [], []
    for k, ev in enumerate(EV):
        d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
        m = (d <= a.window) & (ev['e'] >= THRESH)
        if m.sum() < 5 or 'sig' not in ev:
            continue
        e, sg = ev['e'][m], ev['sig'][m]
        Xs.append(cell_prior_features(e, ev['fr'][m], ev['bk'][m], ev['tf'][m],
                                      d[m].astype(np.float32)))
        ys.append(np.clip(sg / np.clip(e, 1e-6, None), 0.0, 1.0))
        gs.append(np.full(m.sum(), k))
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.float32)
    g = np.concatenate(gs)
    # split by EVENT, so no cell of a held-out event contributes to the fit
    tr, te = g % 2 == 0, g % 2 == 1
    print(f'cells {len(y)} from {len(Xs)} events | train {tr.sum()} test {te.sum()}', flush=True)

    mdl = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.1,
                                        max_depth=a.depth, random_state=0)
    mdl.fit(X[tr], y[tr])
    for nm, s in (('train', tr), ('test', te)):
        p = np.clip(mdl.predict(X[s]), 0, 1)
        print(f'{nm}: corr {np.corrcoef(p, y[s])[0, 1]:.4f}  mae {np.abs(p - y[s]).mean():.4f}')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(mdl, f, protocol=4)
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
