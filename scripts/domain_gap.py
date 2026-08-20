import argparse
import pickle
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from picocal_data import cell_prior_features, THRESH

EXAMPLES = """\
example:
  uv run scripts/domain_gap.py --overlay .scratch/cache/overlay.pkl --real .scratch/cache/minbias_94.pkl

Why this matters now.

The per-cell photon-fraction estimator reaches corr 0.9456 on held-out overlay events, yet feeding
its prediction to the network as an input made both target bins WORSE (15mm low +0.0035, 30mm low
+0.0038; reports/prior_2026-08-16.md). The suspected cause is that the synthetic overlay cells do
not look like real min-bias cells, so a mapping learned on one does not apply to the other -- the
same failure H3 recorded with a domain-gap AUC of 0.74.

This measures the gap on exactly the eight features the estimator uses, by training a classifier
to tell an overlay cell from a real min-bias cell. AUC 0.5 means indistinguishable and the
estimator should transfer; AUC near 1 means the two samples are trivially separable and it cannot.
Per-feature AUC then says WHICH observable betrays the synthetic origin, which is the thing to fix
in make_overlay.py -- a targeted repair rather than another architecture guess.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Measure how separable synthetic overlay cells are from real min-bias cells.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--overlay', default='.scratch/cache/overlay.pkl')
    ap.add_argument('--real', default='.scratch/cache/minbias_94.pkl')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--regions', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--max-events', type=int, default=4000)
    return ap.parse_args()


NAMES = ['log1p(e)', 'time pull', 'pull clipped', 'ring d', 'front-back asym',
         'energy share']


def collect(events, reg, W, cap):
    X = []
    for ev in events:
        if ev['reg'] != reg:
            continue
        d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
        m = (d <= W) & (ev['e'] >= THRESH)
        if m.sum() < 5:
            continue
        X.append(cell_prior_features(ev['e'][m], ev['fr'][m], ev['bk'][m], ev['tf'][m],
                                     d[m].astype(np.float32)))
        if len(X) >= cap:
            break
    return np.concatenate(X) if X else np.zeros((0, len(NAMES)), np.float32)


def main():
    a = parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    syn = pickle.load(open(a.overlay, 'rb'))
    real = pickle.load(open(a.real, 'rb'))

    for reg in a.regions:
        S = collect(syn, reg, a.window, a.max_events)
        R = collect(real, reg, a.window, a.max_events)
        n = min(len(S), len(R))
        if n < 2000:
            print(f'region {reg}: too few cells ({len(S)} syn, {len(R)} real)')
            continue
        X = np.concatenate([S[:n], R[:n]])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        i = np.arange(len(y))
        tr, te = i % 2 == 0, i % 2 == 1
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=0)
        clf.fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        print(f'\n=== {"15mm" if reg == 0 else "30mm"}: {n} cells each side ===')
        print(f'  joint AUC over all {X.shape[1]} features: {auc:.3f}   '
              f'({"transfers" if auc < 0.6 else "does NOT transfer"})')
        print(f'  {"feature":>18s} {"AUC":>6s} {"syn median":>11s} {"real median":>12s}')
        for k in range(X.shape[1]):
            one = roc_auc_score(y[te], X[te][:, k])
            print(f'  {NAMES[k]:>18s} {max(one, 1 - one):6.3f} '
                  f'{np.median(S[:n, k]):11.3f} {np.median(R[:n, k]):12.3f}')


if __name__ == '__main__':
    main()
