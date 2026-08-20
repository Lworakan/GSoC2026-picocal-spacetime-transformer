import argparse
import pickle
from pathlib import Path
import numpy as np

SIGT_E = np.array([5., 20., 55., 175., 550., 1700., 5000.])
SIGT = np.array([.756, .593, .435, .263, .156, .101, .038])

EXAMPLES = """\
example:
  uv run scripts/cell_info_ceiling.py --overlay .scratch/cache/overlay.pkl

The question this answers.

Reaching sigma_eff 0.07 in the two failing bins needs the pileup contribution to the absolute
energy noise cut by 6.1x (15mm, 11-24 GeV) and 3.4x (30mm, 5.6-14 GeV). A time-pull cut buys
about 2x before photon loss starts costing more than the pileup it removes. So the open question
is whether the remaining factor is available in the per-cell observables AT ALL, or whether the
information simply is not there.

This measures it without training the full model. A per-cell classifier is fitted on the paired
overlay sample to predict each cell's photon fraction from observables only -- energy, time pull
against a seed-estimated reference, ring distance, front/back split, local density -- and its
output is then used as the gate in the same weighted-sum readout the network uses. Comparing

  raw sum  <  feature gate  <  oracle gate

separates what is missing from the model (raw -> feature gate) from what is missing from the
detector (feature gate -> oracle). A feature gate close to raw sum means better per-cell
modelling cannot reach 0.07 and the effort belongs elsewhere; close to oracle means it can.

Fitted and evaluated on disjoint events, so the numbers are held out.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Measure how much of the oracle gate benefit is reachable from per-cell '
                    'observables alone.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--overlay', default='.scratch/cache/overlay.pkl')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--thresh', type=float, default=2.49)
    ap.add_argument('--regions', type=int, nargs='+', default=[0, 1])
    return ap.parse_args()


def sigma_t(e):
    return np.interp(np.log(np.clip(e, 1., None)), np.log(SIGT_E), SIGT)


def seff(r):
    r = np.sort(np.asarray(r))
    n = len(r)
    if n < 25:
        return np.nan
    k = int(np.ceil(0.683 * n))
    w = r[k - 1:] - r[:n - k + 1]
    return w[np.argmin(w)] / 2


def cells(ev, W, TH):
    d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
    m = (d <= W) & (ev['e'] >= TH)
    if m.sum() < 5:
        return None
    e, sg, t = ev['e'][m], ev['sig'][m], ev['tf'][m]
    fr, bk = ev['fr'][m], ev['bk'][m]
    dd = d[m].astype(np.float32)
    core = dd <= 1
    if core.sum() < 1 or e[core].sum() <= 0:
        return None
    fin = np.isfinite(t)
    tref = ((e[core & fin] * t[core & fin]).sum() / max(e[core & fin].sum(), 1e-9)
            if (core & fin).any() else 0.0)
    pull = np.where(fin, np.abs(t - tref) / np.maximum(sigma_t(e), 1e-3), 8.0)
    tot = max(e.sum(), 1e-9)
    X = np.stack([np.log1p(e), pull, np.minimum(pull, 4.0), dd,
                  (fr - bk) / np.maximum(fr + bk, 1e-6),
                  e / tot, np.full_like(e, np.log1p(tot)), np.full_like(e, len(e))], 1)
    return X.astype(np.float32), e, sg, float(ev['Etrue']) * 1000.0


def main():
    a = parse_args()
    EV = pickle.load(open(a.overlay, 'rb'))
    from sklearn.ensemble import HistGradientBoostingRegressor

    for reg in a.regions:
        pack = [c for c in (cells(ev, a.window, a.thresh) for ev in EV
                            if ev['reg'] == reg) if c is not None]
        if len(pack) < 500:
            continue
        n = len(pack)
        idx = np.arange(n)
        tr, te = idx[idx % 2 == 0], idx[idx % 2 == 1]
        Xtr = np.concatenate([pack[i][0] for i in tr])
        ytr = np.clip(np.concatenate([pack[i][2] for i in tr])
                      / np.clip(np.concatenate([pack[i][1] for i in tr]), 1e-6, None), 0, 1)
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                            max_depth=6, random_state=0)
        mdl.fit(Xtr, ytr)

        E = np.array([pack[i][3] for i in te])
        lo = E < np.quantile(E, 1 / 3)

        def score(get_w, tag):
            s = np.array([float((get_w(pack[i]) * pack[i][1]).sum()) for i in te])
            x = np.log(np.clip(s, 1e-6, None))
            c = np.polyfit(x, np.log(E), 1)
            p = np.exp(c[0] * x + c[1])
            print(f'{"15mm" if reg == 0 else "30mm":>6s} {tag:>22s} '
                  f'{seff((p - E) / E):9.4f} {seff(((p - E) / E)[lo]):11.4f}')

        print(f'\n{"region":>6s} {"estimator":>22s} {"sigma_eff":>9s} {"bin: low-E":>11s}')
        score(lambda c: np.ones_like(c[1]), 'raw sum')
        score(lambda c: (c[0][:, 1] < 3.0).astype(np.float32), 'time-pull cut < 3')
        score(lambda c: np.clip(mdl.predict(c[0]), 0, 1), 'gate from observables')
        score(lambda c: np.clip(c[2] / np.clip(c[1], 1e-6, None), 0, 1), 'oracle gate (truth)')

        pr = np.clip(mdl.predict(np.concatenate([pack[i][0] for i in te])), 0, 1)
        tv = np.clip(np.concatenate([pack[i][2] for i in te])
                     / np.clip(np.concatenate([pack[i][1] for i in te]), 1e-6, None), 0, 1)
        print(f'{"":>6s} per-cell corr(gate, truth) = {np.corrcoef(pr, tv)[0, 1]:.3f}'
              f'   (the trained network reaches 0.211)')


if __name__ == '__main__':
    main()
