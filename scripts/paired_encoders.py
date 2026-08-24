import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution

# Table h2h reports the encoder comparison as mean +/- seed spread and never attaches a
# significance statement, while every other comparison in the paper carries one. The
# encoder is the only component that differs between these runs, so the comparison can be
# paired event by event: both estimators score the same events, the bootstrap resamples
# events once and scores both on the same draw, and the difference gets its own interval.

D = Path(__file__).resolve().parents[1] / 'reports' / 'predictions'
REF = 'minbias__SubNetW8CleanAuxExDnRcQdEma.csv'
ARMS = {
    'ParticleNet (default)': 'minbias__SubNetW8CleanAuxPnetExDnRcQdEma.csv',
    'ParticleNet lr 1e-4': 'minbias__SubNetW8CleanAuxPnetExDnLr0p0001RcQdEma.csv',
    'ParticleNet lr 6e-4': 'minbias__SubNetW8CleanAuxPnetExDnLr0p0006RcQdEma.csv',
    'ParticleNet k=24': 'minbias__SubNetW8CleanAuxPnetNn24ExDnRcQdEma.csv',
    'GravNet (default)': 'minbias__SubNetW8CleanAuxGravExDnRcQdEma.csv',
    'GravNet lr 1e-4': 'minbias__SubNetW8CleanAuxGravExDnLr0p0001RcQdEma.csv',
    'GravNet lr 6e-4': 'minbias__SubNetW8CleanAuxGravExDnLr0p0006RcQdEma.csv',
    'GravNet k=24': 'minbias__SubNetW8CleanAuxGravNn24ExDnRcQdEma.csv',
}
KEY = ['true_energy', 'region_name']
NBOOT = 400
BINS = (('15mm', 'low'), ('30mm', 'low'))


def load(name):
    t = pd.read_csv(D / name)
    t['true_energy'] = t['true_energy'].astype(np.float32)
    return t


def one_seed(t, sd):
    # single model against single model: averaging our seeds and not the baseline's would
    # hand the comparison an ensembling advantage that table h2h explicitly disclaims
    s = t[t.seed == sd]
    return s.groupby(KEY, sort=False).agg(pred=('pred_energy', 'first')).reset_index()


def low_mask(m):
    keep = np.zeros(len(m), bool)
    for reg, _ in BINS:
        sub = m.region_name == reg
        if sub.any():
            keep |= sub & (m.true_energy <= m.loc[sub, 'true_energy'].quantile(1 / 3))
    return keep


def main():
    ref = load(REF)
    rseeds = sorted(ref.seed.unique())
    print(f'reference {REF}: seeds {rseeds}, {len(ref)} rows')

    rng = np.random.default_rng(0)
    for label, fn in ARMS.items():
        if not (D / fn).exists():
            print(f'{label:24s} MISSING')
            continue
        arm = load(fn)
        worst_p, rows = 0, []
        for asd in sorted(arm.seed.unique()):
            ta = one_seed(arm, asd)
            for rsd in rseeds:
                m = one_seed(ref, rsd).merge(ta, on=KEY, suffixes=('_ref', '_arm'))
                t = m['true_energy'].values
                d = []
                for _ in range(NBOOT):
                    idx = rng.integers(0, len(m), len(m))
                    d.append(resolution(m['pred_arm'].values[idx], t[idx])['sigma_eff']
                             - resolution(m['pred_ref'].values[idx], t[idx])['sigma_eff'])
                d = np.array(d)
                lo = low_mask(m)
                rows.append((resolution(m['pred_ref'].values, t)['sigma_eff'],
                             resolution(m['pred_arm'].values, t)['sigma_eff'],
                             d.mean(), int((d > 0).sum()),
                             resolution(m['pred_ref'].values[lo], t[lo])['sigma_eff'],
                             resolution(m['pred_arm'].values[lo], t[lo])['sigma_eff']))
                worst_p = max(worst_p, NBOOT - rows[-1][3])
        a, b, dm, _, al, bl = np.array(rows).mean(0)
        print(f'{label:24s} pairs={len(rows):2d} n={len(m):6d}  ours {a:.4f}  arm {b:.4f}  '
              f'diff {dm:+.4f}  worst pair: arm better in {worst_p:3d}/{NBOOT}  '
              f'| 15+30 low-E ours {al:.4f} arm {bl:.4f}')


if __name__ == '__main__':
    main()
