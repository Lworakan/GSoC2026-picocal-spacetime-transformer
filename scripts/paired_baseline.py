import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution

# The headline "3.2x over a gradient-boosted reference" compared a ten-fold ensemble
# scored on 72,533 events with a single-seed tree scored on the 12,227 events of the
# fixed split. That is the cross-protocol comparison the rest of the paper forbids.
# This script scores both estimators on the events they share, and pairs the bootstrap
# so the difference carries an uncertainty rather than the two widths separately.

D = Path(__file__).resolve().parents[1] / 'reports' / 'predictions'
MEMBERS = ('minbias__SubNetW8CleanAuxExDnRcK{f}QdEma.csv',
           'minbias__SubNetW8CleanAuxExDnRcK{f}Rr01QdEma.csv',
           'minbias__SubNetW4CleanAuxExDnRcK{f}AcQdEma.csv')
KEY = ['true_energy', 'region_name']
NBOOT = 400


def cv_ensemble():
    out = []
    for f in range(10):
        tabs = [pd.read_csv(D / p.format(f=f)) for p in MEMBERS if (D / p.format(f=f)).exists()]
        allm = pd.concat(tabs)
        out.append(allm.groupby(KEY, sort=False).agg(pred=('pred_energy', 'median')).reset_index())
    return pd.concat(out, ignore_index=True)


def paired(a_pred, b_pred, truth, rng):
    n = len(truth)
    da, db = [], []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        da.append(resolution(a_pred[idx], truth[idx])['sigma_eff'])
        db.append(resolution(b_pred[idx], truth[idx])['sigma_eff'])
    return np.array(da), np.array(db)


def main():
    bdt = pd.read_csv(D / 'minbias__BDT.csv')
    ens = cv_ensemble()
    n_bdt, n_ens = len(bdt), len(ens)

    # the tree writes the target at double precision and the network at single, so an
    # exact join on the energy finds nothing; both are put on the network's grid first
    for t in (bdt, ens):
        t['true_energy'] = t['true_energy'].astype(np.float32)

    m = bdt.merge(ens, on=KEY, how='inner', suffixes=('_bdt', ''))
    m = m.drop_duplicates(subset=KEY)
    truth = m['true_energy'].values
    print(f'BDT rows {n_bdt}, ensemble rows {n_ens}, shared events {len(m)}')

    r_ens = resolution(m['pred'].values, truth)['sigma_eff']
    r_bdt = resolution(m['pred_energy'].values, truth)['sigma_eff']
    print(f'on the shared events: SpaceTformer {r_ens:.4f}  BDT {r_bdt:.4f}  '
          f'ratio {r_bdt / r_ens:.2f}x')

    rng = np.random.default_rng(0)
    da, db = paired(m['pred'].values, m['pred_energy'].values, truth, rng)
    d = db - da
    print(f'paired difference {d.mean():.4f} +/- {d.std():.4f}, '
          f'BDT worse in {(d > 0).sum()}/{NBOOT} resamples')
    print(f'ratio {np.mean(db / da):.2f} +/- {np.std(db / da):.2f}')

    # the per-bin table: an aggregate ratio can hide a bin where the tree is ahead, so
    # every region-energy cell is scored separately and the row is printed as LaTeX
    wins = 0
    for reg in ('15mm', '30mm', '40mm', '60mm', '120mm'):
        s = m[m.region_name == reg]
        q = s.true_energy.quantile([1 / 3, 2 / 3]).values
        cells = (('low', s[s.true_energy <= q[0]]),
                 ('mid', s[(s.true_energy > q[0]) & (s.true_energy <= q[1])]),
                 ('high', s[s.true_energy > q[1]]))
        out = []
        for _, sub in cells:
            tt = sub['true_energy'].values
            a = resolution(sub['pred'].values, tt)['sigma_eff']
            c = resolution(sub['pred_energy'].values, tt)['sigma_eff']
            wins += a < c
            out.append(f'{a:.4f} & {c:.4f} & ${c / a:.2f}\\times$')
        print(f'{reg} & ' + ' & '.join(out) + r'\\')
    print(f'% SpaceTformer ahead in {wins}/15 bins')


if __name__ == '__main__':
    main()
