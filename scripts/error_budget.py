import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH

# Where does the aggregate sigma_eff actually come from, and which part of it is reducible?
# Run before spending GPU time: the answer decides which experiment is worth paying for.

FAMS = ['ExDnRcQdEma', 'ExDnAuxRcQdEma', 'DsExDnRcQdEma', 'ExDnRpQdEma',
        'Sc2ExDnRpQdEma', 'DsExDnAuxRcQdEma', 'AuxRpRr0123QdEma']


def load(names, P):
    out = []
    for f in names:
        p = P / f'minbias__SubNetW8CleanAux{f}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d['split'] == 'test'] if 'split' in d else d
        d['cfg'] = f
        out.append(d)
    return pd.concat(out)


def ens(d):
    return d.groupby(['true_energy', 'region', 'ET'], sort=False)['pred_energy'].median().reset_index()


def bins(g):
    """(region, energy tercile) label per row, terciles taken within each region."""
    lab = np.empty(len(g), object)
    for r in sorted(g['region'].unique()):
        k = (g['region'] == r).values
        le = np.log(g['true_energy'].values[k])
        q1, q2 = np.quantile(le, [1 / 3, 2 / 3])
        t = np.where(le <= q1, 'low', np.where(le <= q2, 'mid', 'high'))
        lab[k] = [f'{int(PITCH[r])}mm {x}' for x in t]
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    P = Path(args.repo) / 'reports' / 'predictions'
    d = load(FAMS, P)
    g = ens(d)
    g['bin'] = bins(g)
    r = (g['pred_energy'].values - g['true_energy'].values) / g['true_energy'].values
    g['res'] = r
    agg = resolution(g['pred_energy'].values, g['true_energy'].values)
    print(f"pooled ensemble: sigma_eff {agg['sigma_eff']}  bias {agg['bias']}  n {len(g)}")

    print('\n== 1. where the error sits, and what removing it would buy ==')
    print(f"{'bin':>12} {'n':>6} {'share':>6} {'sig':>7} {'bias':>7} {'agg if bin perfect':>19}")
    rows = []
    for b, sub in g.groupby('bin'):
        s = resolution(sub['pred_energy'].values, sub['true_energy'].values)
        rr = r.copy()
        rr[(g['bin'] == b).values] = 0.0
        pe = (1 + rr) * g['true_energy'].values
        a2 = resolution(pe, g['true_energy'].values)['sigma_eff']
        rows.append((b, len(sub), len(sub) / len(g), s['sigma_eff'], s['bias'], a2))
    for b, n, sh, s, bi, a2 in sorted(rows, key=lambda x: x[5]):
        print(f'{b:>12} {n:6d} {sh:6.3f} {s:7.4f} {bi:+7.4f} {a2:19.4f}')

    print('\n== 2. free levers: per-bin bias removal (post-hoc, no training) ==')
    rr = r.copy()
    for b, sub in g.groupby('bin'):
        k = (g['bin'] == b).values
        rr[k] = r[k] - np.median(r[k])
    pe = (1 + rr) * g['true_energy'].values
    print(f"  after removing the per-bin median bias: {resolution(pe, g['true_energy'].values)['sigma_eff']:.4f}"
          f"  (from {agg['sigma_eff']})")

    print('\n== 3. tails: how much of the budget is catastrophic ==')
    s0 = agg['sigma_eff']
    for k in (2, 3, 5):
        m = np.abs(r - np.median(r)) > k * s0
        rr = r.copy()
        rr[m] = np.median(r)
        pe = (1 + rr) * g['true_energy'].values
        print(f'  |res| > {k} sigma: {m.mean()*100:5.2f}% of events, '
              f'perfect fix -> {resolution(pe, g["true_energy"].values)["sigma_eff"]:.4f}')

    print('\n== 4. how much is model variance (reducible by averaging) ==')
    per = [resolution(s['pred_energy'].values, s['true_energy'].values)['sigma_eff']
           for _, s in d.groupby(['cfg', 'seed'])]
    print(f'  single members: mean {np.mean(per):.4f}  best {np.min(per):.4f}  n={len(per)}')
    print(f'  pooled ensemble: {agg["sigma_eff"]:.4f}   variance already removed: '
          f'{(np.mean(per) - agg["sigma_eff"]) / np.mean(per) * 100:.1f}%')
    sub = d.groupby(['true_energy', 'region', 'ET'])['pred_energy']
    spread = (sub.std() / sub.median()).median()
    print(f'  median member-to-member spread per event: {spread*100:.2f}% of the prediction')

    print('\n== 5. is the residual still explained by the observables we have ==')
    for name, x in (('log E_true', np.log(g['true_energy'].values)),
                    ('ET', g['ET'].values),
                    ('region', g['region'].values.astype(float))):
        c = np.corrcoef(x, np.abs(r - np.median(r)))[0, 1]
        print(f'  corr(|residual|, {name:10s}) = {c:+.3f}')


if __name__ == '__main__':
    main()
