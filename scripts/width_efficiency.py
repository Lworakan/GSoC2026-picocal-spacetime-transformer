import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution

# sigma_eff vs selection efficiency using the model's own predicted interquartile width as the
# quality flag. Protocol guards against the cheap version of this result: a GLOBAL width cut
# preferentially removes low-energy events (whose width is legitimately larger) and buys
# sigma_eff by emptying the hard bins, not by removing wrong-match events. So the threshold is
# the width quantile WITHIN each (region x validation-energy-decile) cell, fitted on the
# validation split and applied unchanged to the test split.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--name', default='SubNetW8CleanAuxExDnAuxRcQdEma')
    ap.add_argument('--keeps', type=float, nargs='+',
                    default=[1.0, 0.975, 0.95, 0.90, 0.85, 0.80])
    args = ap.parse_args()
    repo = Path(args.repo)
    P = repo / 'reports' / 'predictions'
    w = pd.read_csv(P / f'minbias__{args.name}_width.csv')
    pr = pd.read_csv(P / f'minbias__{args.name}.csv')
    pr = pr[pr['split'] == 'test']
    ens = pr.groupby(['true_energy', 'region', 'ET'], sort=False)['pred_energy'].median().reset_index()

    wv = w[w['split'] == 'val'].copy()
    wt = w[w['split'] == 'test'].copy()
    m = wt.merge(ens, on=['true_energy', 'region', 'ET'])
    edges = {}
    for r, g in wv.groupby('region'):
        le = np.log(g['true_energy'])
        edges[r] = np.quantile(le, np.linspace(0, 1, 11))

    def cell(d):
        le = np.log(d['true_energy'].values)
        e = np.stack([edges[r] for r in d['region']])
        ib = np.array([np.clip(np.searchsorted(e[i], le[i], side='right') - 1, 0, 9)
                       for i in range(len(le))])
        return d['region'].values * 10 + ib

    wv['cell'] = cell(wv)
    m['cell'] = cell(m)
    rows = []
    for keep in args.keeps:
        thr = wv.groupby('cell')['width'].quantile(keep)
        sel = m['width'].values <= thr.reindex(m['cell']).values + 1e-12
        pe, et = m['pred_energy'].values[sel], m['true_energy'].values[sel]
        row = dict(keep_target=keep, eff=round(sel.mean(), 3),
                   sigma_eff=round(resolution(pe, et)['sigma_eff'], 4))
        for r in sorted(m['region'].unique()):
            k = sel & (m['region'].values == r)
            le = np.log(m['true_energy'].values)
            lo = k & (le <= edges[r][int(10 / 3)])
            row[f'r{r}'] = round(resolution(m['pred_energy'].values[k], m['true_energy'].values[k])['sigma_eff'], 4)
            row[f'r{r}low'] = round(resolution(m['pred_energy'].values[lo], m['true_energy'].values[lo])['sigma_eff'], 4)
        rows.append(row)
        print(row, flush=True)
    out = repo / 'reports' / 'width_efficiency.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
