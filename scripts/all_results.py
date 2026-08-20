import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

EXAMPLES = """\
example:
  uv run scripts/all_results.py --sort low15
  uv run scripts/all_results.py --seeds 0 1 --min-seeds 2

One table for every experiment that has a prediction file, scored the same way.

The per-experiment reports each answer one question and are easy to lose track of. This reads
every reports/predictions/minbias__*.csv and prints sigma_eff for the aggregate and for the two
bins that are still above the 0.07 target -- 15mm low-E and 30mm low-E -- with the per-bin
statistical error beside them, so a difference smaller than its error is visible as such.

Rows are compared like-for-like: pass --seeds to restrict every model to the same seeds, since a
five-seed model and a one-seed model are not comparable and the seed count is printed for that
reason. Screening runs (one or two seeds) are marked; nothing from them belongs in a report.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Score every saved experiment into one comparable table.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='reports/predictions')
    ap.add_argument('--sample', default='minbias')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='restrict every model to these seeds for a like-for-like comparison')
    ap.add_argument('--min-seeds', type=int, default=1)
    ap.add_argument('--sort', choices=['all', 'low15', 'low30', 'name'], default='low15')
    ap.add_argument('--baseline', default='SubNetW4CleanAuxExDnQdEma')
    return ap.parse_args()


def seff(r):
    r = np.sort(np.asarray(r))
    n = len(r)
    if n < 25:
        return np.nan
    k = int(np.ceil(0.683 * n))
    w = r[k - 1:] - r[:n - k + 1]
    return float(w[np.argmin(w)] / 2)


def ens_seff(t):
    """sigma_eff of the SEED ENSEMBLE: median prediction per event, then one sigma_eff.

    Pooling the residuals of several seeds instead -- which this script did until 2026-08-17 --
    mixes models with slightly different bias and scale and so widens the combined distribution for
    a reason that has nothing to do with any single model's accuracy. It read worse than the mean of
    the per-seed values in every arm measured (W8 15mm low 0.1243 pooled against 0.1214 mean), and
    it is not the quantity that gets deployed. The ensemble is."""
    if t.seed.nunique() < 2:
        return seff((t.pred_energy - t.true_energy) / t.true_energy)
    g = t.groupby(['true_energy', 'region_name'], sort=False)
    p = g.pred_energy.median()
    e = np.array([k[0] for k in p.index], float)
    return seff((p.to_numpy() - e) / e)


def seed_sd(t):
    """Spread of sigma_eff ACROSS seeds -- the uncertainty that actually matters here.

    The statistical error 0.96*sigma/sqrt(n) computed on pooled rows understates it badly, because
    the same test events repeat for every seed and are therefore not independent. At 15mm low-E
    there are only 411 unique events per seed, and the same configuration returned 0.1138 to 0.1319
    across five seeds -- a spread wider than the remaining distance to the 0.10 target, which is why
    a one-seed number cannot support a claim."""
    if t.seed.nunique() < 2:
        return np.nan
    v = [seff((s.pred_energy - s.true_energy) / s.true_energy) for _, s in t.groupby('seed')]
    v = [x for x in v if x == x]
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def main():
    a = parse_args()
    d = Path(a.dir)
    rows = []
    # Tercile edges are taken from the BASELINE file and reused for every row. Letting each file
    # set its own edge silently scores rows on different energy intervals; taking it from whichever
    # file sorted first made "15mm low" read 0.20 where the champion's own edge gives 0.165.
    edges = {}
    bp = d / f'{a.sample}__{a.baseline}.csv'
    if bp.exists():
        bt = pd.read_csv(bp)
        if a.seeds:
            bt = bt[bt.seed.isin(a.seeds)]
        for reg in ('15mm', '30mm'):
            r = bt[bt.region_name == reg]
            if len(r) >= 100:
                edges[reg] = r.true_energy.quantile(1 / 3)
    for f in sorted(d.glob(f'{a.sample}__*.csv')):
        if '_smoke' in f.name:
            continue
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        if not {'pred_energy', 'true_energy', 'region_name', 'seed'} <= set(t.columns):
            continue
        if a.seeds:
            t = t[t.seed.isin(a.seeds)]
        ns = t.seed.nunique()
        if ns < a.min_seeds or len(t) < 500:
            continue
        t = t.assign(res=(t.pred_energy - t.true_energy) / t.true_energy)
        rec = {'name': f.name.split('__')[1][:-4], 'seeds': ns,
               'all': ens_seff(t), 'all_sd': seed_sd(t)}
        for reg in ('15mm', '30mm'):
            r = t[t.region_name == reg]
            if len(r) < 100:
                rec[f'low{reg[:2]}'] = np.nan
                rec[f'sd{reg[:2]}'] = np.nan
                continue
            # the tercile edge must come from ONE reference model, or each row would be scored
            # on a slightly different energy interval and the column would not be comparable
            if reg not in edges:
                edges[reg] = r.true_energy.quantile(1 / 3)
            s = r[r.true_energy < edges[reg]]
            rec[f'low{reg[:2]}'] = ens_seff(s)
            rec[f'sd{reg[:2]}'] = seed_sd(s)
            rec[f'n{reg[:2]}'] = len(s) // max(ns, 1)
        rows.append(rec)

    if not rows:
        raise SystemExit(f'no prediction files matched in {d}')
    df = pd.DataFrame(rows)
    key = {'all': 'all', 'low15': 'low15', 'low30': 'low30', 'name': 'name'}[a.sort]
    df = df.sort_values(key, na_position='last')
    base = df[df.name == a.baseline]

    print(f'\nsigma_eff of the SEED ENSEMBLE (median prediction per event). +- is the spread of')
    print(f'sigma_eff across seeds, which is the uncertainty that limits a claim here.\n')
    print(f'{"experiment":46s} {"sd":>3s} {"all":>8s} {"15mm low":>16s} {"30mm low":>16s}  vs champ')
    for _, r in df.iterrows():
        d15 = d30 = ''
        if len(base):
            b = base.iloc[0]
            for key, sdkey, slot in (('low15', 'sd15', 'd15'), ('low30', 'sd30', 'd30')):
                if r[key] == r[key] and b[key] == b[key]:
                    x = r[key] - b[key]
                    # significant only against the seed spread of BOTH arms added in quadrature,
                    # not against a statistical error that assumes independent test events
                    sd = np.hypot(r.get(sdkey) or 0.0, b.get(sdkey) or 0.0)
                    star = '*' if sd > 0 and abs(x) > 2 * sd else ' '
                    txt = f'{x:+.4f}{star}'
                    d15, d30 = (txt, d30) if key == 'low15' else (d15, txt)
        mark = '' if r['seeds'] >= 5 else '  (screening)'

        def cell(v, s):
            return f'{v:.4f}+-{s:.4f}' if s == s else f'{v:.4f}        '
        print(f'{r["name"]:46s} {r["seeds"]:3d} {r["all"]:8.4f} '
              f'{cell(r["low15"], r["sd15"]):>16s} {cell(r["low30"], r["sd30"]):>16s}  '
              f'{d15:>9s} {d30:>9s}{mark}')
    print('\n* marks a gap larger than twice the two arms seed spreads in quadrature.')
    print('One-seed rows carry no +-, so nothing about them is significant: at 15mm low-E the same')
    print('configuration spanned 0.1138-0.1319 over five seeds, wider than the gap to the target.')
    print('Target: every bin below 0.07. 13 of 15 bins already are; these two are the task.')


if __name__ == '__main__':
    main()
