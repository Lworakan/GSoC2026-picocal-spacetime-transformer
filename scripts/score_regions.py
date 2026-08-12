import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH

EXAMPLES = """\
examples:
  # score one model per region and per energy tercile
  uv run scripts/score_regions.py reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv

  # compare an experiment against the champion (delta column, negative = better)
  uv run scripts/score_regions.py \\
      reports/predictions/minbias__SubNetW4CleanAuxPhysOccQdEma.csv \\
      --baseline reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv

  # every experiment at once, low-energy bins only
  uv run scripts/score_regions.py reports/predictions/minbias__*.csv \\
      --baseline reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv --bin low

reads the CSVs written by scripts/train_picocal.py (seeds are averaged per file);
energy bins are terciles of true energy computed within each region.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Score prediction CSVs per calorimeter region and per energy bin, '
                    'the diagnostic that isolates the low-energy inner-region failure. '
                    'Prints sigma_eff with its statistical error 0.96*sigma_eff/sqrt(n).',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csvs', nargs='+', help='one or more prediction CSVs')
    ap.add_argument('--baseline', default=None,
                    help='reference CSV; adds a delta column (negative = better than reference)')
    ap.add_argument('--split', default='test', help='data split to score (default: test)')
    ap.add_argument('--bin', choices=['all', 'low', 'mid', 'high'], default='all',
                    help='restrict the energy-bin rows that are printed (default: all)')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='restrict to these seeds before averaging; use to compare a 1-seed '
                         'experiment against one seed of a multi-seed baseline fairly')
    ap.add_argument('--min-n', type=int, default=30,
                    help='skip cells with fewer clusters than this (default: 30)')
    return ap.parse_args()


def load(path, split, keep_seeds=None):
    df = pd.read_csv(path)
    df = df[df['split'] == split].reset_index(drop=True)
    if keep_seeds:
        df = df[df['seed'].isin(keep_seeds)].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f'no rows for seeds {keep_seeds} in {path}')
    seeds = sorted(df['seed'].unique())
    blocks = [df[df['seed'] == s].reset_index(drop=True) for s in seeds]
    t0 = blocks[0]['true_energy'].to_numpy()
    pred = np.mean([b['pred_energy'].to_numpy() for b in blocks], axis=0)
    b0 = blocks[0]
    return dict(name=b0['model'].iloc[0], dataset=b0['dataset'].iloc[0], n_seeds=len(seeds),
                true=t0, pred=pred, region=b0['region_name'].to_numpy())


def score(d, min_n):
    out = {}
    r = resolution(d['pred'], d['true'])
    out[('all', 'all')] = (len(d['true']), r['sigma_eff'], r['bias'])
    for reg in np.unique(d['region']):
        sel = d['region'] == reg
        t, p = d['true'][sel], d['pred'][sel]
        if len(t) >= min_n:
            r = resolution(p, t)
            out[(reg, 'all')] = (len(t), r['sigma_eff'], r['bias'])
        q1, q2 = np.quantile(t, [1 / 3, 2 / 3])
        for lab, m in (('low', t < q1), ('mid', (t >= q1) & (t < q2)), ('high', t >= q2)):
            if int(m.sum()) < min_n:
                continue
            r = resolution(p[m], t[m])
            out[(reg, lab)] = (int(m.sum()), r['sigma_eff'], r['bias'])
    return out


def main():
    args = parse_args()
    base = (score(load(args.baseline, args.split, args.seeds), args.min_n)
            if args.baseline else None)
    order = [f'{int(p)}mm' for p in PITCH] + ['all']
    bins = ['all', 'low', 'mid', 'high'] if args.bin == 'all' else ['all', args.bin]

    for path in args.csvs:
        d = load(path, args.split, args.seeds)
        s = score(d, args.min_n)
        print(f"\n{d['name']} ({d['dataset']}, {d['n_seeds']} seeds)")
        hdr = f"{'region':>8s} {'Ebin':>5s} {'n':>6s} {'sigma_eff':>10s} {'err':>7s} {'bias':>8s}"
        print(hdr + (f" {'delta':>8s}" if base else ''))
        for reg in order:
            for b in bins:
                if (reg, b) not in s:
                    continue
                n, sig, bias = s[(reg, b)]
                row = f'{reg:>8s} {b:>5s} {n:6d} {sig:10.4f} {0.96 * sig / np.sqrt(n):7.4f} {bias:8.4f}'
                if base:
                    ref = base.get((reg, b))
                    row += f' {sig - ref[1]:+8.4f}' if ref else f" {'-':>8s}"
                print(row)


if __name__ == '__main__':
    main()
