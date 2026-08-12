import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH

EXAMPLES = """\
examples:
  uv run scripts/fit_resolution.py reports/predictions/minbias__SubNetW4CleanAuxExDnQdEma.csv

  # compare two models, and ask where each crosses a 5% resolution target
  uv run scripts/fit_resolution.py reports/predictions/minbias__*QdEma.csv --target 0.05

fits sigma_eff(E) = sqrt( (a/sqrt(E))^2 + (b/E)^2 + c^2 ) per calorimeter region:
  a  stochastic term, sampling fluctuations
  b  noise term in GeV -- for a pileup-dominated sample this is the pileup energy
     fluctuation inside the reconstruction window, and it is the term that dominates
     the low-energy resolution
  c  constant term, calibration and leakage
Also reports the energy at which each region crosses the target resolution, and the b
that would be required to reach the target at a chosen energy.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Fit the standard calorimeter resolution parametrisation per region '
                    'and report what limits the low-energy resolution.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csvs', nargs='+', help='prediction CSVs from scripts/train_picocal.py')
    ap.add_argument('--split', default='test')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='restrict to these seeds before averaging, for like-for-like fits')
    ap.add_argument('--bins', type=int, default=8, help='quantile energy bins per region (default: 8)')
    ap.add_argument('--min-n', type=int, default=25, help='skip bins with fewer clusters (default: 25)')
    ap.add_argument('--freeze-ac', action='store_true',
                    help='fit only the noise term b, holding a and c at zero. Use when the '
                         'free 3-parameter fit hits the a>=0 / c>=0 bound, which makes a and c '
                         'meaningless and their errors enormous.')
    ap.add_argument('--target', type=float, default=0.05,
                    help='resolution target used for the crossing energy (default: 0.05)')
    ap.add_argument('--at', type=float, nargs='*', default=[10.0, 20.0, 40.0],
                    help='energies (GeV) at which to quote the fitted resolution')
    return ap.parse_args()


def model(E, a, b, c):
    return np.sqrt((a / np.sqrt(E)) ** 2 + (b / E) ** 2 + c ** 2)


def load(path, split, keep=None):
    df = pd.read_csv(path)
    df = df[df['split'] == split].reset_index(drop=True)
    if keep:
        df = df[df['seed'].isin(keep)].reset_index(drop=True)
    seeds = sorted(df['seed'].unique())
    pred = df.pivot_table(index=df.groupby('seed').cumcount(), columns='seed',
                          values='pred_energy').mean(axis=1).to_numpy()
    b0 = df[df['seed'] == seeds[0]]
    return (b0['true_energy'].to_numpy(), pred, b0['region_name'].to_numpy(),
            b0['model'].iloc[0], len(seeds))


def main():
    args = parse_args()
    order = [f'{int(p)}mm' for p in PITCH]
    for path in args.csvs:
        true, pred, region, name, nseed = load(path, args.split, args.seeds)
        print(f'\n{name} ({nseed} seeds)')
        head = (f"{'region':>7s} {'a':>8s} {'b [GeV]':>13s} {'c':>7s} "
                + ' '.join(f'{f"sig@{e:g}":>7s}' for e in args.at)
                + f" {f'E({args.target:g})':>12s} {'b need':>8s} {'chi2/ndf':>7s}")
        print(head)
        for reg in order:
            sel = region == reg
            t, p = true[sel], pred[sel]
            if len(t) < 8 * args.min_n:
                continue
            edges = np.quantile(t, np.linspace(0, 1, args.bins + 1))
            xs, ys, es = [], [], []
            for i in range(args.bins):
                m = (t >= edges[i]) & (t < edges[i + 1])
                if int(m.sum()) < args.min_n:
                    continue
                sg = resolution(p[m], t[m])['sigma_eff']
                xs.append(float(np.median(t[m])))
                ys.append(sg)
                es.append(0.96 * sg / np.sqrt(int(m.sum())))
            if len(xs) < 3:
                continue
            xs, ys, es = np.array(xs), np.array(ys), np.array(es)
            try:
                if args.freeze_ac:
                    (b,), pcov = curve_fit(lambda E, b: model(E, 0.0, b, 0.0), xs, ys,
                                           p0=[1.0], sigma=es, absolute_sigma=True,
                                           bounds=([0], [50]), maxfev=20000)
                    a = c = 0.0
                    perr = [0.0, float(np.sqrt(pcov[0, 0])), 0.0]
                    ndf = len(xs) - 1
                else:
                    (a, b, c), pcov = curve_fit(model, xs, ys, p0=[0.1, 1.0, 0.02],
                                                sigma=es, absolute_sigma=True,
                                                bounds=([0, 0, 0], [2, 50, 0.5]), maxfev=20000)
                    perr = list(np.sqrt(np.diag(pcov)))
                    ndf = len(xs) - 3
            except RuntimeError:
                print(f'{reg:>7s}  fit did not converge')
                continue
            chi2 = float((((ys - model(xs, a, b, c)) / es) ** 2).sum())
            flag = ''
            if not args.freeze_ac and (a < 1e-6 or c < 1e-6):
                flag = ' [a or c at bound: rerun with --freeze-ac]'
            if ndf > 0 and chi2 / ndf > 5:
                flag += ' [poor fit]'
            grid = np.linspace(1, 100, 4000)
            below = grid[model(grid, a, b, c) < args.target]
            ecross = below[0] if len(below) else float('nan')
            emed = float(np.median(t))
            need = args.target ** 2 - (a / np.sqrt(emed)) ** 2 - c ** 2
            bneed = emed * np.sqrt(need) if need > 0 else float('nan')
            print(f'{reg:>7s} {a:8.4f} {b:6.2f}+-{perr[1]:<5.2f} {c:7.4f} '
                  + ' '.join(f'{model(e, a, b, c):7.3f}' for e in args.at)
                  + f' {ecross:12.1f} {bneed:8.2f} {chi2 / max(ndf, 1):7.1f}{flag}')
        print(f'  "b needed" = noise term that would reach sigma={args.target:g} at the '
              f'region median energy; compare with the fitted b.')


if __name__ == '__main__':
    main()
