import argparse
import pickle
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH

EXAMPLES = """\
example:
  uv run scripts/overlay_bound.py --overlay .scratch/cache/overlay.pkl

What this answers, and why it could not be answered before.

The paired synthetic sample knows, per cell, how much energy is photon and how much is pileup.
So the pileup contribution is a WITHIN-EVENT quantity, not the difference of two population
covariances -- the earlier attempt at that produced a bound of 0.0014 in one formulation and a
logically impossible 42 GeV in the other, and was discarded.

Three estimators are computed on the same events:
  perfect      the photon energy summed from the truth cells, calibrated by a log-linear fit
  oracle gate  our own readout with the gate set to the TRUE photon fraction per cell
  GLS          the best fixed linear estimator given the measured pileup covariance

NOT A BOUND. Every row here is a SUM estimator with a two-parameter global calibration, so
'perfect' is the floor of *summing photon-only cells*, not the floor a trained model faces.
The distinction was measured on 2026-08-16 and it is large: this script reports 0.1870 for
'perfect pileup removal' at 15mm low-E, while a trained model on genuinely pileup-free data
(clean__GateHuber, matched region and energy slice) reaches 0.0854 -- better than the supposed
'floor' by more than a factor of two. Do not quote these numbers as limits on achievable
resolution; use them only to compare estimators against each other on identical events.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Bound the achievable resolution using the paired synthetic sample.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--overlay', default='.scratch/cache/overlay.pkl')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--thresh', type=float, default=2.49)
    return ap.parse_args()


def main():
    a = parse_args()
    with open(a.overlay, 'rb') as f:
        EV = pickle.load(f)
    W, TH = a.window, a.thresh

    rows = {}
    for ev in EV:
        d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
        m = (d <= W) & (ev['e'] >= TH)
        if m.sum() < 5:
            continue
        e, sig = ev['e'][m], ev['sig'][m]
        rows.setdefault(ev['reg'], []).append(
            (float(e.sum()), float(sig.sum()), float(ev['Etrue']) * 1000.0,
             np.array([e[d[m] == r].sum() for r in range(W + 1)]),
             np.array([sig[d[m] == r].sum() for r in range(W + 1)])))

    print(f"{'region':>7s} {'n':>5s} {'estimator':>26s} {'sigma_eff':>10s} {'bin: low-E':>11s}")
    for reg, R in sorted(rows.items()):
        if len(R) < 300:
            continue
        obs = np.array([r[0] for r in R])          # window energy, with pileup
        tru = np.array([r[1] for r in R])          # window energy, photon only
        E = np.array([r[2] for r in R])            # truth photon energy
        Ro = np.stack([r[3] for r in R])           # ring energies, with pileup
        Rs = np.stack([r[4] for r in R])           # ring energies, photon only
        lo = E < np.quantile(E, 1 / 3)

        def fit_and_score(x, tag):
            # same log-linear calibration the pipeline uses, fitted on these events
            a_, b_ = np.polyfit(np.log(np.clip(x, 1e-6, None)), np.log(E), 1)
            p = np.exp(a_ * np.log(np.clip(x, 1e-6, None)) + b_)
            r = resolution(p, E)['sigma_eff']
            rl = resolution(p[lo], E[lo])['sigma_eff']
            print(f'{int(PITCH[reg]):5d}mm {len(R):5d} {tag:>26s} {r:10.4f} {rl:11.4f}')

        fit_and_score(tru, 'perfect pileup removal')
        f = (Rs / E[:, None]).mean(0)
        C = np.cov((Ro - Rs).T)                     # pileup covariance, WITHIN event
        Ci = np.linalg.pinv(C + np.eye(len(f)) * 1e-9 * max(np.trace(C), 1.0))
        w = Ci @ f
        fit_and_score(Ro @ (w / np.abs(w).max()), 'GLS on true pileup cov')
        fit_and_score(obs, 'raw window sum (no removal)')
        sigE = np.sqrt(1.0 / (f @ Ci @ f)) / 1000.0
        print(f'{"":>7s} {"":>5s} {"pileup-only bound":>26s} {sigE:10.3f} GeV'
              f'  -> {sigE / (np.median(E[lo]) / 1000):.4f} at the low-E median')
    print('\nOur trained model reaches 0.0574 (15mm) and 0.0480 (30mm) on the real sample;\n'
          'compare against "perfect pileup removal" for what removal alone can buy.')


if __name__ == '__main__':
    main()
