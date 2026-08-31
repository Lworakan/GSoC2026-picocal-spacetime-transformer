import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution

# The inner regions sit a factor 1.9-2.7 above PicoCal's design resolution
# (10%/sqrt(E) + 1%, PoS(LHCP2024)301), and the excess is attributed to pileup rather than
# sampling: the design stochastic term is 0.10 and the fit reads 0.21 at 15 and 30 mm. If
# that attribution is right, an estimator that separated photon from pileup perfectly would
# land on the design curve. This measures that ceiling directly.
#
# The overlay sample carries per-cell truth: `sig` is the photon's share of each cell. The
# oracle sums the photon's share inside the same window the model sees, and is calibrated
# by the same power law used for every raw sum in this project. What is left in the oracle
# is not pileup -- it is the photon energy that fell outside the window, which no amount of
# pileup subtraction recovers. So the two numbers separate the budget:
#
#   ours - oracle   = what better pileup separation could still buy
#   oracle - design = what only a bigger or better-placed window could buy
#
# NOT YET VALID -- do not quote the numbers this prints. Three things have to be fixed
# first, and two of them may change the answer:
#
#  1. `sig` and `e` are on the same scale (median sig/e = 0.49 per event, which is just the
#     photon's share), and both are MeV against Etrue's GeV. The log-log fit absorbs that
#     constant identically in both columns, so the units are NOT the problem here -- this
#     line is recorded because it was the first suspicion and it was wrong.
#  2. The oracle is calibrated on the region's other two energy thirds and applied to this
#     one, so every number carries an extrapolation across energy that has nothing to do
#     with pileup. Calibrate within the bin, or on a held-out half of the same bin.
#  3. `sig` is energy *deposited* in the recorded cells; Etrue is the *incident* photon
#     energy. Their ratio therefore carries the sampling fluctuation, which the design
#     curve already includes. So "oracle - design" is not a window-quality budget, and the
#     oracle cannot reach the design curve by construction.
#
# And the comparison against the trained model is cross-sample -- the model's 0.0291 is on
# min-bias, this is on overlay -- so "the model beats the oracle" is not measured. Scoring
# the model on these same overlay events is what would make that line real.

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / '.scratch' / 'cache' / 'overlay.pkl'
EPS = 1e-9
PITCH = {0: '15mm', 1: '30mm', 2: '40mm', 3: '60mm', 4: '120mm'}


def design(E):
    return np.sqrt((0.10 / np.sqrt(E)) ** 2 + 0.01 ** 2)


def window(ev, w):
    return (np.abs(ev['di']) <= w) & (np.abs(ev['dj']) <= w)


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ev = pickle.load(open(CACHE, 'rb'))
    reg = np.array([e['reg'] for e in ev])
    etrue = np.array([e['Etrue'] for e in ev], float)
    sig = np.array([e['sig'][window(e, w)].sum() for e in ev], float)
    raw = np.array([e['e'][window(e, w)].sum() for e in ev], float)

    print('overlay events %d, window %dx%d' % (len(ev), 2 * w + 1, 2 * w + 1))
    print({PITCH[k]: int((reg == k).sum()) for k in sorted(set(reg))})
    print()
    print('%-7s %-5s %6s %6s %8s %8s %8s' %
          ('region', 'bin', 'n', 'medE', 'oracle', 'raw sum', 'design'))

    for k in sorted(set(reg)):
        m = reg == k
        if m.sum() < 150:
            continue
        q = np.quantile(etrue[m], [1 / 3, 2 / 3])
        cells = (('low', m & (etrue <= q[0])),
                 ('mid', m & (etrue > q[0]) & (etrue <= q[1])),
                 ('high', m & (etrue > q[1])))
        for lab, c in cells:
            if c.sum() < 50:
                continue
            # calibrate on the other two thirds of the region, score on this one
            fit = m & ~c
            a, b = np.polyfit(np.log(sig[fit] + EPS), np.log(etrue[fit]), 1)
            po = np.exp(a * np.log(sig[c] + EPS) + b)
            a2, b2 = np.polyfit(np.log(raw[fit] + EPS), np.log(etrue[fit]), 1)
            pr = np.exp(a2 * np.log(raw[c] + EPS) + b2)
            E = np.median(etrue[c])
            print('%-7s %-5s %6d %6.1f %8.4f %8.4f %8.4f' %
                  (PITCH[k], lab, c.sum(), E,
                   resolution(po, etrue[c])['sigma_eff'],
                   resolution(pr, etrue[c])['sigma_eff'], design(E)))


if __name__ == '__main__':
    main()
