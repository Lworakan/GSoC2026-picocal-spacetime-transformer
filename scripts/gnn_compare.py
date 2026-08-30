import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / 'reports' / 'predictions'
CACHE = REPO / '.scratch' / 'cache' / 'minbias_94.pkl'
BEST = 'minbias__SubNetW8CleanAuxExDnGs50RcOvV2CrQdEma.csv'
SEEDS = [0, 1, 2]
EPS = 1e-9

# Read off slide 11 of the ICHEP 2026 talk on the graph-network reconstruction for this
# detector (W. Vetens, 31 July 2026), "GNNMP versus Standard Approach". The talk plots
# sigma_eff(dE/E) against transverse energy for cluster seeds in the SpaCal-Pb region, on
# single photons with minimum-bias clusters, at a luminosity of 1.5e34 cm^-2 s^-1.
#
# The numbers below are not tabulated in the talk; they were measured from the figure by
# --digitise, which calibrates on the axis frame and then finds, inside each bin, the row
# of pixels of each series' colour that spans the bin. The vertical scale is 40% over 291
# pixels, so a row is 0.137% and the read-off error is about +/-0.2% absolute. Every value
# but one was found on a bar that is unbroken across its bin; the GNNMP point at 1.4-2.3
# is crossed by two other curves and only a tenth of its bar survives, so it carries a
# larger uncertainty than the rest.
#
# They are someone else's PRELIMINARY numbers. Nothing derived from them belongs in the
# paper until Felipe and Carla have seen this table and agreed both to the values and to
# their being reproduced.
TALK_BINS = [(0.5, 1.4), (1.4, 2.3), (2.3, 3.2), (3.2, 4.1), (4.1, 5.0)]
TALK = {
    '3x3 clustering':    [37.66, 19.11, 13.20, 10.31, 8.38],
    'opt. cluster shapes': [23.92, 13.75, 9.48, 7.70, 6.74],
    'GNNMP model':       [18.97, 8.66, 6.46, 5.36, 4.54],
}


def digitise(path):
    from PIL import Image
    rgb = Image.open(path).convert('RGB')
    hsv = np.asarray(rgb.convert('HSV')).astype(float)[140:545, 40:620]
    H, S, V = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    x0, x1, y0, y1 = 128, 517, 335, 44          # E_T 0.5..5.0 GeV, sigma_eff 0..40 %
    ey = lambda y: (y0 - y) * 40.0 / (y0 - y1)
    px = lambda e: int(round(x0 + (e - 0.5) * (x1 - x0) / 4.5))
    base = (S > 0.15) & (V > 0.3)
    hue = {'3x3 clustering': (150, 190), 'opt. cluster shapes': (15, 45), 'GNNMP model': (230, 265)}
    out = {}
    for name, (h0, h1) in hue.items():
        m = base & (H >= h0) & (H <= h1)
        m[:45] = m[334:] = False
        m[:, :129] = m[:, 518:] = False
        m[:189, 206:] = False                   # the legend, which repeats every colour
        vals = []
        for lo, hi in TALK_BINS:
            a, b = px(lo) + 8, px(hi) - 8
            f, y = max((m[y, a:b].mean(), y) for y in range(46, 333))
            vals.append((round(ey(y), 2), round(f, 2)))
        out[name] = vals
    return out


def sum3(ev):
    k = (np.abs(ev['di']) <= 1) & (np.abs(ev['dj']) <= 1)
    return float(ev['e'][k].sum())


def ours(regions, seeds=SEEDS):
    t = pd.read_csv(PRED / BEST)
    t = t[t.seed.isin(seeds)]
    e = t.groupby(['true_energy', 'region_name', 'ET'], sort=False).agg(
        p=('pred_energy', 'median')).reset_index()
    return e[e.region_name.isin(regions)] if regions else e


def binned(pred, true, et, bins):
    out = []
    for lo, hi in bins:
        k = (et >= lo) & (et < hi)
        out.append((resolution(pred[k], true[k])['sigma_eff'] * 100, int(k.sum()))
                   if k.sum() >= 40 else (float('nan'), int(k.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--digitise', help='path to the talk slide, to re-measure TALK')
    ap.add_argument('--region', default='30mm',
                    help='comma separated; SpaCal-Pb is 30mm and 40mm together')
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    regions = args.region.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    if args.digitise:
        for name, v in digitise(args.digitise).items():
            print(f'{name:22s} ' + '  '.join(f'{s:5.2f}% (bar {f:.2f})' for s, f in v))
        return

    print(f'talk, cluster seeds in the SpaCal-Pb region, luminosity 1.5e34')
    for name, v in TALK.items():
        print(f'  {name:22s} ' + '  '.join(f'{s:5.2f}%' for s in v))

    # Our own 3x3 sum, on our own events, in their bins. This is the anchor: their
    # standard algorithm and ours are the same estimator, so if the two 3x3 curves sit on
    # top of each other the samples are in the same pileup regime and the absolute
    # comparison is allowed. If they do not, only the improvement over 3x3 is comparable.
    ev = pickle.load(open(CACHE, 'rb'))
    s3 = np.array([sum3(e) for e in ev], np.float64)
    etrue = np.array([e['Etrue'] for e in ev], np.float64)
    et = np.array([e['ET'] for e in ev], np.float64)
    reg = np.array([e['reg'] for e in ev])

    o = ours(regions, seeds)
    key = set(zip(np.float32(o.true_energy), np.float32(o.ET)))
    is_test = np.array([(np.float32(a), np.float32(b)) in key for a, b in zip(etrue, et)])
    codes = [{'15mm': 0, '30mm': 1, '40mm': 2, '60mm': 3, '120mm': 4}[r] for r in regions]
    rmask = np.isin(reg, codes)
    tr, te = rmask & ~is_test, rmask & is_test
    print(f'\n{"+".join(regions)}, seeds {seeds}: {tr.sum()} events to calibrate the 3x3 sum, '
          f'{te.sum()} to score it')

    # one calibration per region: the cells differ in size, so they differ in containment
    p3 = np.zeros(te.sum())
    ste = reg[te]
    for c in codes:
        a, b = np.polyfit(np.log(s3[tr & (reg == c)] + EPS), np.log(etrue[tr & (reg == c)]), 1)
        p3[ste == c] = np.exp(a * np.log(s3[te & (reg == c)] + EPS) + b)

    # The gradient-boosted reference, in the same bins. It is the control on how much of the
    # gap between our 3x3 sum and their "3x3 clustering" could be algorithm quality rather
    # than sample difficulty: a moderately better estimator on identical events.
    g = pd.read_csv(PRED / 'minbias__BDT.csv')
    g = g[g.region_name.isin(regions)]

    rows = [('3x3 sum, calibrated (this sample)', binned(p3, etrue[te], et[te], TALK_BINS)),
            ('gradient-boosted trees (this sample)',
             binned(g.pred_energy.values, g.true_energy.values, g.ET.values, TALK_BINS)),
            ('this work', binned(o['p'].values, o.true_energy.values, o.ET.values, TALK_BINS))]
    hdr = '  '.join(f'{lo:.1f}-{hi:.1f}'.rjust(11) for lo, hi in TALK_BINS)
    print(f'\n{"":34s} {hdr}')
    for name, v in rows:
        print(f'{name:34s} ' + '  '.join(f'{s:6.2f}% ({n:4d})' for s, n in v))
    for name, v in TALK.items():
        print(f'{name + " (talk)":34s} ' + '  '.join(f'{s:6.2f}%       ' for s in v))

    # The two 3x3 curves are the same estimator on two samples, so their ratio is what the
    # samples differ by -- pileup, calibration, whatever else. Dividing it out is the
    # comparison that does not depend on knowing their luminosity or ours.
    print('\nimprovement over the 3x3 baseline of the same sample')
    base = np.array([s for s, _ in rows[0][1]])
    ourv = np.array([s for s, _ in rows[-1][1]])
    talk3, talkg = np.array(TALK['3x3 clustering']), np.array(TALK['GNNMP model'])
    print(f'{"this work / our 3x3":34s} ' + '  '.join(f'{v:6.2f}x       ' for v in base / ourv))
    print(f'{"GNNMP / their 3x3":34s} ' + '  '.join(f'{v:6.2f}x       ' for v in talk3 / talkg))
    print(f'\nour sample against theirs, read on the shared 3x3 estimator: '
          + '  '.join(f'{v:.2f}' for v in base / talk3))
    print(f'{"this work, scaled to their sample":34s} '
          + '  '.join(f'{v:6.2f}%       ' for v in ourv * (talk3 / base)))


if __name__ == '__main__':
    main()
