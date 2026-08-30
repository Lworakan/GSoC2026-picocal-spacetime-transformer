import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import resolution

# The GNN work on this detector (Vetens et al., ICHEP 2026, paper forthcoming) plots exactly the
# quantity this project reports -- sigma_eff, the half width of the central 68% interval -- against
# transverse energy, for clusters seeded in the SpaCal-Pb region, on single photons overlaid with
# minimum bias. Our sample already sits in their 0.5-5 GeV E_T range. The region matches because
# their Shashlik modules are 4x4 cm, which is our 40 mm; SpaCal-Pb sits between the tungsten
# SpaCal and the Shashlik, which puts it at our 30 mm.
#
# The plot is drawn so the two can be laid side by side. It is NOT a claim about their numbers:
# the talk reports results as curves without tabulated values, the samples differ (isolated
# photons overlaid with pileup against real minimum-bias events) and so does the pileup condition.

REPO = Path(__file__).resolve().parents[2]
D = REPO / 'reports' / 'predictions'
BEST = 'minbias__SubNetW8CleanAuxExDnGs50RcOvV2CrQdEma.csv'
CTRL = 'minbias__SubNetW8CleanAuxExDnRcQdEma.csv'
SEEDS = [0, 1, 2]
EDGES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])


def curve(f, region):
    t = pd.read_csv(D / f)
    t = t[t.seed.isin(SEEDS)]
    e = t.groupby(['true_energy', 'region_name', 'ET'], sort=False).agg(
        p=('pred_energy', 'median')).reset_index()
    if region:
        e = e[e.region_name == region]
    x, y, n = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (e.ET >= lo) & (e.ET < hi)
        if k.sum() < 40:
            continue
        s = e[k]
        x.append(0.5 * (lo + hi))
        y.append(resolution(s['p'].values, s.true_energy.values)['sigma_eff'])
        n.append(int(k.sum()))
    return np.array(x), np.array(y), n


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, (region, title) in zip(ax, [('30mm', 'SpaCal-Pb equivalent (30 mm)'),
                                   (None, 'all regions')]):
    for f, lab, st in ((CTRL, 'transformer, starting configuration', '--'),
                       (BEST, 'this work', '-')):
        x, y, n = curve(f, region)
        a.plot(x, 100 * y, st, marker='o', ms=4, label=lab)
    a.set_title(title, fontsize=10)
    a.set_xlabel(r'$E_{\mathrm{T}}$ [GeV]')
    a.grid(alpha=.3)
ax[0].set_ylabel(r'$\sigma_{\mathrm{eff}}$ [%]')
ax[0].legend(fontsize=8)
fig.suptitle(r'$\sigma_{\mathrm{eff}}$ (half width of the central 68% interval) vs $E_{\mathrm{T}}$,'
             ' minimum-bias sample', fontsize=10)
fig.tight_layout()
out = REPO / 'paper' / 'figs' / 'vs_gnn.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')

print(f'{"E_T bin":>12s} {"30mm n":>7s} {"30mm":>8s} {"all n":>7s} {"all":>8s}')
xa, ya, na = curve(BEST, '30mm')
xb, yb, nb = curve(BEST, None)
for i in range(len(xa)):
    print(f'{EDGES[i]:5.1f}-{EDGES[i+1]:4.1f} {na[i]:7d} {100*ya[i]:7.2f}% {nb[i]:7d} {100*yb[i]:7.2f}%')
print(f'\nwrote {out}')
