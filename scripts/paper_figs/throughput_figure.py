import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def seff(r):
    r = np.sort(np.asarray(r, float)); n = len(r)
    k = int(np.ceil(0.683*n)); w = r[k-1:] - r[:n-k+1]
    return float(w[np.argmin(w)]/2)

D = Path('reports/predictions')
champ = pd.read_csv(D/'minbias__SubNetW4CleanAuxExDnQdEma.csv')
s0 = champ[champ.seed == champ.seed.unique()[0]]
champ_s0 = seff((s0.pred_energy-s0.true_energy)/s0.true_energy)
g = champ.groupby(['true_energy','region_name'], sort=False).pred_energy.median()
e = np.array([k[0] for k in g.index], float)
champ_ens = seff((g.to_numpy()-e)/e)
cs = pd.read_csv(D/'minbias__CalibratedSum.csv')
cs_s = seff((cs.pred_energy-cs.true_energy)/cs.true_energy)
bdt = pd.read_csv(D/'minbias__BDT.csv')
bdt_s = seff((bdt.pred_energy-bdt.true_energy)/bdt.true_energy)

pts = [
    ('Calibrated sum (analytic)', 5.65e6, cs_s, 'v', 'tab:gray', (-30, -14)),
    ('GBDT (6 features)', 1.66e4, bdt_s, 's', 'tab:orange', (8, 6)),
    ('SpaceTformer $9{\\times}9$, 1 model', 4.96e3, champ_s0, 'o', 'tab:cyan', (8, 6)),
    ('SpaceTformer $9{\\times}9$, 5-seed ens.', 9.9e2, champ_ens, 'D', 'tab:blue', (9, -3)),
    ('SpaceTformer $17{\\times}17$ rec., 1 model', 5.55e2, 0.0397, 'o', 'tab:red', (-8, 9)),
    ('SpaceTformer $17{\\times}17$ rec., 5-member ens.', 1.11e2, 0.0388, '*', 'darkred', (-14, 9)),
]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for name, x, y, mk, col, off in pts:
    ax.scatter([x], [y], marker=mk, s=90 if mk=='*' else 60, zorder=3, color=col, label=name)
    ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=off, fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('inference throughput [clusters/s], CPU (i9-13900HX), batch 64')
ax.set_ylabel(r'aggregate $\sigma_{\mathrm{eff}}$')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=7.5, loc='upper left')
fig.tight_layout()
fig.savefig('paper/figs/accuracy_vs_throughput.pdf')
print('ok', cs_s, bdt_s, champ_s0, champ_ens)
