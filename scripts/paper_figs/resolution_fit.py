import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
def clean_log_axis(ax, axis='x', ticks=(1, 2, 5, 10, 20, 50, 100)):
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_locator(FixedLocator(list(ticks)))
    a.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    a.set_minor_formatter(NullFormatter())


D = Path('reports/predictions')
OUT = Path('paper/figs')
REGIONS = ['15mm', '30mm', '40mm', '60mm', '120mm']
rng = np.random.default_rng(0)


def seff(r):
    r = np.sort(np.asarray(r, float))
    n = len(r)
    if n < 25:
        return np.nan
    k = int(np.ceil(0.683 * n))
    w = r[k - 1:] - r[:n - k + 1]
    return float(w[np.argmin(w)] / 2)


def boot(r, nb=200):
    r = np.asarray(r, float)
    v = [seff(rng.choice(r, len(r), replace=True)) for _ in range(nb)]
    return float(np.std(v, ddof=1))


folds = []
pats = ['minbias__SubNetW8CleanAuxExDnRcK{f}QdEma.csv',
        'minbias__SubNetW8CleanAuxExDnRcK{f}Rr01QdEma.csv',
        'minbias__SubNetW4CleanAuxExDnRcK{f}AcQdEma.csv']
for f in range(10):
    tabs = []
    for pat in pats:
        t = pd.read_csv(D / pat.format(f=f))
        tabs.append(t)
    allm = pd.concat(tabs)
    g = allm.groupby(['true_energy', 'region_name'], sort=False)
    ens = g.agg(pred=('pred_energy', 'median')).reset_index()
    folds.append(ens)
cv = pd.concat(folds)
cv['res'] = (cv.pred - cv.true_energy) / cv.true_energy


def model(E, a, b, c):
    return np.sqrt((a / np.sqrt(E)) ** 2 + (b / E) ** 2 + c ** 2)


EB = np.geomspace(1, 100, 11)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
for reg, col in zip(REGIONS, colors):
    r = cv[cv.region_name == reg]
    xs, ys, es = [], [], []
    for i in range(len(EB) - 1):
        sel = r[(r.true_energy >= EB[i]) & (r.true_energy < EB[i + 1])]
        if len(sel) < 40:
            continue
        xs.append(np.sqrt(EB[i] * EB[i + 1]))
        ys.append(seff(sel.res))
        es.append(boot(sel.res))
    xs, ys, es = map(np.asarray, (xs, ys, es))
    try:
        p, _ = curve_fit(model, xs, ys, p0=[0.1, 0.2, 0.02], sigma=es,
                         bounds=([0, 0, 0], [2, 5, 0.2]), maxfev=20000)
        lab = (f'{reg}: $a$={p[0]:.3f}, $b$={p[1]:.2f}, $c$={p[2]:.3f}')
        xf = np.geomspace(xs.min() * 0.9, xs.max() * 1.1, 200)
        ax.plot(xf, model(xf, *p), '-', color=col, lw=1.2)
    except Exception as e:
        lab = f'{reg}: fit failed'
        print(reg, 'fit failed', e)
    ax.errorbar(xs, ys, yerr=es, fmt='o', ms=4, color=col, capsize=2, label=lab)
    print(reg, 'fit params', lab)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$E_{\mathrm{true}}$ [GeV]')
ax.set_ylabel(r'$\sigma_{\mathrm{eff}}$')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8, title=r'$\sigma_{\mathrm{eff}}(E)=a/\sqrt{E}\ \oplus\ b/E\ \oplus\ c$', title_fontsize=8)
fig.tight_layout()
clean_log_axis(ax, 'x')
clean_log_axis(ax, 'y', (0.02, 0.03, 0.05, 0.1, 0.2))
fig.tight_layout()
fig.savefig(OUT / 'resolution_fit.pdf')
print('wrote resolution_fit.pdf')

champ = pd.read_csv(D / 'minbias__SubNetW4CleanAuxExDnQdEma.csv')
champ['res'] = (champ.pred_energy - champ.true_energy) / champ.true_energy
s0 = champ[champ.seed == champ.seed.unique()[0]]
g = champ.groupby(['true_energy', 'region_name'], sort=False)
p = g.pred_energy.median()
e = np.array([k[0] for k in p.index], float)
champ_ens = seff((p.to_numpy() - e) / e)
champ_s0 = seff(s0.res)

cs = pd.read_csv(D / 'minbias__CalibratedSum.csv')
cs_seff = seff((cs.pred_energy - cs.true_energy) / cs.true_energy)
bdt = pd.read_csv(D / 'minbias__BDT.csv')
bdt_seff = seff((bdt.pred_energy - bdt.true_energy) / bdt.true_energy)
print(f'calibsum {cs_seff:.4f}  bdt {bdt_seff:.4f}  champ_s0 {champ_s0:.4f}  champ_ens {champ_ens:.4f}')

pts = [
    ('Calibrated sum (analytic)', 5.65e6, cs_seff, 'v'),
    ('GBDT (6 features)', 1.66e4, bdt_seff, 's'),
    ('SpaceTformer, 1 model', 4.96e3, champ_s0, 'o'),
    ('SpaceTformer, 5-seed ens.', 9.9e2, champ_ens, 'D'),
]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for name, x, y, mk in pts:
    ax.scatter([x], [y], marker=mk, s=70, zorder=3, label=f'{name}')
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(8, 6), fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('inference throughput [clusters/s], CPU (i9-13900HX), batch 64')
ax.set_ylabel(r'aggregate $\sigma_{\mathrm{eff}}$')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(OUT / 'accuracy_vs_throughput.pdf')
print('wrote accuracy_vs_throughput.pdf')
