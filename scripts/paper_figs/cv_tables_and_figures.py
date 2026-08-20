import numpy as np
import pandas as pd
from pathlib import Path
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
OUT.mkdir(parents=True, exist_ok=True)
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


def boot(r, nb=400):
    r = np.asarray(r, float)
    v = [seff(rng.choice(r, len(r), replace=True)) for _ in range(nb)]
    return float(np.std(v, ddof=1))


base = pd.read_csv(D / 'minbias__SubNetW4CleanAuxExDnQdEma.csv')
edges = {}
for reg in REGIONS:
    e = base[base.region_name == reg].true_energy
    edges[reg] = (e.quantile(1 / 3), e.quantile(2 / 3))

members = {
    'RcK{f}': 'minbias__SubNetW8CleanAuxExDnRcK{f}QdEma.csv',
    'Rr01K{f}': 'minbias__SubNetW8CleanAuxExDnRcK{f}Rr01QdEma.csv',
    'AcK{f}': 'minbias__SubNetW4CleanAuxExDnRcK{f}AcQdEma.csv',
}
folds = []
single_member = []
for f in range(10):
    tabs = []
    mid = 0
    for pat in members.values():
        p = D / pat.format(f=f)
        if not p.exists():
            continue
        t = pd.read_csv(p)
        for sd, s in t.groupby('seed'):
            s = s.copy()
            s['member'] = mid
            mid += 1
            tabs.append(s)
    allm = pd.concat(tabs)
    print(f'fold {f}: members={mid} rows/member={len(allm)//mid}')
    g = allm.groupby(['true_energy', 'region_name'], sort=False)
    ens = g.agg(pred=('pred_energy', 'median')).reset_index()
    ens['fold'] = f
    folds.append(ens)
    for m, s in allm.groupby('member'):
        single_member.append(dict(fold=f, member=m,
                                  df=s[['true_energy', 'region_name', 'pred_energy']]))

cv = pd.concat(folds)
cv['res'] = (cv.pred - cv.true_energy) / cv.true_energy

print('\n== CV table with bootstrap errors ==')
lines = []
for reg in REGIONS:
    r = cv[cv.region_name == reg]
    lo, hi = edges[reg]
    row = [reg]
    for name, sel in (('low', r[r.true_energy < lo]),
                      ('mid', r[(r.true_energy >= lo) & (r.true_energy < hi)]),
                      ('high', r[r.true_energy >= hi])):
        v, b, n = seff(sel.res), boot(sel.res), len(sel)
        row.append((v, b, n))
        print(f'{reg:>6s} {name:4s} {v:.4f} +- {b:.4f}  n={n}')
    lines.append(row)
agg_v, agg_b = seff(cv.res), boot(cv.res)
print(f'aggregate {agg_v:.4f} +- {agg_b:.4f}  n={len(cv)}')

sm_agg = []
sm_bins = {(reg, b): [] for reg in REGIONS for b in ('low', 'mid', 'high')}
for rec in single_member:
    t = rec['df']
    res = (t.pred_energy - t.true_energy) / t.true_energy
    sm_agg.append(seff(res))
    for reg in REGIONS:
        rr = t[t.region_name == reg]
        lo, hi = edges[reg]
        rres = (rr.pred_energy - rr.true_energy) / rr.true_energy
        sm_bins[(reg, 'low')].append(seff(rres[rr.true_energy < lo]))
        sm_bins[(reg, 'mid')].append(seff(rres[(rr.true_energy >= lo) & (rr.true_energy < hi)]))
        sm_bins[(reg, 'high')].append(seff(rres[rr.true_energy >= hi]))
print('\n== single-member spread (std of per-member sigma_eff, 25 member-fold arms) ==')
print(f'aggregate: mean {np.nanmean(sm_agg):.4f} sd {np.nanstd(sm_agg, ddof=1):.4f}')
for reg in ('15mm', '30mm'):
    v = sm_bins[(reg, 'low')]
    print(f'{reg} low: mean {np.nanmean(v):.4f} sd {np.nanstd(v, ddof=1):.4f}')

bdt = pd.read_csv(D / 'minbias__BDT.csv')
bdt['res'] = (bdt.pred_energy - bdt.true_energy) / bdt.true_energy

EB = np.geomspace(1, 100, 9)
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharey=True, sharex=True)
axes = axes.flatten(); axes[5].set_visible(False)
for ax, reg in zip(axes, REGIONS):
    for t, lab, sty in ((cv.rename(columns={'pred': 'pred_energy'}), 'SpaceTformer (CV ens.)', 'o-'),
                        (bdt, 'GBDT reference', 's--')):
        r = t[t.region_name == reg]
        xs, ys, es = [], [], []
        for i in range(len(EB) - 1):
            sel = r[(r.true_energy >= EB[i]) & (r.true_energy < EB[i + 1])]
            if len(sel) < 40:
                continue
            xs.append(np.sqrt(EB[i] * EB[i + 1]))
            ys.append(seff(sel.res))
            es.append(boot(sel.res, 200))
        ax.errorbar(xs, ys, yerr=es, fmt=sty, ms=4, capsize=2, label=lab)
    ax.set_xscale('log')
    ax.set_title(f'{reg} region')
    ax.set_xlabel(r'$E_{\mathrm{true}}$ [GeV]')
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'$\sigma_{\mathrm{eff}}$'); axes[3].set_ylabel(r'$\sigma_{\mathrm{eff}}$')
axes[0].legend(fontsize=8)
fig.tight_layout()
for a in axes[:5]:
    clean_log_axis(a, 'x')
axes[2].tick_params(labelbottom=True)
axes[0].set_xlabel(''); axes[1].set_xlabel('')
fig.tight_layout()
fig.savefig(OUT / 'resolution_vs_e.pdf')
print('wrote resolution_vs_e.pdf')

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey=True, sharex=True)
axes = axes.flatten(); axes[5].set_visible(False)
for ax, reg in zip(axes, REGIONS):
    r = cv[cv.region_name == reg]
    xs, ys, lo68, hi68 = [], [], [], []
    for i in range(len(EB) - 1):
        sel = r[(r.true_energy >= EB[i]) & (r.true_energy < EB[i + 1])]
        if len(sel) < 40:
            continue
        rat = sel.pred / sel.true_energy
        xs.append(np.sqrt(EB[i] * EB[i + 1]))
        ys.append(rat.median())
        lo68.append(rat.quantile(0.16))
        hi68.append(rat.quantile(0.84))
    ax.plot(xs, ys, 'o-', ms=4)
    ax.fill_between(xs, lo68, hi68, alpha=0.2)
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_ylim(0.85, 1.15)
    ax.set_title(f'{reg} region')
    ax.set_xlabel(r'$E_{\mathrm{true}}$ [GeV]')
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'median $E_{\mathrm{pred}}/E_{\mathrm{true}}$'); axes[3].set_ylabel(r'median $E_{\mathrm{pred}}/E_{\mathrm{true}}$')
fig.tight_layout()
for a in axes[:5]:
    clean_log_axis(a, 'x')
axes[2].tick_params(labelbottom=True)
axes[0].set_xlabel(''); axes[1].set_xlabel('')
fig.tight_layout()
fig.savefig(OUT / 'linearity.pdf')
print('wrote linearity.pdf')

fig, ax = plt.subplots(figsize=(5.2, 3.6))
r = cv[cv.region_name == '30mm']
lo, hi = edges['30mm']
res = r[r.true_energy < lo].res
ax.hist(res, bins=np.linspace(-0.5, 0.5, 81), density=True, histtype='stepfilled',
        alpha=0.5, label='30 mm low-E residuals')
x = np.linspace(-0.5, 0.5, 400)
mu = np.median(res)
s = seff(res)
ax.plot(x, np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
        'r--', label=fr'Gaussian, $\sigma=\sigma_{{\mathrm{{eff}}}}={s:.3f}$')
ax.set_yscale('log')
ax.set_ylim(1e-2, 30)
ax.set_xlabel(r'$(E_{\mathrm{pred}}-E_{\mathrm{true}})/E_{\mathrm{true}}$')
ax.set_ylabel('density')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'residuals_30mm_low.pdf')
print('wrote residuals_30mm_low.pdf')

print('\n== tercile edges (GeV) ==')
for reg in REGIONS:
    print(f'{reg}: {edges[reg][0]:.2f} / {edges[reg][1]:.2f}')

print('\n== BDT same-binning table ==')
for reg in REGIONS:
    r = bdt[bdt.region_name == reg]
    lo, hi = edges[reg]
    cells = [seff(r[r.true_energy < lo].res),
             seff(r[(r.true_energy >= lo) & (r.true_energy < hi)].res),
             seff(r[r.true_energy >= hi].res)]
    print(f'{reg}: ' + ' '.join(f'{c:.4f}' for c in cells))
print(f'BDT aggregate: {seff(bdt.res):.4f}  rows={len(bdt)} seeds={bdt.seed.nunique()}')
