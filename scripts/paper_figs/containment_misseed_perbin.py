import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('.')
sys.path.insert(0, str(R / 'scripts'))
from picocal_data import build_grid

OUT = Path('paper/figs')
REGIONS = ['15mm', '30mm', '40mm', '60mm', '120mm']
rng = np.random.default_rng(0)

files = sorted((R / 'data' / 'minimum_bias').glob('*.root'))[:12]
print(f'{len(files)} files')
EV = build_grid(files, label='scan')

WS = np.arange(0, 16)
frac = {c: {r: [] for r in range(5)} for c in ('seed', 'centroid')}
d_seed, d_cent = {r: [] for r in range(5)}, {r: [] for r in range(5)}
for ev in EV:
    r = ev['reg']
    ps = ev['ps']
    di, dj, e = ev['di'].astype(int), ev['dj'].astype(int), ev['e']
    tot = e.sum()
    if tot <= 0:
        continue
    ci = int(round((ev['xc'] - ev['xs']) / ps))
    cj = int(round((ev['yc'] - ev['ys']) / ps))
    ch_s = np.maximum(np.abs(di), np.abs(dj))
    ch_c = np.maximum(np.abs(di - ci), np.abs(dj - cj))
    frac['seed'][r].append([e[ch_s <= w].sum() / tot for w in WS])
    frac['centroid'][r].append([e[ch_c <= w].sum() / tot for w in WS])
    d_seed[r].append(np.hypot(ev['ax'] - ev['xs'], ev['ay'] - ev['ys']) / ps)
    d_cent[r].append(np.hypot(ev['ax'] - ev['xc'], ev['ay'] - ev['yc']) / ps)

fig, ax = plt.subplots(figsize=(7.0, 4.6))
colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
for r, (reg, col) in enumerate(zip(REGIONS, colors)):
    if not frac['centroid'][r]:
        continue
    fc = np.mean(frac['centroid'][r], axis=0)
    fs = np.mean(frac['seed'][r], axis=0)
    ax.plot(WS, fc, '-o', ms=3, color=col, label=f'{reg} (n={len(frac["centroid"][r])})')
    ax.plot(WS, fs, '--', color=col, lw=1)
ax.axvline(4, color='gray', ls=':', lw=1)
ax.axvline(8, color='k', ls=':', lw=1)
ax.text(4.15, 0.02, r'$9{\times}9$ (start)', fontsize=8, rotation=90, va='bottom')
ax.text(8.15, 0.02, r'$17{\times}17$ (this work)', fontsize=8, rotation=90, va='bottom')
ax.set_xlabel('window half-width $w$ [cells]')
ax.set_ylabel('mean contained fraction of cluster energy')
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower right',
          title='solid: barycentre-centred\ndashed: seed-centred', title_fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'containment.pdf')
print('wrote containment.pdf')
for r, reg in enumerate(REGIONS):
    if frac['centroid'][r]:
        print(reg, 'w=4', f"{np.mean(frac['seed'][r], axis=0)[4]:.3f}",
              'w=8c', f"{np.mean(frac['centroid'][r], axis=0)[8]:.3f}")

fig, ax = plt.subplots(figsize=(5.6, 3.8))
bins = np.linspace(0, 12, 49)
ax.hist(np.clip(d_seed[0], 0, 12), bins=bins, histtype='step', lw=1.5,
        label=f'seed cell (median {np.median(d_seed[0]):.2f})', density=True)
ax.hist(np.clip(d_cent[0], 0, 12), bins=bins, histtype='step', lw=1.5,
        label=f'cluster barycentre (median {np.median(d_cent[0]):.2f})', density=True)
ax.set_xlabel('distance from true photon entry point [cells], 15 mm region')
ax.set_ylabel('density')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'misseed_15mm.pdf')
print('wrote misseed_15mm.pdf', 'medians', np.median(d_seed[0]), np.median(d_cent[0]))


def seff(r):
    r = np.sort(np.asarray(r, float))
    n = len(r)
    if n < 25:
        return np.nan
    k = int(np.ceil(0.683 * n))
    w = r[k - 1:] - r[:n - k + 1]
    return float(w[np.argmin(w)] / 2)


D = Path('reports/predictions')
base = pd.read_csv(D / 'minbias__SubNetW4CleanAuxExDnQdEma.csv')
edges = {}
for reg in REGIONS:
    e = base[base.region_name == reg].true_energy
    edges[reg] = (e.quantile(1 / 3), e.quantile(2 / 3))
g = base.groupby(['true_energy', 'region_name'], sort=False)
p = g.pred_energy.median().reset_index()
p.columns = ['true_energy', 'region_name', 'pred']
p['res'] = (p.pred - p.true_energy) / p.true_energy

folds = []
pats = ['minbias__SubNetW8CleanAuxExDnRcK{f}QdEma.csv',
        'minbias__SubNetW8CleanAuxExDnRcK{f}Rr01QdEma.csv',
        'minbias__SubNetW4CleanAuxExDnRcK{f}AcQdEma.csv']
for f in range(10):
    allm = pd.concat([pd.read_csv(D / pt.format(f=f)) for pt in pats])
    ens = allm.groupby(['true_energy', 'region_name'], sort=False).agg(
        pred=('pred_energy', 'median')).reset_index()
    folds.append(ens)
cv = pd.concat(folds)
cv['res'] = (cv.pred - cv.true_energy) / cv.true_energy

labels, sv, fv = [], [], []
for reg in REGIONS:
    lo, hi = edges[reg]
    for nm, lo_, hi_ in (('low', 0, lo), ('mid', lo, hi), ('high', hi, 1e9)):
        bsel = p[(p.region_name == reg) & (p.true_energy >= lo_) & (p.true_energy < hi_)]
        csel = cv[(cv.region_name == reg) & (cv.true_energy >= lo_) & (cv.true_energy < hi_)]
        labels.append(f'{reg} {nm}')
        sv.append(seff(bsel.res))
        fv.append(seff(csel.res))

fig, ax = plt.subplots(figsize=(7.4, 6.2))
ypos = np.arange(len(labels))[::-1]
for y, s, f in zip(ypos, sv, fv):
    ax.plot([s, f], [y, y], '-', color='gray', lw=1)
ax.plot(sv, ypos, 'o', color='tab:red', label='start: $9{\\times}9$ seed-centred (5-seed ens.)')
ax.plot(fv, ypos, 'o', color='tab:blue', label='this work: recentred $17{\\times}17$ (5-fold CV)')
ax.axvline(0.07, color='k', ls=':', lw=1)
ax.text(0.0725, ypos[-1] - 0.3, 'per-bin target 0.07', fontsize=8, rotation=90, va='bottom')
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r'$\sigma_{\mathrm{eff}}$')
ax.grid(alpha=0.3, axis='x')
ax.set_xlim(0.012, 0.19)
ax.legend(fontsize=8, loc='center right')
fig.tight_layout()
fig.savefig(OUT / 'perbin_improvement.pdf')
print('wrote perbin_improvement.pdf')
for l, s, f in zip(labels, sv, fv):
    print(f'{l:12s} {s:.4f} -> {f:.4f}')
