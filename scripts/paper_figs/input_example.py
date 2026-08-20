import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, 'scripts')
from picocal_data import build_grid

files = sorted(Path('data/minimum_bias').glob('*.root'))[:6]
EV = build_grid(files, label='scan')

best, score = None, -1
for ev in EV:
    if ev['reg'] != 0 or ev['Etrue'] > 40:
        continue
    ps = ev['ps']
    ch = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
    out_frac = ev['e'][ch > 4].sum() / max(ev['e'].sum(), 1e-9)
    if 0.35 < out_frac < 0.75 and len(ev['e']) > 120:
        s = len(ev['e']) * out_frac
        if s > score:
            best, score = ev, s

ev = best
ps = ev['ps']
ci = (ev['xc'] - ev['xs']) / ps
cj = (ev['yc'] - ev['ys']) / ps
ai = (ev['ax'] - ev['xs']) / ps
aj = (ev['ay'] - ev['ys']) / ps
W = 8
grid_e = np.full((2*W+1, 2*W+1), np.nan)
grid_t = np.full((2*W+1, 2*W+1), np.nan)
for di, dj, e, tf in zip(ev['di'], ev['dj'], ev['e'], ev['tf']):
    i, j = int(di - round(ci)) + W, int(dj - round(cj)) + W
    if 0 <= i < 2*W+1 and 0 <= j < 2*W+1:
        grid_e[j, i] = e
        if np.isfinite(tf):
            grid_t[j, i] = tf

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
ext = [-W-0.5, W+0.5, -W-0.5, W+0.5]
im0 = axes[0].imshow(np.log10(np.clip(grid_e, 1e-3, None)), origin='lower', extent=ext, cmap='viridis')
plt.colorbar(im0, ax=axes[0], label=r'log$_{10}$ cell energy [MeV]', shrink=0.85)
axes[0].set_title(f'cell energies  (15 mm region, $E_\\gamma$ = {ev["Etrue"]:.1f} GeV)')
im1 = axes[1].imshow(grid_t - np.nanmedian(grid_t), origin='lower', extent=ext, cmap='coolwarm', vmin=-8, vmax=8)
plt.colorbar(im1, ax=axes[1], label='front timestamp $-$ median [ns]', shrink=0.85)
axes[1].set_title(f'timestamps  ({np.isfinite(grid_t).sum()} of {np.isfinite(grid_e).sum()} cells fired)')
for ax in axes:
    sx, sy = -round(ci), -round(cj)
    ax.add_patch(Rectangle((sx-4.5, sy-4.5), 9, 9, fill=False, edgecolor='red', lw=1.8, ls='--'))
    ax.plot(ai - round(ci), aj - round(cj), '*', ms=16, color='magenta', mec='white', label='true photon entry')
    ax.plot(sx, sy, 'x', ms=11, color='red', mew=2.5, label='seed cell (old centre)')
    ax.plot(ci - round(ci), cj - round(cj), '+', ms=13, color='black', mew=2.5, label='barycentre (new centre)')
    ax.set_xlabel('cells'); ax.set_ylabel('cells')
axes[0].legend(fontsize=8, loc='lower left', framealpha=0.9)
axes[0].text(sx-4.3, sy+3.6, 'old $9{\\times}9$', color='red', fontsize=9)
fig.tight_layout()
fig.savefig('paper/figs/input_example.pdf')
ch = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
print('event: Etrue', round(ev['Etrue'],1), 'cells', len(ev['e']),
      'outside-9x9 fraction', round(ev['e'][ch>4].sum()/ev['e'].sum(),3))
