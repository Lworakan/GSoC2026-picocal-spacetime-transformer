import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

EV = pickle.load(open('.scratch/cache/overlay.pkl', 'rb'))

best, score = None, -1
for ev in EV:
    if ev['reg'] != 0 or not (20 < ev['Etrue'] < 60):
        continue
    f = float(ev['sig'].sum() / max(ev['e'].sum(), 1e-9))
    if 0.22 < f < 0.42 and len(ev['e']) > 70:
        s = len(ev['e'])
        if s > score:
            best, score = ev, s
ev = best
frac = float(ev['sig'].sum() / ev['e'].sum())

W = int(max(np.abs(ev['di']).max(), np.abs(ev['dj']).max()))
S = 2 * W + 1
tot = np.full((S, S), np.nan)
sig = np.full((S, S), np.nan)
for di, dj, e, sg in zip(ev['di'], ev['dj'], ev['e'], ev['sig']):
    i, j = int(di) + W, int(dj) + W
    if 0 <= i < S and 0 <= j < S:
        tot[j, i] = e
        sig[j, i] = sg
pil = tot - sig

ai = (ev['ax'] - ev['xs']) / ev['ps']
aj = (ev['ay'] - ev['ys']) / ev['ps']


def lg(a):
    return np.log10(np.clip(a, 1.0, None))


zmax = lg(np.nanmax(tot)) * 1.02
fig = plt.figure(figsize=(12.6, 4.5))
panels = [
    ('(a)  what the detector reads', tot, None, 'frac'),
    ('(b)  the photon alone', sig, '#e8a33d', None),
    ('(c)  the pileup alone', pil, '#7f8794', None),
]
norm = colors.Normalize(0, 1)
cmap = cm.get_cmap('YlOrBr') if hasattr(cm, 'get_cmap') else cm.YlOrBr

for k, (title, grid, flat, mode) in enumerate(panels):
    ax = fig.add_subplot(1, 3, k + 1, projection='3d', computed_zorder=False)
    jj, ii = np.where(np.isfinite(grid) & (grid > 0))
    h = lg(grid[jj, ii])
    if mode == 'frac':
        f = np.clip(sig[jj, ii] / np.clip(tot[jj, ii], 1e-9, None), 0, 1)
        fc = cmap(0.15 + 0.8 * f)
    else:
        fc = np.tile(colors.to_rgba(flat), (len(ii), 1))
    order = np.argsort(-(ii + jj))
    ax.bar3d(ii[order] - W - 0.4, jj[order] - W - 0.4, np.zeros(len(order)),
             0.8, 0.8, h[order], color=fc[order] if fc.ndim == 2 else fc,
             shade=True, edgecolor='none')
    ax.plot([ai, ai], [aj, aj], [zmax * 1.35, 0.02], color='#7c3aed', lw=2.0, zorder=10)
    ax.scatter([ai], [aj], [zmax * 1.35], marker='v', s=70, color='#7c3aed',
               depthshade=False, zorder=11)
    ax.set_xlim(-W - 1, W + 1)
    ax.set_ylim(-W - 1, W + 1)
    ax.set_zlim(0, zmax * 1.45)
    ax.set_box_aspect((1, 1, 0.62))
    ax.view_init(elev=27, azim=-58)
    ax.set_xlabel('cells $i$', labelpad=-4, fontsize=9)
    ax.set_ylabel('cells $j$', labelpad=-4, fontsize=9)
    ax.tick_params(labelsize=7.5, pad=-2)
    if k == 0:
        ax.set_zlabel(r'$\log_{10}$ energy [MeV]', labelpad=2, fontsize=9)
    else:
        ax.set_zticklabels([])
    sh = grid[np.isfinite(grid)].sum() / np.nansum(tot)
    ax.set_title(f'{title}\n{sh*100:.0f}% of the window energy', fontsize=10.5, y=0.97)

sm = cm.ScalarMappable(norm=norm, cmap=colors.LinearSegmentedColormap.from_list(
    'f', [cmap(0.15), cmap(0.95)]))
cb = fig.colorbar(sm, ax=fig.axes[0], shrink=0.55, pad=0.02, aspect=15,
                  orientation='horizontal', location='bottom')
cb.set_label('true photon fraction of the cell', fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.06, wspace=0.0)
fig.savefig('paper/figs/photon_vs_pileup_3d.pdf', bbox_inches='tight', pad_inches=0.02)
print('overlay event: Etrue', round(ev['Etrue'], 1), 'cells', len(ev['e']),
      'photon fraction', round(frac, 3), 'W', W)
