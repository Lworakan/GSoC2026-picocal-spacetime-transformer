import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
sys.path.insert(0, 'scripts')
from picocal_data import build_grid

W = 8
files = sorted(Path('data/minimum_bias').glob('*.root'))[:6]
EV = build_grid(files, label='scan3d')

best, score = None, -1
for ev in EV:
    if ev['reg'] != 0 or ev['Etrue'] > 40:
        continue
    ch = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
    out_frac = ev['e'][ch > 4].sum() / max(ev['e'].sum(), 1e-9)
    if 0.35 < out_frac < 0.75 and len(ev['e']) > 120:
        s = len(ev['e']) * out_frac
        if s > score:
            best, score = ev, s

ev = best
ps = ev['ps']
ci, cj = (ev['xc'] - ev['xs']) / ps, (ev['yc'] - ev['ys']) / ps
ai, aj = (ev['ax'] - ev['xs']) / ps, (ev['ay'] - ev['ys']) / ps
oi, oj = round(ci), round(cj)

S = 2 * W + 1
ge = np.full((S, S), np.nan)
gt = np.full((S, S), np.nan)
gf = np.full((S, S), np.nan)
gb = np.full((S, S), np.nan)
for di, dj, e, tf, fr, bk in zip(ev['di'], ev['dj'], ev['e'], ev['tf'], ev['fr'], ev['bk']):
    i, j = int(di - oi) + W, int(dj - oj) + W
    if 0 <= i < S and 0 <= j < S:
        ge[j, i] = e
        gf[j, i] = fr
        gb[j, i] = bk
        if np.isfinite(tf):
            gt[j, i] = tf

t0 = np.nanmedian(gt)
fig = plt.figure(figsize=(11.0, 5.4))

ax = fig.add_subplot(1, 2, 1, projection='3d', computed_zorder=False)
jj, ii = np.where(np.isfinite(ge))
h = np.log10(np.clip(ge[jj, ii], 1.0, None))
dt = gt[jj, ii] - t0
norm = colors.Normalize(-6, 6)
cmap = cm.coolwarm
fc = np.array([cmap(norm(v)) if np.isfinite(v) else (0.72, 0.72, 0.74, 1.0) for v in dt])
order = np.argsort(-(ii + jj))
ax.bar3d(ii[order] - W - 0.4, jj[order] - W - 0.4, np.zeros(len(order)),
         0.8, 0.8, h[order], color=fc[order], shade=True, edgecolor='none')

zt = h.max() * 1.45
ax.plot([ai - oi, ai - oi], [aj - oj, aj - oj], [zt, 0.02], color='#7c3aed', lw=2.2, zorder=10)
ax.scatter([ai - oi], [aj - oj], [zt], marker='v', s=90, color='#7c3aed',
           depthshade=False, zorder=11)

for half, col, ls, lw in ((4.5, '#d92b2b', '--', 1.9), (W + 0.5, '#111111', '-', 1.6)):
    cx, cy = -oi, -oj
    if half > 5:
        cx = cy = 0.0
    xs = [cx - half, cx + half, cx + half, cx - half, cx - half]
    ys = [cy - half, cy - half, cy + half, cy + half, cy - half]
    ax.plot(xs, ys, [0] * 5, color=col, ls=ls, lw=lw, zorder=1)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color='#d92b2b', ls='--', lw=1.9,
                          label=r'$9{\times}9$ seed-centred (old)'),
                   Line2D([], [], color='#111111', ls='-', lw=1.6,
                          label=r'$17{\times}17$ recentred window'),
                   Line2D([], [], color='#7c3aed', marker='v', lw=0, ms=7,
                          label='true photon entry')],
          fontsize=8, loc='upper left', bbox_to_anchor=(-0.02, 0.90),
          framealpha=0.92, borderpad=0.4, handlelength=1.6)

ax.set_xlabel('cells  $i$', labelpad=-2)
ax.set_ylabel('cells  $j$', labelpad=-2)
ax.set_zlabel(r'$\log_{10}$ cell energy [MeV]', labelpad=2)
ax.set_xlim(-W - 1, W + 1)
ax.set_ylim(-W - 1, W + 1)
ax.set_box_aspect((1, 1, 0.55))
ax.view_init(elev=26, azim=-58)
ax.tick_params(labelsize=8, pad=-1)
ax.set_title('(a)  one cluster as space--time tokens', fontsize=11, y=1.0)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.02, aspect=26,
                  orientation='horizontal', location='bottom')
cb.set_label('cell time $-$ window median [ns]', fontsize=9)
cb.ax.tick_params(labelsize=8)

ax2 = fig.add_subplot(1, 2, 2, projection='3d', computed_zorder=False)
X, Y = np.meshgrid(np.arange(S) - W, np.arange(S) - W)
vmax = np.nanmax(np.log10(np.clip(ge, 1.0, None)))
for grid, z in ((gb, 0.0), (gf, 1.0)):
    g = np.log10(np.clip(np.nan_to_num(grid, nan=0.0), 1.0, None))
    gm = max(np.nanmax(g), 1e-9)
    fcs = cm.viridis(g / gm)
    fcs[..., 3] = np.where(g > 0, 0.95, 0.05)
    ax2.plot_surface(X, Y, np.full_like(X, z, dtype=float), facecolors=fcs,
                     rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

ax2.plot([ai - oi, ai - oi], [aj - oj, aj - oj], [1.62, 0.0], color='#7c3aed', lw=2.2, zorder=10)
ax2.scatter([ai - oi], [aj - oj], [1.62], marker='v', s=90, color='#7c3aed',
            depthshade=False, zorder=11)
ax2.set_xlabel('cells  $i$', labelpad=-2)
ax2.set_ylabel('cells  $j$', labelpad=-2)
ax2.set_zticks([0, 1])
ax2.set_zticklabels(['back', 'front'], fontsize=8)
ax2.set_zlim(-0.15, 1.9)
ax2.set_box_aspect((1, 1, 0.55))
ax2.view_init(elev=22, azim=-58)
ax2.tick_params(labelsize=8, pad=-1)
ax2.set_title('(b)  the two longitudinal segments', fontsize=11, y=1.0)

ch = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
frac = ev['e'][ch > 4].sum() / ev['e'].sum()
fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.06, wspace=0.0)
fig.savefig('paper/figs/event_3d.pdf', bbox_inches='tight', pad_inches=0.02)
print('event Etrue', round(ev['Etrue'], 1), 'cells', len(ev['e']),
      'outside-9x9', round(frac, 3), 'fired-with-time', int(np.isfinite(gt).sum()))
