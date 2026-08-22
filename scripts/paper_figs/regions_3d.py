import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PITCH = [15, 30, 40, 60, 120]
W = 8
S = 2 * W + 1


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def region_stats(EV):
    ncell, ratio = {}, {}
    for r in range(5):
        n, q = [], []
        for e in EV:
            if e['reg'] != r:
                continue
            ch = np.maximum(np.abs(e['di']), np.abs(e['dj']))
            inw = ch <= W
            tot = e['e'][inw].sum()
            if tot <= 0:
                continue
            n.append(int(inw.sum()))
            q.append(tot / max(e['Etrue'] * 1000.0, 1e-9))
        if n:
            ncell[r] = float(np.median(n))
            ratio[r] = float(np.median(q))
    return ncell, ratio


def pick(EV, r):
    """A low-energy event of this region whose window occupancy is typical for it."""
    cand = [e for e in EV if e['reg'] == r]
    if not cand:
        return None
    et = np.array([e['Etrue'] for e in cand])
    cut = np.quantile(et, 1 / 3)
    cand = [e for e in cand if e['Etrue'] <= cut]
    n = np.array([int((np.maximum(np.abs(e['di']), np.abs(e['dj'])) <= W).sum()) for e in cand])
    return cand[int(np.argmin(np.abs(n - np.median(n))))]


def grid(e):
    g = np.full((S, S), np.nan)
    for di, dj, en in zip(e['di'], e['dj'], e['e']):
        i, j = int(di) + W, int(dj) + W
        if 0 <= i < S and 0 <= j < S:
            g[j, i] = en
    return g


def draw(ax, g, colour, zmax, e):
    jj, ii = np.where(np.isfinite(g) & (g > 0))
    h = np.log10(np.clip(g[jj, ii], 1.0, None))
    order = np.argsort(-(ii + jj))
    ax.bar3d(ii[order] - W - 0.4, jj[order] - W - 0.4, np.zeros(len(order)),
             0.8, 0.8, h[order], color=colour, shade=True, edgecolor='none')
    ai = (e['ax'] - e['xs']) / e['ps']
    aj = (e['ay'] - e['ys']) / e['ps']
    ax.plot([ai, ai], [aj, aj], [zmax * 1.3, 0.02], color='#7c3aed', lw=1.6, zorder=10)
    ax.scatter([ai], [aj], [zmax * 1.3], marker='v', s=34, color='#7c3aed',
               depthshade=False, zorder=11)
    ax.set_xlim(-W - 1, W + 1)
    ax.set_ylim(-W - 1, W + 1)
    ax.set_zlim(0, zmax * 1.4)
    ax.set_box_aspect((1, 1, 0.62))
    ax.view_init(elev=28, azim=-58)
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])
    ax.set_zticks([0, 2, 4, 6])
    ax.tick_params(labelsize=6.5, pad=-3)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.15)


def main():
    clean = load('.scratch/cache/clean-aux_100.pkl')
    mb = load('.scratch/cache/minbias_94.pkl')
    ncell_c, _ = region_stats(clean)
    ncell_m, ratio_m = region_stats(mb)

    fig = plt.figure(figsize=(13.6, 5.9))
    zmax = 6.0
    for c, r in enumerate(range(5)):
        for row, (EV, colour, lab) in enumerate((
                (clean, '#e8a33d', 'photon only'),
                (mb, '#7f8794', 'under pileup'))):
            e = pick(EV, r)
            ax = fig.add_subplot(2, 5, row * 5 + c + 1, projection='3d', computed_zorder=False)
            if e is not None:
                draw(ax, grid(e), colour, zmax, e)
            if row == 0:
                ax.set_title(f'{PITCH[r]} mm', fontsize=12, weight='bold', y=0.99)
            if c == 0:
                ax.text2D(-0.06, 0.5, lab, transform=ax.transAxes, rotation=90,
                          va='center', ha='center', fontsize=10, weight='bold',
                          color='#b8791f' if row == 0 else '#4b5563')
            if row == 1:
                ax.text2D(0.5, -0.06,
                          f'{ncell_m.get(r, 0):.0f} cells    '
                          + r'$E_{\rm win}/E_\gamma$ = ' + f'{ratio_m.get(r, 0):.1f}',
                          transform=ax.transAxes, ha='center', fontsize=8.5)
            else:
                ax.text2D(0.5, -0.06, f'{ncell_c.get(r, 0):.0f} cells',
                          transform=ax.transAxes, ha='center', fontsize=8.5)
    fig.text(0.5, 0.012, 'cell index within the $17{\\times}17$ window; '
                         r'bar height $= \log_{10}$ cell energy [MeV]',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.97, bottom=0.09,
                        wspace=0.0, hspace=0.14)
    fig.savefig('paper/figs/regions_3d.pdf', bbox_inches='tight', pad_inches=0.02)
    print('medians  region: clean cells / minbias cells / E_win over E_photon')
    for r in range(5):
        print(f'  {PITCH[r]:3d}mm  {ncell_c.get(r, 0):5.0f}  {ncell_m.get(r, 0):5.0f}  '
              f'{ratio_m.get(r, 0):5.2f}')


if __name__ == '__main__':
    main()
