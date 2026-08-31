import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import resolution

# Three figures the argument needs and did not have.
#
#  (a) resolution_vs_et  -- the per-region curves against TRANSVERSE energy. Mentor item of
#      24 August: the physics selections are written in E_T, so the per-region figure has to
#      be readable in that axis too.
#  (b) design_gap        -- the same curves against true energy with PicoCal's own design
#      resolution drawn on top, sigma/E = 10%/sqrt(E) + 1% (PoS(LHCP2024)301), and the 0.08
#      line. This is the figure that makes the outer-region claim land: 60 and 120 mm sit
#      within a sixth of the number the detector was specified to, and no estimator reaches
#      0.02 there because the design curve is already above it at those energies.
#  (c) perbin_bars       -- the fifteen region-energy cells, this work against the control,
#      so "wins everywhere" is a picture rather than a table to be scanned.

REPO = Path(__file__).resolve().parents[2]
D = REPO / 'reports' / 'predictions'
OUT = REPO / 'paper' / 'figs'
BEST = 'minbias__SubNetW8CleanAuxExDnGs50RcOvV2CrQdEma.csv'
CTRL = 'minbias__SubNetW8CleanAuxExDnRcQdEma.csv'
REGIONS = ['15mm', '30mm', '40mm', '60mm', '120mm']
SEED = 0


def design(E):
    return np.sqrt((0.10 / np.sqrt(E)) ** 2 + 0.01 ** 2)


def load(f, seed=SEED):
    t = pd.read_csv(D / f)
    t = t[t.seed == seed]
    return t.groupby(['true_energy', 'region_name', 'ET'], sort=False).agg(
        p=('pred_energy', 'median')).reset_index()


def curve(s, xcol, nbin=6, minn=60):
    q = np.quantile(s[xcol], np.linspace(0, 1, nbin + 1))
    x, y, e, ex = [], [], [], []
    for lo, hi in zip(q[:-1], q[1:]):
        k = (s[xcol] >= lo) & (s[xcol] < hi if hi < q[-1] else s[xcol] <= hi)
        if k.sum() < minn:
            continue
        c = s[k]
        r = resolution(c.p.values, c.true_energy.values)['sigma_eff']
        x.append(np.median(c[xcol]))
        y.append(r)
        e.append(0.96 * r / np.sqrt(k.sum()))
        ex.append(np.median(c.true_energy))
    return np.array(x), np.array(y), np.array(e), np.array(ex)


def panels(best, ctrl, xcol, xlabel, fname, with_design):
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for a, reg in zip(ax, REGIONS):
        for tab, lab, st in ((ctrl, 'starting configuration', '--'), (best, 'this work', '-')):
            s = tab[tab.region_name == reg]
            x, y, e, ex = curve(s, xcol)
            a.errorbar(x, y, yerr=e, fmt='o' + st, ms=3.5, lw=1.2, label=lab)
        if with_design:
            s = best[best.region_name == reg]
            x, y, e, ex = curve(s, xcol)
            o = np.argsort(x)
            a.plot(x[o], design(ex[o]), ':', color='k', lw=1.4,
                   label=r'design $10\%/\sqrt{E}\oplus 1\%$')
            a.axhline(0.08, color='crimson', lw=0.9, alpha=.7)
        a.set_title(reg, fontsize=10)
        a.set_xlabel(xlabel)
        a.set_xscale('log')
        a.grid(alpha=.3)
    ax[0].set_ylabel(r'$\sigma_{\mathrm{eff}}$')
    ax[0].set_ylim(0, None)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{fname}.{ext}', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('wrote', OUT / f'{fname}.pdf')


def bars(best, ctrl):
    labels, b, c = [], [], []
    for reg in REGIONS:
        s, t = best[best.region_name == reg], ctrl[ctrl.region_name == reg]
        q = s.true_energy.quantile([1 / 3, 2 / 3]).values
        for lab, kb, kc in (('low', s.true_energy <= q[0], t.true_energy <= q[0]),
                            ('mid', (s.true_energy > q[0]) & (s.true_energy <= q[1]),
                             (t.true_energy > q[0]) & (t.true_energy <= q[1])),
                            ('high', s.true_energy > q[1], t.true_energy > q[1])):
            labels.append(f'{reg}\n{lab}')
            b.append(resolution(s[kb].p.values, s[kb].true_energy.values)['sigma_eff'])
            c.append(resolution(t[kc].p.values, t[kc].true_energy.values)['sigma_eff'])
    b, c = np.array(b), np.array(c)
    i = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(i - 0.2, c, 0.4, label='starting configuration', color='#9aa5b1')
    ax.bar(i + 0.2, b, 0.4, label='this work', color='#2f6f9f')
    ax.axhline(0.08, color='crimson', lw=1.0)
    ax.text(len(labels) - 0.4, 0.0815, '0.08', color='crimson', fontsize=8, ha='right')
    ax.set_xticks(i)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r'$\sigma_{\mathrm{eff}}$')
    ax.set_title(f'Every region--energy cell, single model, seed {SEED} '
                 f'({int((b < c).sum())} of {len(b)} improved)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'perbin_bars.{ext}', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'wrote {OUT / "perbin_bars.pdf"}  improved {int((b < c).sum())}/{len(b)}, '
          f'worst bin {b.max():.4f}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    best, ctrl = load(BEST), load(CTRL)
    panels(best, ctrl, 'ET', r'$E_{\mathrm{T}}$ [GeV]', 'resolution_vs_et', False)
    panels(best, ctrl, 'true_energy', r'$E$ [GeV]', 'design_gap', True)
    bars(best, ctrl)


if __name__ == '__main__':
    main()
