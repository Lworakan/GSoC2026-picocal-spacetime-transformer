import numpy as np, pandas as pd
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
rng = np.random.default_rng(0)
def seff(r):
    r=np.sort(np.asarray(r,float)); n=len(r)
    if n<25: return np.nan
    k=int(np.ceil(0.683*n)); w=r[k-1:]-r[:n-k+1]; return float(w[np.argmin(w)]/2)
def boot(r, nb=200):
    r=np.asarray(r,float)
    return float(np.std([seff(rng.choice(r,len(r),True)) for _ in range(nb)],ddof=1))

def load(f):
    t=pd.read_csv(D/f)
    g=t.groupby(['true_energy','region_name'],sort=False).agg(pred=('pred_energy','median')).reset_index()
    g['res']=(g.pred-g.true_energy)/g.true_energy
    g['seedn']=t.seed.nunique()
    return g

pairs = [('clean (no pileup)', 'clean__CleanQuantW2.csv', 'clean__CleanQuantW2NoTime.csv'),
         ('minimum bias (pileup)', 'minbias__SubNetW4CleanAuxExDnQdEma.csv',
          'minbias__SubNetW4CleanAuxExDnNoTimeQdEma.csv')]
EB = np.geomspace(1, 100, 9)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, (title, fw, fo) in zip(axes, pairs):
    for f, lab, sty, col in ((fw, 'with timestamps', 'o-', 'tab:blue'),
                             (fo, 'timestamps removed', 's--', 'tab:red')):
        t = load(f)
        agg = seff(t.res)
        xs, ys, es = [], [], []
        for i in range(len(EB)-1):
            sel = t[(t.true_energy>=EB[i])&(t.true_energy<EB[i+1])]
            if len(sel)<60: continue
            xs.append(np.sqrt(EB[i]*EB[i+1])); ys.append(seff(sel.res)); es.append(boot(sel.res))
        ax.errorbar(xs, ys, yerr=es, fmt=sty, ms=4, capsize=2, color=col,
                    label=f'{lab} (agg {agg:.4f})')
        print(title, lab, f'{agg:.4f}', 'seeds', t.seedn.iloc[0])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(r'$E_{\mathrm{true}}$ [GeV]')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=9)
axes[0].set_ylabel(r'$\sigma_{\mathrm{eff}}$')
for a in axes:
    clean_log_axis(a, 'x')
    clean_log_axis(a, 'y', (0.02, 0.03, 0.05, 0.1, 0.2, 0.3))
fig.tight_layout()
fig.savefig('paper/figs/timing_ablation.pdf')
print('wrote timing_ablation.pdf')
