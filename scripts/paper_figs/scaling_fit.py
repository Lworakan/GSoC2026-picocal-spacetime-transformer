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
def seff(r):
    r=np.sort(np.asarray(r,float)); n=len(r)
    if n<25: return np.nan
    k=int(np.ceil(0.683*n)); w=r[k-1:]-r[:n-k+1]; return float(w[np.argmin(w)]/2)

base = pd.read_csv(D/'minbias__SubNetW4CleanAuxExDnQdEma.csv')
lo15 = base[base.region_name=='15mm'].true_energy.quantile(1/3)
lo30 = base[base.region_name=='30mm'].true_energy.quantile(1/3)

FULL_TRAIN = 50787
pts = {0.25:'minbias__SubNetW8CleanAuxFr25ExDnRcQdEma.csv',
       0.50:'minbias__SubNetW8CleanAuxFr50ExDnRcQdEma.csv',
       0.75:'minbias__SubNetW8CleanAuxFr75ExDnRcQdEma.csv',
       1.00:'minbias__SubNetW8CleanAuxExDnRcQdEma.csv'}
rows=[]
for fr,f in pts.items():
    t=pd.read_csv(D/f)
    seeds=sorted(t.seed.unique())[:3]
    per=[]
    for sd in seeds:
        s=t[t.seed==sd]
        per.append(seff((s.pred_energy-s.true_energy)/s.true_energy))
    t3=t[t.seed.isin(seeds)]
    g=t3.groupby(['true_energy','region_name'],sort=False).pred_energy.median()
    e=np.array([k[0] for k in g.index],float)
    ens=seff((g.to_numpy()-e)/e)
    r15=t3[(t3.region_name=='15mm')&(t3.true_energy<lo15)]
    g15=r15.groupby('true_energy').pred_energy.median()
    e15=seff((g15.to_numpy()-g15.index.to_numpy())/g15.index.to_numpy())
    rows.append(dict(fr=fr,N=int(fr*FULL_TRAIN),nseeds=len(seeds),
                     mean_single=np.mean(per),sd_single=np.std(per,ddof=1) if len(per)>1 else np.nan,
                     ens=ens,ens15=e15))
    print(f"frac {fr}: seeds {len(seeds)} single {np.mean(per):.4f}+-{(np.std(per,ddof=1) if len(per)>1 else 0):.4f} ens3 {ens:.4f} 15low {e15:.4f}")

df=pd.DataFrame(rows)
for col,lab in (('mean_single','single-model mean'),('ens','3-seed ensemble')):
    p=np.polyfit(np.log(df.N),np.log(df[col]),1)
    print(f'{lab}: exponent {p[0]:.3f}; 3x data -> {np.exp(np.polyval(p,np.log(3*FULL_TRAIN))):.4f}')

fig,ax=plt.subplots(figsize=(5.8,4.2))
ax.errorbar(df.N,df.mean_single,yerr=df.sd_single,fmt='o',capsize=3,label='single model (mean of 3 seeds)')
ax.plot(df.N,df.ens,'s',label='3-seed ensemble')
for col,sty in (('mean_single','-'),('ens','--')):
    p=np.polyfit(np.log(df.N),np.log(df[col]),1)
    xs=np.geomspace(df.N.min()*0.9,3.2*FULL_TRAIN,100)
    ax.plot(xs,np.exp(np.polyval(p,np.log(xs))),sty,lw=1,alpha=0.7,
            label=f'fit: $\\sigma \\propto N^{{{p[0]:.2f}}}$')
ax.axvline(3*FULL_TRAIN,color='gray',ls=':',lw=1)
ax.text(3*FULL_TRAIN*0.94,0.0455,'3$\\times$ data',fontsize=8,rotation=90,va='bottom')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('training events $N$')
ax.set_ylabel(r'aggregate $\sigma_{\mathrm{eff}}$')
ax.grid(alpha=0.3,which='both')
ax.legend(fontsize=8)
clean_log_axis(ax, 'x', (10000, 20000, 50000, 100000, 200000))
clean_log_axis(ax, 'y', (0.03, 0.04, 0.05, 0.06))
fig.tight_layout()
fig.savefig('paper/figs/scaling_curve.pdf')
print('wrote scaling_curve.pdf')
