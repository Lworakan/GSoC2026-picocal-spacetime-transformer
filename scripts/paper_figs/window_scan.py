import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

led = Path('/tmp/claude-1000/-home-lworakan-Documents-GitHub-GSoC2026-picocal-spacetime-transformer/0d08fdbc-c8f5-4fff-af19-d1b7eab0024a/scratchpad/ledger.txt').read_text().splitlines()
rows = {}
for ln in led:
    m = re.match(r'^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)(?:\+-([\d.]+))?\s+([\d.]+)(?:\+-([\d.]+))?', ln.strip())
    if m:
        rows[m.group(1)] = dict(seeds=int(m.group(2)), agg=float(m.group(3)),
                                l15=float(m.group(4)), sd15=m.group(5),
                                l30=float(m.group(6)), sd30=m.group(7))

ws = {2: 'SubNetW2CleanAuxExDnQdEma', 4: 'SubNetW4CleanAuxExDnQdEma', 5: 'SubNetW5CleanAuxExDnQdEma',
      6: 'SubNetW6CleanAuxExDnQdEma', 7: 'SubNetW7CleanAuxExDnQdEma', 8: 'SubNetW8CleanAuxExDnQdEma',
      9: 'SubNetW9CleanAuxExDnQdEma', 10: 'SubNetW10CleanAuxExDnQdEma'}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for key, lab, col in (('l15', '15 mm low-E', 'tab:red'), ('l30', '30 mm low-E', 'tab:blue'), ('agg', 'aggregate', 'tab:gray')):
    x = sorted(ws)
    y = [rows[ws[w]][key] for w in x]
    e = [float(rows[ws[w]]['sd15' if key == 'l15' else 'sd30']) if key != 'agg' and rows[ws[w]].get('sd15' if key == 'l15' else 'sd30') else 0 for w in x]
    ax.errorbar(x, y, yerr=e, fmt='o-', ms=4, capsize=2, label=lab, color=col)
rc = rows['SubNetW8CleanAuxExDnRcQdEma']
ax.plot([7.85], [rc['l15']], '*', ms=15, color='darkred', zorder=5, label='W8 + recentring (15 mm low)')
ax.plot([8.15], [rc['l30']], '*', ms=15, color='darkblue', zorder=5, label='W8 + recentring (30 mm low)')
ax.set_xlabel('window half-width $w$ (seed-centred, except stars)')
ax.set_ylabel(r'$\sigma_{\mathrm{eff}}$ (fixed-split ensemble)')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('paper/figs/window_scan.pdf')
print('wrote window_scan.pdf')

hdr = r'''\begin{longtable}{lrlll}
\caption{Complete experiment ledger: every configuration with a saved
prediction set on the minimum-bias sample, scored identically (seed-ensemble
\seff; $\pm$ is the spread across seeds). Configuration names are the run
identifiers from the training script; one- and two-seed rows are screening
runs.}\label{tab:ledger}\\
\toprule
configuration & seeds & aggregate & 15\,mm low-E & 30\,mm low-E\\
\midrule
\endfirsthead
\toprule
configuration & seeds & aggregate & 15\,mm low-E & 30\,mm low-E\\
\midrule
\endhead
'''
out = [hdr]
for ln in led:
    m = re.match(r'^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)(\+-[\d.]+)?\s+([\d.]+)(\+-[\d.]+)?', ln.strip())
    if not m:
        continue
    name = m.group(1).replace('_', r'\_')
    def cell(v, s):
        return f'${v}\\pm{s[2:]}$' if s else f'${v}$'
    out.append(f'\\texttt{{{name}}} & {m.group(2)} & ${m.group(3)}$ & '
               f'{cell(m.group(4), m.group(5))} & {cell(m.group(6), m.group(7))}\\\\\n')
out.append('\\bottomrule\n\\end{longtable}\n')
Path('paper/ledger_table.tex').write_text(''.join(out))
print('wrote ledger_table.tex,', len(out) - 2, 'rows')
