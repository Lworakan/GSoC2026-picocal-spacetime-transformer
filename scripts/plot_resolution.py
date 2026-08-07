import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution

pio.templates.default = 'plotly_white'

INK = '#1F2328'
MUT = '#59636E'
GRID = '#E6E9EF'


def polish(fig, title, xlabel, ylabel):
    fig.update_layout(
        font=dict(family="ui-sans-serif, system-ui, 'Segoe UI', Helvetica, Arial", size=13, color=INK),
        title=dict(text=title, font=dict(size=16), x=0.02, xanchor='left'),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=30, t=70, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1.0,
                    font=dict(size=12), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified')
    for ax in (fig.update_xaxes, fig.update_yaxes):
        ax(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks='outside',
           tickcolor=GRID, title_font=dict(size=13, color=MUT), tickfont=dict(size=12, color=MUT))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, tickformat='.3f')
    return fig


def export(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs='cdn')
    print('wrote', out)
    try:
        png = out.with_suffix('.png')
        fig.write_image(png, width=1100, height=560, scale=2)
        print('wrote', png)
    except Exception as e:
        print(f'(png export skipped: {e})')


EXAMPLES = """\
examples:
  # single model, resolution vs true energy
  uv run scripts/plot_resolution.py reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv

  # model evolution: every development round on one plot
  uv run scripts/plot_resolution.py \\
      reports/predictions/minbias__GateHuber.csv \\
      reports/predictions/minbias__SubNetW4.csv \\
      reports/predictions/minbias__SubNetW4CleanAux.csv \\
      reports/predictions/minbias__SubNetW4CleanAuxQuant.csv \\
      reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv \\
      --labels "Huber baseline" "+physics readout" "+clean-aux" "+quantile head" "+qd loss & EMA" \\
      --title "Model evolution on minimum bias" --out reports/figures/evolution.html

  # vs transverse energy, 10 bins, with an ideal-resolution reference curve
  uv run scripts/plot_resolution.py reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv \\
      --x ET --bins 10 --ideal 0.10 0.0 0.01

  # add residual histograms alongside the resolution curves
  uv run scripts/plot_resolution.py reports/predictions/minbias__SubNetW4CleanAuxQdEma.csv --residuals

input CSV schema (one row per cluster per seed, written by scripts/train_picocal.py):
  model,dataset,seed,split,true_energy,pred_energy,pred_bias,region,region_name,ET

output: interactive HTML plus a static PNG next to it (also printed: an aggregate
sigma_eff / bias summary table for every input file).
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Compare energy resolution across models: overlay sigma_eff curves '
                    'from any number of prediction CSVs, with per-bin statistical '
                    'uncertainty 0.96*sigma_eff/sqrt(n). Seeds are averaged per file.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csvs', nargs='+',
                    help='one or more prediction CSVs (each becomes one curve)')
    ap.add_argument('--bins', type=int, default=8,
                    help='number of quantile bins along the x axis (default: 8)')
    ap.add_argument('--x', choices=['E', 'ET'], default='E',
                    help="x axis: 'E' = true energy, 'ET' = transverse energy (default: E)")
    ap.add_argument('--split', default='test',
                    help="which data split to plot: train / val / test (default: test)")
    ap.add_argument('--min-n', type=int, default=50,
                    help='skip bins with fewer clusters than this (default: 50)')
    ap.add_argument('--labels', nargs='*', default=None,
                    help='legend label per CSV, same order as the file list '
                         '(default: "<model> (<dataset>)" from the CSV itself)')
    ap.add_argument('--title', default=None,
                    help='plot title (default: auto-generated from axis and split)')
    ap.add_argument('--ideal', type=float, nargs=3, metavar=('A', 'B', 'C'), default=None,
                    help='overlay the design resolution sigma/E = A/sqrt(E) (+) B/E (+) C '
                         '(quadrature sum, fractions; e.g. --ideal 0.10 0.0 0.01)')
    ap.add_argument('--residuals', action='store_true',
                    help='also write a (E_pred-E_true)/E_true histogram figure')
    ap.add_argument('--out', default=None,
                    help='output HTML path; a PNG is written next to it '
                         '(default: ./resolution_<x>_<bins>bins.html)')
    return ap.parse_args()


def load(path, split):
    df = pd.read_csv(path)
    df = df[df['split'] == split].reset_index(drop=True)
    seeds = sorted(df['seed'].unique())
    blocks = [df[df['seed'] == s].reset_index(drop=True) for s in seeds]
    t0 = blocks[0]['true_energy'].to_numpy()
    for b in blocks[1:]:
        assert np.allclose(b['true_energy'].to_numpy(), t0), f'seed blocks in {path} are not aligned'
    pred = np.mean([b['pred_energy'].to_numpy() for b in blocks], axis=0)
    b0 = blocks[0]
    name = f"{b0['model'].iloc[0]} ({b0['dataset'].iloc[0]})"
    return dict(true=t0, pred=pred, ET=b0['ET'].to_numpy(), name=name, n_seeds=len(seeds))


def qbin(x, true, pred, nbins, min_n):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    xs, ys, es = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n < min_n:
            continue
        s = resolution(pred[sel], true[sel])['sigma_eff']
        xs.append(float(np.median(x[sel])))
        ys.append(s)
        es.append(0.96 * s / np.sqrt(n))
    return xs, ys, es


def main():
    args = parse_args()
    data = [load(p, args.split) for p in args.csvs]
    if args.labels:
        for d, lab in zip(data, args.labels):
            d['name'] = lab
    palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2

    print(f"{'model':40s} {'n':>7s} {'seeds':>5s} {'sigma_eff':>9s} {'rob_sigma':>9s} {'bias':>8s}")
    for d in data:
        r = resolution(d['pred'], d['true'])
        res = (d['pred'] - d['true']) / d['true']
        d['sigma_eff'] = r['sigma_eff']
        d['rob_sigma'] = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        print(f"{d['name']:40s} {len(d['true']):7d} {d['n_seeds']:5d} "
              f"{r['sigma_eff']:9.4f} {d['rob_sigma']:9.4f} {r['bias']:8.4f}")

    fig = go.Figure()
    for i, d in enumerate(data):
        x = d['true'] if args.x == 'E' else d['ET']
        xs, ys, es = qbin(x, d['true'], d['pred'], args.bins, args.min_n)
        fig.add_scatter(x=xs, y=ys, mode='lines+markers', name=d['name'],
                        error_y=dict(type='data', array=es, thickness=1.2, width=3),
                        line=dict(color=palette[i % len(palette)], width=2.5),
                        marker=dict(size=8, line=dict(color='white', width=1.5)))
    if args.ideal is not None:
        a, b, c = args.ideal
        allx = np.concatenate([(d['true'] if args.x == 'E' else d['ET']) for d in data])
        xe = np.linspace(np.quantile(allx, 0.01), np.quantile(allx, 0.99), 200)
        ideal = np.sqrt((a / np.sqrt(xe)) ** 2 + (b / xe) ** 2 + c ** 2)
        fig.add_scatter(x=xe, y=ideal, mode='lines', name='design resolution',
                        line=dict(color=MUT, dash='dot', width=2))
    xlabel = 'true energy [GeV]' if args.x == 'E' else 'E_T [GeV]'
    polish(fig, args.title or f'&#963;_eff vs {xlabel} &#183; {args.bins} bins &#183; {args.split} split',
           xlabel, 'sigma_eff  (lower is better)')
    fig.update_layout(height=560)
    out = Path(args.out) if args.out else Path.cwd() / f'resolution_{args.x}_{args.bins}bins.html'
    export(fig, out)

    if args.residuals:
        fig2 = go.Figure()
        for i, d in enumerate(data):
            res = (d['pred'] - d['true']) / d['true']
            lab = f"{d['name']}  &#963;_eff={d['sigma_eff']:.4f}  rob&#963;={d['rob_sigma']:.4f}"
            fig2.add_histogram(x=np.clip(res, -0.5, 0.5), nbinsx=120, name=lab,
                               opacity=0.55, marker_color=palette[i % len(palette)])
        polish(fig2, '&#916;E/E distribution &#183; (E_pred &#8722; E_true) / E_true', 'relative residual', 'events')
        fig2.update_layout(barmode='overlay', height=560, yaxis_tickformat='d')
        export(fig2, out.with_name(out.stem + '_residuals.html'))


if __name__ == '__main__':
    main()
