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


def parse_args():
    ap = argparse.ArgumentParser(
        description='Overlay sigma_eff curves from a list of prediction CSVs (schema: model,dataset,seed,split,true_energy,pred_energy,pred_bias,region,region_name,ET)')
    ap.add_argument('csvs', nargs='+')
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--x', choices=['E', 'ET'], default='E')
    ap.add_argument('--split', default='test')
    ap.add_argument('--min-n', type=int, default=50)
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--title', default=None)
    ap.add_argument('--ideal', type=float, nargs=3, metavar=('A', 'B', 'C'), default=None,
                    help='ideal curve sigma/E = A/sqrt(E) (+) B/E (+) C, as fractions')
    ap.add_argument('--residuals', action='store_true')
    ap.add_argument('--out', default=None)
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

    print(f"{'model':40s} {'n':>7s} {'seeds':>5s} {'sigma_eff':>9s} {'bias':>8s}")
    for d in data:
        r = resolution(d['pred'], d['true'])
        print(f"{d['name']:40s} {len(d['true']):7d} {d['n_seeds']:5d} {r['sigma_eff']:9.4f} {r['bias']:8.4f}")

    fig = go.Figure()
    for i, d in enumerate(data):
        x = d['true'] if args.x == 'E' else d['ET']
        xs, ys, es = qbin(x, d['true'], d['pred'], args.bins, args.min_n)
        fig.add_scatter(x=xs, y=ys, error_y=dict(type='data', array=es), mode='lines+markers',
                        name=d['name'], line=dict(color=palette[i % len(palette)]))
    if args.ideal is not None:
        a, b, c = args.ideal
        allx = np.concatenate([(d['true'] if args.x == 'E' else d['ET']) for d in data])
        xe = np.linspace(np.quantile(allx, 0.01), np.quantile(allx, 0.99), 200)
        ideal = np.sqrt((a / np.sqrt(xe)) ** 2 + (b / xe) ** 2 + c ** 2)
        fig.add_scatter(x=xe, y=ideal, mode='lines', name='design resolution',
                        line=dict(color='black', dash='dot'))
    xlabel = 'true energy [GeV]' if args.x == 'E' else 'ET [GeV]'
    fig.update_layout(height=500, title=args.title or f'sigma_eff vs {xlabel} ({args.bins} bins, split={args.split})',
                      xaxis_title=xlabel, yaxis_title='sigma_eff', legend_title='model')
    out = Path(args.out) if args.out else Path.cwd() / f'resolution_{args.x}_{args.bins}bins.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs='cdn')
    print('wrote', out)

    if args.residuals:
        fig2 = go.Figure()
        for i, d in enumerate(data):
            res = (d['pred'] - d['true']) / d['true']
            fig2.add_histogram(x=np.clip(res, -0.5, 0.5), nbinsx=120, name=d['name'],
                               opacity=0.55, marker_color=palette[i % len(palette)])
        fig2.update_layout(barmode='overlay', height=500, title='(E_pred - E_true) / E_true',
                           xaxis_title='relative residual', yaxis_title='events', legend_title='model')
        out2 = out.with_name(out.stem + '_residuals.html')
        fig2.write_html(out2, include_plotlyjs='cdn')
        print('wrote', out2)


if __name__ == '__main__':
    main()
