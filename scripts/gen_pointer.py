import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from picocal_data import prep
from picocal_models import NG, load_model
from train_picocal import cached_grid

# Stage 1 of the two-stage architecture: run the trained --aux members over EVERY event and
# save the seed-frame predicted photon offset in cells, for --rc-mode pred. The aux target ya
# is (ax - xs)/ps, so unstandardising with YA_std puts the prediction in the same frame the
# window recentring uses. Pointer error is 0.13-0.20 cells against the seed's 0.43 median and
# 8.5-cell p90 tail at 15mm. Predictions on the training split are made by models that saw
# those events (stage-2 training windows only); the test-split windows come from held-out
# predictions, which is what the reported numbers use.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--name', default='SubNetW8CleanAuxExDnAuxRcQdEma')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default='.scratch/pointer.pkl')
    args = ap.parse_args()
    repo = Path(args.repo)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mb = sorted((repo / 'data' / 'minimum_bias').glob('*.root'))
    cl = sorted((repo / 'data' / 'full').glob('matched_*.root'))
    if args.smoke:
        mb, cl = mb[:4], cl[:2]
    cdir = repo / '.scratch' / 'cache'
    sfx = '_smoke' if args.smoke else ''
    main_ev = cached_grid(mb, 'minbias' + sfx, cdir)
    aux_ev = cached_grid(cl, 'clean-aux' + sfx, cdir)
    D = prep(8, main_ev, aux_ev, prior=None, rings=0, halo=0, globpe=0, patch=0, patch_side=3,
             recenter=True, fold=None, nfold=5, mmgeo=False, rc_regions=None, rc_mode='centroid',
             tcomb=False, allcells=False, ng=NG + 8, phys=False, occ=False, extra=True,
             dens=True, rho=False, tpull=False, aux=True, depth=False, orho=False, abst=False)
    X = torch.from_numpy(D['X']).to(device)
    M = torch.from_numpy(D['M']).to(device)
    G = torch.from_numpy(D['G']).to(device)
    E = torch.from_numpy(D['Eraw']).to(device)
    P = torch.from_numpy(D['POS']).to(device)
    SL = torch.from_numpy(D['SL']).to(device)
    N = len(D['X'])
    outs = []
    for s in args.seeds:
        model, st = load_model(repo / 'models' / f'{args.name}_s{s}.pt', device)
        o = []
        with torch.no_grad():
            for j in range(0, N, 512):
                b = torch.arange(j, min(j + 512, N), device=device)
                o.append(model(X[b], M[b], G[b], E[b], P[b], SL[b])[:, 3:5].cpu().numpy())
        outs.append(np.concatenate(o))
        print(f'seed {s} done', flush=True)
    pred_cells = np.median(np.stack(outs), 0) * D['YA_std'][:2]
    n_mb = N - len(D['ctr'])
    pm = np.full((len(main_ev), 2), np.nan, np.float32)
    pm[D['keep']] = pred_cells[:n_mb]
    pa = np.full((len(aux_ev), 2), np.nan, np.float32)
    pa[D['keep_aux']] = pred_cells[n_mb:]
    outp = repo / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'wb') as f:
        pickle.dump(dict(main=pm, aux=pa), f, protocol=4)
    d = np.hypot(*(pred_cells[:n_mb] - np.stack([
        [(e['ax'] - e['xs']) / e['ps'] for e in main_ev],
        [(e['ay'] - e['ys']) / e['ps'] for e in main_ev]], -1)[D['keep']]).T)
    print(f'-> {outp}  pointer error (cells): median {np.median(d):.3f}  p90 {np.quantile(d, .9):.3f}')


if __name__ == '__main__':
    main()
