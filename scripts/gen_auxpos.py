import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import PITCH
from picocal_data import prep
from picocal_models import NG, load_model
from train_picocal import cached_grid

# How well does the --aux head localise the photon? The oracle-centre run prices perfect
# centring at -4% aggregate and -10% in the weak bins, so the question that decides whether a
# two-stage learned recentring is worth building is the aux head's position error against the
# centring error it would replace. Needs prep(aux=True) for the YA truth targets.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--name', default='SubNetW8CleanAuxExDnAuxRcQdEma')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    repo = Path(args.repo)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mb = sorted((repo / 'data' / 'minimum_bias').glob('*.root'))
    cl = sorted((repo / 'data' / 'full').glob('matched_*.root'))
    if args.smoke:
        mb, cl = mb[:4], cl[:2]
    cdir = repo / '.scratch' / 'cache'
    sfx = '_smoke' if args.smoke else ''
    D = prep(8, cached_grid(mb, 'minbias' + sfx, cdir), cached_grid(cl, 'clean-aux' + sfx, cdir),
             prior=None, rings=0, halo=0, globpe=0, patch=0, patch_side=3, recenter=True,
             fold=None, nfold=5, mmgeo=False, rc_regions=None, rc_mode='centroid', tcomb=False,
             allcells=False, ng=NG + 8, phys=False, occ=False, extra=True, dens=True, rho=False,
             tpull=False, aux=True, depth=False, orho=False, abst=False)
    X = torch.from_numpy(D['X']).to(device)
    M = torch.from_numpy(D['M']).to(device)
    G = torch.from_numpy(D['G']).to(device)
    E = torch.from_numpy(D['Eraw']).to(device)
    P = torch.from_numpy(D['POS']).to(device)
    SL = torch.from_numpy(D['SL']).to(device)
    kte = np.asarray(D['kte'])
    outs = []
    for s in args.seeds:
        model, st = load_model(repo / 'models' / f'{args.name}_s{s}.pt', device)
        o = []
        with torch.no_grad():
            for j in range(0, len(kte), 512):
                b = torch.from_numpy(kte[j:j + 512]).to(device)
                o.append(model(X[b], M[b], G[b], E[b], P[b], SL[b])[:, 3:6].cpu().numpy())
        outs.append(np.concatenate(o))
        print(f'seed {s} done', flush=True)
    pred = np.median(np.stack(outs), 0)
    true = D['YA'][kte]
    reg = D['reg'][kte]
    print('\naux position error (standardised YA units, per region):')
    for r in np.unique(reg):
        k = reg == r
        d = np.sqrt(((pred[k, :2] - true[k, :2]) ** 2).sum(1))
        base = np.sqrt((true[k, :2] ** 2).sum(1))
        print(f'  {int(PITCH[r])}mm: median |pred-true| {np.median(d):.3f}  '
              f'median |true-0| (predict-nothing) {np.median(base):.3f}  '
              f'ratio {np.median(d) / max(np.median(base), 1e-9):.2f}', flush=True)
    d = np.sqrt(((pred[:, :2] - true[:, :2]) ** 2).sum(1))
    base = np.sqrt((true[:, :2] ** 2).sum(1))
    print(f'  all: median |pred-true| {np.median(d):.3f}  baseline {np.median(base):.3f}  '
          f'ratio {np.median(d) / np.median(base):.2f}')
    np.save(repo / 'reports' / 'predictions' / 'auxpos_test.npy',
            dict(pred=pred, true=true, reg=reg), allow_pickle=True)
    print('-> reports/predictions/auxpos_test.npy')


if __name__ == '__main__':
    main()
