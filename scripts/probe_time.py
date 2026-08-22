import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH
from picocal_data import prep
from picocal_models import NG, load_model, width_binned_calibration
from train_picocal import cached_grid

# Interventional probes on the trained model's use of the two time channels. The ablation
# (retrain without time: 0.0485 vs 0.0390) proves timing carries 20% of the resolution, but it
# cannot say HOW the model reads it: a crude per-event average would survive a within-event
# shuffle, while a pairwise same-vertex structure would not. That difference decides whether an
# explicit relative-time mechanism still has room, so it is measured here instead of argued.

TCOLS = slice(7, 11)


def build_final_D(repo, smoke=False):
    repo = Path(repo)
    mb = sorted((repo / 'data' / 'minimum_bias').glob('*.root'))
    cl = sorted((repo / 'data' / 'full').glob('matched_*.root'))
    if smoke:
        mb, cl = mb[:4], cl[:2]
    cdir = repo / '.scratch' / 'cache'
    sfx = '_smoke' if smoke else ''
    main_ev = cached_grid(mb, 'minbias' + sfx, cdir)
    aux_ev = cached_grid(cl, 'clean-aux' + sfx, cdir)
    return prep(8, main_ev, aux_ev, prior=None, rings=0, halo=0, globpe=0, patch=0,
                patch_side=3, recenter=True, fold=None, nfold=5, mmgeo=False, rc_regions=None,
                rc_mode='centroid', tcomb=False, allcells=False, ng=NG + 8, phys=False,
                occ=False, extra=True, dens=True, rho=False, tpull=False, aux=False,
                depth=False, orho=False, abst=False)


def variants(D):
    X, M = D['X'], D['M']
    pf = X[:, :, 9] > 0.5
    pb = X[:, :, 10] > 0.5
    rng = np.random.default_rng(2026)

    def perm():
        Xp = X.copy()
        for i in range(len(X)):
            v = np.flatnonzero(M[i])
            if len(v) > 1:
                Xp[i, v, TCOLS] = X[i, rng.permutation(v), TCOLS]
        return Xp

    def shift(d):
        Xs = X.copy()
        Xs[:, :, 7] = np.where(pf & M, X[:, :, 7] + d, X[:, :, 7])
        Xs[:, :, 8] = np.where(pb & M, X[:, :, 8] + d, X[:, :, 8])
        return Xs

    def coarsen(q):
        Xc = X.copy()
        Xc[:, :, 7] = np.where(pf & M, np.round(X[:, :, 7] / q) * q, X[:, :, 7])
        Xc[:, :, 8] = np.where(pb & M, np.round(X[:, :, 8] / q) * q, X[:, :, 8])
        return Xc

    def notime():
        Xz = X.copy()
        Xz[:, :, TCOLS] = 0.0
        return Xz

    yield 'base', X
    yield 'zeroed', notime()
    yield 'permuted', perm()
    for d in (0.5, 1.0, 2.0):
        yield f'shift_{d:g}sd', shift(d)
    for q in (0.25, 0.5, 1.0, 2.0):
        yield f'coarsen_{q:g}sd', coarsen(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--model', default='SubNetW8CleanAuxExDnRcQdEma_s0.pt')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    repo = Path(args.repo)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    D = build_final_D(repo, args.smoke)
    model, st = load_model(repo / 'models' / args.model, device)
    if st['in_dim'] != D['IN_DIM']:
        raise SystemExit(f"model in_dim {st['in_dim']} != data {D['IN_DIM']}: prep config drifted")
    M = torch.from_numpy(D['M']).to(device)
    G = torch.from_numpy(D['G']).to(device)
    E = torch.from_numpy(D['Eraw']).to(device)
    P = torch.from_numpy(D['POS']).to(device)
    SL = torch.from_numpy(D['SL']).to(device)
    kva, kte = np.asarray(D['kva']), np.asarray(D['kte'])
    yva = D['y'][kva]
    Et = D['Et'][kte]
    reg = D['reg'][kte]

    def run(Xt, idx, bs=512):
        out = []
        with torch.no_grad():
            for j in range(0, len(idx), bs):
                b = torch.from_numpy(idx[j:j + bs]).to(device)
                out.append(model(Xt[b], M[b], G[b], E[b], P[b], SL[b])[:, :3].cpu().numpy())
        return np.concatenate(out)

    rows = []
    for name, Xv in variants(D):
        Xt = torch.from_numpy(Xv).to(device)
        pe = width_binned_calibration(run(Xt, kva), run(Xt, kte), yva)
        del Xt
        row = dict(variant=name, sigma_eff=resolution(pe, Et)['sigma_eff'])
        for r in np.unique(reg):
            k = reg == r
            row[f'{int(PITCH[r])}mm'] = resolution(pe[k], Et[k])['sigma_eff']
        rows.append(row)
        print(' | '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k}={v}'
                         for k, v in row.items()), flush=True)
    out = Path(args.out or repo / 'reports' / 'probe_time.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
