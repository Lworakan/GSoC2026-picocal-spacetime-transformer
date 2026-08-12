import argparse
import copy
import pickle
import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH
from picocal_data import build_grid, prep
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from picocal_models import (CFG, NG, SubNetFQ, CNNSub, QUANTILES, pinball_loss, qd_pinball_loss,
                            width_binned_calibration, linear_calibration, save_model)


def parse_args():
    ap = argparse.ArgumentParser(description='Train the PicoCal SubNet family from a configuration')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--sample', choices=['minbias', 'clean'], default='minbias')
    ap.add_argument('--cleanaux', action='store_true')
    ap.add_argument('--objective', choices=['qd', 'quant', 'huber'], default='qd')
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--gate', choices=['learned', 'time', 'signed', 'off'], default='learned')
    ap.add_argument('--no-time', action='store_true')
    ap.add_argument('--phys', action='store_true')
    ap.add_argument('--occ', action='store_true')
    ap.add_argument('--d4aug', action='store_true')
    ap.add_argument('--tta', action='store_true',
                    help='average the prediction over the 8 D4 transforms of the window at '
                         'EVALUATION time. Group averaging reduces output variance and is a '
                         'different mechanism from train-time augmentation; it was previously '
                         'claimed in reports but never implemented in this pipeline.')
    ap.add_argument('--trim', type=float, default=0.0,
                    help='trimmed-risk objective: ignore the worst FRAC of per-sample losses in '
                         'each batch (0 = off). sigma_eff is itself a trimmed statistic, and '
                         'dropping the worst 10%% of residuals lowers it by 30%% in our data, so '
                         'the unweighted mean loss is misaligned with the metric.')
    ap.add_argument('--arch', choices=['std', 'geo', 'cnn'], default='std')
    ap.add_argument('--qpool', action='store_true')
    ap.add_argument('--film', action='store_true')
    ap.add_argument('--nfour', type=int, default=0,
                    help='number of Fourier frequencies applied to the in-cell offset channels '
                         '(0 = off). Targets the sub-cell impact position, which our own '
                         'measurements identify as the hidden variable.')
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--dens', action='store_true')
    ap.add_argument('--gx', action='store_true')
    ap.add_argument('--rho', action='store_true')
    ap.add_argument('--tpull', action='store_true')
    ap.add_argument('--depth', action='store_true')
    ap.add_argument('--abst', action='store_true',
                    help='keep the ABSOLUTE window time plus a late-energy fraction and time '
                         'spread as global features. In-time pileup is unreachable at our per-'
                         'cell resolution, but out-of-time pileup sits 25 ns away and the '
                         'pipeline currently median-subtracts the only feature that sees it.')
    ap.add_argument('--orho', action='store_true',
                    help='per-cell pileup subtraction using a density estimated from the cells '
                         'OUTSIDE the window, which are nearly photon-free (the photon is 99% '
                         'contained inside 9x9 in clean data)')
    ap.add_argument('--wlow', type=float, default=0.0,
                    help='inverse-density loss weighting exponent over (region, log E). 0 = off. '
                         'The metric is per-region per-energy-bin sigma_eff, but an unweighted '
                         'loss optimises mostly the abundant easy events; 0.5-1.0 shifts capacity '
                         'to the rare low-energy inner-region clusters.')
    ap.add_argument('--aux', action='store_true')
    ap.add_argument('--aux-w', type=float, default=0.2)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--mode', choices=['full', 'smoke'], default='full')
    ap.add_argument('--device', default=None)
    ap.add_argument('--name', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--models-dir', default=None)
    ap.add_argument('--ckpt-dir', default=None)
    return ap.parse_args()


def cached_grid(files, tag, cache_dir, no_cache=False):
    if no_cache:
        return build_grid(files, tag)
    cp = Path(cache_dir) / f'{tag}_{len(files)}.pkl'
    if cp.exists():
        with open(cp, 'rb') as f:
            ev = pickle.load(f)
        print(f'{tag}: {len(ev)} events (cache)')
        return ev
    ev = build_grid(files, tag)
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(ev, f, protocol=4)
    tmp.replace(cp)
    return ev


def d4_apply(xb, mb, mean, std, phys, pidx, k=None, vec_idx=()):
    # Every channel that is a COMPONENT OF A VECTOR must transform with the window, not just
    # di/dj: the --extra block carries the cell position relative to the reconstructed shower
    # centroid, and --phys carries mm offsets. Rotating only some of them feeds the model a
    # self-contradictory event, which silently invalidates both train-time augmentation and
    # test-time averaging.
    if k is None:
        k = int(torch.randint(0, 8, (1,)).item())
    if k == 0:
        return xb
    sx = -1.0 if k & 1 else 1.0
    sy = -1.0 if k & 2 else 1.0
    sw = bool(k & 4)
    x = xb.clone()
    di = x[:, :, 3] * std[3] + mean[3]
    dj = x[:, :, 4] * std[4] + mean[4]
    ndi, ndj = (dj, di) if sw else (di, dj)
    x[:, :, 3] = (sx * ndi - mean[3]) / std[3]
    x[:, :, 4] = (sy * ndj - mean[4]) / std[4]
    pairs = list(vec_idx) + ([pidx] if phys else [])
    for i0 in pairs:
        ux = x[:, :, i0].clone()
        uy = x[:, :, i0 + 1].clone()
        nux, nuy = (uy, ux) if sw else (ux, uy)
        x[:, :, i0] = sx * nux
        x[:, :, i0 + 1] = sy * nuy
    x[~mb] = 0.0
    return x


def sample_weights(D, alpha):
    if alpha <= 0:
        return np.ones(len(D['Et']), np.float32)
    le = np.log(np.clip(D['Et'], 1e-3, None))
    edges = np.quantile(le[D['ktr']], np.linspace(0, 1, 11))
    ib = np.clip(np.searchsorted(edges, le, side='right') - 1, 0, len(edges) - 2)
    key = D['reg'] * (len(edges) - 1) + ib
    cnt = np.bincount(key[D['ktr']], minlength=key.max() + 1).astype(np.float64)
    dens = np.clip(cnt[key], 1.0, None) / max(len(D['ktr']), 1)
    w = dens ** (-alpha)
    w = w / w[D['ktr']].mean()
    return np.clip(w, 0.2, 8.0).astype(np.float32)


def build_model(D, args, device):
    if args.arch == 'cnn':
        return CNNSub(D['IN_DIM'], D['la0'], D['lb0'], ng=D['G'].shape[1],
                      side=D['S']).to(device)
    return SubNetFQ(D['IN_DIM'], D['la0'], D['lb0'], ng=D['G'].shape[1], gate=args.gate,
                    arch=args.arch, qpool=args.qpool, gx=args.gx, aux=args.aux, film=args.film,
                    nfour=args.nfour).to(device)


def train_one(T, D, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(D, args, device)
    ema = None if args.no_ema else AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    qs = torch.tensor(QUANTILES, device=device)
    tr_idx = np.concatenate([np.asarray(D['ktr']), D['ctr']]) if len(D['ctr']) else np.asarray(D['ktr'])
    ck = Path(args.ckpt_dir) / f'{args.name}_s{seed}.pt'

    def batches(idx, bs, sh):
        idx = np.asarray(idx)
        if sh:
            idx = rng.permutation(idx)
        for j in range(0, len(idx), bs):
            yield torch.from_numpy(idx[j:j + bs]).to(device)

    def fwd(b):
        xb = T['X'][b]
        if args.d4aug:
            xb = d4_apply(xb, T['M'][b], T['mean'], T['std'], args.phys,
                          16 + (2 if args.orho else 0) + (2 if args.depth else 0) + (3 if args.tpull else 0) + (1 if args.rho else 0) + (2 if args.extra else 0) + (3 if args.dens else 0))
        return model(xb, T['M'][b], T['G'][b], T['E'][b], T['P'][b])

    def loss_fn(qa, yb, b=None):
        q = qa[:, :3]
        extra = 0.0
        if args.aux and b is not None:
            extra = args.aux_w * nn.functional.huber_loss(qa[:, 3:], T['YA'][b], delta=1.0)
        w = T['W'][b] if (args.wlow > 0 and b is not None) else None
        return base_loss(q, yb, w) + extra

    def base_loss(q, yb, w=None):
        if args.trim > 0:
            with torch.no_grad():
                per = (yb - q[:, 1:2]).abs().squeeze(1)
                k = max(int(per.numel() * (1.0 - args.trim)), 8)
                thr = torch.kthvalue(per, k).values
                keep = (per <= thr).float()
            w = keep if w is None else w * keep
        if args.objective == 'qd':
            return qd_pinball_loss(q, yb, qs, w=w)
        if args.objective == 'quant':
            return pinball_loss(q, yb, qs, w=w)
        if w is None:
            return nn.functional.huber_loss(q[:, 1:2], yb, delta=CFG['huber_delta'])
        h = nn.functional.huber_loss(q[:, 1:2], yb, delta=CFG['huber_delta'], reduction='none')
        return (h.squeeze(1) * w).sum() / w.sum().clamp(min=1e-6)

    def eval_model():
        return ema.module if ema is not None else model

    def run(idx):
        m = eval_model()
        m.eval()
        out = []
        with torch.no_grad():
            for b in batches(idx, 256, False):
                out.append(m(T['X'][b], T['M'][b], T['G'][b], T['E'][b], T['P'][b]).cpu().numpy())
        return np.concatenate(out)

    def vloss():
        m = eval_model()
        m.eval()
        s, k = 0.0, 0
        with torch.no_grad():
            for b in batches(D['kva'], 256, False):
                q = m(T['X'][b], T['M'][b], T['G'][b], T['E'][b], T['P'][b])
                s += pinball_loss(q[:, :3], T['Y'][b], qs).item()
                k += 1
        return s / max(k, 1)

    best, bstate, wait, ep0 = 1e9, None, 0, 0
    if ck.exists():
        st = torch.load(ck, map_location=device)
        model.load_state_dict(st['model'])
        if ema is not None and st.get('ema') is not None:
            ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        best, bstate, wait, ep0 = st['best'], st['bstate'], st['wait'], st['ep'] + 1
        rng = np.random.default_rng(seed + 1000 * ep0)
        print(f'  resume {args.name} s{seed} from epoch {ep0}', flush=True)
    for ep in range(ep0, args.epochs):
        model.train()
        for b in batches(tr_idx, CFG['batch'], True):
            opt.zero_grad()
            loss_fn(fwd(b), T['Y'][b], b).backward()
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
        sched.step()
        vv = vloss()
        if vv < best - 1e-4:
            best, bstate, wait = vv, copy.deepcopy(eval_model().state_dict()), 0
        else:
            wait += 1
        torch.save(dict(model=model.state_dict(), ema=None if ema is None else ema.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(),
                        best=best, bstate=bstate, wait=wait, ep=ep), ck)
        if wait >= args.patience:
            break
    final = build_model(D, args, device)
    final.load_state_dict(bstate)
    final.eval()

    off = 16 + (2 if args.orho else 0) + (2 if args.depth else 0) + (3 if args.tpull else 0)
    vec_idx = (off,) if args.extra else ()          # extra: cell position vs reco centroid
    pidx = off + (1 if args.rho else 0) + (2 if args.extra else 0) + (3 if args.dens else 0)

    def run_final(idx):
        out = []
        ks = range(8) if args.tta else (0,)
        with torch.no_grad():
            for b in batches(idx, 256, False):
                acc = None
                for k in ks:
                    xb = T['X'][b] if k == 0 else d4_apply(T['X'][b], T['M'][b], T['mean'],
                                                           T['std'], args.phys, pidx, k, vec_idx)
                    o = final(xb, T['M'][b], T['G'][b], T['E'][b], T['P'][b])
                    acc = o if acc is None else acc + o
                out.append((acc / len(list(ks))).cpu().numpy())
        return np.concatenate(out)

    qv, qt = run_final(D['kva'])[:, :3], run_final(D['kte'])[:, :3]
    yva = D['y'][D['kva']]
    if args.objective in ('quant', 'qd'):
        pe = width_binned_calibration(qv, qt, yva)
    else:
        pe = linear_calibration(qv, qt, yva)
    return final, pe


def main():
    args = parse_args()
    repo = Path(args.repo)
    args.out = Path(args.out or repo / 'reports' / 'predictions')
    args.models_dir = Path(args.models_dir or repo / 'models')
    args.ckpt_dir = Path(args.ckpt_dir or repo / '.scratch' / 'ckpt')
    for d in (args.out, args.models_dir, args.ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.name is None:
        base = 'SubNetW%d' % args.window
        suffix = {'qd': 'Qd', 'quant': 'Quant', 'huber': ''}[args.objective] + ('' if args.no_ema else 'Ema')
        args.name = (base + ('CleanAux' if args.cleanaux else '') +
                     ('Phys' if args.phys else '') + ('Occ' if args.occ else '') +
                     ('Tgate' if args.gate == 'time' else '') + ('Sgn' if args.gate == 'signed' else '') +
                     ('D4' if args.d4aug else '') + ('Geo' if args.arch == 'geo' else '') + ('Cnn' if args.arch == 'cnn' else '') + ('Ex' if args.extra else '') + ('Dn' if args.dens else '') + ('Gx' if args.gx else '') + ('Rho' if args.rho else '') + ('Tp' if args.tpull else '') + ('Aux' if args.aux else '') + ('Dep' if args.depth else '') + ('Orh' if args.orho else '') + ('Abs' if args.abst else '') + (f'W{args.wlow:g}'.replace('.', '') if args.wlow > 0 else '') +
                     ('Qp' if args.qpool else '') + ('Tta' if args.tta else '') + (f'Tr{int(args.trim*100)}' if args.trim > 0 else '') + ('Fm' if args.film else '') + (f'F{args.nfour}' if args.nfour else '') + suffix)
    if args.mode == 'smoke':
        args.name += '_smoke'
        args.epochs, args.patience = 2, 99

    mb_files = sorted((repo / 'data' / 'minimum_bias').glob('*.root'))
    clean_files = sorted((repo / 'data' / 'full').glob('matched_*.root'))
    if args.mode == 'smoke':
        mb_files, clean_files = mb_files[:4], clean_files[:2]
    main_files = mb_files if args.sample == 'minbias' else clean_files
    cdir = repo / '.scratch' / 'cache'
    tag = args.sample + ('_smoke' if args.mode == 'smoke' else '')
    main_ev = cached_grid(main_files, tag, cdir, args.no_cache)
    aux_ev = (cached_grid(clean_files, 'clean-aux' + ('_smoke' if args.mode == 'smoke' else ''),
                          cdir, args.no_cache)
              if (args.cleanaux and args.sample == 'minbias') else None)
    D = prep(args.window, main_ev, aux_ev,
             ng=NG + (2 if args.occ else 0) + (8 if args.extra else 0)
             + (2 if args.rho else 0) + (3 if args.abst else 0),
             phys=args.phys, occ=args.occ, extra=args.extra, dens=args.dens, rho=args.rho,
             tpull=args.tpull, aux=args.aux, depth=args.depth, orho=args.orho, abst=args.abst)
    if args.no_time:
        D['X'][:, :, 7:11] = 0.0
    T = dict(X=torch.from_numpy(D['X']).to(device), M=torch.from_numpy(D['M']).to(device),
             G=torch.from_numpy(D['G']).to(device), Y=torch.from_numpy(D['y']).unsqueeze(1).to(device),
             E=torch.from_numpy(D['Eraw']).to(device),
             P=torch.from_numpy(D['POS']).to(device),
             YA=torch.from_numpy(D['YA']).to(device),
             W=torch.from_numpy(sample_weights(D, args.wlow)).to(device),
             mean=torch.from_numpy(np.asarray(D['mean'], np.float32)).to(device),
             std=torch.from_numpy(np.asarray(D['std'], np.float32)).to(device))
    print(f'device {device} | {args.name} | objective {args.objective} | seeds {args.seeds}')

    csvp = args.out / f'{args.sample}__{args.name}.csv'
    done = set()
    if csvp.exists():
        done = set(pd.read_csv(csvp)['seed'].unique())
        print('resume, done seeds:', sorted(done))
    kte = D['kte']
    for seed in args.seeds:
        if seed in done:
            print('skip seed', seed)
            continue
        t0 = time.time()
        model, pe = train_one(T, D, seed, args, device)
        sig = resolution(pe, D['Et'][kte])['sigma_eff']
        save_model(args.models_dir / f'{args.name}_s{seed}.pt', model,
                   dict(in_dim=D['IN_DIM'], la0=D['la0'], lb0=D['lb0'], ng=D['G'].shape[1],
                        mean=D['mean'], std=D['std'], cfg=CFG, window=args.window,
                        objective=args.objective, sample=args.sample, cleanaux=args.cleanaux,
                        gate=args.gate, arch=args.arch, qpool=args.qpool, gx=args.gx,
                        aux=args.aux, film=args.film, nfour=args.nfour, seed=seed,
                        feats=dict(extra=args.extra, dens=args.dens, orho=args.orho,
                                   tpull=args.tpull, depth=args.depth, phys=args.phys,
                                   occ=args.occ, rho=args.rho)))
        df = pd.DataFrame(dict(
            model=args.name, dataset=args.sample, seed=seed, split='test',
            true_energy=D['Et'][kte], pred_energy=pe,
            pred_bias=pe / D['Et'][kte] - 1.0,
            region=D['reg'][kte],
            region_name=[f'{int(PITCH[r])}mm' for r in D['reg'][kte]],
            ET=D['ET'][kte]))
        df.to_csv(csvp, mode='a', header=not csvp.exists() or csvp.stat().st_size == 0, index=False)
        print(f'{args.name} seed {seed}: sigma_eff {sig:.4f} ({time.time() - t0:.0f}s) -> {csvp.name}', flush=True)


if __name__ == '__main__':
    main()
