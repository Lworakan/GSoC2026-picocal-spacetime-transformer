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
    ap.add_argument('--arch', choices=['std', 'geo', 'cnn', 'pnet', 'gravnet', 'convnext', 'spacetime', 'mixer'], default='std')
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
    ap.add_argument('--gatesup', type=float, default=0.0,
                    help='weight of per-cell gate supervision using the synthetic overlay '
                         'sample (0 = off). The overlay knows the true photon fraction per '
                         'cell; measured against it the learned gate correlates only 0.21, and '
                         'an oracle gate would improve 30mm low-E by 1.9x, so this trains the '
                         'gate directly instead of hoping it emerges from the energy loss.')
    ap.add_argument('--synaux', action='store_true',
                    help='add the synthetic overlay events to training WITHOUT the gate loss. '
                         'This is the control arm --gatesup needs: turning gatesup on adds both '
                         'a loss term and ~9.7k extra events, so without this run a gain cannot '
                         'be attributed to supervision rather than to the extra data.')
    ap.add_argument('--aperture-mm', type=float, default=0.0,
                    help='keep only cells within this PHYSICAL radius of the window centre, on '
                         'top of the square crop. 0 = off. The square window is counted in cells '
                         'while a shower has a fixed width in mm (Moliere radius ~35mm), so at '
                         'fixed W the aperture spans 60mm at 15mm pitch and 480mm at 120mm -- '
                         'and the benefit of widening is ordered exactly by that physical size '
                         '(Spearman 1.000 over the five regions). A radius cut makes the aperture '
                         'the same everywhere and drops the square corners, which reach W*sqrt(2) '
                         'cells and hold essentially no photon.')
    ap.add_argument('--allcells', action='store_true',
                    help='no square crop: every cluster cell is a token. A window is a square '
                         'crop of a round cluster, so even W8 discards corners and tail, and its '
                         'padded grid (289 slots) costs more than the ~270 real cells at 15mm. '
                         'Attention encoders only; the CNN grid needs the square.')
    ap.add_argument('--tcomb', action='store_true',
                    help='inverse-variance combination of front and back cell times, replacing the '
                         'two raw time channels with (combined residual, log sigma). ~0.82x on '
                         'per-cell sigma_t from the measured resolution table; the one timing item '
                         'never implemented, and the only one that stays raw rather than engineered.')
    ap.add_argument('--rc-pred', default='',
                    help='pickle with {"main": (N,2), "aux": (N,2)} predicted photon offsets '
                         'from the seed in cells (scripts/gen_pointer.py), aligned to the raw '
                         'event lists. Used by --rc-mode pred: the two-stage architecture where '
                         'a first pass points at the photon and the window is re-cut around it.')
    ap.add_argument('--rc-mode', choices=['centroid', 'oracle', 'pred'], default='centroid',
                    help='oracle centres the window on the TRUE photon entry. Diagnostic only: '
                         'it bounds what any centre estimator could achieve, and it separates the '
                         'coverage theory of the recentring gain from the localisation theory.')
    ap.add_argument('--rc-regions', type=int, nargs='*', default=None,
                    help='restrict --recenter to these region indices. Measured at seed 0: moving the '
                         'centre to the cluster centroid gains -0.0371 at 15mm low-E and -0.0086 at '
                         '30mm low-E but LOSES at 40mm (+0.0020) and 120mm (+0.0034), because the '
                         'centroid is dragged by pileup tails where the loudest cell was already the '
                         'photon. Occupancy decides which estimator is better, so the choice is '
                         'per-region and follows the measured mis-seeding rate, not the test metric.')
    ap.add_argument('--mmgeo', action='store_true',
                    help='express the per-cell geometry in millimetres scaled by the shower size '
                         'instead of in cell counts. Measured on the clean sample: the photon is '
                         '99.7-100%% contained within 120 mm in every region, so the shower has one '
                         'physical width while its width in CELLS varies eightfold, and the model is '
                         'currently forced to learn five radial functions for what physics says is '
                         'one. Replaces the cell-unit channels rather than adding to them.')
    ap.add_argument('--fold', type=int, default=None,
                    help='k-fold cross-validation fold index instead of the fixed 70/15/15 split. '
                         'The fixed split trains on 50,787 of 72,554 min-bias events; the missing 35%% '
                         'is worth about 1.086 on sigma_eff at our measured N^-0.28, and the test set '
                         'holds only 411 events per seed at 15mm low-E, which is why one config spanned '
                         '0.1138-0.1319 over five seeds. With k folds every event is tested once.')
    ap.add_argument('--nfold', type=int, default=5)
    ap.add_argument('--recenter', action='store_true',
                    help='centre the window on the reconstruction cluster centroid instead of the '
                         'loudest cell. Measured against the truth entry point, at 15mm the photon '
                         'lands more than two cells from the argmax seed in 17.6%% of events (p90 = '
                         '8.28 cells) because a low-energy photon loses the loudness contest to a '
                         'pileup cell where occupancy is highest; 120mm shows 0.1%%. Never varied in '
                         '60+ experiments.')
    ap.add_argument('--lr', type=float, default=None,
                    help='learning rate (default 3e-4). The optimiser recipe has never been swept '
                         'in 60+ experiments: every arm changed inputs or encoders on a frozen '
                         'training formula. At this data size the recipe is worth measuring.')
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--cosine', action='store_true', help='cosine LR decay over the epoch budget')
    ap.add_argument('--cjit', type=float, default=0.0,
                    help='train-time gaussian jitter (in cells) on the token coordinate channels. '
                         'The centring study showed the model is sensitive to where the window '
                         'origin lands; jitter teaches invariance to that estimation error.')
    ap.add_argument('--dim', type=int, default=None,
                    help='transformer width (default 128). Provided to TEST the capacity question '
                         'rather than to assume an answer: 624k parameters against 50,787 training '
                         'events is already 12 parameters per event, and the aggregate has sat at '
                         '0.0379-0.0402 across ~60 experiments spanning transformer, CNN, GNN, EFN '
                         'and pairwise encoders, which is the signature of an information floor '
                         'rather than a capacity one.')
    ap.add_argument('--layers', type=int, default=None, help='transformer depth (default 3)')
    ap.add_argument('--globpe', type=int, default=0,
                    help='Fourier frequencies on the ABSOLUTE detector position of the seed cell '
                         '(0 = off). ClusTEX (arXiv:2603.18172) separates the local window '
                         'coordinate from a global detector coordinate carried by a learnable '
                         'embedding, whose weights they observe converging to a cosine of the '
                         'detector extent; this supplies that basis directly. Our tokens have only '
                         'ever seen local offsets, the region one-hot and the pitch.')
    ap.add_argument('--patch', type=int, default=0,
                    help='patch-hierarchical tokens out to ring PATCH: cells stay individual inside '
                         'the window, and outside it each PATCH_SIDE x PATCH_SIDE block becomes one '
                         'token. Follows arXiv:2605.21789 (2026). Preferred over --halo because a '
                         'ring collapses everything at one radius into a single number and so '
                         'deletes angular structure, which the ring arms showed the model uses.')
    ap.add_argument('--patch-side', type=int, default=3)
    ap.add_argument('--halo', type=int, default=0,
                    help='radial multi-scale tokens: individual cells inside the window, plus ONE '
                         'token per ring from window+1 to HALO carrying that ring summed energy '
                         'and its energy-weighted timing. The 15mm cluster reaches ring 15 but the '
                         'far rings are diffuse pileup, so paying O(n^2) attention per cell out '
                         'there is waste -- W15 is 961 tokens against 92 this way.')
    ap.add_argument('--only-region', type=int, default=None,
                    help='train and score on one region index only (0=15mm .. 4=120mm). The best '
                         'window is region-dependent -- the 15mm cluster reaches ring 15 while '
                         '60mm and 120mm end by ring 3 and 1 -- so one shared window is wrong for '
                         'everyone. A 15mm specialist can afford a wide window because the other '
                         'regions do not pay for it, and it trains faster on ~11% of the events.')
    ap.add_argument('--rings', type=int, default=0,
                    help='add per-ring energy sums for rings window+1..RINGS as global features. '
                         'The 15mm cluster reaches ring 15 while a 9x9 window sees 37.6% of its '
                         'energy, and enlarging the window moved 15mm low-E by 34%; this buys the '
                         'same containment information without paying O(n^2) attention on hundreds '
                         'of extra tokens.')
    ap.add_argument('--prior-feat', action='store_true',
                    help='append the fitted per-cell photon-fraction estimate as an input '
                         'feature (see scripts/fit_cell_prior.py)')
    ap.add_argument('--prior-teach', action='store_true',
                    help='use that estimate as the gate target on EVERY event. Truth-based gate '
                         'supervision only ever reached the synthetic events, so the gate on real '
                         'min-bias stayed at corr 0.211; this target is a function of observables '
                         'and therefore applies to real events too. Combine with --gatesup W to '
                         'set the weight.')
    ap.add_argument('--prior-path', default='.scratch/cell_prior.pkl')
    ap.add_argument('--overlay', default='.scratch/cache/overlay.pkl')
    ap.add_argument('--aux', action='store_true')
    ap.add_argument('--aux-w', type=float, default=0.2)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--mode', choices=['full', 'smoke'], default='full')
    ap.add_argument('--frac', type=float, default=0.0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--preln', action='store_true')
    ap.add_argument('--knn', type=int, default=0)
    ap.add_argument('--tfour', type=int, default=0)
    ap.add_argument('--sum-core', type=int, default=0,
                    help='restrict the gated energy sum to cells within Chebyshev SUM_CORE of '
                         'the window centre, while the encoder still attends over the whole '
                         'window. Measured on the clean sample, the photon is 99.8%% contained '
                         'by w=4 already, so the outer cells of a w=8 window carry no signal to '
                         'sum -- they carry the pileup field the estimate has to subtract. This '
                         'separates the two roles instead of letting one weighted sum do both.')
    ap.add_argument('--slots', type=int, default=0,
                    help='slot-decomposition readout with SLOTS competing queries: each cell '
                         'energy is split by a softmax responsibility over the slots and only '
                         'slot 0 enters the energy sum. The event is physically a sum of showers, '
                         'and no ledger encoder carries that superposition as structure.')
    ap.add_argument('--distill', default='',
                    help='path to a teacher .npy of shape (N, 3): the seed-averaged raw quantile '
                         'outputs of the 5-seed ensemble over every event (scripts/gen_teacher.py). '
                         'The ensemble scores 0.0388 against the single member 0.0402; this trains '
                         'a single member toward the average it would otherwise need 5 passes for.')
    ap.add_argument('--distill-w', type=float, default=1.0)
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
    if args.arch in ('cnn', 'convnext'):
        return CNNSub(D['IN_DIM'], D['la0'], D['lb0'], ng=D['G'].shape[1],
                      side=D['S'], modern=args.arch == 'convnext').to(device)
    return SubNetFQ(D['IN_DIM'], D['la0'], D['lb0'], ng=D['G'].shape[1], gate=args.gate,
                    arch=args.arch, qpool=args.qpool, gx=args.gx, aux=args.aux, film=args.film,
                    nfour=args.nfour, tfour=args.tfour, slots=args.slots).to(device)


def train_one(T, D, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(D, args, device)
    ema = None if args.no_ema else AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs - args.warmup, 1))
    sched = (torch.optim.lr_scheduler.SequentialLR(
                 opt, [torch.optim.lr_scheduler.LinearLR(
                           opt, start_factor=0.1, total_iters=args.warmup), cos],
                 milestones=[args.warmup])
             if args.warmup else cos)
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
        if args.cjit > 0 and model.training:
            # jitter the standardised coordinate channels (cols 3:5 = di, dj). The centring study
            # showed accuracy tracks window-origin error; a jittered origin trains invariance to
            # the centre estimator's noise.
            xb = xb.clone()
            xb[:, :, 3:5] = xb[:, :, 3:5] + torch.randn_like(xb[:, :, 3:5]) * (args.cjit / T['std'][3:5].clamp(min=1e-6))
        if args.d4aug:
            xb = d4_apply(xb, T['M'][b], T['mean'], T['std'], args.phys,
                          16 + (2 if args.orho else 0) + (2 if args.depth else 0) + (3 if args.tpull else 0) + (1 if args.rho else 0) + (2 if args.extra else 0) + (3 if args.dens else 0))
        return model(xb, T['M'][b], T['G'][b], T['E'][b], T['P'][b], T['SL'][b])

    def loss_fn(qa, yb, b=None):
        q = qa[:, :3]
        extra = 0.0
        if args.aux and b is not None:
            extra = args.aux_w * nn.functional.huber_loss(qa[:, 3:], T['YA'][b], delta=1.0)
        w = T['W'][b] if (args.wlow > 0 and b is not None) else None
        if args.gatesup > 0 and b is not None and getattr(model, 'last_w', None) is not None:
            hf = T['HASF'][b]
            if hf.sum() > 0:
                mm = T['M'][b].float() * hf.unsqueeze(1)
                d2 = ((model.last_w - T['FR'][b]) ** 2 * mm).sum() / mm.sum().clamp(min=1.0)
                extra = extra + args.gatesup * d2
        if args.distill and b is not None:
            extra = extra + args.distill_w * nn.functional.huber_loss(q, T['TD'][b], delta=0.1)
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
                out.append(m(T['X'][b], T['M'][b], T['G'][b], T['E'][b], T['P'][b], T['SL'][b]).cpu().numpy())
        return np.concatenate(out)

    def vloss():
        m = eval_model()
        m.eval()
        s, k = 0.0, 0
        with torch.no_grad():
            for b in batches(D['kva'], 256, False):
                q = m(T['X'][b], T['M'][b], T['G'][b], T['E'][b], T['P'][b], T['SL'][b])
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
                    o = final(xb, T['M'][b], T['G'][b], T['E'][b], T['P'][b], T['SL'][b])
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
    if args.allcells and args.arch in ('cnn', 'convnext'):
        raise SystemExit('--allcells has no square grid for the CNN path; use an attention encoder')
    if args.mmgeo and args.arch in ('cnn', 'convnext'):
        raise SystemExit('--mmgeo rescales the geometry channels, and the CNN path builds its grid '
                         'index from them: int() of a millimetre offset collapses every token onto '
                         'one pixel. Use --mmgeo with an attention encoder.')
    if args.lr:
        CFG['lr'] = args.lr
    if args.batch:
        CFG['batch'] = args.batch
    if args.dim:
        CFG['d'] = args.dim
    if args.layers:
        CFG['layers'] = args.layers
    CFG['preln'] = args.preln
    if args.knn:
        CFG['knn'] = args.knn
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
                     ('D4' if args.d4aug else '') + ('Geo' if args.arch == 'geo' else '') + ('Cnn' if args.arch == 'cnn' else '') + ('Cnx' if args.arch == 'convnext' else '') + ('St' if args.arch == 'spacetime' else '') + ('Pnet' if args.arch == 'pnet' else '') + ('Grav' if args.arch == 'gravnet' else '') + ('Mxr' if args.arch == 'mixer' else '') + (f'Fr{int(round(args.frac*100))}' if args.frac and args.frac < 1.0 else '') + (f'Wu{args.warmup}' if args.warmup else '') + ('Pln' if args.preln else '') + (f'Nn{args.knn}' if args.knn else '') + (f'Tf{args.tfour}' if args.tfour else '') + (f'Sc{args.sum_core}' if args.sum_core else '') + (f'Sk{args.slots}' if args.slots else '') + ('Ds' if args.distill else '') + ('Ex' if args.extra else '') + ('Dn' if args.dens else '') + ('Gx' if args.gx else '') + ('Rho' if args.rho else '') + ('Tp' if args.tpull else '') + ('Aux' if args.aux else '') + (f'Gs{int(args.gatesup*10)}' if args.gatesup > 0 else '') + ('Sx' if args.synaux else '') + ('Pf' if args.prior_feat else '') + ('Pt' if args.prior_teach else '') + (f'R{args.rings}' if args.rings else '') + (f'Rg{args.only_region}' if args.only_region is not None else '') + (f'H{args.halo}' if args.halo else '') + (f'P{args.patch}s{args.patch_side}' if args.patch else '') + (f'Gp{args.globpe}' if args.globpe else '') + (f'D{args.dim}' if args.dim else '') + (f'Lr{args.lr:g}'.replace('.','p').replace('-','m') if args.lr else '') + (f'B{args.batch}' if args.batch else '') + ('Cos' if args.cosine else '') + (f'Cj{args.cjit:g}'.replace('.','p') if args.cjit else '') + (f'L{args.layers}' if args.layers else '') + (({'oracle': 'Ro', 'pred': 'Rp'}.get(args.rc_mode, 'Rc')) if args.recenter else '') + (f'K{args.fold}' if args.fold is not None else '') + ('Mm' if args.mmgeo else '') + ('Tc' if args.tcomb else '') + (f'Ap{int(args.aperture_mm)}'.replace('-','C') if args.aperture_mm else '') + ('Ac' if args.allcells else '') + (('Rr' + ''.join(map(str, args.rc_regions))) if args.rc_regions else '') + ('Dep' if args.depth else '') + ('Orh' if args.orho else '') + ('Abs' if args.abst else '') + (f'W{args.wlow:g}'.replace('.', '') if args.wlow > 0 else '') +
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
    if args.only_region is not None:
        # filtered AFTER the cache load so the cache stays shared with every other run
        n0 = len(main_ev)
        main_ev = [e for e in main_ev if e['reg'] == args.only_region]
        print(f'region {args.only_region} only: {len(main_ev)} of {n0} events', flush=True)
        if not main_ev:
            raise SystemExit(f'no events in region {args.only_region}')
    syn_ev = []
    if args.gatesup > 0 or args.synaux:
        with open(args.overlay, 'rb') as f:
            syn_ev = pickle.load(f)
        print(f'overlay: {len(syn_ev)} synthetic events with per-cell truth', flush=True)
    aux_ev = (cached_grid(clean_files, 'clean-aux' + ('_smoke' if args.mode == 'smoke' else ''),
                          cdir, args.no_cache)
              if (args.cleanaux and args.sample == 'minbias') else None)
    if args.rc_mode == 'pred':
        if not args.rc_pred:
            raise SystemExit('--rc-mode pred needs --rc-pred (scripts/gen_pointer.py output)')
        with open(args.rc_pred, 'rb') as f:
            rp = pickle.load(f)
        for evs, key in ((main_ev, 'main'), (aux_ev or [], 'aux')):
            arr = rp[key]
            if len(arr) != len(evs):
                raise SystemExit(f'--rc-pred {key} rows {len(arr)} != events {len(evs)}: '
                                 'regenerate the pointer file against the current caches')
            for e, (px, py) in zip(evs, arr):
                e['px'], e['py'] = float(px), float(py)
        print(f'pointer offsets attached from {args.rc_pred}', flush=True)
    if syn_ev:
        aux_ev = (aux_ev or []) + syn_ev
    if args.only_region is not None and aux_ev:
        aux_ev = [e for e in aux_ev if e['reg'] == args.only_region]
    prior = None
    if args.prior_feat or args.prior_teach:
        with open(args.prior_path, 'rb') as f:
            prior = dict(model=pickle.load(f), feat=args.prior_feat, teach=args.prior_teach)
        print(f'cell prior: {args.prior_path} (feat={args.prior_feat} teach={args.prior_teach})',
              flush=True)
    D = prep(args.window, main_ev, aux_ev, prior=prior, rings=args.rings, halo=args.halo,
             globpe=args.globpe, patch=args.patch, patch_side=args.patch_side, recenter=args.recenter, fold=args.fold, nfold=args.nfold, mmgeo=args.mmgeo, rc_regions=(set(args.rc_regions) if args.rc_regions else None), rc_mode=args.rc_mode, tcomb=args.tcomb, allcells=args.allcells, aperture_mm=args.aperture_mm,
             ng=NG + (2 if args.occ else 0) + (8 if args.extra else 0)
             + (2 if args.rho else 0) + (3 if args.abst else 0)
             + (max(args.rings - args.window, 0)) + (4 * args.globpe),
             phys=args.phys, occ=args.occ, extra=args.extra, dens=args.dens, rho=args.rho,
             tpull=args.tpull, aux=args.aux, depth=args.depth, orho=args.orho, abst=args.abst)
    if args.frac and args.frac < 1.0:
        fr = np.random.default_rng(12345)
        for k in ('ktr', 'ctr'):
            idx = np.asarray(D[k])
            if len(idx):
                D[k] = np.sort(idx[fr.permutation(len(idx))[:int(round(args.frac * len(idx)))]])
        print(f'frac {args.frac}: train {len(D["ktr"])} main + {len(D["ctr"])} aux')
    if args.no_time:
        D['X'][:, :, 7:11] = 0.0
    if args.sum_core:
        S = D['S']
        gi, gj = np.divmod(np.clip(D['POS'], 0, S * S - 1), S)
        core = (np.maximum(np.abs(gi - args.window), np.abs(gj - args.window)) <= args.sum_core)
        D['Eraw'] = D['Eraw'] * (core & D['M'])
        print(f'sum-core {args.sum_core}: energy sum over '
              f'{(2*args.sum_core+1)**2} of {S*S} grid positions', flush=True)
    T = dict(X=torch.from_numpy(D['X']).to(device), M=torch.from_numpy(D['M']).to(device),
             G=torch.from_numpy(D['G']).to(device), Y=torch.from_numpy(D['y']).unsqueeze(1).to(device),
             E=torch.from_numpy(D['Eraw']).to(device),
             P=torch.from_numpy(D['POS']).to(device),
             SL=torch.from_numpy(D['SL']).to(device),
             YA=torch.from_numpy(D['YA']).to(device),
             FR=torch.from_numpy(D['FR']).to(device),
             HASF=torch.from_numpy(D['HASF']).to(device),
             TD=(torch.from_numpy(np.load(args.distill).astype(np.float32)).to(device)
                 if args.distill else torch.zeros(1)),
             W=torch.from_numpy(sample_weights(D, args.wlow)).to(device),
             mean=torch.from_numpy(np.asarray(D['mean'], np.float32)).to(device),
             std=torch.from_numpy(np.asarray(D['std'], np.float32)).to(device))
    if args.distill and T['TD'].shape[0] != D['X'].shape[0]:
        raise SystemExit(f'teacher rows {T["TD"].shape[0]} != events {D["X"].shape[0]}: the '
                         'teacher file was generated against a different prep configuration')
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
                        aux=args.aux, film=args.film, nfour=args.nfour, tfour=args.tfour,
                        slots=args.slots, sum_core=args.sum_core, seed=seed,
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
