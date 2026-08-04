import argparse
import copy
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
from picocal_models import (CFG, NG, SubNetFQ, QUANTILES, pinball_loss, qd_pinball_loss,
                            width_binned_calibration, linear_calibration, save_model)


def parse_args():
    ap = argparse.ArgumentParser(description='Train the PicoCal SubNet family from a configuration')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--sample', choices=['minbias', 'clean'], default='minbias')
    ap.add_argument('--cleanaux', action='store_true')
    ap.add_argument('--objective', choices=['qd', 'quant', 'huber'], default='qd')
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--gate', choices=['learned', 'off'], default='learned')
    ap.add_argument('--no-time', action='store_true')
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


def train_one(T, D, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SubNetFQ(D['IN_DIM'], D['la0'], D['lb0'], gate=args.gate).to(device)
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
        return model(T['X'][b], T['M'][b], T['G'][b], T['E'][b])

    def loss_fn(q, yb):
        if args.objective == 'qd':
            return qd_pinball_loss(q, yb, qs)
        if args.objective == 'quant':
            return pinball_loss(q, yb, qs)
        return nn.functional.huber_loss(q[:, 1:2], yb, delta=CFG['huber_delta'])

    def eval_model():
        return ema.module if ema is not None else model

    def run(idx):
        m = eval_model()
        m.eval()
        out = []
        with torch.no_grad():
            for b in batches(idx, 256, False):
                out.append(m(T['X'][b], T['M'][b], T['G'][b], T['E'][b]).cpu().numpy())
        return np.concatenate(out)

    def vloss():
        m = eval_model()
        m.eval()
        s, k = 0.0, 0
        with torch.no_grad():
            for b in batches(D['kva'], 256, False):
                q = m(T['X'][b], T['M'][b], T['G'][b], T['E'][b])
                s += pinball_loss(q, T['Y'][b], qs).item()
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
            loss_fn(fwd(b), T['Y'][b]).backward()
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
    final = SubNetFQ(D['IN_DIM'], D['la0'], D['lb0'], gate=args.gate).to(device)
    final.load_state_dict(bstate)
    final.eval()

    def run_final(idx):
        out = []
        with torch.no_grad():
            for b in batches(idx, 256, False):
                out.append(final(T['X'][b], T['M'][b], T['G'][b], T['E'][b]).cpu().numpy())
        return np.concatenate(out)

    qv, qt = run_final(D['kva']), run_final(D['kte'])
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
        args.name = base + ('CleanAux' if args.cleanaux else '') + suffix
    if args.mode == 'smoke':
        args.name += '_smoke'
        args.epochs, args.patience = 2, 99

    mb_files = sorted((repo / 'data' / 'minimum_bias').glob('*.root'))
    clean_files = sorted((repo / 'data' / 'full').glob('matched_*.root'))
    if args.mode == 'smoke':
        mb_files, clean_files = mb_files[:4], clean_files[:2]
    main_files = mb_files if args.sample == 'minbias' else clean_files
    main_ev = build_grid(main_files, args.sample)
    aux_ev = build_grid(clean_files, 'clean-aux') if (args.cleanaux and args.sample == 'minbias') else None
    D = prep(args.window, main_ev, aux_ev, ng=NG)
    if args.no_time:
        D['X'][:, :, 7:11] = 0.0
    T = dict(X=torch.from_numpy(D['X']).to(device), M=torch.from_numpy(D['M']).to(device),
             G=torch.from_numpy(D['G']).to(device), Y=torch.from_numpy(D['y']).unsqueeze(1).to(device),
             E=torch.from_numpy(D['Eraw']).to(device))
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
                   dict(in_dim=D['IN_DIM'], la0=D['la0'], lb0=D['lb0'], ng=NG,
                        mean=D['mean'], std=D['std'], cfg=CFG, window=args.window,
                        objective=args.objective, sample=args.sample, cleanaux=args.cleanaux, seed=seed))
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
