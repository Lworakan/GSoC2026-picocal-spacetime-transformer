import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from picocal_models import load_model
from probe_time import build_final_D

# Writes the (N, 3) seed-averaged raw quantile outputs of the five headline members over every
# event, in prep() order, for --distill. The average is taken in the model's own output space,
# BEFORE calibration: each seed's calibration map differs, and averaging calibrated energies
# would fold five different maps into one target.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--name', default='SubNetW8CleanAuxExDnRcQdEma')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default='.scratch/teacher_w8rc.npy')
    args = ap.parse_args()
    repo = Path(args.repo)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    D = build_final_D(repo, args.smoke)
    X = torch.from_numpy(D['X']).to(device)
    M = torch.from_numpy(D['M']).to(device)
    G = torch.from_numpy(D['G']).to(device)
    E = torch.from_numpy(D['Eraw']).to(device)
    P = torch.from_numpy(D['POS']).to(device)
    SL = torch.from_numpy(D['SL']).to(device)
    N = len(D['X'])
    acc = np.zeros((N, 3), np.float64)
    for s in args.seeds:
        model, st = load_model(repo / 'models' / f'{args.name}_s{s}.pt', device)
        if st['in_dim'] != D['IN_DIM']:
            raise SystemExit(f"seed {s}: in_dim {st['in_dim']} != data {D['IN_DIM']}")
        out = []
        with torch.no_grad():
            for j in range(0, N, 512):
                b = torch.arange(j, min(j + 512, N), device=device)
                out.append(model(X[b], M[b], G[b], E[b], P[b], SL[b])[:, :3].cpu().numpy())
        acc += np.concatenate(out)
        print(f'seed {s} done', flush=True)
    outp = repo / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, (acc / len(args.seeds)).astype(np.float32))
    print(f'-> {outp} ({N} events, {len(args.seeds)} seeds)')


if __name__ == '__main__':
    main()
