import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import PITCH
from picocal_models import load_model
from probe_time import build_final_D

# Per-event predicted interquartile width of the aux ensemble, saved next to the prediction
# CSVs. The quantile head already prices its own uncertainty -- predicted width flags the
# unreliable events at AUC 0.93 -- but the width is a raw model output and was never persisted.
# This writes it for the sigma_eff-vs-efficiency curve: the cut threshold must come from the
# VALIDATION split, so both splits are saved.


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
    D = build_final_D(repo, args.smoke)
    X = torch.from_numpy(D['X']).to(device)
    M = torch.from_numpy(D['M']).to(device)
    G = torch.from_numpy(D['G']).to(device)
    E = torch.from_numpy(D['Eraw']).to(device)
    P = torch.from_numpy(D['POS']).to(device)
    SL = torch.from_numpy(D['SL']).to(device)
    idx = np.concatenate([np.asarray(D['kva']), np.asarray(D['kte'])])
    split = np.array(['val'] * len(D['kva']) + ['test'] * len(D['kte']))
    qs = []
    for s in args.seeds:
        model, st = load_model(repo / 'models' / f'{args.name}_s{s}.pt', device)
        out = []
        with torch.no_grad():
            for j in range(0, len(idx), 512):
                b = torch.from_numpy(idx[j:j + 512]).to(device)
                out.append(model(X[b], M[b], G[b], E[b], P[b], SL[b])[:, :3].cpu().numpy())
        qs.append(np.concatenate(out))
        print(f'seed {s} done', flush=True)
    qm = np.median(np.stack(qs), 0)
    df = pd.DataFrame(dict(
        split=split, true_energy=D['Et'][idx], region=D['reg'][idx],
        region_name=[f'{int(PITCH[r])}mm' for r in D['reg'][idx]], ET=D['ET'][idx],
        width=qm[:, 2] - qm[:, 0], q50=qm[:, 1]))
    outp = repo / 'reports' / 'predictions' / f'minbias__{args.name}_width.csv'
    df.to_csv(outp, index=False)
    print(f'-> {outp} ({len(df)} rows)')


if __name__ == '__main__':
    main()
