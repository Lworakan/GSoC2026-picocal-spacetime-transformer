import argparse
import platform
import time
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from picocal_data import build_grid, prep
from picocal_models import NG, load_model

EXAMPLES = """\
examples:
  # best model, seed 0, default timing set (4 minbias files)
  uv run scripts/benchmark_inference.py

  # compare the best checkpoints, larger timing set
  uv run scripts/benchmark_inference.py \\
      models/SubNetW4CleanAuxQdEma_s0.pt \\
      models/SubNetW4CleanAuxQuant_s0.pt \\
      models/CleanHuberW4_s0.pt --files 8

  # sweep batch sizes and save the table
  uv run scripts/benchmark_inference.py --batch-sizes 32 128 512 2048 \\
      --out reports/benchmark_inference.csv

metric: clusters/second (== events/second in these samples: 1 cluster per event).
Per model the table also derives the 5-seed ensemble rate (/5) and the
5-seed + D4 test-time-augmentation rate (/40) used by the full stack.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Measure inference throughput (clusters per second) of saved PicoCal '
                    'models on CPU or GPU: loads a timing set from ROOT files, runs '
                    'warmup passes, then reports the median rate over repeated passes.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='*', default=['models/SubNetW4CleanAuxQdEma_s0.pt'],
                    help='checkpoint paths under models/ (default: best model, seed 0)')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]),
                    help='repository root (default: parent of scripts/)')
    ap.add_argument('--sample', choices=['minbias', 'clean'], default='minbias',
                    help='which sample to draw the timing clusters from (default: minbias)')
    ap.add_argument('--files', type=int, default=4,
                    help='number of ROOT files to load for the timing set (default: 4)')
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[64, 256, 1024],
                    help='batch sizes to sweep (default: 64 256 1024)')
    ap.add_argument('--repeats', type=int, default=5,
                    help='timed passes over the full set; median is reported (default: 5)')
    ap.add_argument('--warmup', type=int, default=2,
                    help='untimed warmup passes before measuring (default: 2)')
    ap.add_argument('--threads', type=int, default=None,
                    help='torch CPU thread count (default: torch default)')
    ap.add_argument('--device', default='cpu',
                    help="inference device (default: cpu; e.g. cuda, cuda:0)")
    ap.add_argument('--out', default=None,
                    help='optional CSV path for the results table')
    return ap.parse_args()


def cpu_name():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def timed_pass(model, T, bs, device):
    n = T['X'].shape[0]
    with torch.inference_mode():
        t0 = time.perf_counter()
        for j in range(0, n, bs):
            sl = slice(j, j + bs)
            model(T['X'][sl], T['M'][sl], T['G'][sl], T['E'][sl])
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        return time.perf_counter() - t0


def main():
    args = parse_args()
    repo = Path(args.repo)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = args.device

    ckpts = []
    for p in args.models:
        path = Path(p) if Path(p).exists() else repo / p
        if not path.exists():
            raise SystemExit(f'checkpoint not found: {p}')
        model, st = load_model(path, device)
        ckpts.append((path.stem, model, st))
    windows = {st.get('window', 4) for _, _, st in ckpts}
    if len(windows) != 1:
        raise SystemExit(f'mixed window sizes across checkpoints: {sorted(windows)}')
    window = windows.pop()

    sub = 'minimum_bias' if args.sample == 'minbias' else 'full'
    pat = '*.root' if args.sample == 'minbias' else 'matched_*.root'
    files = sorted((repo / 'data' / sub).glob(pat))[:args.files]
    if not files:
        raise SystemExit(f'no ROOT files under data/{sub}')
    ev = build_grid(files, args.sample)
    D = prep(window, ev, None, ng=NG)
    T = dict(X=torch.from_numpy(D['X']).to(device), M=torch.from_numpy(D['M']).to(device),
             G=torch.from_numpy(D['G']).to(device), E=torch.from_numpy(D['Eraw']).to(device))
    n = T['X'].shape[0]

    print(f'device {device} | torch {torch.__version__} | threads {torch.get_num_threads()}')
    print(f'cpu {cpu_name()}')
    print(f'timing set: {n} clusters from {len(files)} {args.sample} files | window {window} '
          f'({(2 * window + 1)}x{(2 * window + 1)} cells)\n')

    header = (f"{'model':34s} {'params':>8s} {'batch':>6s} {'clusters/s':>11s} "
              f"{'ms/cluster':>10s} {'5-seed/s':>9s} {'5s+TTA/s':>9s}")
    print(header)
    rows = []
    for name, model, _ in ckpts:
        nparams = sum(p.numel() for p in model.parameters())
        for bs in args.batch_sizes:
            for _ in range(args.warmup):
                timed_pass(model, T, bs, device)
            times = [timed_pass(model, T, bs, device) for _ in range(args.repeats)]
            cps = n / float(np.median(times))
            rows.append(dict(model=name, params=nparams, batch=bs, clusters_per_s=cps,
                             ms_per_cluster=1e3 / cps, ensemble5_per_s=cps / 5,
                             ensemble5_tta_per_s=cps / 40,
                             device=device, threads=torch.get_num_threads(),
                             n_clusters=n, repeats=args.repeats))
            print(f'{name:34s} {nparams:8d} {bs:6d} {cps:11.0f} {1e3 / cps:10.3f} '
                  f'{cps / 5:9.0f} {cps / 40:9.0f}', flush=True)

    if args.out:
        import pandas as pd
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f'\nwrote {outp}')


if __name__ == '__main__':
    main()
