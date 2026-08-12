import argparse
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import resolution, PITCH
from picocal_data import prep
from picocal_models import NG, load_model, width_binned_calibration
from train_picocal import cached_grid, d4_apply

EXAMPLES = """\
example:
  uv run scripts/eval_tta.py models/SubNetW4CleanAuxExDnQdEma_s*.pt --extra --dens

Test-time D4 averaging costs nothing to try: it reuses trained weights and only changes
inference. Group averaging over the 8 dihedral transforms of the window reduces the variance
of the output, which is a different mechanism from train-time augmentation. The feature flags
must match those the checkpoints were trained with, since they determine the input layout.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Score saved checkpoints with and without test-time D4 averaging, '
                    'per region and per energy bin. CPU is fine.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--dens', action='store_true')
    ap.add_argument('--orho', action='store_true')
    ap.add_argument('--tpull', action='store_true')
    ap.add_argument('--phys', action='store_true')
    ap.add_argument('--occ', action='store_true')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    return ap.parse_args()


def main():
    a = parse_args()
    repo = Path(a.repo)
    cdir = repo / '.scratch' / 'cache'
    mb = cached_grid(sorted((repo / 'data' / 'minimum_bias').glob('*.root')), 'minbias', cdir)
    cl = cached_grid(sorted((repo / 'data' / 'full').glob('matched_*.root')), 'clean-aux', cdir)
    ng = NG + (2 if a.occ else 0) + (8 if a.extra else 0)
    D = prep(a.window, mb, cl, ng=ng, phys=a.phys, occ=a.occ, extra=a.extra, dens=a.dens,
             orho=a.orho, tpull=a.tpull)
    dev = a.device
    T = {k: torch.from_numpy(D[k]).to(dev) for k in ('X', 'M', 'G', 'Eraw', 'POS')}
    mean = torch.from_numpy(np.asarray(D['mean'], np.float32)).to(dev)
    std = torch.from_numpy(np.asarray(D['std'], np.float32)).to(dev)
    off = 16 + (2 if a.orho else 0) + (3 if a.tpull else 0)
    vec_idx = (off,) if a.extra else ()
    pidx = off + (2 if a.extra else 0) + (3 if a.dens else 0)

    def predict(model, idx, tta):
        out = []
        with torch.no_grad():
            for j in range(0, len(idx), 256):
                b = torch.from_numpy(np.asarray(idx[j:j + 256])).to(dev)
                acc = None
                for k in (range(8) if tta else (0,)):
                    xb = T['X'][b] if k == 0 else d4_apply(T['X'][b], T['M'][b], mean, std,
                                                           a.phys, pidx, k, vec_idx)
                    o = model(xb, T['M'][b], T['G'][b], T['Eraw'][b], T['POS'][b])
                    acc = o if acc is None else acc + o
                out.append((acc / (8 if tta else 1)).cpu().numpy())
        return np.concatenate(out)[:, :3]

    def deming(x, y, lam=1.0):
        # Errors-in-variables slope. Ordinary least squares attenuates it because q50 is itself
        # noisy, which compresses predictions toward the mean and appears as a positive bias at
        # low energy and a negative one at high energy -- exactly the pattern we measure.
        sxx, syy = float(np.var(x)), float(np.var(y))
        sxy = float(np.cov(x, y)[0, 1])
        if abs(sxy) < 1e-12:
            return 1.0, float(np.mean(y) - np.mean(x))
        a_ = ((syy - lam * sxx) + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
        return a_, float(np.mean(y) - a_ * np.mean(x))

    def binned(qv, qt, yva, key, n_groups, fit):
        kv, kt = key(qv), key(qt)
        cuts = np.quantile(kv, np.linspace(0, 1, n_groups + 1)[1:-1])
        gv, gt = np.digitize(kv, cuts), np.digitize(kt, cuts)
        pe = np.empty(len(qt))
        for g in range(n_groups):
            sv, stt = gv == g, gt == g
            aa, bb = fit(qv[sv, 1], yva[sv]) if (sv.sum() >= 30 and stt.sum()) \
                else fit(qv[:, 1], yva)
            pe[stt] = np.exp(aa * qt[stt, 1] + bb)
        return pe

    def ols(x, y):
        a_, b_ = np.polyfit(x, y, 1)
        return float(a_), float(b_)

    width = lambda q: q[:, 2] - q[:, 0]
    value = lambda q: q[:, 1]

    def joint(qv, qt, yva, nv=4, nw=3, rv=None, rt=None):
        # Calibrate inside cells of (optional region) x predicted-value x predicted-width.
        # The slope of log E_true against q50 varies along the response curve and differs by
        # region, and a single global fit per width bin cannot follow either.
        def cell(q, r, cuts_v, cuts_w):
            c = np.digitize(value(q), cuts_v) * nw + np.digitize(width(q), cuts_w)
            return c if r is None else c * 8 + r
        cuts_v = np.quantile(value(qv), np.linspace(0, 1, nv + 1)[1:-1])
        cuts_w = np.quantile(width(qv), np.linspace(0, 1, nw + 1)[1:-1])
        cv, ct = cell(qv, rv, cuts_v, cuts_w), cell(qt, rt, cuts_v, cuts_w)
        pe = np.empty(len(qt))
        ga, gb = ols(qv[:, 1], yva)
        for c in np.unique(ct):
            sv, stt = cv == c, ct == c
            aa, bb = ols(qv[sv, 1], yva[sv]) if sv.sum() >= 40 else (ga, gb)
            pe[stt] = np.exp(aa * qt[stt, 1] + bb)
        return pe

    rva, rte = D['reg'][D['kva']], D['reg'][D['kte']]
    variants = {
        'width-bin OLS (current)': lambda qv, qt, y: width_binned_calibration(qv, qt, y),
        'width-bin Deming': lambda qv, qt, y: binned(qv, qt, y, width, 3, deming),
        'joint q50 x width': lambda qv, qt, y: joint(qv, qt, y),
        'region x q50 x width': lambda qv, qt, y: joint(qv, qt, y, rv=rva, rt=rte),
    }

    kva, kte = D['kva'], D['kte']
    yva, Ete, reg = D['y'][kva], D['Et'][kte], D['reg'][kte]
    Q = []
    for c in a.ckpts:
        model, _ = load_model(c, dev)
        Q.append((predict(model, kva, False), predict(model, kte, False)))
    Qt = [(predict(load_model(c, dev)[0], kva, True), predict(load_model(c, dev)[0], kte, True))
          for c in a.ckpts] if False else None

    res = {}
    for tag, fn in variants.items():
        res[tag] = np.mean([fn(qv, qt, yva) for qv, qt in Q], 0)
        r = resolution(res[tag], Ete)
        print(f'{tag:26s} aggregate sigma_eff {r["sigma_eff"]:.4f}   bias {r["bias"]:+.4f}',
              flush=True)

    keys = list(variants)
    print('\n' + f"{'region':>7s} {'bin':>5s} {'n':>6s} "
          + ' '.join(f'{k[:15]:>16s}' for k in keys) + '   (sigma_eff / bias)')
    for r, name in enumerate([f'{int(p)}mm' for p in PITCH]):
        sel = reg == r
        if sel.sum() < 200:
            continue
        t = Ete[sel]
        q1, q2 = np.quantile(t, [1 / 3, 2 / 3])
        for lab, m in (('all', np.ones(sel.sum(), bool)), ('low', t < q1),
                       ('mid', (t >= q1) & (t < q2)), ('high', t >= q2)):
            vals = [resolution(res[k][sel][m], t[m]) for k in keys]
            print(f'{name:>7s} {lab:>5s} {int(m.sum()):6d} '
                  + ' '.join(f'{v["sigma_eff"]:8.4f}/{v["bias"]:+6.3f}' for v in vals))


if __name__ == '__main__':
    main()
