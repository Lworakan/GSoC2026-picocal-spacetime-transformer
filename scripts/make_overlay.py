import argparse
import pickle
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import PITCH

EXAMPLES = """\
example:
  uv run scripts/make_overlay.py --regions 0 1 --per-event 3 --out .scratch/cache/overlay.pkl

Builds a paired sample without asking anyone for new simulation.

The cells OUTSIDE the 9x9 window of a min-bias cluster are almost pure pileup: in clean data
the photon is 99% contained inside the window, so whatever lies outside is background. Their
per-cell density matches the pileup density measured INSIDE the window to within 1% in the
15mm and 30mm regions (867 vs 880 MeV/cell and 824 vs 824), which is what makes the transplant
legitimate. Patches are moved as blocks so the spatial correlation of pileup survives -- ring
to ring correlations of 0.45-0.63 are a real feature of the noise and scattering cells
independently would produce a much easier problem than the real one.

Each output event carries the clean photon cells, the transplanted pileup, and therefore the
per-cell truth of which energy belongs to the photon -- the quantity the real samples cannot
give us, and the one that unlocks denoising targets, per-cell supervision and a forward model
for simulation-based inference.
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description='Transplant real pileup from outside-window cells onto clean photon '
                    'clusters, producing paired events with per-cell truth.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--regions', type=int, nargs='+', default=[0, 1],
                    help='region indices to build (default: 0 1, the two where the density '
                         'check passes; 60mm and 120mm have almost no outside cells)')
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--patch-inner-mm', type=float, default=105.0,
                    help='inner radius, in MILLIMETRES, of the annulus pileup patches are cut '
                         'from. The annulus has to start beyond the photon so the patches are '
                         'photon-free, and containment is set by the Moliere radius (~35 mm), not '
                         'by a cell count: at a fixed inner radius of W=4 cells the annulus is '
                         '60 mm out at 15 mm pitch but 240 mm out at 60 mm, where a cluster has '
                         'almost no cells left -- which is why the library held 20,081 patches at '
                         '30 mm and 173 at 60 mm, with none at 120 mm. Three Moliere radii is '
                         'photon-free everywhere and leaves cells to cut in the coarse regions.')
    ap.add_argument('--per-event', type=int, default=3,
                    help='pileup patches transplanted per clean event (default: 3)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='.scratch/cache/overlay.pkl')
    ap.add_argument('--report-only', action='store_true',
                    help='validate the synthetic statistics against real min-bias and stop')
    return ap.parse_args()


def load(repo, tag, n):
    p = Path(repo) / '.scratch' / 'cache' / f'{tag}_{n}.pkl'
    if not p.exists():
        raise SystemExit(f'missing {p}; run any training once to build the event cache')
    with open(p, 'rb') as f:
        return pickle.load(f)


def patches(mb_events, regions, W, rng, inner_mm=105.0):
    """Pileup blocks lifted from outside the window, keyed by region."""
    lib = {r: [] for r in regions}
    for ev in mb_events:
        r = ev['reg']
        if r not in lib:
            continue
        d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
        inner = max(2, int(np.ceil(inner_mm / float(ev['ps']))))
        out = d > inner
        if out.sum() < 4:
            continue
        di, dj = ev['di'][out].astype(int), ev['dj'][out].astype(int)
        e, fr, bk = ev['e'][out], ev['fr'][out], ev['bk'][out]
        # Times are stored RELATIVE to the donor event's own reference. Copying the donor's
        # absolute timestamps put a random offset -- the difference between two unrelated events'
        # t0 -- between the photon cells and the pileup cells, so --gatesup was teaching the gate
        # that "far from the photon in time" means pileup using examples where that distance was
        # an artefact of pairing. Real pileup is in the same bunch crossing: the offset is zero on
        # average, and what separates it is the SPREAD, which relative times preserve.
        tref = np.nanmedian(ev['tf']) if np.isfinite(ev['tf']).any() else 0.0
        tf, tb = ev['tf'][out] - tref, ev['tb'][out] - tref
        # translate a block of the annulus into the window, keeping relative geometry
        for _ in range(2):
            oi = int(rng.integers(-2 * W - 1, 2 * W + 2))
            oj = int(rng.integers(-2 * W - 1, 2 * W + 2))
            ni, nj = di + oi, dj + oj
            keep = (np.abs(ni) <= W) & (np.abs(nj) <= W)
            if keep.sum() >= 2:
                lib[r].append((ni[keep], nj[keep], e[keep], fr[keep], bk[keep],
                               tf[keep], tb[keep]))
    return lib


def stats(events, W):
    """Statistics the real min-bias sample is measured by, so the synthetic can be checked."""
    seedf, phot, ncell = [], [], []
    for ev in events:
        d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
        m = (d <= W) & (ev['e'] >= 2.49)
        if m.sum() < 5:
            continue
        e = ev['e'][m]
        seedf.append(e.max() / max(e.sum(), 1e-6))
        ncell.append(int(m.sum()))
        # denominator must be the WINDOW energy for both samples: the real events' 'tot' is
        # the whole cluster, which the 9x9 window only partly covers, so comparing against it
        # would make the synthetic look twice as clean as it is.
        phot.append(ev['Etrue'] * 1000.0 / max(float(e.sum()), 1e-6))
    return (float(np.median(seedf)), float(np.median(phot)), float(np.median(ncell)))


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    W = a.window
    mb = load(a.repo, 'minbias', 94)
    cl = load(a.repo, 'clean-aux', 100)
    lib = patches(mb, set(a.regions), W, rng, a.patch_inner_mm)
    print('pileup patches per region: '
          + ', '.join(f'{int(PITCH[r])}mm={len(v)}' for r, v in lib.items()), flush=True)

    synth = []
    for ev in cl:
        r = ev['reg']
        if r not in lib or not lib[r]:
            continue
        d = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
        base = d <= W
        di, dj = ev['di'][base].astype(int), ev['dj'][base].astype(int)
        e, fr, bk = ev['e'][base].copy(), ev['fr'][base].copy(), ev['bk'][base].copy()
        tf, tb = ev['tf'][base].copy(), ev['tb'][base].copy()
        pc = ev['pc'][base].copy()
        sig = e.copy()                                   # per-cell photon truth
        grid = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(di, dj))}
        t0 = float(np.nanmedian(tf)) if np.isfinite(tf).any() else 0.0
        for _ in range(a.per_event):
            pi, pj, pe, pfr, pbk, ptf, ptb = lib[r][int(rng.integers(len(lib[r])))]
            ptf, ptb = ptf + t0, ptb + t0        # donor times onto the recipient's clock
            for k in range(len(pi)):
                key = (int(pi[k]), int(pj[k]))
                if key in grid:
                    q = grid[key]
                    e[q] += pe[k]; fr[q] += pfr[k]; bk[q] += pbk[k]
                    if not np.isfinite(tf[q]):
                        tf[q], tb[q] = ptf[k], ptb[k]
                else:
                    grid[key] = len(e)
                    di = np.append(di, pi[k]); dj = np.append(dj, pj[k])
                    e = np.append(e, pe[k]); fr = np.append(fr, pfr[k]); bk = np.append(bk, pbk[k])
                    tf = np.append(tf, ptf[k]); tb = np.append(tb, ptb[k])
                    pc = np.append(pc, ev['ps']); sig = np.append(sig, 0.0)
        n = dict(ev)
        n.update(di=di.astype(np.int16), dj=dj.astype(np.int16),
                 e=e.astype(np.float32), fr=fr.astype(np.float32), bk=bk.astype(np.float32),
                 tf=tf.astype(np.float32), tb=tb.astype(np.float32), pc=pc.astype(np.float32),
                 sig=sig.astype(np.float32),                       # per-cell photon energy
                 tot_syn=float(e.sum()), ncl=float(len(e)),
                 x=np.resize(ev['x'][base], len(e)).astype(np.float32),
                 y=np.resize(ev['y'][base], len(e)).astype(np.float32))
        synth.append(n)

    reg_sel = [e for e in mb if e['reg'] in set(a.regions)]
    cl_sel = [e for e in cl if e['reg'] in set(a.regions)]
    print(f'\nsynthetic events: {len(synth)}')
    print(f"{'sample':>12s} {'seed frac':>10s} {'photon/cluster':>15s} {'cells in window':>16s}")
    for tag, evs in (('clean', cl_sel), ('synthetic', synth), ('real min-bias', reg_sel)):
        s, p, c = stats(evs, W)
        print(f'{tag:>12s} {s:10.3f} {p:15.3f} {c:16.0f}')
    print('\nthe synthetic row should sit close to real min-bias, not to clean')

    if not a.report_only:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(synth, f, protocol=4)
        print(f'\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
