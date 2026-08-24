import sys
from pathlib import Path
import numpy as np
import uproot
import awkward as ak

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import split, PITCH, EPS

TK = ['cell_x', 'cell_y', 'energy', 'cell_energies_front', 'cell_energies_back',
      'cell_times_front', 'cell_times_back', 'imodx', 'jmody']
AUX = ['sig_flux_prod_vertex_z', 'sig_flux_eTot', 'sig_flux_px', 'sig_flux_py', 'sig_flux_pz']
EXTRA = ['total_energy', 'total_energy_front', 'total_energy_back', 'x_cluster', 'y_cluster',
         'nenergy']
# truth-only, used as auxiliary regression TARGETS and never as inputs
AUXT = ['sig_flux_entry_x', 'sig_flux_entry_y', 'sig_flux_timing']
THRESH = 2.49
NC = 9
SIGT_E = np.array([5.0, 20.0, 55.0, 175.0, 550.0, 1700.0, 5000.0])
SIGT = np.array([0.756, 0.593, 0.435, 0.263, 0.156, 0.101, 0.038])


def sigma_t(e):
    return np.interp(np.log(np.clip(e, 1.0, None)), np.log(SIGT_E), SIGT)


def ref_time(t, e):
    ok = np.isfinite(t)
    if not ok.any():
        return 0.0
    w = e[ok] / np.clip(sigma_t(e[ok]), 1e-3, None) ** 2
    cut = np.quantile(e[ok], 0.9) if ok.sum() >= 5 else 0.0
    hi = e[ok] >= cut
    if hi.sum() >= 1:
        return float(np.average(t[ok][hi], weights=w[hi]))
    return float(np.average(t[ok], weights=w))


def event_geom(cc):
    x, yy, e = cc['cell_x'], cc['cell_y'], cc['energy']
    ix, iy = cc['imodx'], cc['jmody']
    seed = int(np.argmax(e))
    pts = np.stack([x, yy], 1)
    pitch = np.full(len(x), np.nan)
    for key in {(int(p), int(q)) for p, q in zip(ix, iy)}:
        sel = (ix == key[0]) & (iy == key[1])
        p = pts[sel]
        if len(p) >= 2:
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            d[d == 0] = np.inf
            pitch[sel] = np.median(np.min(d, axis=1))
    fill = np.nanmedian(pitch) if np.isfinite(pitch).any() else 120.0
    pitch[~np.isfinite(pitch)] = fill
    ps = pitch[seed]
    ei = (x - x[seed]) / ps
    ej = (yy - yy[seed]) / ps
    di = np.round(ei).astype(int)
    dj = np.round(ej).astype(int)
    ok = (np.abs(ei - di) < 0.15) & (np.abs(ej - dj) < 0.15)
    return seed, ps, di, dj, ok, pitch


def build_grid(files, label='', vertex_max=100.0, emin=1.0, emax=100.0):
    EV = []
    for path in files:
        with uproot.open(path) as f:
            a = f['clusters_matched'].arrays(TK + AUX + EXTRA + AUXT, library='ak')
        ex = {k: ak.to_numpy(a[k]).astype(float) for k in EXTRA + AUXT}
        vz = ak.to_numpy(a['sig_flux_prod_vertex_z']).astype(float)
        et_all = ak.to_numpy(a['sig_flux_eTot']).astype(float)
        px = ak.to_numpy(a['sig_flux_px']).astype(float)
        py = ak.to_numpy(a['sig_flux_py']).astype(float)
        pz = ak.to_numpy(a['sig_flux_pz']).astype(float)
        pt = np.hypot(px, py)
        p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        etv = et_all * pt / np.maximum(p, EPS)
        for i in np.flatnonzero((vz < vertex_max) & (et_all >= emin) & (et_all <= emax)):
            cc = {k: np.asarray(ak.to_numpy(a[k][i])).astype(float) for k in TK}
            e = cc['energy']
            if len(e) < 3:
                continue
            seed, ps, di, dj, ok, pitch = event_geom(cc)
            if ok.mean() < 0.5 or not ok[seed]:
                continue
            tf = cc['cell_times_front']
            tb = cc['cell_times_back']
            tf = np.where(np.isfinite(tf) & (tf != 0) & (np.abs(tf) < 1e4), tf, np.nan)
            tb = np.where(np.isfinite(tb) & (tb != 0) & (np.abs(tb) < 1e4), tb, np.nan)
            EV.append(dict(di=di[ok].astype(np.int16), dj=dj[ok].astype(np.int16),
                           tf_abs=tf[ok].astype(np.float32),
                           x=cc['cell_x'][ok].astype(np.float32), y=cc['cell_y'][ok].astype(np.float32),
                           e=e[ok].astype(np.float32),
                           fr=cc['cell_energies_front'][ok].astype(np.float32),
                           bk=cc['cell_energies_back'][ok].astype(np.float32),
                           tf=tf[ok].astype(np.float32), tb=tb[ok].astype(np.float32),
                           pc=pitch[ok].astype(np.float32),
                           ps=float(ps), reg=int(np.argmin(np.abs(PITCH - ps))),
                           Etrue=float(et_all[i]), ET=float(etv[i]),
                           tot=float(ex['total_energy'][i]),
                           totf=float(ex['total_energy_front'][i]),
                           totb=float(ex['total_energy_back'][i]),
                           xc=float(ex['x_cluster'][i]), yc=float(ex['y_cluster'][i]),
                           xs=float(cc['cell_x'][seed]), ys=float(cc['cell_y'][seed]),
                           ncl=float(ex['nenergy'][i]),
                           ax=float(ex['sig_flux_entry_x'][i]),
                           ay=float(ex['sig_flux_entry_y'][i]),
                           at=float(ex['sig_flux_timing'][i])))
    if label:
        print(f'{label}: {len(EV)} events')
    return EV


def cell_prior_features(e, fr, bk, tf, d):
    """Per-cell observables a photon-fraction estimator can use. No truth, so it is computable
    at inference. Kept in one place because the fitter and the pipeline MUST see identical
    columns -- a silent reordering here would look like a domain gap rather than a bug.

    Deliberately CELL-level only. Two event-level columns (log window energy, cell count) were
    included at first and measured with scripts/domain_gap.py: they carried almost no extra
    predictive power (corr 0.9324 -> 0.9310 at 15mm) while pushing the synthetic-versus-real
    separability from AUC 0.714 to 0.931. They encode how crowded the event is, which the overlay
    gets wrong by construction -- 77 cells against 65 in real data -- so the estimator was partly
    learning "this is a synthetic event", which means nothing at inference. The network already
    receives window totals through its global features, so nothing is lost by withholding them
    here."""
    core = d <= 1
    fin = np.isfinite(tf)
    cf = core & fin
    tref = (float((e[cf] * tf[cf]).sum() / max(float(e[cf].sum()), 1e-9)) if cf.any()
            else (float(np.nanmedian(tf[fin])) if fin.any() else 0.0))
    pull = np.where(fin, np.abs(tf - tref) / np.clip(sigma_t(e), 1e-3, None), 8.0)
    tot = max(float(e.sum()), 1e-9)
    return np.stack([np.log1p(np.clip(e, 0, None)), pull, np.minimum(pull, 4.0),
                     d.astype(np.float32),
                     (fr - bk) / np.clip(fr + bk, 1e-6, None),
                     e / tot], 1).astype(np.float32)


def halo_tokens(ev, W, halo):
    """One pseudo-cell per ring beyond the window, carrying that ring's summed energy and its
    energy-weighted timing.

    Motivated by the measurement, not by architecture fashion: the 15mm cluster reaches ring 15
    and a 9x9 window sees 37.6% of its energy, but the far rings are diffuse pileup rather than
    structure -- energy per ring varies slowly and there is little per-cell information out there.
    Giving every one of those ~600 cells its own token costs O(n^2) attention on the least
    informative part of the event; W15 is 961 tokens against 81. Collapsing each ring to a single
    token keeps the containment information, and the timing summary keeps the one handle that
    separates pileup from photon (pileup time rms 13.2 ns against 1.6 ns at 15mm), at 92 tokens.
    """
    dall = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
    above = ev['e'] >= THRESH
    out = []
    for r in range(W + 1, halo + 1):
        s = above & (dall == r)
        if not s.any():
            continue
        e, fr, bk = ev['e'][s], ev['fr'][s], ev['bk'][s]
        w = np.clip(e, 1e-6, None)
        def wmean(t):
            f = np.isfinite(t)
            return float(np.average(t[f], weights=w[f])) if f.any() else np.nan
        out.append((r, float(e.sum()), float(fr.sum()), float(bk.sum()),
                    wmean(ev['tf'][s]), wmean(ev['tb'][s]),
                    float(np.median(ev['pc'][s])), float(np.median(ev['x'][s])),
                    float(np.median(ev['y'][s]))))
    return out


def patch_tokens(ev, W, reach, side):
    """One token per SIDE x SIDE block of cells outside the window, out to `reach`.

    This is the patch-hierarchical construction of arXiv:2605.21789 (2026) adapted to our grid:
    fine detail where structure lives, pooled summaries outside, attention over the union. It
    replaces the ring pooling in halo_tokens for a measured reason -- ring sums collapse everything
    at one radius into a single number and so delete the angular structure, and the ring arms showed
    the model does use per-cell detail out there (R15 at W4 gives -0.0037 where a real window gives
    -0.0608). A 2D block keeps the angular position that a ring throws away.
    """
    dall = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
    above = (ev['e'] >= THRESH) & (dall > W) & (dall <= reach)
    if not above.any():
        return []
    bi = np.floor_divide(ev['di'][above].astype(int) + reach, side)
    bj = np.floor_divide(ev['dj'][above].astype(int) + reach, side)
    e, fr, bk = ev['e'][above], ev['fr'][above], ev['bk'][above]
    tf, tb, pc = ev['tf'][above], ev['tb'][above], ev['pc'][above]
    x, y = ev['x'][above], ev['y'][above]
    out = []
    for key in {(int(a), int(b)) for a, b in zip(bi, bj)}:
        s = (bi == key[0]) & (bj == key[1])
        w = np.clip(e[s], 1e-6, None)

        def wmean(t):
            f = np.isfinite(t[s])
            return float(np.average(t[s][f], weights=w[f])) if f.any() else np.nan
        # the token sits at the energy-weighted centre of its own block, so its di/dj stay
        # geometrically meaningful rather than snapping to a block corner
        out.append((float(np.average(ev['di'][above][s], weights=w)),
                    float(np.average(ev['dj'][above][s], weights=w)),
                    float(e[s].sum()), float(fr[s].sum()), float(bk[s].sum()),
                    wmean(tf), wmean(tb), float(np.median(pc[s])),
                    float(np.average(x[s], weights=w)), float(np.average(y[s], weights=w))))
    return out


def make_windows(W, EVS, phys=False, extra=False, dens=False, rho=False, tpull=False,
                 depth=False, orho=False, abst=False, prior=None, rings=0, halo=0,
                 patch=0, patch_side=3, recenter=False, mmgeo=False, rc_regions=None, rc_mode='centroid',
                 tcomb=False, allcells=False, aperture_mm=0.0):
    rows = []
    keep = []
    for i, ev in enumerate(EVS):
        ci = cj = 0
        if recenter and (rc_regions is None or ev['reg'] in rc_regions):
            # The window has always been centred on the loudest cell (event_geom: seed = argmax(e)).
            # Measured 2026-08-17 against the truth entry point: at 15mm the photon lands more than
            # two cells from that seed in 17.6% of events, with a 90th percentile of 8.28 cells,
            # because a low-energy photon loses the loudness contest to a pileup cell exactly where
            # occupancy is highest. 120mm, where occupancy is low, shows 0.1%.
            # Recentring on the reconstruction's own cluster centroid needs no truth and no cache
            # rebuild, since x_cluster/y_cluster and the seed position are both already stored.
            if rc_mode == 'oracle':
                # TRUE photon entry as the centre. Diagnostic ONLY -- truth is not available at
                # inference -- but it measures the ceiling of ANY centre estimator in one arm,
                # and it discriminates the two theories of the recentring gain: if the photon-
                # centred window beats the barycentre-centred one, a learned position regressor
                # (the ax/ay aux targets already exist) is worth building; if not, coverage is
                # confirmed as the mechanism and the centre question is closed.
                ax, ay = ev.get('ax', np.nan), ev.get('ay', np.nan)
                if np.isfinite(ax) and np.isfinite(ay):
                    ci = int(np.rint((ax - ev.get('xs', 0.0)) / ev['ps']))
                    cj = int(np.rint((ay - ev.get('ys', 0.0)) / ev['ps']))
                else:
                    ci = cj = 0
            elif rc_mode == 'pred':
                # Learned pointer: the aux head's predicted photon offset FROM THE SEED, in
                # cells (the same frame ya is defined in), attached to the event by the caller.
                # Median pointer error is 0.13-0.20 cells against the seed's 0.43 and the
                # centroid's 3.08 at 15mm, and unlike the seed it has no wrong-photon tail.
                px, py = ev.get('px', 0.0), ev.get('py', 0.0)
                ci = int(np.rint(px)) if np.isfinite(px) else 0
                cj = int(np.rint(py)) if np.isfinite(py) else 0
            else:
                ci = int(np.rint((ev.get('xc', 0.0) - ev.get('xs', 0.0)) / ev['ps']))
                cj = int(np.rint((ev.get('yc', 0.0) - ev.get('ys', 0.0)) / ev['ps']))
            ci, cj = int(np.clip(ci, -W, W)), int(np.clip(cj, -W, W))
        if allcells:
            # No square crop at all: every cell of the cluster is a token. The window is a square
            # crop of a roughly circular cluster, so even W8 discards corners and tail; and the
            # padded grid costs MORE than the real cells (W8 pads to 289 slots where the 15mm
            # cluster averages ~270 real cells). This is the reconstruction-level version of the
            # window ladder: the model sees exactly what the upstream clustering saw.
            m = ev['e'] >= THRESH
        else:
            m = (np.maximum(np.abs(ev['di'] - ci), np.abs(ev['dj'] - cj)) <= W) & (ev['e'] >= THRESH)
            if aperture_mm > 0:
                # The square crop is a geometric accident, not a physical one. A shower has a
                # fixed width in millimetres (the Moliere radius is ~35 mm everywhere) while the
                # window is counted in cells, so a half-width of W spans W*pitch mm and that
                # varies eightfold across the regions. Cutting on a physical radius instead makes
                # the aperture the same size in every region AND drops the square's corners,
                # which reach W*sqrt(2) cells -- for the photon those are empty, since 99.8% of
                # it sits inside 4 cells of a correctly centred window.
                r_mm = np.hypot(ev['di'] - ci, ev['dj'] - cj) * ev['ps']
                m &= r_mm <= aperture_mm
        if m.sum() < 1:
            continue
        di, dj, e, fr, bk, tf, tb = (v[m] for v in (ev['di'], ev['dj'], ev['e'], ev['fr'], ev['bk'], ev['tf'], ev['tb']))
        if ci or cj:
            # offsets must be relative to the NEW centre: the token features (di, dj, radius, phys
            # dx/dy) and the CNN grid index all assume the centre is at zero, and leaving them
            # relative to the old seed would push POS outside the grid, where the clip would silently
            # collide several cells onto one pixel.
            di = (di - ci).astype(ev['di'].dtype)
            dj = (dj - cj).astype(ev['dj'].dtype)
        pcw, xw, yw = ev['pc'][m], ev['x'][m], ev['y'][m]
        nwin = int(m.sum())
        if halo > W or patch > W:
            ht = halo_tokens(ev, W, halo)
            if ht:
                a = np.array(ht, np.float32)
                di = np.concatenate([di, a[:, 0].astype(di.dtype)])
                dj = np.concatenate([dj, np.zeros(len(a), dj.dtype)])
                e = np.concatenate([e, a[:, 1]])
                fr = np.concatenate([fr, a[:, 2]])
                bk = np.concatenate([bk, a[:, 3]])
                tf = np.concatenate([tf, a[:, 4]])
                tb = np.concatenate([tb, a[:, 5]])
                pcw = np.concatenate([pcw, a[:, 6]])
                xw = np.concatenate([xw, a[:, 7]])
                yw = np.concatenate([yw, a[:, 8]])
        if patch > W:
            pt = patch_tokens(ev, W, patch, patch_side)
            if pt:
                a = np.array(pt, np.float32)
                di = np.concatenate([di, np.rint(a[:, 0]).astype(di.dtype)])
                dj = np.concatenate([dj, np.rint(a[:, 1]).astype(dj.dtype)])
                e = np.concatenate([e, a[:, 2]])
                fr = np.concatenate([fr, a[:, 3]])
                bk = np.concatenate([bk, a[:, 4]])
                tf = np.concatenate([tf, a[:, 5]])
                tb = np.concatenate([tb, a[:, 6]])
                pcw = np.concatenate([pcw, a[:, 7]])
                xw = np.concatenate([xw, a[:, 8]])
                yw = np.concatenate([yw, a[:, 9]])
        if tpull:
            t0f, t0b = ref_time(tf, fr), ref_time(tb, bk)
        else:
            t0f = np.nanmedian(tf) if np.isfinite(tf).any() else 0.0
            t0b = np.nanmedian(tb) if np.isfinite(tb).any() else 0.0
        tfc = np.where(np.isfinite(tf), tf - t0f, 0.0)
        htf = np.isfinite(tf).astype(np.float32)
        tbc = np.where(np.isfinite(tb), tb - t0b, 0.0)
        htb = np.isfinite(tb).astype(np.float32)
        if tcomb:
            # Inverse-variance combination of the front and back cell times. The two samples
            # measure the same arrival with different noise (sigma_t depends on the energy seen by
            # each layer); combining with weights 1/sigma^2 improves the per-cell resolution by
            # ~0.82x when both fire, from the measured sigma_t(E) table -- pure physics, nothing
            # learned. The two time channels are REPLACED by (combined residual, its log-sigma):
            # raw timestamps are the one timing delivery with measured value (20% ablation), and
            # every engineered pull has failed, so this stays as close to raw as possible.
            wf = np.where(np.isfinite(tf), 1.0 / np.clip(sigma_t(fr), 1e-3, None) ** 2, 0.0)
            wb = np.where(np.isfinite(tb), 1.0 / np.clip(sigma_t(bk), 1e-3, None) ** 2, 0.0)
            wsum = wf + wb
            both = wsum > 0
            tcm = np.where(both, (np.nan_to_num(tfc) * wf + np.nan_to_num(tbc) * wb)
                           / np.clip(wsum, 1e-9, None), 0.0)
            sig = np.where(both, 1.0 / np.sqrt(np.clip(wsum, 1e-9, None)), 1.0)
            tfc = tcm
            tbc = np.log(np.clip(sig, 1e-3, None))
            htf = (wsum > 0).astype(np.float32)
            htb = ((wf > 0) & (wb > 0)).astype(np.float32)
        rdr = np.hypot(di, dj)
        gi, gj, gr = di.astype(np.float32), dj.astype(np.float32), rdr
        if mmgeo:
            # Geometry in MILLIMETRES rather than cell counts, scaled by the shower's own physical
            # size. Measured on the clean sample: the photon is 99.7-100% contained within 120 mm in
            # EVERY region, so the shower has a fixed physical width while its width in CELLS varies
            # eightfold (8 cells at 15mm pitch, 1 cell at 120mm). In cell units the model therefore
            # has to learn five different radial functions, each from a fraction of the data, for
            # what physics says is one universal function. This REPLACES the cell-unit channels
            # rather than adding to them: --phys already added mm offsets alongside and was neutral,
            # which is what a redundant mixed representation would do.
            s = float(ev['ps']) / 120.0
            gi, gj, gr = gi * s, gj * s, gr * s
        cont = np.stack([np.log1p(np.clip(e, 0, None)), np.log1p(np.clip(fr, 0, None)),
                         np.log1p(np.clip(bk, 0, None)), gi, gj,
                         gr, np.full(len(e), np.log(ev['ps'])), np.clip(tfc, -5, 5), np.clip(tbc, -5, 5)], 1)
        oh = np.zeros((len(e), len(PITCH)), np.float32)
        oh[:, ev['reg']] = 1.0
        tok = np.concatenate([cont, htf[:, None], htb[:, None], oh], 1).astype(np.float32)
        if halo > W or patch > W:
            # the flag matters: a ring token holds the SUM over ~25 cells, so without it the model
            # would read a halo token as one very energetic cell
            hf_flag = np.zeros((len(e), 1), np.float32)
            hf_flag[nwin:] = 1.0
            tok = np.concatenate([tok, hf_flag], 1)
        add = []
        if orho:
            # Cells OUTSIDE the window are a pileup sample from this same event with almost
            # no photon in them: clean data shows the photon is 99% contained inside 9x9.
            # ring-rho failed because it estimated the density from rings 3-4, which are
            # inside the window and still hold about a third of the shower.
            ar = (np.clip(pcw, 1.0, None) / 15.0) ** 2
            nout = max(ev.get('ncl', 0.0) - float(m.sum()), 0.0)
            eout = max(ev.get('tot', 0.0) - float(e.sum()), 0.0)
            rho_o = eout / max(nout, 1.0)
            ec = np.clip(e - rho_o * ar, 0.0, None)
            add += [np.log1p(ec), np.full(len(e), np.log1p(rho_o))]
        if depth:
            dr = np.log((np.clip(fr, 0, None) + 1.0) / (np.clip(bk, 0, None) + 1.0))
            sd = int(np.argmax(e))
            add += [dr, dr - dr[sd]]
        if tpull:
            pf = np.where(np.isfinite(tf), (tf - t0f) / np.clip(sigma_t(fr), 1e-3, None), 0.0)
            pb = np.where(np.isfinite(tb), (tb - t0b) / np.clip(sigma_t(bk), 1e-3, None), 0.0)
            add += [np.clip(pf, -10, 10), np.clip(pb, -10, 10),
                    np.log(np.clip(sigma_t(e), 1e-3, None))]
        rho_g = (0.0, 1.0)
        if rho:
            pcr = pcw
            ar = (np.clip(pcr, 1.0, None) / 15.0) ** 2
            ring = np.maximum(np.abs(di), np.abs(dj)) > 2
            rv = float(np.median(e[ring] / ar[ring])) if ring.sum() >= 3 else 0.0
            ec = np.clip(e - rv * ar, 0.0, None)
            add += [np.log1p(ec)]
            rho_g = (rv, float(ec.sum()) / (float(e.sum()) + EPS))
        if extra or dens:
            xs, ys, pc = xw, yw, pcw
            if extra:
                add += [(xs - ev.get('xc', 0.0)) / ev['ps'], (ys - ev.get('yc', 0.0)) / ev['ps']]
            if dens:
                lp = 2.0 * np.log(np.clip(pc, 1.0, None) / 15.0)
                add += [np.log1p(np.clip(e, 0, None)) - lp,
                        np.log1p(np.clip(fr, 0, None)) - lp,
                        np.log1p(np.clip(bk, 0, None)) - lp]
        if add:
            tok = np.concatenate([tok, np.stack(add, 1).astype(np.float32)], 1)
        if phys:
            xs, ys, pc = xw, yw, pcw
            s = int(np.argmax(e))
            dx = (xs - xs[s]) / 100.0
            dy = (ys - ys[s]) / 100.0
            pr = np.log(np.clip(pc, 1.0, None) / ev['ps'])
            tok = np.concatenate([tok, np.stack([dx, dy, pr], 1).astype(np.float32)], 1)
        ya = ((ev.get('ax', 0.0) - ev.get('xs', 0.0)) / ev['ps'],
              (ev.get('ay', 0.0) - ev.get('ys', 0.0)) / ev['ps'], ev.get('at', 0.0))
        ab = (0.0, 0.0, 0.0)
        if abst:
            # We median-subtract the time of every window, which deletes the ABSOLUTE time --
            # the only handle on out-of-time pileup (25 ns away, easily separable), as opposed
            # to in-time pileup which is quantitatively closed at our resolution.
            ta = ev.get('tf_abs')
            ta = tf if ta is None else ta[m] if len(ta) == len(ev['e']) else tf
            fin = np.isfinite(ta)
            if fin.any():
                w_ = e[fin]
                t0a = float(np.average(ta[fin], weights=np.clip(w_, 1e-6, None)))
                late = float(e[fin][ta[fin] > t0a + 1.0].sum() / max(float(e.sum()), 1e-6))
                spread = float(np.std(ta[fin])) if fin.sum() > 2 else 0.0
                ab = (t0a, late, spread)
        # Time slice per cell, for the divided space-time attention. Kept as its own array rather
        # than an extra token column because the column index would shift with every feature flag,
        # and a silently wrong index here would look like the architecture failing.
        pull_sl = np.abs(tf - (t0f if tpull else np.nanmedian(tf[np.isfinite(tf)])
                              if np.isfinite(tf).any() else 0.0)) / np.clip(sigma_t(e), 1e-3, None)
        sl = np.where(np.isfinite(tf), np.digitize(pull_sl, [0.5, 1.0, 2.0, 4.0]), 5).astype(np.int8)
        rsum = ()
        if rings > W:
            # Measured 2026-08-17: the 15mm cluster reaches ring 15 and a 9x9 window sees only
            # 37.6% of its energy, which is why enlarging the window moved 15mm low-E by 34% --
            # five times more than anything else ever tried. But tokens cost O(n^2) attention, and
            # W12 would be 625 tokens against 81. The far rings are diffuse pileup rather than
            # structure, so their per-ring SUMS carry what the model needs at the price of a few
            # numbers. Regions whose cluster ends earlier simply report zeros.
            dall = np.maximum(np.abs(ev['di']), np.abs(ev['dj']))
            above = ev['e'] >= THRESH
            rsum = tuple(float(ev['e'][above & (dall == r)].sum()) for r in range(W + 1, rings + 1))
        sg = ev.get('sig')
        frac = (np.clip(sg[m] / np.clip(e, 1e-6, None), 0.0, 1.0).astype(np.float32)
                if sg is not None else None)
        if prior is not None:
            dw = np.maximum(np.abs(di), np.abs(dj)).astype(np.float32)
            pr = np.clip(prior['model'].predict(
                cell_prior_features(e, fr, bk, tf, dw)), 0.0, 1.0).astype(np.float32)
            if prior.get('feat'):
                tok = np.concatenate([tok, pr[:, None]], 1)
            if prior.get('teach'):
                # The estimator is fitted on the synthetic pair, but its OUTPUT is a function of
                # observables only, so using it as the gate target works on real events too --
                # which is the point: the truth-based target only ever reached overlay events.
                frac = pr
        rows.append((tok, float(e.sum()), float(e.max()), ev['Etrue'], ev['reg'], ev['ET'],
                     (ev.get('tot', 0.0), ev.get('totf', 0.0), ev.get('totb', 0.0),
                      (ev.get('xc', 0.0) - ev.get('xs', 0.0)) / ev['ps'],
                      (ev.get('yc', 0.0) - ev.get('ys', 0.0)) / ev['ps'],
                      ev.get('ncl', 0.0), rho_g[0], rho_g[1]) + ab + rsum
                     + (ev.get("xs", 0.0), ev.get("ys", 0.0)), ya, frac, sl))
        keep.append(i)
    return rows, np.array(keep)


def splits_for(keep, n_events, fold=None, nfold=5):
    remap = -np.ones(n_events, int)
    remap[keep] = np.arange(len(keep))
    a, b, t = split(n_events, fold=fold, nfold=nfold)
    return (remap[a][remap[a] >= 0], remap[b][remap[b] >= 0], remap[t][remap[t] >= 0])


def prep(W, main_events, aux_events=None, ng=6, phys=False, occ=False, extra=False,
         dens=False, rho=False, tpull=False, aux=False, depth=False, orho=False,
         abst=False, prior=None, rings=0, halo=0, globpe=0, patch=0, patch_side=3,
         recenter=False, fold=None, nfold=5, mmgeo=False, rc_regions=None, rc_mode='centroid',
         tcomb=False, allcells=False, aperture_mm=0.0):
    if aux and main_events and 'ax' not in main_events[0]:
        raise SystemExit('--aux needs truth position/time branches missing from these events: '
                         'delete .scratch/cache/*.pkl so the cache is rebuilt')
    if extra and main_events and 'tot' not in main_events[0]:
        raise SystemExit('--extra needs cluster-level branches missing from these events: '
                         'delete .scratch/cache/*.pkl so the cache is rebuilt')
    rows, keep = make_windows(W, main_events, phys, extra, dens, rho, tpull, depth, orho, abst,
                              prior, rings, halo, patch, patch_side, recenter, mmgeo, rc_regions, rc_mode, tcomb,
                              allcells, aperture_mm)
    ktr, kva, kte = splits_for(keep, len(main_events), fold, nfold)
    n_mb = len(rows)
    ctr = np.array([], int)
    keep_aux = np.array([], int)
    if aux_events is not None:
        crows, keep_aux = make_windows(W, aux_events, phys, extra, dens, rho, tpull, depth, orho, abst,
                                prior, rings, halo, patch, patch_side, recenter, mmgeo, rc_regions, rc_mode, tcomb,
                              allcells, aperture_mm)
        rows = rows + crows
        ctr = np.arange(n_mb, len(rows))
    N = len(rows)
    # room for the ring tokens on top of the window grid; the CNN path indexes POS into the grid
    # only, so halo tokens are clipped there and reach the model through attention instead
    L = (2 * W + 1) ** 2 + max(halo - W, 0) + (0 if patch <= W else ((2 * patch + 1) // patch_side + 1) ** 2)
    if allcells:
        L = max(r[0].shape[0] for r in rows)
    IN_DIM = rows[0][0].shape[1]
    y = np.array([np.log(max(r[3], 1e-3)) for r in rows], np.float32)
    Et = np.array([r[3] for r in rows], np.float32)
    sumE = np.array([r[1] for r in rows], np.float32)
    reg = np.array([r[4] for r in rows], int)
    ET = np.array([r[5] for r in rows], np.float32)
    S = 2 * W + 1
    POS = np.full((N, L), L, np.int64)
    X = np.zeros((N, L, IN_DIM), np.float32)
    M = np.zeros((N, L), np.bool_)
    G = np.zeros((N, ng), np.float32)
    Eraw = np.zeros((N, L), np.float32)
    YA = np.zeros((N, 3), np.float32)
    FR = np.zeros((N, L), np.float32)
    HASF = np.zeros(N, np.float32)
    SL = np.zeros((N, L), np.int64)
    for i, (tok, se, sde, et, rg, etv, ex, ya, frac, sl) in enumerate(rows):
        if frac is not None:
            FR[i, :len(frac)] = frac
            HASF[i] = 1.0
        n = tok.shape[0]
        X[i, :n] = tok
        SL[i, :n] = sl[:n] if len(sl) >= n else np.pad(sl, (0, n - len(sl)))
        M[i, :n] = True
        YA[i] = ya
        e = np.expm1(tok[:, 0])
        Eraw[i, :n] = e
        gi = (tok[:, 3].astype(int) + W) * S + (tok[:, 4].astype(int) + W)
        POS[i, :n] = np.clip(gi, 0, L - 1)
        lat = float(np.sqrt((e * tok[:, 5] ** 2).sum() / (e.sum() + EPS)))
        fbr = float(np.expm1(tok[:, 1]).sum() / (np.expm1(tok[:, 2]).sum() + EPS))
        g = [np.log1p(se), np.log1p(sde), np.log(n), fbr, lat]
        if occ:
            core = float(e[(np.abs(tok[:, 3]) <= 1) & (np.abs(tok[:, 4]) <= 1)].sum())
            g += [1.0 - core / (float(e.sum()) + EPS), n / L]
        if abst:
            g += [ex[8] / 50.0, ex[9], min(ex[10], 20.0) / 5.0]
        if rho:
            g += [np.log1p(max(ex[6], 0.0)), ex[7]]
        if extra:
            tot, totf, totb, dxc, dyc, ncl = ex[:6]
            sf = float(np.expm1(tok[:, 1]).sum())
            sb = float(np.expm1(tok[:, 2]).sum())
            g += [np.log1p(max(tot, 0.0)), se / (tot + EPS),
                  sf / (totf + EPS), sb / (totb + EPS),
                  np.log((totf + EPS) / (totb + EPS)), np.log1p(max(ncl, 0.0)),
                  dxc, dyc]
        if rings > W:
            # ex[11:] holds the far-ring sums; scaled like the other energies so one feature
            # cannot dominate the normalisation
            g += [np.log1p(max(v, 0.0)) for v in ex[11:11 + (rings - W)]]
        if globpe:
            # ClusTEX (arXiv:2603.18172) separates the LOCAL coordinate inside the window from a
            # GLOBAL detector coordinate, carried by a learnable embedding; their Fig. 15 shows
            # those learned weights converging to a cosine whose period is the detector extent.
            # Supplying that basis directly costs no new module. Our tokens only ever saw local
            # offsets, the region one-hot and the pitch -- never where in the detector the cluster
            # sits, which is what sets the local occupancy and the response.
            xs, ys = ex[11 + max(rings - W, 0)], ex[12 + max(rings - W, 0)]
            for k in range(globpe):
                for u in (xs / 4000.0, ys / 4000.0):
                    g += [np.sin(2.0 ** k * np.pi * u), np.cos(2.0 ** k * np.pi * u)]
        G[i] = (g + [float(i >= n_mb)])[:ng]
    la0, lb0 = np.polyfit(np.log1p(0.5 * sumE[ktr]), y[ktr], 1)
    k = ng - 1
    G[:, :k] = (G[:, :k] - G[ktr, :k].mean(0)) / (G[ktr, :k].std(0) + EPS)
    cont = X[ktr][:, :, :NC].reshape(-1, NC)[M[ktr].reshape(-1)]
    mean = cont.mean(0)
    std = cont.std(0) + EPS
    X[:, :, :NC] = (X[:, :, :NC] - mean) / std
    X[~M] = 0.0
    print(f'W={W}: N {N} (main {n_mb} + aux {len(ctr)}), tr/va/te {len(ktr)}/{len(kva)}/{len(kte)}, IN_DIM {IN_DIM}')
    YA[:, 2] -= YA[ktr, 2].mean()
    ya_std = YA[ktr].std(0) + EPS
    YA /= ya_std
    return dict(X=X, M=M, G=G, y=y, Et=Et, ET=ET, reg=reg, Eraw=Eraw, POS=POS, S=S, YA=YA,
                FR=FR, HASF=HASF, SL=SL, YA_std=ya_std, keep=keep, keep_aux=keep_aux,
                ktr=ktr, kva=kva, kte=kte, ctr=ctr, IN_DIM=IN_DIM,
                la0=float(la0), lb0=float(lb0), mean=mean, std=std)
