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


def make_windows(W, EVS, phys=False, extra=False, dens=False, rho=False, tpull=False,
                 depth=False, orho=False, abst=False):
    rows = []
    keep = []
    for i, ev in enumerate(EVS):
        m = (np.maximum(np.abs(ev['di']), np.abs(ev['dj'])) <= W) & (ev['e'] >= THRESH)
        if m.sum() < 1:
            continue
        di, dj, e, fr, bk, tf, tb = (v[m] for v in (ev['di'], ev['dj'], ev['e'], ev['fr'], ev['bk'], ev['tf'], ev['tb']))
        if tpull:
            t0f, t0b = ref_time(tf, fr), ref_time(tb, bk)
        else:
            t0f = np.nanmedian(tf) if np.isfinite(tf).any() else 0.0
            t0b = np.nanmedian(tb) if np.isfinite(tb).any() else 0.0
        tfc = np.where(np.isfinite(tf), tf - t0f, 0.0)
        htf = np.isfinite(tf).astype(np.float32)
        tbc = np.where(np.isfinite(tb), tb - t0b, 0.0)
        htb = np.isfinite(tb).astype(np.float32)
        rdr = np.hypot(di, dj)
        cont = np.stack([np.log1p(np.clip(e, 0, None)), np.log1p(np.clip(fr, 0, None)),
                         np.log1p(np.clip(bk, 0, None)), di.astype(np.float32), dj.astype(np.float32),
                         rdr, np.full(len(e), np.log(ev['ps'])), np.clip(tfc, -5, 5), np.clip(tbc, -5, 5)], 1)
        oh = np.zeros((len(e), len(PITCH)), np.float32)
        oh[:, ev['reg']] = 1.0
        tok = np.concatenate([cont, htf[:, None], htb[:, None], oh], 1).astype(np.float32)
        add = []
        if orho:
            # Cells OUTSIDE the window are a pileup sample from this same event with almost
            # no photon in them: clean data shows the photon is 99% contained inside 9x9.
            # ring-rho failed because it estimated the density from rings 3-4, which are
            # inside the window and still hold about a third of the shower.
            ar = (np.clip(ev['pc'][m], 1.0, None) / 15.0) ** 2
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
            pcr = ev['pc'][m]
            ar = (np.clip(pcr, 1.0, None) / 15.0) ** 2
            ring = np.maximum(np.abs(di), np.abs(dj)) > 2
            rv = float(np.median(e[ring] / ar[ring])) if ring.sum() >= 3 else 0.0
            ec = np.clip(e - rv * ar, 0.0, None)
            add += [np.log1p(ec)]
            rho_g = (rv, float(ec.sum()) / (float(e.sum()) + EPS))
        if extra or dens:
            xs, ys, pc = (v[m] for v in (ev['x'], ev['y'], ev['pc']))
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
            xs, ys, pc = (v[m] for v in (ev['x'], ev['y'], ev['pc']))
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
        rows.append((tok, float(e.sum()), float(e.max()), ev['Etrue'], ev['reg'], ev['ET'],
                     (ev.get('tot', 0.0), ev.get('totf', 0.0), ev.get('totb', 0.0),
                      (ev.get('xc', 0.0) - ev.get('xs', 0.0)) / ev['ps'],
                      (ev.get('yc', 0.0) - ev.get('ys', 0.0)) / ev['ps'],
                      ev.get('ncl', 0.0), rho_g[0], rho_g[1]) + ab, ya))
        keep.append(i)
    return rows, np.array(keep)


def splits_for(keep, n_events):
    remap = -np.ones(n_events, int)
    remap[keep] = np.arange(len(keep))
    a, b, t = split(n_events)
    return (remap[a][remap[a] >= 0], remap[b][remap[b] >= 0], remap[t][remap[t] >= 0])


def prep(W, main_events, aux_events=None, ng=6, phys=False, occ=False, extra=False,
         dens=False, rho=False, tpull=False, aux=False, depth=False, orho=False,
         abst=False):
    if aux and main_events and 'ax' not in main_events[0]:
        raise SystemExit('--aux needs truth position/time branches missing from these events: '
                         'delete .scratch/cache/*.pkl so the cache is rebuilt')
    if extra and main_events and 'tot' not in main_events[0]:
        raise SystemExit('--extra needs cluster-level branches missing from these events: '
                         'delete .scratch/cache/*.pkl so the cache is rebuilt')
    rows, keep = make_windows(W, main_events, phys, extra, dens, rho, tpull, depth, orho, abst)
    ktr, kva, kte = splits_for(keep, len(main_events))
    n_mb = len(rows)
    ctr = np.array([], int)
    if aux_events is not None:
        crows, _ = make_windows(W, aux_events, phys, extra, dens, rho, tpull, depth, orho, abst)
        rows = rows + crows
        ctr = np.arange(n_mb, len(rows))
    N = len(rows)
    L = (2 * W + 1) ** 2
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
    for i, (tok, se, sde, et, rg, etv, ex, ya) in enumerate(rows):
        n = tok.shape[0]
        X[i, :n] = tok
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
    YA /= (YA[ktr].std(0) + EPS)
    return dict(X=X, M=M, G=G, y=y, Et=Et, ET=ET, reg=reg, Eraw=Eraw, POS=POS, S=S, YA=YA,
                ktr=ktr, kva=kva, kte=kte, ctr=ctr, IN_DIM=IN_DIM,
                la0=float(la0), lb0=float(lb0), mean=mean, std=std)
