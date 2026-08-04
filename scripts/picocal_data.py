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
THRESH = 2.49
NC = 9


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
    return seed, ps, di, dj, ok


def build_grid(files, label='', vertex_max=100.0, emin=1.0, emax=100.0):
    EV = []
    for path in files:
        with uproot.open(path) as f:
            a = f['clusters_matched'].arrays(TK + AUX, library='ak')
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
            seed, ps, di, dj, ok = event_geom(cc)
            if ok.mean() < 0.5 or not ok[seed]:
                continue
            tf = cc['cell_times_front']
            tb = cc['cell_times_back']
            tf = np.where(np.isfinite(tf) & (tf != 0) & (np.abs(tf) < 1e4), tf, np.nan)
            tb = np.where(np.isfinite(tb) & (tb != 0) & (np.abs(tb) < 1e4), tb, np.nan)
            EV.append(dict(di=di[ok].astype(np.int16), dj=dj[ok].astype(np.int16),
                           x=cc['cell_x'][ok].astype(np.float32), y=cc['cell_y'][ok].astype(np.float32),
                           e=e[ok].astype(np.float32),
                           fr=cc['cell_energies_front'][ok].astype(np.float32),
                           bk=cc['cell_energies_back'][ok].astype(np.float32),
                           tf=tf[ok].astype(np.float32), tb=tb[ok].astype(np.float32),
                           ps=float(ps), reg=int(np.argmin(np.abs(PITCH - ps))),
                           Etrue=float(et_all[i]), ET=float(etv[i])))
    if label:
        print(f'{label}: {len(EV)} events')
    return EV


def make_windows(W, EVS):
    rows = []
    keep = []
    for i, ev in enumerate(EVS):
        m = (np.maximum(np.abs(ev['di']), np.abs(ev['dj'])) <= W) & (ev['e'] >= THRESH)
        if m.sum() < 1:
            continue
        di, dj, e, fr, bk, tf, tb = (v[m] for v in (ev['di'], ev['dj'], ev['e'], ev['fr'], ev['bk'], ev['tf'], ev['tb']))
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
        rows.append((tok, float(e.sum()), float(e.max()), ev['Etrue'], ev['reg'], ev['ET']))
        keep.append(i)
    return rows, np.array(keep)


def splits_for(keep, n_events):
    remap = -np.ones(n_events, int)
    remap[keep] = np.arange(len(keep))
    a, b, t = split(n_events)
    return (remap[a][remap[a] >= 0], remap[b][remap[b] >= 0], remap[t][remap[t] >= 0])


def prep(W, main_events, aux_events=None, ng=6):
    rows, keep = make_windows(W, main_events)
    ktr, kva, kte = splits_for(keep, len(main_events))
    n_mb = len(rows)
    ctr = np.array([], int)
    if aux_events is not None:
        crows, _ = make_windows(W, aux_events)
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
    X = np.zeros((N, L, IN_DIM), np.float32)
    M = np.zeros((N, L), np.bool_)
    G = np.zeros((N, ng), np.float32)
    Eraw = np.zeros((N, L), np.float32)
    for i, (tok, se, sde, et, rg, etv) in enumerate(rows):
        n = tok.shape[0]
        X[i, :n] = tok
        M[i, :n] = True
        e = np.expm1(tok[:, 0])
        Eraw[i, :n] = e
        lat = float(np.sqrt((e * tok[:, 5] ** 2).sum() / (e.sum() + EPS)))
        fbr = float(np.expm1(tok[:, 1]).sum() / (np.expm1(tok[:, 2]).sum() + EPS))
        G[i] = [np.log1p(se), np.log1p(sde), np.log(n), fbr, lat, float(i >= n_mb)][:ng]
    la0, lb0 = np.polyfit(np.log1p(0.5 * sumE[ktr]), y[ktr], 1)
    G[:, :5] = (G[:, :5] - G[ktr, :5].mean(0)) / (G[ktr, :5].std(0) + EPS)
    cont = X[ktr][:, :, :NC].reshape(-1, NC)[M[ktr].reshape(-1)]
    mean = cont.mean(0)
    std = cont.std(0) + EPS
    X[:, :, :NC] = (X[:, :, :NC] - mean) / std
    X[~M] = 0.0
    print(f'W={W}: N {N} (main {n_mb} + aux {len(ctr)}), tr/va/te {len(ktr)}/{len(kva)}/{len(kte)}, IN_DIM {IN_DIM}')
    return dict(X=X, M=M, G=G, y=y, Et=Et, ET=ET, reg=reg, Eraw=Eraw,
                ktr=ktr, kva=kva, kte=kte, ctr=ctr, IN_DIM=IN_DIM,
                la0=float(la0), lb0=float(lb0), mean=mean, std=std)
