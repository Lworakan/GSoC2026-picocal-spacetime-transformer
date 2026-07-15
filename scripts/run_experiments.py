import sys
import json
import argparse
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import HistGradientBoostingRegressor

PITCH = np.array([15.0, 30.0, 40.0, 60.0, 120.0])
EPS = 1e-6
CELL_KEYS = ["cell_x", "cell_y", "cell_energies_front", "cell_energies_back", "energy", "imodx", "jmody"]


def derive_geom(c):
    x = c["cell_x"]; y = c["cell_y"]; ix = c["imodx"]; iy = c["jmody"]
    pts = np.stack([x, y], 1)
    pitch = np.full(len(x), np.nan)
    for key in {(int(a), int(b)) for a, b in zip(ix, iy)}:
        sel = (ix == key[0]) & (iy == key[1])
        p = pts[sel]
        if len(p) >= 2:
            d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
            d[d == 0] = np.inf
            pitch[sel] = np.median(np.min(d, axis=1))
    fill = np.nanmedian(pitch) if np.isfinite(pitch).any() else 120.0
    pitch[~np.isfinite(pitch)] = fill
    seed = int(np.nanargmax(c["energy"]))
    rel_x = x - x[seed]; rel_y = y - y[seed]; rel_dr = np.hypot(rel_x, rel_y)
    mod = np.array([int(np.argmin(np.abs(PITCH - p))) for p in pitch], dtype=int)
    return pitch, mod, rel_x, rel_y, rel_dr, seed


def select_window(c, window):
    pitch, mod, rel_x, rel_y, rel_dr, seed = derive_geom(c)
    ps = pitch[seed]
    half = window // 2
    return (np.abs(np.round(rel_x / ps)) <= half) & (np.abs(np.round(rel_y / ps)) <= half)


def select_knn(c, k):
    pitch, mod, rel_x, rel_y, rel_dr, seed = derive_geom(c)
    order = np.argsort(rel_dr)
    keep = np.zeros(len(rel_dr), dtype=bool)
    keep[order[:k]] = True
    return keep


def tokens(e, fr, bk, rel_x, rel_y, rel_dr, pitch, mod):
    cont = np.stack([np.log1p(np.clip(e, 0, None)), np.log1p(np.clip(fr, 0, None)),
                     np.log1p(np.clip(bk, 0, None)), rel_x / pitch, rel_y / pitch,
                     rel_dr / pitch, np.log(pitch)], 1)
    onehot = np.zeros((len(e), len(PITCH)))
    onehot[np.arange(len(e)), mod] = 1.0
    return np.concatenate([cont, onehot], 1).astype(np.float32)


def variant_tokens(raw, variant):
    e, fr, bk, rx, ry, rdr, pitch, mod, cx, cy = raw
    le = np.log1p(np.clip(e, 0, None)); lf = np.log1p(np.clip(fr, 0, None)); lbk = np.log1p(np.clip(bk, 0, None))
    oh = np.zeros((len(e), len(PITCH))); oh[np.arange(len(e)), mod] = 1.0
    if variant == "drop_fb":
        cont = np.stack([le, rx / pitch, ry / pitch, rdr / pitch, np.log(pitch)], 1)
        return np.concatenate([cont, oh], 1).astype(np.float32), cont.shape[1]
    if variant == "raw_e":
        cont = np.stack([np.clip(e, 0, None), np.clip(fr, 0, None), np.clip(bk, 0, None),
                         rx / pitch, ry / pitch, rdr / pitch, np.log(pitch)], 1)
        return np.concatenate([cont, oh], 1).astype(np.float32), cont.shape[1]
    if variant == "drop_onehot":
        cont = np.stack([le, lf, lbk, rx / pitch, ry / pitch, rdr / pitch, np.log(pitch)], 1)
        return cont.astype(np.float32), cont.shape[1]
    if variant == "abs_coords":
        cont = np.stack([le, lf, lbk, cx, cy, np.hypot(cx, cy), np.log(pitch)], 1)
        return np.concatenate([cont, oh], 1).astype(np.float32), cont.shape[1]
    raise ValueError(variant)


def build(files, window, vertex_max, selector=None):
    out = {k: [] for k in ["region", "tok_seed", "tok_cluster", "raw", "agg", "sumE", "seedE",
                            "total_energy", "y", "Etrue"]}
    for path in files:
        with uproot.open(path) as f:
            a = f["clusters_matched"].arrays(
                CELL_KEYS + ["sig_flux_prod_vertex_z", "sig_flux_eTot", "total_energy",
                             "x_cluster", "y_cluster"], library="ak")
        vz = ak.to_numpy(a["sig_flux_prod_vertex_z"]).astype(float)
        keep = np.flatnonzero(vz < vertex_max)
        for i in keep:
            c = {k: np.asarray(ak.to_numpy(a[k][i])).astype(float) for k in CELL_KEYS}
            sel = selector(c) if selector is not None else select_window(c, window)
            cw = {k: v[sel] for k, v in c.items()}
            if len(cw["energy"]) == 0:
                continue
            pitch, mod, rel_x, rel_y, rel_dr, seed = derive_geom(cw)
            e = cw["energy"]; fr = cw["cell_energies_front"]; bk = cw["cell_energies_back"]
            region = int(mod[seed])
            out["tok_seed"].append(tokens(e, fr, bk, rel_x, rel_y, rel_dr, pitch, mod))
            xcl = float(a["x_cluster"][i]); ycl = float(a["y_cluster"][i])
            rx = cw["cell_x"] - xcl; ry = cw["cell_y"] - ycl; rdr = np.hypot(rx, ry)
            out["tok_cluster"].append(tokens(e, fr, bk, rx, ry, rdr, pitch, mod))
            out["raw"].append((e, fr, bk, rel_x, rel_y, rel_dr, pitch, mod, cw["cell_x"], cw["cell_y"]))
            sumE = float(e.sum()); seedE = float(e[seed])
            lat = float(np.sqrt((e * rel_dr ** 2).sum() / (sumE + EPS)))
            fb = float(fr.sum() / (bk.sum() + EPS))
            out["agg"].append([np.log1p(sumE), fb, len(e), np.log1p(seedE), lat, region])
            out["sumE"].append(sumE); out["seedE"].append(seedE)
            out["total_energy"].append(float(a["total_energy"][i]))
            etrue = float(a["sig_flux_eTot"][i])
            out["y"].append(np.log(max(etrue, 1e-3))); out["Etrue"].append(etrue)
            out["region"].append(region)
    for k in ["region", "agg", "sumE", "seedE", "total_energy", "y", "Etrue"]:
        out[k] = np.array(out[k])
    return out


def resolution(E_pred, E_true):
    r = (np.asarray(E_pred) - E_true) / E_true
    rs = np.sort(r); n = len(rs); k = max(1, int(np.ceil(0.683 * n)))
    w = rs[k - 1:] - rs[:n - k + 1]
    sigma_eff = float(w.min() / 2) if len(w) else float("nan")
    iqr = float((np.percentile(r, 75) - np.percentile(r, 25)) / 1.349)
    return {"sigma_eff": round(sigma_eff, 4), "iqr": round(iqr, 4), "bias": round(float(np.median(r)), 4)}


def split(n, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    a = int(0.70 * n); b = int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


class TokenDS(Dataset):
    def __init__(self, toks, y, mean, std, ncont=7):
        self.toks = toks; self.y = y; self.mean = mean; self.std = std; self.ncont = ncont

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        t = self.toks[i].copy()
        c = self.ncont
        t[:, :c] = (t[:, :c] - self.mean) / self.std
        return torch.tensor(t), torch.tensor([self.y[i]], dtype=torch.float32)


def collate(batch):
    feats, ys = zip(*batch)
    Lmax = max(f.shape[0] for f in feats); F = feats[0].shape[1]; B = len(batch)
    X = torch.zeros(B, Lmax, F); m = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, f in enumerate(feats):
        X[i, : f.shape[0]] = f; m[i, : f.shape[0]] = True
    return X, m, torch.stack(ys)


class Transformer(nn.Module):
    def __init__(self, in_dim, d=32, nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=64, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, m):
        h = self.enc(self.embed(x), src_key_padding_mask=~m)
        w = m.unsqueeze(-1).float()
        return self.head((h * w).sum(1) / w.sum(1).clamp(min=1))


class DeepSets(nn.Module):
    def __init__(self, in_dim, d=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, m):
        h = self.phi(x); w = m.unsqueeze(-1).float()
        return self.head((h * w).sum(1) / w.sum(1).clamp(min=1))


def train_eval(model, toks, y, tr, va, te, Etrue, epochs, device, bs=64, seed=0, ncont=7):
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)
    cont = np.concatenate([toks[i][:, :ncont] for i in tr], 0)
    mean = cont.mean(0); std = cont.std(0) + EPS
    dl_tr = DataLoader(TokenDS([toks[i] for i in tr], y[tr], mean, std, ncont),
                       batch_size=bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for X, m, yb in dl_tr:
            X, m, yb = X.to(dev), m.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X, m), yb)
            loss.backward(); opt.step()
    model.eval()
    preds = np.zeros(len(toks))
    with torch.no_grad():
        for idx in (va, te):
            dl = DataLoader(TokenDS([toks[i] for i in idx], y[idx], mean, std, ncont),
                            batch_size=256, shuffle=False, collate_fn=collate)
            p = []
            for X, m, _ in dl:
                p.append(model(X.to(dev), m.to(dev)).cpu().numpy().ravel())
            preds[idx] = np.concatenate(p)
    return resolution(np.exp(preds[te]), Etrue[te]), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--vertex", type=float, default=100.0)
    ap.add_argument("--region", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu", flush=True)
        args.device = "cpu"
    print(f"device={args.device}", flush=True)

    repo = Path(__file__).resolve().parent.parent
    files = sorted((repo / "data" / "full").glob("matched_*.root"))[: args.files]
    print(f"building from {len(files)} files ...", flush=True)
    D = build(files, args.window, args.vertex)
    n = len(D["y"])
    ridx = np.flatnonzero(D["region"] == args.region)
    print(f"kept {n} clusters; region R{args.region} = {len(ridx)}", flush=True)

    results = {}
    rtr, rva, rte = (ridx[s] for s in split(len(ridx)))
    Et = D["Etrue"]

    def calib(xtr, xte, ytr_log, Ete):
        a, b = np.polyfit(np.log(xtr + EPS), ytr_log, 1)
        return resolution(np.exp(a * np.log(xte + EPS) + b), Ete)

    results["E0_B0_sum_calib"] = calib(D["sumE"][rtr], D["sumE"][rte], D["y"][rtr], Et[rte])
    results["E0_B1_seed_calib"] = calib(D["seedE"][rtr], D["seedE"][rte], D["y"][rtr], Et[rte])
    results["E0_B2_total_energy_calib"] = calib(D["total_energy"][rtr], D["total_energy"][rte], D["y"][rtr], Et[rte])

    gb = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    gb.fit(D["agg"][rtr], D["y"][rtr])
    results["E0b_BDT"] = resolution(np.exp(gb.predict(D["agg"][rte])), Et[rte])

    in_dim = D["tok_seed"][0].shape[1]
    results["E1_transformer_seed"] = train_eval(Transformer(in_dim), D["tok_seed"], D["y"],
                                                rtr, rva, rte, Et, args.epochs, args.device, args.batch)[0]
    results["E2_transformer_cluster"] = train_eval(Transformer(in_dim), D["tok_cluster"], D["y"],
                                                   rtr, rva, rte, Et, args.epochs, args.device, args.batch)[0]
    results["E11_deepsets_seed"] = train_eval(DeepSets(in_dim), D["tok_seed"], D["y"],
                                              rtr, rva, rte, Et, args.epochs, args.device, args.batch)[0]

    atr, ava, ate = split(n)
    res_all, preds_all = train_eval(Transformer(in_dim), D["tok_seed"], D["y"],
                                    atr, ava, ate, Et, args.epochs, args.device, args.batch)
    results["E4_all_regions_overall"] = res_all
    te_r = ate[D["region"][ate] == args.region]
    if len(te_r) > 20:
        results["E4_all_regions_on_R" + str(args.region)] = resolution(np.exp(preds_all[te_r]), Et[te_r])

    for name, var in [("E12a_drop_frontback", "drop_fb"), ("E12b_raw_energy", "raw_e"),
                      ("E12c_drop_region_onehot", "drop_onehot"), ("E12d_abs_coords", "abs_coords")]:
        toks_v = [None] * n
        nc = 7
        for i in ridx:
            tk, nc = variant_tokens(D["raw"][i], var)
            toks_v[i] = tk
        vdim = toks_v[int(ridx[0])].shape[1]
        results[name] = train_eval(Transformer(vdim), toks_v, D["y"], rtr, rva, rte,
                                   Et, args.epochs, args.device, args.batch, ncont=nc)[0]

    out = {"config": vars(args), "n_kept": int(n), "n_region": int(len(ridx)), "results": results}
    rep = repo / "reports"; rep.mkdir(exist_ok=True)
    suffix = ("_" + args.tag) if args.tag else ""
    (rep / f"experiment-results{suffix}.json").write_text(json.dumps(out, indent=2))
    lines = ["# Experiment results", "", f"config: {vars(args)}",
             f"kept {n} clusters; R{args.region} = {len(ridx)}", "",
             "| experiment | sigma_eff | IQR | bias |", "|---|---|---|---|"]
    for k, v in results.items():
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['sigma_eff']} | {v['iqr']} | {v['bias']} |")
    (rep / f"experiment-results{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
