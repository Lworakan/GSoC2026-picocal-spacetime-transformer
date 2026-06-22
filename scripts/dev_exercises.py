import glob, time, os
import numpy as np
import awkward as ak
import uproot
import torch
import torch.nn as nn

RNG = np.random.default_rng(0)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FILES = int(os.environ.get("DEV_FILES", "50"))
SKIP_XF = os.environ.get("SKIP_XF", "0") == "1"
COOLDOWN = float(os.environ.get("COOLDOWN", "0.25"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "35000"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
TBAD = -1e30
TREF = 85.0
KMAX = 128

SCALARS = ["sig_flux_eTot", "sig_flux_entry_x", "sig_flux_entry_y",
           "sig_dr_matched", "sig_flux_pdgID",
           "x_cluster", "y_cluster", "total_energy", "total_energy_front", "total_energy_back"]
CELLS = ["energy", "cell_energies_front", "cell_energies_back",
         "cell_x", "cell_y", "cell_times_front", "cell_times_back"]


def load(files):
    return uproot.concatenate([f"{fp}:clusters_matched" for fp in files], SCALARS + CELLS, library="ak")


def quality_mask(a):
    eT = ak.to_numpy(a["sig_flux_eTot"])
    dr = ak.to_numpy(a["sig_dr_matched"])
    raw = ak.to_numpy(a["total_energy"])
    pdg = ak.to_numpy(a["sig_flux_pdgID"])
    ncell = ak.to_numpy(ak.num(a["energy"], axis=1))
    return (pdg == 22) & (dr < 40.0) & (eT > 0) & (raw > 0) & (ncell >= 9)


def robust_sigma(r):
    q16, q84 = np.percentile(r, [16, 84])
    return 0.5 * (q84 - q16)


def calib_loglog(raw_tr, et_tr, deg=3):
    return np.polyfit(np.log10(raw_tr), np.log10(et_tr), deg)


def apply_loglog(coef, raw):
    return 10 ** np.polyval(coef, np.log10(raw))


def binned_resolution(pred, true, edges):
    c, sig, bias, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (true >= lo) & (true < hi)
        if m.sum() < 30:
            continue
        r = (pred[m] - true[m]) / true[m]
        c.append(np.sqrt(lo * hi)); sig.append(robust_sigma(r))
        bias.append(np.median(r)); ns.append(int(m.sum()))
    return np.array(c), np.array(sig), np.array(bias), np.array(ns)


def fit_reso(E, soe):
    A = np.vstack([1.0 / E, np.ones_like(E)]).T
    sol, *_ = np.linalg.lstsq(A, soe ** 2, rcond=None)
    a2, c2 = sol
    return np.sqrt(max(a2, 0)), np.sqrt(max(c2, 0))


def build_padded(a, k=KMAX):
    e = a["energy"]
    order = ak.argsort(e, axis=1, ascending=False)
    def pick(name):
        v = a[name][order][:, :k]
        v = ak.pad_none(v, k, axis=1, clip=True)
        return ak.to_numpy(ak.fill_none(v, 0.0)).astype(np.float32)
    es = pick("energy"); ef = pick("cell_energies_front"); eb = pick("cell_energies_back")
    cx = pick("cell_x"); cy = pick("cell_y"); tf = pick("cell_times_front"); tb = pick("cell_times_back")
    xcl = ak.to_numpy(a["x_cluster"]).astype(np.float32)[:, None]
    ycl = ak.to_numpy(a["y_cluster"]).astype(np.float32)[:, None]
    mask = es > 0
    rel_x = (cx - xcl) * mask / 50.0
    rel_y = (cy - ycl) * mask / 50.0
    vt_f = (tf > TBAD) & mask
    vt_b = (tb > TBAD) & mask
    feat = {
        "log_e": np.log1p(np.clip(es, 0, None)) * mask,
        "log_ef": np.log1p(np.clip(ef, 0, None)) * mask,
        "log_eb": np.log1p(np.clip(eb, 0, None)) * mask,
        "rel_x": rel_x, "rel_y": rel_y,
        "rel_dr": np.sqrt(rel_x ** 2 + rel_y ** 2),
        "t_front": np.where(vt_f, (tf - TREF) / 20.0, 0.0).astype(np.float32),
        "t_back": np.where(vt_b, (tb - TREF) / 20.0, 0.0).astype(np.float32),
    }
    return feat, mask, es


def stack(feat, keys):
    return np.stack([feat[k] for k in keys], axis=-1).astype(np.float32)


class CellTransformer(nn.Module):
    def __init__(self, fdim, d=64, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(fdim, d), nn.GELU(), nn.LayerNorm(d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True,
                                         dropout=0.0, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.pool_norm = nn.LayerNorm(3 * d)
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask):
        h = self.embed(x)
        h = self.enc(h, src_key_padding_mask=~mask)
        w = mask.unsqueeze(-1).float()
        s = (h * w).sum(1)
        m = s / w.sum(1).clamp(min=1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        pooled = self.pool_norm(torch.cat([s / 16.0, m, mx], dim=-1))
        return self.head(pooled).squeeze(-1)


def run_transformer(X, M, y, itr, iva, epochs=EPOCHS, tag=""):
    fdim = X.shape[-1]
    flat = X[itr][M[itr]]
    mu = flat.mean(0); sd = flat.std(0) + 1e-6
    Xn = ((X - mu) / sd) * M[..., None]
    ym, ys = y[itr].mean(), y[itr].std() + 1e-6
    yn = (y - ym) / ys
    Xt = torch.from_numpy(Xn).to(DEV); Mt = torch.from_numpy(M).to(DEV)
    yt = torch.from_numpy(yn.astype(np.float32)).to(DEV)
    tr = torch.from_numpy(itr).to(DEV); va = torch.from_numpy(iva).to(DEV)
    model = CellTransformer(fdim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 512
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=epochs,
                steps_per_epoch=int(np.ceil(len(tr) / bs)))
    lossf = nn.SmoothL1Loss()
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr), device=DEV)]
        for s, i in enumerate(range(0, len(perm), bs)):
            b = perm[i:i + bs]
            opt.zero_grad(); out = model(Xt[b], Mt[b])
            loss = lossf(out, yt[b]); loss.backward(); opt.step(); sched.step()
            if DEV == "cuda" and s % 10 == 9:
                torch.cuda.synchronize(); time.sleep(0.03)
        if DEV == "cuda":
            torch.cuda.synchronize(); time.sleep(COOLDOWN)
    def predict(sel):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(sel), 2048):
                b = sel[i:i + 2048]
                out.append(model(Xt[b], Mt[b]).cpu().numpy())
        return np.concatenate(out) * ys + ym
    pv_tr = predict(tr); pv_va = predict(va)
    del Xt, Mt, yt, model, opt
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return 10 ** pv_tr, 10 ** pv_va


def main():
    files = sorted(glob.glob("data/full/*.root"))[:N_FILES]
    t = time.time(); a = load(files)
    print(f"loaded {len(files)} files / {len(a)} clusters in {time.time()-t:.1f}s")
    a = a[quality_mask(a)]
    et = ak.to_numpy(a["sig_flux_eTot"]).astype(np.float64) * 1000.0
    raw = ak.to_numpy(a["total_energy"]).astype(np.float64)
    emax = ak.to_numpy(ak.fill_none(ak.max(a["energy"], axis=1), 0)).astype(np.float64)
    n = len(et); idx = RNG.permutation(n)
    ntr = min(int(0.7 * n), MAX_TRAIN); nva = int(0.15 * n)
    itr, iva = idx[:ntr], idx[ntr:ntr + nva]
    edges = np.geomspace(1000, 150000, 16)
    print(f"clean clusters: {n}  | E_true median {np.median(et)/1000:.2f} GeV")

    cs = calib_loglog(raw[itr], et[itr]); sum_pred = apply_loglog(cs, raw)
    cm = calib_loglog(emax[itr], et[itr]); max_pred = apply_loglog(cm, emax)
    for name, pred in [("sum-E", sum_pred), ("max-cell", max_pred)]:
        E, sig, bias, ns = binned_resolution(pred[iva], et[iva], edges)
        a_, c_ = fit_reso(E / 1000, sig)
        print(f"[{name:8s}] median robustσ={np.median(sig):.3f} | a={a_*100:.1f}%/√E c={c_*100:.1f}%")

    if SKIP_XF:
        print("SKIP_XF set -> baselines only"); return

    feat, mask, es = build_padded(a)
    y = np.log10(et).astype(np.float32)
    sets = {
        "E-only": ["log_e"],
        "E+pos": ["log_e", "rel_x", "rel_y"],
        "E+pos+depth": ["log_e", "log_ef", "log_eb", "rel_x", "rel_y", "rel_dr"],
        "space-time": ["log_e", "log_ef", "log_eb", "rel_x", "rel_y", "rel_dr", "t_front", "t_back"],
    }
    t = time.time()
    for tag, keys in sets.items():
        ptr, pva = run_transformer(stack(feat, keys), mask, y, itr, iva, tag=tag)
        scale = 10 ** np.median(np.log10(et[itr]) - np.log10(ptr))
        pva = pva * scale
        E, sig, bias, ns = binned_resolution(pva, et[iva], edges)
        a_, c_ = fit_reso(E / 1000, sig)
        print(f"[xformer {tag:12s}] median robustσ={np.median(sig):.3f} | a={a_*100:.1f}%/√E c={c_*100:.1f}% | medbias={np.median(bias):+.3f}")
    print(f"transformer block {time.time()-t:.1f}s on {DEV}")
    print("OK")


if __name__ == "__main__":
    main()
