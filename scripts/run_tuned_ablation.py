import sys
import json
import copy
import argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import build, split, resolution, collate, TokenDS, EPS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingRegressor


class TunedTransformer(nn.Module):
    def __init__(self, in_dim, d, nhead, layers, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4 * d, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x, m):
        h = self.enc(self.embed(x), src_key_padding_mask=~m)
        w = m.unsqueeze(-1).float()
        return self.head((h * w).sum(1) / w.sum(1).clamp(min=1))


class TunedDeepSets(nn.Module):
    def __init__(self, in_dim, d, dropout):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x, m):
        h = self.phi(x); w = m.unsqueeze(-1).float()
        return self.head((h * w).sum(1) / w.sum(1).clamp(min=1))


def train_cfg(make_model, train_idx, toks, y, rva, rte, Et, dev, cfg, seed):
    torch.manual_seed(seed)
    cont = np.concatenate([toks[i][:, :7] for i in train_idx], 0)
    mean = cont.mean(0); std = cont.std(0) + EPS

    def loader(idx, sh):
        return DataLoader(TokenDS([toks[i] for i in idx], y[idx], mean, std, 7),
                          batch_size=cfg["batch"], shuffle=sh, collate_fn=collate)

    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    dl_tr, dl_va = loader(train_idx, True), loader(rva, False)

    def vloss():
        model.eval(); t = 0.0; n = 0
        with torch.no_grad():
            for X, m, yb in dl_va:
                t += nn.functional.mse_loss(model(X.to(dev), m.to(dev)), yb.to(dev)).item(); n += 1
        return t / max(n, 1)

    best = float("inf"); best_state = None; wait = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for X, m, yb in dl_tr:
            opt.zero_grad()
            nn.functional.mse_loss(model(X.to(dev), m.to(dev)), yb.to(dev)).backward(); opt.step()
        sched.step()
        v = vloss()
        if v < best - 1e-4:
            best = v; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)

    def predict(idx):
        model.eval(); out = []
        with torch.no_grad():
            for X, m, _ in loader(idx, False):
                out.append(model(X.to(dev), m.to(dev)).cpu().numpy().ravel())
        return np.concatenate(out)

    pv, pt = predict(rva), predict(rte)
    a, b = np.polyfit(pv, y[rva], 1)
    return resolution(np.exp(a * pt + b), Et[rte])["sigma_eff"]


def multiseed(make_model, train_idx, toks, y, rva, rte, Et, dev, cfg, seeds, label):
    vals = []
    for s in range(seeds):
        v = train_cfg(make_model, train_idx, toks, y, rva, rte, Et, dev, cfg, s)
        vals.append(v)
        print(f"  {label} seed {s}: {v:.4f}", flush=True)
    a = np.array(vals)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "vals": [round(v, 4) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    dev = torch.device(args.device)
    cfg = {"d": 96, "nhead": 4, "layers": 3, "dropout": 0.1,
           "epochs": args.epochs, "patience": args.patience, "lr": 3e-4, "wd": 1e-4, "batch": 128}

    repo = Path(__file__).resolve().parent.parent
    files = sorted((repo / "data" / "full").glob("matched_*.root"))[: args.files]
    D = build(files, 3, 100.0)
    y = D["y"]; Et = D["Etrue"]; toks = D["tok_seed"]
    ridx = np.flatnonzero(D["region"] == 3)
    rtr, rva, rte = (ridx[s] for s in split(len(ridx)))
    ttr = np.setdiff1d(np.setdiff1d(np.arange(len(y)), rte), rva)
    in_dim = toks[int(ridx[0])].shape[1]
    print(f"train_all={len(ttr)} train_r3={len(rtr)} test={len(rte)}", flush=True)

    mk_tf = lambda: TunedTransformer(in_dim, cfg["d"], cfg["nhead"], cfg["layers"], cfg["dropout"])
    mk_ds = lambda: TunedDeepSets(in_dim, cfg["d"], cfg["dropout"])

    res = {}
    res["B_transformer_r3only"] = multiseed(mk_tf, rtr, toks, y, rva, rte, Et, dev, cfg, args.seeds, "B")
    res["C_deepsets_allregion"] = multiseed(mk_ds, ttr, toks, y, rva, rte, Et, dev, cfg, args.seeds, "C")

    a_path = repo / "reports" / "tuned_3x3_allregion.json"
    if a_path.exists():
        A = json.loads(a_path.read_text())["tuned_transformer"]
        res["A_transformer_allregion"] = {"mean": A["mean"], "std": A["std"], "vals": A["vals"]}

    gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(D["agg"][rtr], y[rtr])
    bdt = resolution(np.exp(gb.predict(D["agg"][rte])), Et[rte])["sigma_eff"]

    out = {"config": {**cfg, "window": 3, "seeds": args.seeds}, "bar_BDT": round(bdt, 4),
           "ablation": res}
    (repo / "reports" / "tuned_ablation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
