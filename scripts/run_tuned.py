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
    def __init__(self, in_dim, d_model, nhead, layers, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x, m):
        h = self.enc(self.embed(x), src_key_padding_mask=~m)
        w = m.unsqueeze(-1).float()
        return self.head((h * w).sum(1) / w.sum(1).clamp(min=1))


def train_tuned(toks, y, ttr, rva, rte, Et, dev, cfg, seed):
    torch.manual_seed(seed)
    cont = np.concatenate([toks[i][:, :7] for i in ttr], 0)
    mean = cont.mean(0); std = cont.std(0) + EPS

    def loader(idx, sh):
        return DataLoader(TokenDS([toks[i] for i in idx], y[idx], mean, std, 7),
                          batch_size=cfg["batch"], shuffle=sh, collate_fn=collate)

    model = TunedTransformer(toks[ttr[0]].shape[1], cfg["d_model"], cfg["nhead"],
                             cfg["layers"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    dl_tr, dl_va = loader(ttr, True), loader(rva, False)

    def vloss():
        model.eval(); t = 0.0; n = 0
        with torch.no_grad():
            for X, m, yb in dl_va:
                t += nn.functional.mse_loss(model(X.to(dev), m.to(dev)), yb.to(dev)).item(); n += 1
        return t / max(n, 1)

    best = float("inf"); best_state = None; wait = 0; ran = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for X, m, yb in dl_tr:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X.to(dev), m.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
        sched.step(); ran = ep + 1
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
    return resolution(np.exp(a * pt + b), Et[rte])["sigma_eff"], ran


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--region", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    dev = torch.device(args.device)
    cfg = {"d_model": 96, "nhead": 4, "layers": 3, "dropout": 0.1,
           "epochs": args.epochs, "patience": args.patience, "lr": 3e-4, "wd": 1e-4, "batch": 128}
    print(f"device={args.device} window={args.window} seeds={args.seeds}", flush=True)

    repo = Path(__file__).resolve().parent.parent
    files = sorted((repo / "data" / "full").glob("matched_*.root"))[: args.files]
    D = build(files, args.window, 100.0)
    y = D["y"]; Et = D["Etrue"]; toks = D["tok_seed"]
    ridx = np.flatnonzero(D["region"] == args.region)
    rtr, rva, rte = (ridx[s] for s in split(len(ridx)))
    ttr = np.setdiff1d(np.setdiff1d(np.arange(len(y)), rte), rva)
    print(f"train(all-region)={len(ttr)} R{args.region}_test={len(rte)}", flush=True)

    vals = []
    for s in range(args.seeds):
        se, ran = train_tuned(toks, y, ttr, rva, rte, Et, dev, cfg, s)
        vals.append(se)
        print(f"  seed {s}: sigma_eff={se:.4f} (epochs {ran})", flush=True)
    vals = np.array(vals)

    gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(D["agg"][rtr], y[rtr])
    bdt = resolution(np.exp(gb.predict(D["agg"][rte])), Et[rte])["sigma_eff"]
    te = None
    ca, cb = np.polyfit(np.log(D["total_energy"][rtr] + EPS), y[rtr], 1)
    te = resolution(np.exp(ca * np.log(D["total_energy"][rte] + EPS) + cb), Et[rte])["sigma_eff"]

    mean = float(vals.mean()); std = float(vals.std())
    bar = min(bdt, te)
    delta = mean - bar
    pooled = std
    out = {"config": {**cfg, "window": args.window, "all_region": True, "seeds": args.seeds},
           "tuned_transformer": {"mean": round(mean, 4), "std": round(std, 4),
                                 "vals": [round(v, 4) for v in vals.tolist()]},
           "bar_BDT": round(bdt, 4), "bar_total_energy": round(te, 4),
           "delta_vs_bar": round(delta, 4),
           "sigma": round(delta / pooled, 1) if pooled else 0.0,
           "verdict": "WINS (firm)" if delta < -2 * pooled else
                      ("wins (mean, not firm)" if delta < 0 else "does not beat bar")}
    rep = repo / "reports"; rep.mkdir(exist_ok=True)
    (rep / "tuned_3x3_allregion.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
