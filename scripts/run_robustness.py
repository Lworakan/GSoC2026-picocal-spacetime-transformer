import sys
import json
import argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (build, split, Transformer, DeepSets, train_eval,
                             variant_tokens, EPS)


def build_variant(D, var, ridx, n):
    toks_v = [None] * n
    nc = 7
    for i in ridx:
        tk, nc = variant_tokens(D["raw"][i], var)
        toks_v[i] = tk
    return toks_v, nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--vertex", type=float, default=100.0)
    ap.add_argument("--region", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"device={args.device} seeds={args.seeds}", flush=True)

    repo = Path(__file__).resolve().parent.parent
    files = sorted((repo / "data" / "full").glob("matched_*.root"))[: args.files]
    print(f"building from {len(files)} files ...", flush=True)
    D = build(files, args.window, args.vertex)
    n = len(D["y"]); Et = D["Etrue"]
    ridx = np.flatnonzero(D["region"] == args.region)
    rtr, rva, rte = (ridx[s] for s in split(len(ridx)))
    print(f"kept {n}; R{args.region}={len(ridx)} (test={len(rte)})", flush=True)

    def ms(make, toks, ncont, tr=rtr, va=rva, te=rte):
        vals = []
        for sd in range(args.seeds):
            r = train_eval(make(), toks, D["y"], tr, va, te, Et, args.epochs,
                           args.device, args.batch, seed=sd, ncont=ncont)[0]
            vals.append(r["sigma_eff"])
            print(f"  seed {sd}: {r['sigma_eff']:.4f}", flush=True)
        a = np.array(vals)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                "vals": [round(float(v), 4) for v in vals]}

    d12 = D["tok_seed"][0].shape[1]
    robust = {}
    print("E1 transformer", flush=True)
    robust["E1_transformer_seed"] = ms(lambda: Transformer(d12), D["tok_seed"], 7)
    print("E11 deepsets", flush=True)
    robust["E11_deepsets_seed"] = ms(lambda: DeepSets(d12), D["tok_seed"], 7)
    for name, var in [("E12a_drop_frontback", "drop_fb"), ("E12b_raw_energy", "raw_e"),
                      ("E12c_drop_region_onehot", "drop_onehot"), ("E12d_abs_coords", "abs_coords")]:
        print(name, flush=True)
        toks_v, nc = build_variant(D, var, ridx, n)
        vdim = toks_v[int(ridx[0])].shape[1]
        robust[name] = ms(lambda v=vdim: Transformer(v), toks_v, nc)

    print("controlled E4 (same R3 test set)", flush=True)
    all_train = np.setdiff1d(np.arange(n), rte)
    e4 = {}
    e4["R3only_on_rte"] = ms(lambda: Transformer(d12), D["tok_seed"], 7, tr=rtr)
    e4["allregions_on_rte"] = ms(lambda: Transformer(d12), D["tok_seed"], 7, tr=all_train)

    out = {"config": vars(args), "n_kept": n, "n_region": len(ridx), "n_test_R3": len(rte),
           "robust": robust, "controlled_E4": e4}
    rep = repo / "reports"; rep.mkdir(exist_ok=True)
    (rep / "robustness.json").write_text(json.dumps(out, indent=2))

    e1 = robust["E1_transformer_seed"]["mean"]
    lines = ["# Robustness — multi-seed and controlled E4", "",
             f"config: {vars(args)}", f"R{args.region} test set = {len(rte)} clusters", "",
             f"All over {args.seeds} seeds. E1 baseline mean = {e1:.4f}.", "",
             "| experiment | mean sigma_eff | std | delta vs E1 | significant? |",
             "|---|---|---|---|---|"]
    for k, v in robust.items():
        d = v["mean"] - e1
        sig = "yes" if abs(d) > 2 * (v["std"] + robust["E1_transformer_seed"]["std"]) else "within noise"
        tag = "(baseline)" if k == "E1_transformer_seed" else f"{d:+.4f}"
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {tag} | {'-' if k=='E1_transformer_seed' else sig} |")
    lines += ["", "## Controlled E4 — same R3 test set", "",
              "| model | mean sigma_eff | std |", "|---|---|---|"]
    for k, v in e4.items():
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |")
    da = e4["allregions_on_rte"]["mean"] - e4["R3only_on_rte"]["mean"]
    pooled = e4["allregions_on_rte"]["std"] + e4["R3only_on_rte"]["std"]
    lines += ["", f"all-region vs R3-only on the same test set: delta = {da:+.4f} "
              f"({'significant' if abs(da) > 2 * pooled else 'within noise'}).", ""]
    (rep / "robustness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
