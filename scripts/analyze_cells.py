import sys
import json
import argparse
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import derive_geom, PITCH

RKEYS = ["cell_x", "cell_y", "energy", "imodx", "jmody"]
RNAMES = ["R0_15mm", "R1_30mm", "R2_40mm", "R3_60mm", "R4_120mm"]


def collect(files, vertex_max):
    ncells = []
    region = []
    dist_mm = []
    dist_pitch = []
    dist_region = []
    cover = []
    cover_region = []
    KMAX = 40
    for path in files:
        with uproot.open(path) as f:
            a = f["clusters_matched"].arrays(RKEYS + ["sig_flux_prod_vertex_z"], library="ak")
        vz = ak.to_numpy(a["sig_flux_prod_vertex_z"]).astype(float)
        keep = np.flatnonzero(vz < vertex_max)
        for i in keep:
            c = {k: np.asarray(ak.to_numpy(a[k][i])).astype(float) for k in RKEYS}
            if len(c["energy"]) == 0:
                continue
            pitch, mod, rel_x, rel_y, rel_dr, seed = derive_geom(c)
            reg = int(mod[seed])
            ncells.append(len(c["energy"]))
            region.append(reg)
            dist_mm.extend(rel_dr.tolist())
            dist_pitch.extend((rel_dr / pitch[seed]).tolist())
            dist_region.extend([reg] * len(rel_dr))
            order = np.argsort(rel_dr)
            e_sorted = c["energy"][order]
            tot = e_sorted.sum() + 1e-9
            cum = np.cumsum(e_sorted) / tot
            padded = np.ones(KMAX)
            m = min(len(cum), KMAX)
            padded[:m] = cum[:m]
            cover.append(padded)
            cover_region.append(reg)
    return (np.array(ncells), np.array(region), np.array(dist_mm), np.array(dist_pitch),
            np.array(dist_region), np.array(cover), np.array(cover_region), KMAX)


def summarize(ncells, region, cover, cover_region, KMAX):
    out = {"overall": {}, "per_region": {}}
    windows = {"3x3 (k<=9)": 9, "5x5 (k<=25)": 25, "k=13": 13}
    out["overall"]["n_clusters"] = int(len(ncells))
    out["overall"]["ncells_median"] = float(np.median(ncells))
    out["overall"]["ncells_mean"] = round(float(ncells.mean()), 2)
    out["overall"]["ncells_p95"] = float(np.percentile(ncells, 95))
    out["overall"]["ncells_max"] = int(ncells.max())
    for label, k in windows.items():
        out["overall"]["cover_" + label] = round(float(cover[:, k - 1].mean()), 4)
    for r in range(len(PITCH)):
        m = region == r
        cm = cover_region == r
        if m.sum() == 0:
            continue
        d = {"n_clusters": int(m.sum()), "ncells_median": float(np.median(ncells[m])),
             "ncells_mean": round(float(ncells[m].mean()), 2),
             "ncells_p95": float(np.percentile(ncells[m], 95)),
             "ncells_max": int(ncells[m].max())}
        for label, k in windows.items():
            d["cover_" + label] = round(float(cover[cm][:, k - 1].mean()), 4)
        out["per_region"][RNAMES[r]] = d
    return out


def figures(ncells, region, dist_pitch, dist_region, cover, cover_region, KMAX, repo):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    present = [r for r in range(len(PITCH)) if (region == r).sum() > 0]

    figA = go.Figure()
    maxc = int(np.percentile(ncells, 99))
    edges = np.arange(0.5, maxc + 20, 4)
    ctr = (edges[:-1] + edges[1:]) / 2
    for r in present:
        v = ncells[region == r]
        h, _ = np.histogram(v, bins=edges)
        figA.add_trace(go.Bar(x=ctr, y=h, name=RNAMES[r], marker_color=palette[r], opacity=0.75))
    figA.update_layout(barmode="overlay", template="plotly_white", width=900, height=460,
                       title="Number of non-empty cells per cluster (by seed region)",
                       xaxis_title="cells per cluster", yaxis_title="clusters",
                       legend_title="region")
    figA.write_html(str(repo / "reports" / "cell_ncells.html"))

    figB = go.Figure()
    dedges = np.arange(0, 6.01, 0.2)
    dctr = (dedges[:-1] + dedges[1:]) / 2
    for r in present:
        v = dist_pitch[dist_region == r]
        h, _ = np.histogram(v, bins=dedges, density=True)
        figB.add_trace(go.Bar(x=dctr, y=h, name=RNAMES[r], marker_color=palette[r], opacity=0.6))
    figB.update_layout(barmode="overlay", template="plotly_white", width=900, height=460,
                       title="Cell distance to seed (in seed-pitch units, by region)",
                       xaxis_title="distance / seed pitch", yaxis_title="density",
                       legend_title="region")
    figB.add_vline(x=1, line_dash="dot", line_color="#888", annotation_text="3x3 edge")
    figB.add_vline(x=2, line_dash="dash", line_color="#888", annotation_text="5x5 edge")
    figB.write_html(str(repo / "reports" / "cell_distance.html"))

    figC = go.Figure()
    ks = np.arange(1, KMAX + 1)
    for r in present:
        cm = cover_region == r
        mean_cov = cover[cm].mean(0)
        figC.add_trace(go.Scatter(x=ks, y=mean_cov, mode="lines+markers",
                                  name=RNAMES[r], line=dict(color=palette[r])))
    figC.add_trace(go.Scatter(x=ks, y=cover.mean(0), mode="lines",
                              name="all", line=dict(color="black", width=3, dash="dot")))
    figC.add_vline(x=9, line_dash="dot", line_color="#888")
    figC.add_vline(x=25, line_dash="dash", line_color="#888")
    figC.add_hline(y=0.99, line_dash="dot", line_color="crimson",
                   annotation_text="99%", annotation_position="top left")
    figC.add_annotation(x=27, y=0.75, showarrow=False, align="left",
                        text="···· vertical line 1:  3x3 = 9 cells<br>– – – vertical line 2:  5x5 = 25 cells",
                        bgcolor="rgba(255,255,255,0.75)", bordercolor="#ccc", borderwidth=1,
                        font=dict(size=12, color="#555"))
    figC.update_layout(template="plotly_white", width=900, height=480,
                       title="Energy captured vs number of nearest cells kept<br><sub>decision curve: how many cells to keep</sub>",
                       xaxis_title="nearest cells kept (sorted by distance to seed)",
                       yaxis_title="mean fraction of cluster energy captured",
                       legend_title="region", yaxis_range=[0.6, 1.005])
    figC.write_html(str(repo / "reports" / "cell_coverage.html"))

    for fig, name in [(figA, "cell_ncells"), (figB, "cell_distance"), (figC, "cell_coverage")]:
        try:
            fig.write_image(str(repo / "reports" / (name + ".png")), scale=2)
        except Exception:
            pass

    pd = {"regions": {RNAMES[r]: r for r in present}, "colors": palette,
          "ncells": {"centers": ctr.tolist(),
                     "counts": {RNAMES[r]: np.histogram(ncells[region == r], bins=edges)[0].tolist() for r in present}},
          "distance": {"centers": dctr.tolist(),
                       "density": {RNAMES[r]: np.histogram(dist_pitch[dist_region == r], bins=dedges, density=True)[0].tolist() for r in present}},
          "coverage": {"ks": ks.tolist(),
                       "mean": {RNAMES[r]: cover[cover_region == r].mean(0).tolist() for r in present},
                       "all": cover.mean(0).tolist()}}
    (repo / "reports" / "cell_selection_plotdata.json").write_text(json.dumps(pd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--vertex", type=float, default=100.0)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    files = sorted((repo / "data" / "full").glob("matched_*.root"))[: args.files]
    print(f"reading {len(files)} files, vertex<{args.vertex} ...", flush=True)
    (ncells, region, dist_mm, dist_pitch, dist_region, cover, cover_region, KMAX) = collect(files, args.vertex)
    summ = summarize(ncells, region, cover, cover_region, KMAX)
    (repo / "reports" / "cell_selection_stats.json").write_text(json.dumps(summ, indent=2))
    figures(ncells, region, dist_pitch, dist_region, cover, cover_region, KMAX, repo)
    print(json.dumps(summ, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
