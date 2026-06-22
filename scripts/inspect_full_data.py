from pathlib import Path
import glob
import numpy as np
import awkward as ak
import uproot

DATA = Path("data/full")
files = sorted(glob.glob(str(DATA / "*.root")))
print(f"# files: {len(files)}")
f0 = files[0]
print(f"first file: {Path(f0).name}")

with uproot.open(f0) as f:
    print("top-level keys:", [k for k in f.keys()])
    tree = f["clusters_matched"]
    print("num_entries (file0):", tree.num_entries)
    print("\n=== BRANCHES (name | typename) ===")
    for name in tree.keys():
        br = tree[name]
        print(f"  {name:28s} | {br.typename}")

    arrays = tree.arrays(library="ak")
    print("\n=== FIELD KINDS ===")
    jagged, scalar = [], []
    for name in arrays.fields:
        try:
            ndim = arrays[name].ndim
        except Exception:
            ndim = "?"
        (jagged if ndim != 1 else scalar).append(name)
    print("scalar fields:", scalar)
    print("jagged fields:", jagged)

    print("\n=== SCALAR RANGES (file0) ===")
    for name in scalar:
        try:
            v = ak.to_numpy(arrays[name])
            v = v[np.isfinite(v)]
            if v.size:
                print(f"  {name:28s} min={np.min(v):12.4g} mean={np.mean(v):12.4g} max={np.max(v):12.4g}")
        except Exception as e:
            print(f"  {name:28s} <{e}>")

    print("\n=== JAGGED CELL MULTIPLICITY (file0) ===")
    for name in jagged:
        try:
            n = ak.to_numpy(ak.num(arrays[name], axis=1))
            print(f"  {name:28s} cells/clu: min={n.min()} median={int(np.median(n))} max={n.max()}")
        except Exception as e:
            print(f"  {name:28s} <{e}>")

total = 0
for fp in files:
    with uproot.open(fp) as f:
        total += f["clusters_matched"].num_entries
print(f"\nTOTAL matched clusters across {len(files)} files: {total}")
