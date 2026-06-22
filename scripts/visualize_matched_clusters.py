"""Create neutral event displays from the provided matched-cluster simulation."""

from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


REPO = Path(__file__).resolve().parents[1]
INPUT_FILES = sorted((REPO / "data" / "full").glob("matched_*.root"))
OUTPUT_DIR = REPO / "figures"

BRANCHES = [
    "event",
    "x_cluster",
    "y_cluster",
    "sig_flux_entry_x",
    "sig_flux_entry_y",
    "sig_flux_eTot",
    "cell_x",
    "cell_y",
    "energy",
]


def load_arrays(input_file: Path) -> ak.Array:
    """Load the branches needed for one event display."""
    with uproot.open(input_file, handler=uproot.source.file.MemmapSource) as root_file:
        return root_file["clusters_matched"].arrays(BRANCHES, library="ak")


def save_cluster_display(
    arrays: ak.Array,
    input_file: Path,
    index: int,
    output_name: str,
    title: str,
    truth_indices: np.ndarray | list[int] | None = None,
) -> None:
    """Plot stored cell readout coordinates and reconstruction positions."""
    if truth_indices is None:
        truth_indices = [index]

    cell_x = np.asarray(arrays.cell_x[index])
    cell_y = np.asarray(arrays.cell_y[index])
    energy = np.asarray(arrays.energy[index])

    point_size = float(np.clip(4500 / len(cell_x), 10, 220))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    points = ax.scatter(
        cell_x,
        cell_y,
        c=energy,
        s=point_size,
        marker="o",
        cmap="viridis",
        edgecolors="black",
        linewidths=0.35,
        label="Cell readout positions",
    )
    fig.colorbar(points, ax=ax, label="Cell energy [MeV]")

    ax.scatter(
        float(arrays.x_cluster[index]),
        float(arrays.y_cluster[index]),
        marker="+",
        s=260,
        linewidths=3,
        color="red",
        label="Reconstructed cluster position",
        zorder=4,
    )

    colors = ["tab:orange", "tab:blue", "tab:purple", "tab:brown"]
    for order, truth_index in enumerate(truth_indices):
        label = "Truth photon entry"
        if len(truth_indices) > 1:
            label = f"Truth photon entry {order + 1}"
        ax.scatter(
            float(arrays.sig_flux_entry_x[truth_index]),
            float(arrays.sig_flux_entry_y[truth_index]),
            marker="x",
            s=170,
            linewidths=3,
            color=colors[order % len(colors)],
            label=label,
            zorder=4,
        )

    span = max(float(np.ptp(cell_x)), float(np.ptp(cell_y)), 1.0)
    margin = 0.08 * span
    ax.set_xlim(float(np.min(cell_x) - margin), float(np.max(cell_x) + margin))
    ax.set_ylim(float(np.min(cell_y) - margin), float(np.max(cell_y) + margin))
    ax.set_xlabel("ECAL x [mm]")
    ax.set_ylabel("ECAL y [mm]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        0.025,
        f"Source: {input_file.name} — provided Geant4 simulation",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.11)
    fig.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Example 1: a true single-entry event from the first ROOT file.
single_file = INPUT_FILES[0]
single_arrays = load_arrays(single_file)
single_events = np.asarray(single_arrays.event)
unique_events, event_counts = np.unique(single_events, return_counts=True)
single_event_ids = unique_events[event_counts == 1]
single_index = int(np.flatnonzero(single_events == single_event_ids[0])[0])
save_cluster_display(
    arrays=single_arrays,
    input_file=single_file,
    index=single_index,
    output_name="cluster_example_single_entry.png",
    title=(
        f"Single-entry event — event {single_events[single_index]}, "
        f"entry {single_index}"
    ),
)

# Example 2: the largest cell window in the middle ROOT file.
high_granularity_file = INPUT_FILES[len(INPUT_FILES) // 2]
high_granularity_arrays = load_arrays(high_granularity_file)
high_granularity_events = np.asarray(high_granularity_arrays.event)
cell_counts = np.asarray(ak.num(high_granularity_arrays.energy, axis=1))
high_granularity_index = int(np.argmax(cell_counts))
save_cluster_display(
    arrays=high_granularity_arrays,
    input_file=high_granularity_file,
    index=high_granularity_index,
    output_name="cluster_example_high_granularity.png",
    title=(
        f"Large cell window — event {high_granularity_events[high_granularity_index]}, "
        f"entry {high_granularity_index}, {cell_counts[high_granularity_index]} cells"
    ),
)

# Example 3: an identical reconstruction shared by truth entries in the last ROOT file.
shared_file = INPUT_FILES[-1]
shared_arrays = load_arrays(shared_file)
shared_events = np.asarray(shared_arrays.event)
shared_x = np.asarray(shared_arrays.x_cluster)
shared_y = np.asarray(shared_arrays.y_cluster)
keys = np.rec.fromarrays([shared_events, shared_x, shared_y], names="event,x,y")
_, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
for shared_group in np.flatnonzero(counts >= 2):
    shared_indices = np.flatnonzero(inverse == shared_group)
    reference = int(shared_indices[0])
    if all(
        ak.almost_equal(shared_arrays.energy[reference], shared_arrays.energy[index])
        for index in shared_indices[1:]
    ):
        break
else:
    raise RuntimeError(f"No identical shared reconstruction found in {shared_file.name}")

save_cluster_display(
    arrays=shared_arrays,
    input_file=shared_file,
    index=reference,
    output_name="cluster_example_shared_reconstruction.png",
    title=(
        f"Shared reconstructed cluster — event {shared_events[reference]}, "
        f"{len(shared_indices)} truth-photon entries"
    ),
    truth_indices=shared_indices,
)

for path in sorted(OUTPUT_DIR.glob("cluster_example_*.png")):
    print(path)
