"""Fig. 3: test AUC against head capacity, rows = datasets, columns = feature
families, +/-1 SD bands over the 10 seeds, raw pixels as a grey reference. A
gold star marks, at each head, the leader of the validation-selected quantum
and classical representatives (val_selected_margins.py).

    python experiments/plot_capacity_grid.py [--out STEM]
Writes STEM.pdf and STEM.png (default outputs/figures/capacity_sweep_grid).
"""
import os
import numpy as np
import val_selected_margins as val_sel
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType fonts in the PDF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=None,
                help="output stem (default outputs/figures/capacity_sweep_grid)")
ARGS = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HLAB = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]

# scale key -> (linestyle, marker)
SCALE_STYLE = {
    "z":  ((0, (1, 1.4)), "v"),
    "1k": ("solid",       "o"),
    "4k": ((0, (5, 2)),   "s"),
}

# column families: (title, color, [(source, scale)])
FAMILIES = [
    ("Quantum (digital)", "#08306b",
     [("digital_z_1k", "z"), ("digital_zz_1k", "1k"), ("digital_zz_4k", "4k")]),
    ("Quantum (analog)", "#c0392b",
     [("analog_z_1k", "z"), ("analog_zz_1k", "1k"), ("analog_zz_4k", "4k")]),
    ("Random projection", "#bcbd22",
     [("random_9", "z"), ("random_45", "1k"), ("random_180", "4k")]),
    ("Classical nonlinear", "#e67e22",
     [("poly2_45", "z"), ("rff_45", "1k"), ("rff_180", "4k")]),
]

# per-seed JSONs (they carry the validation accuracy the selection needs)
DATASETS = val_sel.DATASETS

# columns 0, 1 = quantum camp; 2, 3 = classical camp
SRC_COL = {}
for ci, (title, color, lines) in enumerate(FAMILIES):
    for src, scale in lines:
        SRC_COL[src] = ci
QUANTUM_SRCS = [s for s, c in SRC_COL.items() if c in (0, 1)]
CLASSICAL_SRCS = [s for s, c in SRC_COL.items() if c in (2, 3)]


def series(cells, src):
    """Per-head mean and SD of test AUC over the 10 validation seeds."""
    means, sds = [], []
    for head in HEADS:
        cell = cells.get((head, src))
        if cell is None:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        vals = np.array(cell["test_aucs"])
        means.append(vals.mean())
        sds.append(vals.std())
    return np.array(means), np.array(sds)


assert sorted(QUANTUM_SRCS) == sorted(val_sel.QUANTUM)
assert sorted(CLASSICAL_SRCS) == sorted(val_sel.CLASSICAL)

data = [(name, val_sel.load_cells(sweep_dir)) for name, sweep_dir in DATASETS]
x = np.arange(len(HEADS))

fig, axes = plt.subplots(3, 4, figsize=(7.1, 3.9), sharex=True)

for ri, (dname, d) in enumerate(data):
    raw_m, _ = series(d, "raw")
    allv = [float(np.mean(cell["test_aucs"])) for cell in d.values()]
    ylo, yhi = min(allv) - 0.020, max(allv) + 0.020

    for ci, (title, color, lines) in enumerate(FAMILIES):
        ax = axes[ri, ci]
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.25, lw=0.5)
        ax.plot(x, raw_m, color="#999999", ls=(0, (1, 1)), lw=1.0, alpha=0.7,
                marker="x", markersize=3, zorder=1)
        any_data = False
        for src, scale in lines:
            ls, mk = SCALE_STYLE[scale]
            m, s = series(d, src)
            if np.isnan(m).all():
                continue
            any_data = True
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0, zorder=2)
            ax.plot(x, m, color=color, ls=ls, lw=1.6, marker=mk, markersize=3.6, zorder=3)
        if not any_data:
            ax.text(0.5, 0.5, "not run\n(digital-only)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#999999", style="italic")
        if ri == 0:
            ax.set_title(title, fontsize=8.5, color=color, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{dname}\nTest AUC", fontsize=8)
        ax.tick_params(labelsize=6.5)

    # star on the better of the two validation-selected camp representatives
    for hi, head in enumerate(HEADS):
        def camp_rep(srcs):
            src = val_sel.pick(d, head, srcs, "val_acc_mean")
            return (series(d, src)[0][hi], src) if src is not None else None
        bq, bc = camp_rep(QUANTUM_SRCS), camp_rep(CLASSICAL_SRCS)
        if bq is None or bc is None:
            continue
        quantum_wins = bq[0] >= bc[0]
        best_v, best_src = bq if quantum_wins else bc
        ci = SRC_COL[best_src]
        axes[ri, ci].plot(hi, best_v, marker="*", ms=8.5, mfc="gold", mec="black",
                          mew=0.6, ls="none", zorder=6)

for ax in axes[2]:
    ax.set_xticks(x)
    ax.set_xticklabels(HLAB, fontsize=7, rotation=90, ha="center", va="top")

handles = [
    Line2D([0], [0], color="#444", ls=SCALE_STYLE["z"][0], marker="v", ms=4,
           label=r"$\langle Z\rangle$ / $\times$9 / poly-2"),
    Line2D([0], [0], color="#444", ls="solid", marker="o", ms=4, label="single-kernel, 45-dim"),
    Line2D([0], [0], color="#444", ls=(0, (5, 2)), marker="s", ms=4, label="four-kernel, 180-dim"),
    Line2D([0], [0], color="#999", ls=(0, (1, 1)), marker="x", ms=4, label="raw pixels"),
    Line2D([0], [0], color="black", marker="*", mfc="gold", mec="black", ms=8,
           ls="none", label="val-selected leader"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
           frameon=False, handlelength=2.6, bbox_to_anchor=(0.5, -0.03))

fig.tight_layout(rect=[0, 0.05, 1, 1.0])
out = ARGS.out or os.path.join(ROOT, "outputs/figures/capacity_sweep_grid")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight", dpi=175)
print(f"wrote {out}.pdf / .png")
