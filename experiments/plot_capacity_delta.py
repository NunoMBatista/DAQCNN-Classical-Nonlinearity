"""Fig. 4: test AUC of each quantum source minus the validation-selected
classical source at the same head, 3 datasets x 5 quantum sources (rows) by
5 heads (columns).

    python experiments/plot_capacity_delta.py [--out STEM]
Writes STEM.pdf and STEM.png (default outputs/figures/capacity_delta).
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import val_selected_margins as val_sel
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["pdf.fonttype"] = 42   # TrueType fonts in the PDF

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="outputs/figures/capacity_delta", help="output stem")
ARGS = ap.parse_args()

# (sweep dir, short label)
DATASETS = [(val_sel.DATASETS[0][1], "Breast"),
            (val_sel.DATASETS[1][1], "Pneumonia"),
            (val_sel.DATASETS[2][1], "Tissue")]
HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]
HEAD_LABELS = ["Linear", "MLP-1", "MLP-2", "CNN-16", "CNN-64"]
QUANT = ["digital_zz_1k", "digital_zz_4k", "analog_zz_1k", "analog_zz_4k", "digital_z_1k"]
QLABELS = ["Dig-ZZ 1k", "Dig-ZZ 4k", "Ana-ZZ 1k", "Ana-ZZ 4k", "Dig-Z 1k"]
CLASS = val_sel.CLASSICAL

M = np.full((len(DATASETS) * len(QUANT), len(HEADS)), np.nan)
row_labels = []
for di, (sweep_dir, _) in enumerate(DATASETS):
    cells = val_sel.load_cells(sweep_dir)
    mean_auc = {k: float(np.mean(c["test_aucs"])) for k, c in cells.items()}
    for ri, q in enumerate(QUANT):
        row_labels.append(QLABELS[ri])
        for ci, h in enumerate(HEADS):
            ref = val_sel.pick(cells, h, CLASS, "val_acc_mean")
            if ref is not None and (h, q) in mean_auc:
                M[di * len(QUANT) + ri, ci] = mean_auc[(h, q)] - mean_auc[(h, ref)]

fig, ax = plt.subplots(figsize=(3.4, 2.75))
norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)
cmap = plt.get_cmap("RdBu_r")
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        if not np.isnan(M[r, c]):
            # white text on dark cells
            rr, gg, bb, _ = cmap(norm(M[r, c]))
            lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
            ax.text(c, r, f"{M[r, c]:+.3f}", ha="center", va="center",
                    fontsize=6.2, color="white" if lum < 0.5 else "black")

# dataset separators and labels
for b in (1, 2):
    ax.axhline(b * len(QUANT) - 0.5, color="black", lw=1.1)
for di, (_, name) in enumerate(DATASETS):
    ax.text(-2.05, di * len(QUANT) + (len(QUANT) - 1) / 2, name, rotation=90,
            ha="center", va="center", fontsize=7.5, fontweight="bold")

ax.set_xticks(range(len(HEADS)))
ax.set_xticklabels(HEAD_LABELS, fontsize=6.5, rotation=30, ha="right")
ax.set_yticks(range(M.shape[0]))
ax.set_yticklabels(row_labels, fontsize=6.0)
ax.tick_params(length=0)

cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.04, pad=0.03)
cb.set_label("Quantum $-$ val-selected classical AUC", fontsize=6.0, rotation=270,
             labelpad=9)
cb.ax.tick_params(labelsize=6.0)

os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
fig.savefig(ARGS.out + ".pdf", bbox_inches="tight")
fig.savefig(ARGS.out + ".png", dpi=170, bbox_inches="tight")
print("wrote", ARGS.out + ".pdf")
