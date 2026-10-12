"""Fig. 2: TikZ panels of the 8 topologies. Each panel draws the 3x3 patch
pixels as nodes, with edge width and opacity set by the coupling 1/r^6 of the
topology's atom coordinates on a log scale (couplings below THRESH not drawn).
Writes a standalone .tex file (compile with pdflatex).

    python experiments/gen_atom_topologies_tikz.py [--out FILE]
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.physics.kernel_topologies import get_3x3_kernel_set

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="outputs/figures/atom_topologies.tex")
ARGS = ap.parse_args()

ORDER = ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"]
NCOL = 4                      # panels per row -> 4x2 layout
PX, PY = 1.75, 1.95          # panel pitch (cm) horizontal / vertical
G = 0.5                       # node grid spacing within a panel (cm)
THRESH = 0.008                # smallest coupling V_ij that is drawn
LOGMIN, LOGMAX = -2.0, 1.0    # log10(V) span mapped from ghost line to strong bond


def edges(coords):
    """[(i, j, V)] for pairs with V = 1/r^6 above THRESH. Styles use the absolute
    V, not a per-panel maximum, so grid (kings at 1.5x spacing) draws fainter."""
    out = []
    for i in range(9):
        for j in range(i + 1, 9):
            r = np.linalg.norm(coords[i] - coords[j])
            v = 1.0 / r**6 if r > 0 else 0.0
            if v > THRESH:
                out.append((i, j, v))
    return out


def edge_style(v):
    """Line width (pt) and opacity from log10(V)."""
    t = (np.log10(v) - LOGMIN) / (LOGMAX - LOGMIN)
    t = min(1.0, max(0.0, t))
    return round(0.2 + 1.4 * t, 2), round(0.22 + 0.73 * t, 2)


ks = get_3x3_kernel_set()
lines = [
    r"\documentclass[tikz,border=3pt]{standalone}",
    r"\usepackage{tikz}",
    r"\begin{document}",
    r"\begin{tikzpicture}[x=1cm,y=1cm]",
    r"  \tikzset{atomdot/.style={circle,draw=black,fill=white,line width=0.4pt,"
    r"inner sep=0pt,minimum size=4.2pt}}",
]

for k, name in enumerate(ORDER):
    r, c = divmod(k, NCOL)
    ox, oy = c * PX, -r * PY          # top-left node of the panel

    def pos(p):
        return ox + (p % 3) * G, oy - (p // 3) * G

    lines.append(f"  % --- {name} ---")
    panel_edges = edges(ks[name])
    # atoms with at least one drawn edge (ring's centre has none)
    coupled = {i for i, j, w in panel_edges} | {j for i, j, w in panel_edges}
    for i, j, w in panel_edges:
        lw, op = edge_style(w)
        xi, yi = pos(i); xj, yj = pos(j)
        # pixel on the straight segment between i and j, if any
        ri, ci, rj, cj = i // 3, i % 3, j // 3, j % 3
        mid = None
        if (ri + rj) % 2 == 0 and (ci + cj) % 2 == 0:
            m = ((ri + rj) // 2) * 3 + (ci + cj) // 2
            if m not in (i, j):
                mid = m
        # bend an edge that would pass through a decoupled atom
        path = "to[bend left=30]" if (mid is not None and mid not in coupled) else "--"
        lines.append(f"  \\draw[black,line width={lw}pt,opacity={op}] "
                     f"({xi:.3f},{yi:.3f}) {path} ({xj:.3f},{yj:.3f});")
    for p in range(9):
        x, y = pos(p)
        lines.append(f"  \\node[atomdot] at ({x:.3f},{y:.3f}) {{}};")
    lx = ox + G
    ly = oy - 2 * G - 0.34
    lines.append(f"  \\node[font=\\footnotesize] at ({lx:.3f},{ly:.3f}) {{{name}}};")

lines += [r"\end{tikzpicture}", r"\end{document}", ""]

os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
with open(ARGS.out, "w") as f:
    f.write("\n".join(lines))
print("wrote", ARGS.out)
