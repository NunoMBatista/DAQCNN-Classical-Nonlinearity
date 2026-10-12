"""Readout of capacity_repeat_val_auc.py: does the selection objective matter?

1. Within run: for each cell, both selections were validated on the same ten
   seeds, so their difference gets a paired t interval.
2. With --run1-log (the stdout of an earlier run of the same script): the
   absolute difference between the two runs' values of the same quantity, a
   reproducibility floor (the non-quantum sources are built from an unseeded,
   shuffled train loader, so runs differ). A within-run effect below it cannot
   be acted on.
Then the best-quantum minus best-classical margin per head under both rules,
and the paper's fixed pair (digital_zz_1k minus random_45) at MLP-1 / MLP-2.

    python experiments/capacity_repeat_readout.py [--run1-log RUN1.log]
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.stats import paired_t_interval

QUANTUM = {"digital_z_1k", "digital_zz_1k", "digital_zz_4k",
           "analog_z_1k", "analog_zz_1k", "analog_zz_4k"}
HEADS = ["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"]


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run1-log", default=None,
                help="stdout of an earlier capacity_repeat_val_auc.py run (enables the floor)")
args = ap.parse_args()

cells = {}
for f in glob.glob("outputs/camera_ready/a8/cells/*/result.json"):
    r = json.load(open(f))
    cells[(r["head"], r["source"])] = r

# ---- reproducibility floor: run 1's log versus this run's JSONs ------------
floor = None
if args.run1_log:
    run1 = {}
    pat = re.compile(r"^\s+(\S+)\s+(\S+)\s+(?:DIFFERENT trial|same trial)\s+"
                     r"AUC\(val_acc-sel\)=([\d.]+)\s+AUC\(val_auc-sel\)=([\d.]+)")
    for line in open(args.run1_log):
        m = pat.match(line)
        if m:
            run1[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))

    floor = []
    for k, r in cells.items():
        if k in run1:
            floor.append(abs(r["auc_sel_acc"] - run1[k][0]))
            floor.append(abs(r["auc_sel_auc"] - run1[k][1]))
    floor.sort()
    n = len(floor)
    print(f"REPRODUCIBILITY FLOOR (same quantity, two independent runs, n={n})")
    print(f"  median |diff| {floor[n//2]:.4f}   90th pct {floor[int(0.9*n)]:.4f}   "
          f"max {floor[-1]:.4f}")
    print(f"  |diff| exceeds 0.005 in {sum(x > 0.005 for x in floor)}/{n} comparisons")

# ---- within-run selection-objective effect ---------------------------------
print(f"\nSELECTION OBJECTIVE, within run (val_auc-selected minus val_acc-selected)")
sig = 0
big = []
for k, r in sorted(cells.items()):
    m, lo, hi, _ = paired_t_interval(r["test_auc_values_sel_auc"], r["test_auc_values_sel_acc"])
    nonzero = not (lo <= 0 <= hi)
    sig += nonzero
    if nonzero:
        big.append((abs(m), k, m, lo, hi))
print(f"  cells whose 95% paired interval excludes zero: {sig}/{len(cells)}")
print(f"  cells where the two rules chose the same trial: "
      f"{sum(r['same_trial'] for r in cells.values())}/{len(cells)}")
big.sort(reverse=True)
print("  largest resolved effects:")
for _, k, m, lo, hi in big[:5]:
    print(f"    {k[0]:10s} {k[1]:12s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]")

# ---- camp margins per head ---------------------------------------------------
print(f"\nCAMP MARGIN (best quantum minus best classical) per head")
print(f"  {'head':10s} {'val_acc-sel':>12s} {'val_auc-sel':>12s} {'shift':>8s} "
      + (f"{'> floor?':>9s}" if floor is not None else ""))
flips = 0
for h in HEADS:
    recs = [(s, r) for (hh, s), r in cells.items() if hh == h]
    def margin(key):
        q = max(r[key] for s, r in recs if s in QUANTUM)
        c = max(r[key] for s, r in recs if s not in QUANTUM)
        return q - c
    a, b = margin("auc_sel_acc"), margin("auc_sel_auc")
    flip = (a > 0) != (b > 0)
    flips += flip
    shift = b - a
    above = ""
    if floor is not None:
        above = f"{'yes' if abs(shift) > floor[int(0.9*n)] else 'NO':>9s}"
    print(f"  {h:10s} {a:+12.4f} {b:+12.4f} {shift:+8.4f} " + above
          + ("   SIGN FLIP" if flip else ""))
if floor is not None:
    print(f"\n  sign flips: {flips}/5, but see the floor column: a flip whose shift")
    print(f"  is below the reproducibility floor is not an effect of the objective.")
else:
    print(f"\n  sign flips: {flips}/5")

# ---- the paper's low-capacity claim: a fixed pair ---------------------------
print(f"\nPAPER'S LOW-CAPACITY CLAIM (digital_zz_1k minus random_45, fixed pair):")
for h in ("mlp1", "mlp2"):
    q = cells[(h, "digital_zz_1k")]
    r45 = cells[(h, "random_45")]
    for key, label in (("test_auc_values_sel_acc", "val_acc-selected"),
                       ("test_auc_values_sel_auc", "val_auc-selected")):
        m, lo, hi, _ = paired_t_interval(q[key], r45[key])
        print(f"  {h:6s} {label:16s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"{'  excludes 0' if not (lo <= 0 <= hi) else ''}")
