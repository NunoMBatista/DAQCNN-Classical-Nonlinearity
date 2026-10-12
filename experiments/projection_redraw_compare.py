"""Compare the frozen-draw and per-seed-draw projection cells of projection_redraw.py.

Per cell: mean and SD under each mode, sd_ratio (per-seed SD / frozen SD; above
1 means the published spread understates the variability) and mean_shift
(per-seed mean minus frozen mean, with a paired t interval: both modes share
the training seed at each index).

    python experiments/projection_redraw_compare.py --dataset breast_mnist
Reads outputs/camera_ready/a12/<dataset>_{frozen,varying}/, writes
outputs/camera_ready/a12_compare_<dataset>.csv.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))
from src.utils.stats import paired_t_interval


def load_mode(root):
    """(head, source) -> list of test AUCs ordered by seed, for one mode."""
    cells = {}
    for cell_dir in sorted(glob.glob(str(root / "*__*"))):
        head, _, source = os.path.basename(cell_dir).partition("__")
        recs = []
        for f in sorted(glob.glob(os.path.join(cell_dir, "seed_*.json")),
                        key=lambda p: int(os.path.basename(p)[5:-5])):
            recs.append(json.load(open(f)))
        if recs:
            cells[(head, source)] = [r["test_auc"] for r in recs]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_mnist")
    ap.add_argument("--dir", default="outputs/camera_ready/a12")
    args = ap.parse_args()

    base = ROOT / args.dir
    frozen = load_mode(base / f"{args.dataset}_frozen")
    varying = load_mode(base / f"{args.dataset}_varying")
    shared = sorted(set(frozen) & set(varying))
    if not shared:
        raise SystemExit(f"no cells with both modes under {base}")

    # an SD ratio from two samples of ten is weak: print the 95% band of a true ratio of 1
    n = len(next(iter(frozen.values())))
    f_lo = stats.f.ppf(0.975, n - 1, n - 1)
    f_hi = stats.f.ppf(0.025, n - 1, n - 1)
    print(f"=== {args.dataset}: {len(shared)} cells with both modes, "
          f"{n} seeds each ===")
    print(f"an sd_ratio of exactly 1 has 95% CI "
          f"[{math.sqrt(1/f_lo):.2f}, {math.sqrt(1/f_hi):.2f}] at this n: "
          f"ratios inside that band are not evidence of anything\n")
    print(f"{'head':<11} {'source':<12} {'frozen':>16} {'varying':>16} "
          f"{'sd_ratio':>9} {'mean_shift':>11} {'shift 95% CI':>20}")
    ratios, shifts, rows, real_shift = [], [], [], []
    for head, source in shared:
        f, v = frozen[(head, source)], varying[(head, source)]
        fm, fs = statistics.mean(f), statistics.stdev(f)
        vm, vs = statistics.mean(v), statistics.stdev(v)
        ratio = vs / fs if fs > 0 else float("nan")
        dm, lo, hi, _ = paired_t_interval(v, f)
        nonzero = not (lo <= 0 <= hi)
        if nonzero:
            real_shift.append((head, source, dm))
        ratios.append(ratio)
        shifts.append(dm)
        rows.append((head, source, fm, fs, vm, vs, ratio, dm, lo, hi, len(f), len(v)))
        print(f"{head:<11} {source:<12} {fm:>8.4f}+-{fs:<6.4f} {vm:>8.4f}+-{vs:<6.4f} "
              f"{ratio:>9.2f} {dm:>+11.4f} [{lo:>+7.4f},{hi:>+7.4f}]"
              + ("  *" if nonzero else ""))

    wider = sum(1 for r in ratios if r > 1)
    print(f"\nspread widens in {wider}/{len(ratios)} cells; "
          f"median sd_ratio {statistics.median(ratios):.2f}")
    print(f"mean shift: median {statistics.median(shifts):+.4f}, "
          f"{sum(1 for s in shifts if s > 0)}/{len(shifts)} positive "
          f"(positive = the published frozen draw was unlucky for that cell)")
    print(f"shifts whose paired interval excludes zero: {len(real_shift)}/{len(rows)}"
          + ("".join(f"\n   {h} {s}: {d:+.4f}" for h, s, d in real_shift)
             if real_shift else ""))

    out = ROOT / "outputs" / "camera_ready" / f"a12_compare_{args.dataset}.csv"
    with open(out, "w") as fh:
        fh.write("head,source,frozen_mean,frozen_sd,varying_mean,varying_sd,"
                 "sd_ratio,mean_shift,shift_ci_lo,shift_ci_hi,n_frozen,n_varying\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]}," + ",".join(f"{x:.6f}" for x in r[2:10])
                     + f",{r[10]},{r[11]}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
