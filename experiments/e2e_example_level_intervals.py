"""Bootstrap over test examples for the end-to-end contrasts on BreastMNIST and
PneumoniaMNIST, from the per-example test probabilities of e2e_val_reinference.py
(and of the frozen-filter Random rows). Three 95% intervals per contrast,
A minus B test AUC:
  seeds     paired t over the ten seeds, test split fixed (retraining only)
  examples  percentile bootstrap over test examples, models fixed (test sampling only)
  both      seeds and examples resampled together
Checks first: a model against itself gives [0, 0], and the bootstrap statistic
on the full index set equals the seed-level mean difference.

    python experiments/e2e_example_level_intervals.py
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from paired_seed_intervals import E2E_CONTRASTS          # noqa: E402
sys.path.insert(0, ROOT)
from src.utils.stats import example_bootstrap_intervals  # noqa: E402

REINF = os.path.join(ROOT, "outputs", "camera_ready", "e2e_val_reinference")
FIXED = os.path.join(ROOT, "outputs", "camera_ready", "e2e_random_fixed")
OUT = os.path.join(ROOT, "outputs", "camera_ready", "e2e_example_level_intervals.csv")
FIXED_ROWS = {(d, m) for d in ("BreastMNIST", "PneumoniaMNIST") for m in ("Random x45", "Random x180")}
N_SEEDS = 10
B = 2000

# the validation-selected pairs of val_selected_e2e.py --random-fixed
SELECTOR_CONTRASTS = [
    ("S-breast-valauc", "BreastMNIST", "Digital-ZZ 4k", "RFF x45", "ten-seed val-AUC-selected representatives"),
    ("S-breast-valacc", "BreastMNIST", "Digital-ZZ 1k", "Random x45", "ten-seed val-acc-selected representatives"),
    ("S-pneumonia-valauc", "PneumoniaMNIST", "Analog-ZZ 4k", "Raw pixels", "ten-seed val-AUC-selected representatives"),
    ("S-pneumonia-valacc", "PneumoniaMNIST", "Analog-ZZ 4k", "Trainable x180", "ten-seed val-acc-selected representatives"),
]


def load_scores(dataset, model):
    """Positive-class test scores, shape (N_SEEDS, n_test), and the labels."""
    scores, labels = [], None
    if (dataset, model) in FIXED_ROWS:
        for k in range(N_SEEDS):
            z = np.load(os.path.join(FIXED, f"{dataset}__{model.replace(' ', '_')}", f"seed_{k}", "test_preds.npz"))
            scores.append(np.asarray(z["probs"])[:, 1])
            lab = np.asarray(z["labels"]).reshape(-1)
            assert labels is None or np.array_equal(labels, lab)
            labels = lab
    else:
        z = np.load(os.path.join(REINF, f"{dataset}__{model.replace(' ', '_').replace('/', '_')}_preds.npz"))
        for k in range(N_SEEDS):
            scores.append(np.asarray(z[f"seed_{k}_test_probs"])[:, 1])
            lab = np.asarray(z[f"seed_{k}_test_labels"]).reshape(-1)
            assert labels is None or np.array_equal(labels, lab)
            labels = lab
    return np.stack(scores).astype(np.float64), labels.astype(np.int64)


def auc_matrix(scores, labels, idx):
    """AUC of every seed on the example subset idx, shape (N_SEEDS,)."""
    y = labels[idx]
    return np.array([roc_auc_score(y, s[idx]) for s in scores])


def run_contrast(cid, dataset, a, b, sentence, rng):
    sa, la = load_scores(dataset, a)
    sb, lb = load_scores(dataset, b)
    assert np.array_equal(la, lb), (cid, "labels differ between arms")
    full = np.arange(len(la))
    d_full = auc_matrix(sa, la, full) - auc_matrix(sb, la, full)
    m, lo_s, hi_s, lo_e, hi_e, lo_b, hi_b = example_bootstrap_intervals(
        d_full, lambda idx: auc_matrix(sa, la, idx) - auc_matrix(sb, la, idx), len(la), rng, B)
    return dict(contrast=cid, dataset=dataset, arm_a=a, arm_b=b, n_test=len(la), n_pos=int(la.sum()),
                mean_diff=m, seeds_lo=lo_s, seeds_hi=hi_s, examples_lo=lo_e, examples_hi=hi_e,
                both_lo=lo_b, both_hi=hi_b,
                width_ratio_examples_over_seeds=(hi_e - lo_e) / (hi_s - lo_s),
                nonzero_seeds=not (lo_s <= 0 <= hi_s), nonzero_examples=not (lo_e <= 0 <= hi_e),
                nonzero_both=not (lo_b <= 0 <= hi_b), sentence=sentence)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rng = np.random.default_rng(0)

    sa, la = load_scores("BreastMNIST", "Digital-ZZ 4k")
    full = np.arange(len(la))
    self_diff = auc_matrix(sa, la, full) - auc_matrix(sa, la, full)
    assert np.all(self_diff == 0.0)
    _, _, _, lo_e, hi_e, lo_b, hi_b = example_bootstrap_intervals(
        self_diff, lambda idx: auc_matrix(sa, la, idx) - auc_matrix(sa, la, idx), len(la),
        np.random.default_rng(1), B)
    assert lo_e == hi_e == 0.0 and lo_b == hi_b == 0.0, (lo_e, hi_e, lo_b, hi_b)
    sb, lb = load_scores("BreastMNIST", "Random x45")
    d_full = (auc_matrix(sa, la, full) - auc_matrix(sb, lb, full)).mean()
    d_stat = (auc_matrix(sa, la, full) - auc_matrix(sb, lb, full)).mean()   # bootstrap statistic, full index set
    assert abs(d_full - d_stat) < 1e-12
    print(f"controls OK: self-contrast intervals are [0,0]; full-index statistic = seed-mean diff = {d_full:+.4f}")

    rows = []
    contrasts = [c for c in E2E_CONTRASTS if c[1] != "TissueMNIST"] + SELECTOR_CONTRASTS
    print(f"\n{'contrast':20s} {'diff':>8s} {'seeds 95%':>20s} {'examples 95%':>20s} {'both 95%':>20s} {'w_ex/w_seed':>11s}")
    for cid, dataset, a, b, sentence in contrasts:
        r = run_contrast(cid, dataset, a, b, sentence, rng)
        rows.append(r)
        flag = lambda k: "*" if r[k] else " "
        print(f"{cid:20s} {r['mean_diff']:+8.4f} [{r['seeds_lo']:+.4f},{r['seeds_hi']:+.4f}]{flag('nonzero_seeds')} "
              f"[{r['examples_lo']:+.4f},{r['examples_hi']:+.4f}]{flag('nonzero_examples')} "
              f"[{r['both_lo']:+.4f},{r['both_hi']:+.4f}]{flag('nonzero_both')} "
              f"{r['width_ratio_examples_over_seeds']:11.2f}", flush=True)
    print("(* = interval excludes zero)")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
