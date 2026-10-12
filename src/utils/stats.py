"""Seed and example-level intervals shared by the camera-ready analysis scripts."""
import math

import numpy as np
from scipy import stats


def paired_t_interval(a, b, level=0.95):
    """t interval for the mean of a - b over paired seeds (equal-length lists).

    Returns (mean, lo, hi, sd), sd being the SD of the differences.
    """
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return mean, mean - t * se, mean + t * se, sd


def example_bootstrap_intervals(d_full, seed_diffs, n_examples, rng, n_boot):
    """The three 95% intervals for a seed-mean AUC difference A minus B.

    d_full: per-seed AUC differences on the full test set.
    seed_diffs(idx): per-seed AUC differences on the test-example resample idx.
    Returns mean, then lo, hi over seeds (t interval), over test examples
    (percentile bootstrap), and over both (examples and seeds resampled).
    """
    n_seeds = len(d_full)
    m = float(d_full.mean())
    if n_seeds < 2:
        lo_s, hi_s = float("nan"), float("nan")
    else:
        se = d_full.std(ddof=1) / math.sqrt(n_seeds)
        t = stats.t.ppf(0.975, n_seeds - 1)
        lo_s, hi_s = m - t * se, m + t * se

    # examples only first, then examples and seeds; the rng draws keep this order
    boot = []
    for resample_seeds in (False, True):
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            d = seed_diffs(rng.integers(0, n_examples, n_examples))
            if resample_seeds:
                d = d[rng.integers(0, n_seeds, n_seeds)]
            diffs[b] = d.mean()
        boot += [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    lo_e, hi_e, lo_b, hi_b = boot
    return m, lo_s, hi_s, lo_e, hi_e, lo_b, hi_b
