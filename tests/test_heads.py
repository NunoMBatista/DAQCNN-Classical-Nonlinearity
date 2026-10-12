"""Head parameter counts against the paper's Table I (binary tasks)."""

import torch

from head_capacity_sweep import HEAD_NAMES, _head_factory, _param_count
from src.models.classical_baseline import ClassicalBaselineCNN

# input shape of each capacity-sweep source (C, H, W)
SOURCE_SHAPES = {
    "raw": (1, 28, 28),
    "z_or_random_9": (9, 9, 9),        # Digital-Z, Analog-Z, Random x9
    "one_kernel_45": (45, 9, 9),       # Z+ZZ 1k, Random x45, poly-2, RFF x45
    "four_kernel_180": (180, 9, 9),    # Z+ZZ 4k, Random x180, RFF x180
}

# Table I, "Params" column, in thousands
TABLE_I = {"linear": (1.5, 29), "mlp1": (23, 467), "mlp2": (49, 935),
           "cnn_small": (2.0, 13), "cnn_large": (20, 64)}


def _count(head, shape):
    builder, _ = _head_factory(head)
    return _param_count(builder, 2, shape[0], torch.zeros(1, *shape))


def test_exact_counts():
    d = 9 * 9 * 9
    assert _count("linear", SOURCE_SHAPES["z_or_random_9"]) == 2 * d + 2
    assert _count("mlp1", SOURCE_SHAPES["z_or_random_9"]) == 32 * d + 32 + 32 * 2 + 2
    assert _count("mlp2", SOURCE_SHAPES["z_or_random_9"]) == 64 * d + 64 + 64 * 32 + 32 + 32 * 2 + 2
    # conv(9 -> 16, 2x2) + BN + conv(16 -> 16, 2x2) + linear(16 * 3 * 3 -> 2)
    assert _count("cnn_small", SOURCE_SHAPES["z_or_random_9"]) == (9 * 4 * 16 + 16) + 32 + (16 * 16 * 4 + 16) + (144 * 2 + 2)


def test_table_one_ranges():
    for head in HEAD_NAMES:
        counts = [_count(head, shape) for shape in SOURCE_SHAPES.values()]
        lo, hi = TABLE_I[head]
        digits = 1 if lo < 10 else 0
        assert round(min(counts) / 1000, digits) == lo, (head, counts)
        assert round(max(counts) / 1000) == hi, (head, counts)


def test_raw_pixel_baseline_size():
    model = ClassicalBaselineCNN(num_classes=2, raw_features=True)
    model(torch.zeros(1, 1, 28, 28))
    assert sum(p.numel() for p in model.parameters()) == 35330


def test_frozen_random_filters_are_not_trainable():
    model = ClassicalBaselineCNN(num_classes=2, out_channels=45, fixed_random_filters=True)
    assert not model.conv.weight.requires_grad
    assert all(p.requires_grad for p in model.head.parameters())
