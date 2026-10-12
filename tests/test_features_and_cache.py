"""Classical feature maps and the cache lookup / channel slicing, on synthetic data."""

import json
from itertools import combinations

import numpy as np
import torch

from src.utils.classical_nonlinear_features import poly2_features, rff_features
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset

TOPOS = ["kings", "horizontal", "vertical", "cross"]


def test_poly2_is_pixels_then_pairwise_products():
    img = torch.rand(1, 1, 3, 3)
    feat = poly2_features(img)[0, :, 0, 0].numpy()
    x = img.reshape(9).numpy()
    expected = list(x) + [x[i] * x[j] for i, j in combinations(range(9), 2)]
    np.testing.assert_allclose(feat, expected, rtol=1e-6)


def test_rff_approximates_rbf_kernel():
    torch.manual_seed(0)
    img = torch.rand(1, 1, 3, 6)                  # two 3x3 patches side by side
    z = rff_features(img, 20000, gamma=0.5, seed=1)[0, :, 0, :].numpy()
    x, y = img[0, 0, :, :3].reshape(9).numpy(), img[0, 0, :, 3:].reshape(9).numpy()
    np.testing.assert_allclose(z[:, 0] @ z[:, 1], np.exp(-0.5 * np.sum((x - y) ** 2)), atol=0.03)
    assert np.array_equal(rff_features(img, 45, seed=3), rff_features(img, 45, seed=3))


def _metadata(topologies, correlators=True):
    cmap = []
    for t in topologies:
        for q in range(9):
            cmap.append({"channel": len(cmap), "kernel": t, "qubit": q, "measurement": "Z"})
        if correlators:
            for i, j in combinations(range(9), 2):
                cmap.append({"channel": len(cmap), "kernel": t, "qubit_pair": [i, j], "measurement": "ZZ"})
    return {"dataset_name": "breast_mnist", "image_size": 28, "color_space": "GRAYSCALE",
            "kernel_size": 3, "stride": 3, "kernel_topology_names": topologies,
            "num_kernels": len(topologies), "scaling_factor": 1.0, "evolution_time": 2.5,
            "out_channels": len(cmap), "include_correlators": correlators, "encoding_mode": "digital",
            "noise_enabled": False, "channel_kernel_map": cmap}


def _cfg(topologies):
    return {"dataset": {"name": "breast_mnist", "color_space": "GRAYSCALE"},
            "model": {"kernel_size": 3, "stride": 3, "kernel_topology_names": topologies,
                      "scaling_factor": 1.0, "evolution_time": 2.5, "include_correlators": True,
                      "encoding_mode": "digital"}}


def test_lookup_prefers_exact_match_and_accepts_supersets(tmp_path):
    for name, topos in (("a_super", TOPOS), ("b_exact", ["cross"])):
        (tmp_path / f"{name}.json").write_text(json.dumps(_metadata(topos)))
        (tmp_path / f"{name}.npz").write_bytes(b"")
    assert find_cached_quantum_dataset(_cfg(["cross"]), tmp_path).name == "b_exact.npz"
    assert find_cached_quantum_dataset(_cfg(["kings", "vertical"]), tmp_path).name == "a_super.npz"
    assert find_cached_quantum_dataset(_cfg(["star"]), tmp_path) is None
    noisy = dict(_metadata(["star"]), noise_enabled=True)
    (tmp_path / "c_noisy.json").write_text(json.dumps(noisy))
    (tmp_path / "c_noisy.npz").write_bytes(b"")
    assert find_cached_quantum_dataset(_cfg(["star"]), tmp_path) is None


def test_subset_keeps_cache_channel_order(tmp_path):
    meta = _metadata(TOPOS)
    feats = np.arange(4 * 180 * 2 * 2, dtype=np.float32).reshape(4, 180, 2, 2)
    labels = np.array([0, 1, 0, 1])
    path = tmp_path / "c.npz"
    np.savez_compressed(path, metadata=json.dumps(meta), train_features=feats, train_labels=labels,
                        val_features=feats, val_labels=labels, test_features=feats, test_labels=labels)
    # requested in the "wrong" order: the result follows the cache order (kings, then cross)
    _, val, _, n_classes, out_meta = load_cached_quantum_dataset(path, batch_size=4, num_workers=0,
                                                                 requested_kernels=["cross", "kings"])
    x, _ = next(iter(val))
    expected = np.concatenate([feats[:, 0:45], feats[:, 135:180]], axis=1)
    assert np.array_equal(x.numpy(), expected)
    assert out_meta["kernel_topology_names"] == ["kings", "cross"]
    assert [e["channel"] for e in out_meta["channel_kernel_map"]] == list(range(90))
    assert n_classes == 2
