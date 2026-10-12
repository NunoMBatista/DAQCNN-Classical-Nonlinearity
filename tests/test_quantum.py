"""Analytic checks of the quantum feature map."""

from itertools import combinations

import numpy as np
import torch

from src.layers.daqk import DAQKLayer
from src.layers.quantum_convolution import QuantumConv2d

PAIRS = list(combinations(range(9), 2))


def _patches(n=4, seed=0):
    return torch.tensor(np.random.default_rng(seed).random((n, 9)) * np.pi)


def test_digital_features_are_sin_x_as_T_goes_to_0():
    # RY(x) then H gives <Z> = sin x per qubit, and a product state, so <Z_i Z_j> = sin x_i sin x_j
    x = _patches()
    layer = DAQKLayer(n_qubits=9, grid_size=3, evolution_time=1e-6,
                      kernel_topology_names=["kings"], include_correlators=True)
    out = layer(x).numpy()
    s = np.sin(x.numpy())
    expected = np.concatenate([s, np.stack([s[:, i] * s[:, j] for i, j in PAIRS], axis=1)], axis=1)
    assert out.shape == (4, 45)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_z_channels_do_not_depend_on_measuring_zz():
    # derive_z_from_zz.py relies on this: Z-only features = first 9 channels of Z+ZZ
    x = _patches()
    for enc in ("digital", "analog"):
        z = DAQKLayer(n_qubits=9, grid_size=3, evolution_time=0.2, kernel_topology_names=["star"],
                      encoding_mode=enc)(x).numpy()
        zz = DAQKLayer(n_qubits=9, grid_size=3, evolution_time=0.2, kernel_topology_names=["star"],
                       include_correlators=True, encoding_mode=enc)(x).numpy()
        np.testing.assert_allclose(z, zz[:, :9], atol=1e-12)


def test_decoupled_rows_stay_uncorrelated():
    # "horizontal": rows 100 units apart (coupling 1e-12), so ZZ across rows factorises
    x = _patches(3)
    out = DAQKLayer(n_qubits=9, grid_size=3, evolution_time=1.0, kernel_topology_names=["horizontal"],
                    include_correlators=True)(x).numpy()
    z, zz = out[:, :9], out[:, 9:]
    for k, (i, j) in enumerate(PAIRS):
        connected = zz[:, k] - z[:, i] * z[:, j]
        if i // 3 != j // 3:
            assert np.abs(connected).max() < 1e-9
    # within a row the atoms do interact
    k01 = PAIRS.index((0, 1))
    assert np.abs(zz[:, k01] - z[:, 0] * z[:, 1]).max() > 1e-3


def test_features_are_expectation_values():
    out = DAQKLayer(n_qubits=9, grid_size=3, evolution_time=2.5, kernel_topology_names=["kings"],
                    include_correlators=True, encoding_mode="analog")(_patches(2)).numpy()
    assert np.all(np.abs(out) <= 1 + 1e-9)


def test_quantum_conv_is_per_image_normalised_and_row_major():
    torch.manual_seed(0)
    img = torch.rand(1, 1, 6, 6) * 0.5 + 0.2
    conv = QuantumConv2d(kernel_size=3, stride=3, kernel_topology_names=["cross", "ring"],
                         evolution_time=0.2, include_correlators=True)
    norm = conv._normalize_inputs(img)
    assert torch.isclose(norm.min(), torch.tensor(0.0)) and torch.isclose(norm.max(), torch.tensor(np.pi))

    out = conv(img)
    assert out.shape == (1, 90, 2, 2) and conv.out_channels == 90
    # patch (r, c) of the 2x2 grid = pixels [3r:3r+3, 3c:3c+3], flattened row-major
    patches = torch.stack([norm[0, 0, 3 * r:3 * r + 3, 3 * c:3 * c + 3].reshape(9)
                           for r in range(2) for c in range(2)])
    direct = conv.quantum_kernel(patches)                      # (4, 90)
    np.testing.assert_allclose(out[0].reshape(90, 4).T.numpy(), direct.numpy(), atol=1e-12)
