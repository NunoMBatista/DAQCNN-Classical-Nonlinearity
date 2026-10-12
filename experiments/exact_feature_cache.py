"""Feature caches from exact (ODE) evolution instead of the dt = 0.05 Trotter steps,
for the Trotter-error check of the probe (exact_probe.py).

Same images, normalisation, patches and channel layout as the production
caches. Written to outputs/camera_ready/exact_caches/ with an "__exact" suffix,
so the cache lookup never picks them up.

Digital: U(T) does not depend on the data, so it is integrated once per
topology (qml.evolve, tol 1e-11) and applied to every patch's product state.
Analog: one scipy DOP853 solve per patch (rtol 1e-9), over a process pool.

Checks before writing: labels equal the production cache's; production
features recomputed here for two test images equal the stored ones; and, if the
reference file of trotter_convergence_check.py exists, the exact features of
BreastMNIST test image 0 match it.

    python experiments/exact_feature_cache.py --dataset breast_mnist --enc digital --T 0.2 0.5 1.0 2.5 4.0
"""
import argparse, json, os, sys, time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
import pennylane as qml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.trotter_convergence_check import (          # noqa: E402
    WIRES, ZZ_PAIRS, N_QUBITS, build_static_pieces, kron_all, scipy_features)
from experiments.create_quantum_dataset import generate_output_filename  # noqa: E402
import src.physics.hamiltonian as phys                                    # noqa: E402
import src.physics.kernel_topologies as topologies                        # noqa: E402
from src.layers.quantum_convolution import QuantumConv2d                  # noqa: E402
from src.utils.data import load_medmnist_dataset                          # noqa: E402
from src.utils.quantum_dataset_cache import (                             # noqa: E402
    find_cached_quantum_dataset, load_cached_quantum_dataset)
from experiments.linear_probing_topology_sweep import make_cfg            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "camera_ready", "exact_caches")
CHECK_DIR = os.path.join(ROOT, "outputs", "camera_ready", "trotter_convergence")

TOPOLOGIES = {                       # the evolution-time sweep topologies
    "breast_mnist":    ["kings", "cross", "star"],
    "pneumonia_mnist": ["kings", "horizontal", "star"],
}


def split_patches(conv, dataset, split):
    """All patches of one split, (N, 81, 9), plus labels (N,), in dataset order."""
    ds = load_medmnist_dataset(dataset, split, os.path.join(ROOT, "data"))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    patches, labels = [], []
    for images, lbl in loader:
        x = conv._normalize_inputs(images.double())              # (B, 1, 28, 28) -> [0, pi]
        p = conv.unfold(x).transpose(1, 2)                        # (B, 81, 9)
        patches.append(p.numpy())
        labels.append(lbl.numpy().reshape(-1))
    return np.concatenate(patches), np.concatenate(labels)


def z_matrix(pieces):
    """(512, 45) matrix whose columns are the diagonals of Z_i and Z_i Z_j."""
    _, _, _, _, z_i = pieces
    cols = [z_i[i] for i in WIRES] + [z_i[i] * z_i[j] for i, j in ZZ_PAIRS]
    return np.stack(cols, axis=1)


def digital_unitary(coords, T):
    """U(T) for the data-independent digital Hamiltonian, tight ODE tolerance."""
    H = phys.get_rydberg_hamiltonian(WIRES, coords, scaling_factor=1.0, use_local_detuning=False)
    ev = qml.evolve(H)(params=np.array([1.0, 1.0, 1.0]), t=np.array([0.0, T]), atol=1e-11, rtol=1e-11)
    U = np.asarray(ev.matrix(wire_order=WIRES), dtype=complex)
    unit_err = np.max(np.abs(U.conj().T @ U - np.eye(2 ** N_QUBITS)))
    assert unit_err < 1e-6, unit_err
    return U


def digital_exact_features(U, zmat, patches, chunk=8192):
    """(n, 45) features: product states H RY(x_i)|0> of every patch, evolved by U (in chunks)."""
    p_all = np.asarray(patches, dtype=float).reshape(-1, N_QUBITS)
    out = []
    for start in range(0, len(p_all), chunk):
        p = p_all[start:start + chunk]
        c, s = np.cos(p / 2), np.sin(p / 2)
        up, dn = (c + s) / np.sqrt(2), (c - s) / np.sqrt(2)    # Hadamard after RY
        psi = np.ones((p.shape[0], 1), dtype=complex)
        for i in range(N_QUBITS):                              # kron over wires, wire 0 leftmost
            site = np.stack([up[:, i], dn[:, i]], axis=1)      # (n, 2)
            psi = (psi[:, :, None] * site[:, None, :]).reshape(p.shape[0], -1)
        prob = np.abs(psi @ U.T) ** 2                          # |U psi|^2 for every row
        out.append(prob @ zmat)                                # (n, 45)
    return np.concatenate(out)


_WORKER = {}                                                   # set once per pool worker


def _init_worker(pieces, T):
    _WORKER["pieces"], _WORKER["T"] = pieces, T


def _analog_worker(chunk):
    return scipy_features(_WORKER["pieces"], "analog", _WORKER["T"], chunk, rtol=1e-9, atol=1e-11)


def analog_exact_features(pieces, T, patches, workers):
    """One ODE per patch, over a process pool; returns (n, 45) in input order."""
    p = np.asarray(patches, dtype=float).reshape(-1, N_QUBITS)
    chunks = [p[i:i + 250] for i in range(0, len(p), 250)]
    out = []
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(pieces, T)) as pool:
        for k, feats in enumerate(pool.imap(_analog_worker, chunks)):
            out.append(feats)
            if k % 20 == 0:
                print(f"    analog chunk {k+1}/{len(chunks)} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(out)


def production_reference(dataset, topos, enc, T):
    """Path of the production Trotter cache holding these topologies at this T."""
    cfg = make_cfg(topos, enc, True, dataset_name=dataset)
    cfg["model"]["evolution_time"] = T
    path = find_cached_quantum_dataset(cfg)
    assert path is not None, f"no production cache for {dataset} {topos} {enc} T={T}"
    assert "tissue" not in str(path), "never load a Tissue cache here"
    return path


def check_labels_and_pipeline(dataset, topos, enc, T, labels, conv):
    """Labels and two images' features against the production cache (Breast/Pneumonia only)."""
    path = production_reference(dataset, topos, enc, T)
    ref = np.load(path, allow_pickle=True)                     # small for these two datasets
    for split in ("train", "val", "test"):
        assert np.array_equal(ref[f"{split}_labels"].reshape(-1), labels[split]), f"label mismatch {split}"
    tr, _, te, _, meta = load_cached_quantum_dataset(path, batch_size=2, num_workers=0, requested_kernels=topos)
    cached, _ = next(iter(te))                                 # (2, 45*k, 9, 9) float32
    ds = load_medmnist_dataset(dataset, "test", os.path.join(ROOT, "data"))
    imgs = torch.stack([ds[0][0], ds[1][0]]).double()
    with torch.no_grad():
        mine = conv(imgs).numpy()                              # production layer, default.qubit
    # the loader returns the requested kernels in cache channel order
    order = meta["kernel_topology_names"]
    if order != topos:
        idx = [topos.index(t) for t in order]
        mine = np.concatenate([mine[:, 45 * i:45 * (i + 1)] for i in idx], axis=1)
    err = float(np.max(np.abs(mine - cached.numpy())))
    print(f"  control 1 (labels) PASSED; control 2 (pipeline vs cached Trotter, 2 images): max |diff| = {err:.1e}")
    assert err < 1e-5, err                                     # float32 storage + device noise
    return order


def check_solver(enc, T, topo, exact_test_feats_topo):
    """BreastMNIST test image 0 against the reference file of trotter_convergence_check.py."""
    f = os.path.join(CHECK_DIR, f"{enc}__breast_test0__{topo}__T{T}.npz")
    if not os.path.exists(f):
        print(f"  control 3 skipped (no {os.path.basename(f)})")
        return
    ref = np.load(f)
    # reference: (n, 45) per patch; ours: (N, 45, 9, 9) -> image 0 -> (81, 45)
    n = ref["exact"].shape[0]
    mine = exact_test_feats_topo[0].reshape(45, 81).T[:n]
    err_ex = float(np.max(np.abs(mine - ref["exact"])))
    err_sp = float(np.max(np.abs(mine - ref["scipy"])))
    print(f"  control 3 ({topo}, test image 0, {n} patches, vs check-script exact / scipy): {err_ex:.1e} / {err_sp:.1e}")
    assert err_sp < 1e-6, err_sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TOPOLOGIES))
    ap.add_argument("--enc", required=True, choices=["digital", "analog"])
    ap.add_argument("--T", type=float, nargs="+", default=[2.5])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    topos = TOPOLOGIES[args.dataset]
    n_per_topo = N_QUBITS + len(ZZ_PAIRS)                      # 45

    for T in args.T:
        t_start = time.time()
        print(f"=== {args.dataset} {args.enc} T={T} topologies={topos} ===", flush=True)
        conv = QuantumConv2d(kernel_size=3, stride=3, kernel_topology_names=topos, evolution_time=T,
                             mode="trotter", include_correlators=True, encoding_mode=args.enc,
                             quantum_device="default.qubit")
        patches, labels = {}, {}
        for split in ("train", "val", "test"):
            patches[split], labels[split] = split_patches(conv, args.dataset, split)
            print(f"  {split}: {patches[split].shape[0]} images", flush=True)
        order = check_labels_and_pipeline(args.dataset, topos, args.enc, T, labels, conv)

        feats = {s: [] for s in patches}
        for topo in order:
            coords = topologies.build_kernel_coordinate_sets(3, [topo])[0]
            pieces = build_static_pieces(coords)
            if args.enc == "digital":
                U = digital_unitary(coords, T)
                zmat = z_matrix(pieces)
            for split in ("train", "val", "test"):
                p = patches[split]
                if args.enc == "digital":
                    f = digital_exact_features(U, zmat, p)
                else:
                    f = analog_exact_features(pieces, T, p, args.workers)
                feats[split].append(f.reshape(p.shape[0], 81, n_per_topo))
            print(f"  {topo} done ({time.time()-t_start:.0f}s)", flush=True)

        # same layout as the production forward: (N, 81, C) -> (N, C, 9, 9)
        out = {}
        for split in ("train", "val", "test"):
            arr = np.concatenate(feats[split], axis=2)         # (N, 81, 45*k)
            out[f"{split}_features"] = np.ascontiguousarray(arr.transpose(0, 2, 1).reshape(arr.shape[0], -1, 9, 9))
            out[f"{split}_labels"] = labels[split]
        if args.dataset == "breast_mnist":
            for i, topo in enumerate(order):
                check_solver(args.enc, T, topo, out["test_features"][:1, 45 * i:45 * (i + 1)])

        channel_kernel_map = []
        for topo in order:
            for q in range(N_QUBITS):
                channel_kernel_map.append({"channel": len(channel_kernel_map), "kernel": topo, "qubit": q, "measurement": "Z"})
            for qi, qj in ZZ_PAIRS:
                channel_kernel_map.append({"channel": len(channel_kernel_map), "kernel": topo, "qubit_pair": [qi, qj], "measurement": "ZZ"})
        meta = {
            "dataset_name": args.dataset, "image_size": 28, "in_channels": 1, "color_space": "GRAYSCALE",
            "kernel_size": 3, "stride": 3, "kernel_topology_names": order, "num_kernels": len(order),
            "scaling_factor": 1.0, "evolution_time": T, "out_channels": 45 * len(order),
            "quantum_out_channels": 45 * len(order), "include_correlators": True, "encoding_mode": args.enc,
            "noise_enabled": False, "noise_T1_us": None, "noise_T2_us": None, "noise_p_gate_1q": None,
            "noise_omega_mhz": None, "channel_kernel_map": channel_kernel_map,
            "evolution_mode": "exact",
            "exact_solver": ("pennylane qml.evolve jax odeint atol=rtol=1e-11, U(T) applied to product states"
                             if args.enc == "digital" else "scipy DOP853 rtol=1e-9 atol=1e-11 per patch"),
            "created_at": datetime.now().isoformat(),
            "train_samples": int(len(labels["train"])), "val_samples": int(len(labels["val"])),
            "test_samples": int(len(labels["test"])),
        }
        base = generate_output_filename(meta)[:-len(".npz")]
        path = os.path.join(OUT_DIR, base + "__exact.npz")
        np.savez_compressed(path, metadata=json.dumps(meta), **out)
        with open(path[:-4] + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  wrote {path} ({out['train_features'].shape}) in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
