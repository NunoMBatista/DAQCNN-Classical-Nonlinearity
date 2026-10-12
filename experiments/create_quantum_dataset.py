"""Run a MedMNIST dataset through the quantum kernels once and cache the features.

Writes data/quantum_datasets/<name>.npz (features and labels per split, plus
metadata) and a .json sidecar with the same metadata, which the training and
probe scripts use to find the cache.

Chunk mode (--split/--start/--end/--chunk-dir) processes one index range of one
split and writes a partial .chunk.npz; merge_quantum_chunks.py reassembles the
chunks into a normal cache.

    python experiments/create_quantum_dataset.py --config configs/breast_mnist/cache_generation/digital_zz.yml
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.layers.quantum_convolution import QuantumConv2d
from src.utils.data import load_medmnist_dataset

# used when no --config is given; a config overrides any of them
DEFAULTS = {
    "dataset_name": "breast_mnist",
    "color_space": "GRAYSCALE",
    "kernel_size": 3,
    "stride": 3,
    "kernel_topology_names": ["kings", "horizontal", "vertical", "cross", "ring", "chain", "star", "grid"],
    "scaling_factor": 1.0,
    "evolution_time": 2.5,
    "mode": "trotter",
    "encoding_mode": "digital",
    "include_correlators": False,
    "quantum_device": "default.qubit",  # lightning.qubit is about 2x faster for 3x3 kernels
    "batch_size": 8,
}
INTERFACE = "torch"
SPLITS = ["train", "val", "test"]


def load_params(config_path):
    params = dict(DEFAULTS)
    if config_path is None:
        return params
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    for cfg_key, key in (("name", "dataset_name"), ("color_space", "color_space"), ("batch_size", "batch_size")):
        if cfg_key in dataset_cfg:
            params[key] = dataset_cfg[cfg_key]
    for key in ("kernel_size", "stride", "kernel_topology_names", "scaling_factor", "evolution_time",
                "mode", "encoding_mode", "include_correlators", "quantum_device"):
        if key in model_cfg:
            params[key] = model_cfg[key]
    print(f"Loaded parameters from {config_path}")
    return params


def generate_output_filename(params):
    """Cache file name, e.g. breast_mnist__k3_s3_tkin-cro-sta_ev2.50_sc1_gray_zz_analog.npz."""
    topo_short = "-".join([t[:3] for t in params["kernel_topology_names"]])
    filename = (
        f"{params['dataset_name']}__"
        f"k{params['kernel_size']}_"
        f"s{params['stride']}_"
        f"t{topo_short}_"
        f"ev{params['evolution_time']:.2f}_"
        f"sc{params['scaling_factor']:.0f}"
    )
    if params.get("color_space", "RGB") == "GRAYSCALE":
        filename += "_gray"
    if params.get("include_correlators", False):
        filename += "_zz"
    if params.get("encoding_mode", "digital") == "analog":
        filename += "_analog"
    return filename + ".npz"


def process_dataset_through_quantum(dataset, q_conv, batch_size, desc="Processing"):
    """(features (N, C, H_out, W_out), labels (N,)) for a dataset, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features = []
    all_labels = []
    for images, labels in tqdm(loader, desc=desc):
        with torch.no_grad():
            q_out = q_conv(images)
        all_features.append(q_out.cpu().numpy())
        lbl = labels.numpy()
        if lbl.ndim > 1:
            lbl = lbl.squeeze(-1)
        all_labels.append(lbl)
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate a quantum feature cache")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--split", type=str, default=None,
                        help="Chunk mode: process only this split (train/val/test)")
    parser.add_argument("--start", type=int, default=None,
                        help="Chunk mode: first image index (default 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="Chunk mode: one-past-last image index (default = split size)")
    parser.add_argument("--chunk-dir", type=str, default=None,
                        help="Chunk mode: directory to write the chunk file into")
    args = parser.parse_args()

    p = load_params(args.config)
    for key, value in p.items():
        print(f"  {key:22s} {value}")

    data_root = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "data" / "quantum_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    q_conv = QuantumConv2d(
        kernel_size=p["kernel_size"],
        stride=p["stride"],
        kernel_topology_names=p["kernel_topology_names"],
        scaling_factor=p["scaling_factor"],
        evolution_time=p["evolution_time"],
        mode=p["mode"],
        quantum_device=p["quantum_device"],
        interface=INTERFACE,
        include_correlators=p["include_correlators"],
        encoding_mode=p["encoding_mode"],
    )
    out_channels = q_conv.out_channels

    # which kernel and measurement produced each output channel
    n_qubits = p["kernel_size"] * p["kernel_size"]
    channel_kernel_map = []
    for topo_name in p["kernel_topology_names"]:
        for qubit_idx in range(n_qubits):
            channel_kernel_map.append({"channel": len(channel_kernel_map), "kernel": topo_name,
                                       "qubit": qubit_idx, "measurement": "Z"})
        if p["include_correlators"]:
            for qi, qj in combinations(range(n_qubits), 2):
                channel_kernel_map.append({"channel": len(channel_kernel_map), "kernel": topo_name,
                                           "qubit_pair": [qi, qj], "measurement": "ZZ"})

    # Cache identity. The noise_* fields are constant: kept so that new
    # sidecars have the same keys as the published caches.
    base_metadata = {
        "dataset_name": p["dataset_name"],
        "image_size": 28,
        "in_channels": 1,
        "color_space": p["color_space"],
        "kernel_size": p["kernel_size"],
        "stride": p["stride"],
        "kernel_topology_names": p["kernel_topology_names"],
        "num_kernels": len(p["kernel_topology_names"]),
        "scaling_factor": p["scaling_factor"],
        "evolution_time": p["evolution_time"],
        "out_channels": out_channels,
        "quantum_out_channels": out_channels,
        "include_correlators": p["include_correlators"],
        "encoding_mode": p["encoding_mode"],
        "noise_enabled": False,
        "noise_T1_us": None,
        "noise_T2_us": None,
        "noise_p_gate_1q": None,
        "noise_omega_mhz": None,
        "channel_kernel_map": channel_kernel_map,
    }

    if args.split is not None:
        split = args.split
        ds = load_medmnist_dataset(p["dataset_name"], split, data_root)
        split_total = len(ds)
        start = 0 if args.start is None else args.start
        end = split_total if args.end is None else args.end
        assert 0 <= start < end <= split_total, (
            f"Bad range [{start}, {end}) for split '{split}' of size {split_total}"
        )

        print(f"--- CHUNK: {split}[{start}:{end}] of {split_total} ---")
        features, labels = process_dataset_through_quantum(
            Subset(ds, range(start, end)), q_conv, p["batch_size"], desc=f"  {split}[{start}:{end}]"
        )
        assert features.shape[0] == end - start == labels.shape[0]

        chunk_meta = {
            **base_metadata,
            "split": split,
            "start": start,
            "end": end,
            "split_total": split_total,
            "created_at": datetime.now().isoformat(),
        }
        chunk_dir = Path(args.chunk_dir) if args.chunk_dir else output_dir
        chunk_dir.mkdir(parents=True, exist_ok=True)
        cache_base = generate_output_filename(base_metadata)[:-len(".npz")]
        chunk_path = chunk_dir / f"{cache_base}__{split}_{start:06d}-{end:06d}.chunk.npz"
        np.savez_compressed(chunk_path, features=features, labels=labels, chunk_meta=json.dumps(chunk_meta))
        print(f"Saved chunk to: {chunk_path}")
        print(f"  shape {features.shape}, covers {split}[{start}:{end}] / {split_total}")
        return

    results = {}
    for split in SPLITS:
        ds = load_medmnist_dataset(p["dataset_name"], split, data_root)
        print(f"--- {split}: {len(ds)} images ---")
        features, labels = process_dataset_through_quantum(ds, q_conv, p["batch_size"], desc=f"  {split}")
        results[f"{split}_features"] = features
        results[f"{split}_labels"] = labels

    metadata = {
        **base_metadata,
        "created_at": datetime.now().isoformat(),
        "train_samples": len(results["train_labels"]),
        "val_samples": len(results["val_labels"]),
        "test_samples": len(results["test_labels"]),
    }
    output_path = output_dir / generate_output_filename(metadata)
    np.savez_compressed(output_path, **results, metadata=json.dumps(metadata))
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {output_path} and its .json sidecar")


if __name__ == "__main__":
    main()
