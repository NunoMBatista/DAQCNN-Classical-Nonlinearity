"""Find and load the precomputed feature caches in data/quantum_datasets/.

A cache is an .npz (features and labels per split, plus metadata) with the
same metadata in a .json sidecar. Lookup reads only the sidecars.
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import QUANTUM_DATASETS_DIR
from src.utils.kernel_mapping import build_kernel_to_channels_map


def _match_metadata(meta, cfg):
    """"exact", "subset" (the config's topologies are a subset of the cache's) or False."""
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})

    if meta.get("dataset_name") != dataset_cfg.get("name"):
        return False
    # caches without the field are 28x28
    if meta.get("image_size", 28) != 28:
        return False
    # a missing color_space means RGB, on both sides
    if meta.get("color_space", "RGB") != dataset_cfg.get("color_space", "RGB"):
        return False
    if meta.get("kernel_size") != model_cfg.get("kernel_size"):
        return False
    if meta.get("stride") != model_cfg.get("stride"):
        return False

    cached_topos = sorted(meta.get("kernel_topology_names", []))
    config_topos = sorted(model_cfg.get("kernel_topology_names", []))
    exact_match = cached_topos == config_topos
    subset_match = set(config_topos).issubset(set(cached_topos))
    if not exact_match and not subset_match:
        return False

    if abs(float(meta.get("scaling_factor", 0)) - float(model_cfg.get("scaling_factor", 0))) > 1e-6:
        return False
    if abs(float(meta.get("evolution_time", 0)) - float(model_cfg.get("evolution_time", 0))) > 1e-6:
        return False
    if meta.get("include_correlators", False) != model_cfg.get("include_correlators", False):
        return False
    if meta.get("encoding_mode", "digital") != model_cfg.get("encoding_mode", "digital"):
        return False
    # never use a cache generated with the (removed) noise model
    if meta.get("noise_enabled", False):
        return False

    return "exact" if exact_match else "subset"


def find_cached_quantum_dataset(cfg, datasets_dir=None):
    """Path of the .npz whose sidecar matches cfg (an exact match wins over a superset), or None."""
    if datasets_dir is None:
        datasets_dir = QUANTUM_DATASETS_DIR
    if not datasets_dir.is_dir():
        return None

    exact_match = None
    subset_match = None
    for json_path in sorted(datasets_dir.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        match_type = _match_metadata(meta, cfg)
        npz_path = json_path.with_suffix(".npz")
        if match_type and npz_path.is_file():
            if match_type == "exact":
                exact_match = npz_path
                break
            subset_match = npz_path

    return exact_match if exact_match else subset_match


def load_cached_quantum_dataset(npz_path, batch_size=32, num_workers=2, requested_kernels=None):
    """(train_loader, val_loader, test_loader, n_classes, metadata) from a cache.

    If requested_kernels is a proper subset of the cached topologies, only their
    channels are kept, in the cache's channel order (not the requested order).
    Loads whole arrays into memory: never call this on the TissueMNIST caches
    on a small machine.
    """
    data = np.load(npz_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))

    cached_kernels = metadata.get("kernel_topology_names", [])
    extract_subset = (
        requested_kernels is not None
        and set(requested_kernels) != set(cached_kernels)
        and set(requested_kernels).issubset(set(cached_kernels))
    )

    if extract_subset:
        cached_channel_map = metadata["channel_kernel_map"]
        kernel_to_channels = build_kernel_to_channels_map(cached_channel_map)

        # channel order, not alphabetical or requested order: the channels of a
        # multi-kernel cache follow its generation order
        requested_sorted_by_channel = sorted(
            requested_kernels, key=lambda name: kernel_to_channels[name][0]
        )

        channel_indices = []
        new_channel_map = []
        for kernel_name in requested_sorted_by_channel:
            for old_idx in kernel_to_channels[kernel_name]:
                channel_indices.append(old_idx)
                original_entry = next(e for e in cached_channel_map if e["channel"] == old_idx)
                new_entry = dict(original_entry)
                new_entry["channel"] = len(new_channel_map)
                new_channel_map.append(new_entry)

        metadata = metadata.copy()
        metadata["kernel_topology_names"] = requested_sorted_by_channel
        metadata["num_kernels"] = len(requested_kernels)
        metadata["out_channels"] = len(channel_indices)
        metadata["channel_kernel_map"] = new_channel_map

        print(
            f"- Extracting subset from cached dataset:"
            f"{len(requested_kernels)}/{len(cached_kernels)} kernels "
            f"({len(channel_indices)}/{len(cached_channel_map)} channels)"
        )
        print(f"   Requested: {requested_sorted_by_channel}")
        print(f"   Available: {sorted(cached_kernels)}")
    else:
        channel_indices = None

    def _make_loader(split, shuffle):
        features = torch.from_numpy(data[f"{split}_features"]).float()
        if channel_indices is not None:
            features = features[:, channel_indices, :, :]
        labels = torch.from_numpy(data[f"{split}_labels"]).long()
        return DataLoader(TensorDataset(features, labels), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

    train_loader = _make_loader("train", shuffle=True)
    val_loader = _make_loader("val", shuffle=False)
    test_loader = _make_loader("test", shuffle=False)

    n_classes = len(set(data["train_labels"].tolist()))

    return train_loader, val_loader, test_loader, n_classes, metadata
