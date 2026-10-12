"""Write a poly-2 or RFF feature cache in the quantum-cache format.

The cache gets a synthetic topology name ("poly2", "rff45", "rff180"), so the
end-to-end HP search finds and consumes it exactly like a quantum cache.
Features come from the same dataloader and feature maps as the capacity sweep.

    python experiments/create_classical_cache.py --dataset breast_mnist --source poly2_45
"""
import os, sys, json, argparse
from datetime import datetime
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data import get_dataloaders
from src.utils.classical_nonlinear_features import poly2_features, rff_features

RFF_SEED = 42  # the draw of the published caches (= RANDOM_FILTER_SEED in head_capacity_sweep.py)

# source -> (synthetic topology name, feature map of (images, rff_seed))
SOURCES = {
    "poly2_45": ("poly2",  lambda x, seed: poly2_features(x)),
    "rff_45":   ("rff45",  lambda x, seed: rff_features(x, 45,  seed=seed)),
    "rff_180":  ("rff180", lambda x, seed: rff_features(x, 180, seed=seed)),
}


def stack(loader, fmap, rff_seed):
    xs, ys = [], []
    for x, y in loader:
        xs.append(fmap(x.float(), rff_seed))
        ys.append(y.long().squeeze())
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--out-dir", default="data/quantum_datasets")
    # the draw is not part of the cache identity or file name: write other draws
    # to their own --out-dir
    ap.add_argument("--rff-seed", type=int, default=RFF_SEED)
    args = ap.parse_args()

    topo, fmap = SOURCES[args.source]
    cfg = {"dataset": {"name": args.dataset, "data_root": "./data", "batch_size": 1000,
                       "num_workers": 0, "download": True, "color_space": "GRAYSCALE"}}
    train_loader, val_loader, test_loader, _ = get_dataloaders(cfg)

    feats, labs = {}, {}
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        feats[split], labs[split] = stack(loader, fmap, args.rff_seed)
        print(f"  {split}: {feats[split].shape}")

    F = feats["train"].shape[1]
    ckm = [{"channel": i, "kernel": topo} for i in range(F)]
    meta = {
        "dataset_name": args.dataset, "image_size": 28, "in_channels": 1,
        "color_space": "GRAYSCALE", "kernel_size": 3, "stride": 3,
        "kernel_topology_names": [topo], "num_kernels": 1,
        "scaling_factor": 1.0, "evolution_time": 2.5,
        "out_channels": F, "quantum_out_channels": F,
        "include_correlators": False, "encoding_mode": "digital",
        "noise_enabled": False, "noise_T1_us": None, "noise_T2_us": None,
        "noise_p_gate_1q": None, "noise_omega_mhz": None,
        "channel_kernel_map": ckm,
        "created_at": datetime.now().isoformat(),
        "train_samples": len(labs["train"]), "val_samples": len(labs["val"]),
        "test_samples": len(labs["test"]),
        "feature_source": args.source,
        "rff_seed": args.rff_seed,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    base = f"{args.dataset}__classical_{topo}"
    npz = os.path.join(args.out_dir, base + ".npz")
    np.savez_compressed(
        npz,
        train_features=feats["train"], train_labels=labs["train"],
        val_features=feats["val"], val_labels=labs["val"],
        test_features=feats["test"], test_labels=labs["test"],
        metadata=json.dumps(meta),
    )
    with open(npz[:-4] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {npz}  ({F} channels, topology '{topo}')")


if __name__ == "__main__":
    main()
