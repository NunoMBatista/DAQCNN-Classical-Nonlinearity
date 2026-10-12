"""Reassemble the .chunk.npz files written by create_quantum_dataset.py in chunk
mode into one cache, identical to what a single run would have written.

Nothing is written unless every check passes:
  1. all chunks agree on the cache identity and channel map;
  2. per split, the chunks tile [0, split_total) exactly (no gap, no overlap);
  3. each chunk's labels equal the dataset's own labels for its index range
     (a free check that every chunk sits at the right position);
  4. all three splits are present.

    python experiments/merge_quantum_chunks.py --chunk-dir DIR --out-dir data/quantum_datasets
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.create_quantum_dataset import generate_output_filename
from src.utils.data import load_medmnist_dataset

# cache identity; every chunk must agree on these (the noise_* keys are constant
# but part of the published sidecar format)
IDENTITY_KEYS = [
    "dataset_name", "image_size", "in_channels", "color_space",
    "kernel_size", "stride", "kernel_topology_names", "num_kernels",
    "scaling_factor", "evolution_time", "out_channels", "quantum_out_channels",
    "include_correlators", "encoding_mode",
    "noise_enabled", "noise_T1_us", "noise_T2_us", "noise_p_gate_1q", "noise_omega_mhz",
]
SPLITS = ["train", "val", "test"]


def ground_truth_labels(dataset_name, split, data_root):
    """Labels of a split in dataset order."""
    ds = load_medmnist_dataset(dataset_name, split, data_root)
    labels = np.asarray(ds.labels).reshape(-1).astype(np.int64)
    assert len(labels) == len(ds), "label array length disagrees with dataset length"
    return labels


def main():
    ap = argparse.ArgumentParser(description="Merge quantum-dataset chunks")
    ap.add_argument("--chunk-dir", required=True, help="Directory containing *.chunk.npz")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "quantum_datasets"))
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    args = ap.parse_args()

    chunk_dir = Path(args.chunk_dir)
    chunk_files = sorted(chunk_dir.glob("*.chunk.npz"))
    assert chunk_files, f"No *.chunk.npz files found in {chunk_dir}"
    print(f"Found {len(chunk_files)} chunk file(s) in {chunk_dir}")

    chunks = []
    for f in chunk_files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["chunk_meta"]))
        chunks.append({
            "path": f, "meta": meta,
            "features": z["features"], "labels": z["labels"],
            "split": meta["split"], "start": meta["start"],
            "end": meta["end"], "split_total": meta["split_total"],
        })

    # ---- Check 1: all chunks agree on cache identity ----
    ref = chunks[0]["meta"]
    for c in chunks[1:]:
        for k in IDENTITY_KEYS:
            assert c["meta"].get(k) == ref.get(k), (
                f"Chunk {c['path'].name} disagrees on '{k}': "
                f"{c['meta'].get(k)} != {ref.get(k)}"
            )
    assert all(c["meta"]["channel_kernel_map"] == ref["channel_kernel_map"]
               for c in chunks), "channel_kernel_map differs across chunks"
    print("Check 1 OK: all chunks share the same cache parameters and channel map.")

    dataset_name = ref["dataset_name"]

    merged = {}
    counts = {}
    for split in SPLITS:
        split_chunks = sorted([c for c in chunks if c["split"] == split],
                              key=lambda c: c["start"])
        assert split_chunks, f"Check 4 FAILED: no chunks for split '{split}'"

        split_total = split_chunks[0]["split_total"]
        assert all(c["split_total"] == split_total for c in split_chunks), (
            f"split_total disagreement within split '{split}'"
        )

        # ---- Check 2: exact tiling of [0, split_total) ----
        cursor = 0
        for c in split_chunks:
            assert c["start"] == cursor, (
                f"Check 2 FAILED ({split}): gap/overlap — expected start {cursor}, "
                f"got {c['start']} from {c['path'].name}"
            )
            assert c["features"].shape[0] == c["end"] - c["start"] == c["labels"].shape[0], (
                f"Check 2 FAILED ({split}): {c['path'].name} row count != range width"
            )
            cursor = c["end"]
        assert cursor == split_total, (
            f"Check 2 FAILED ({split}): chunks cover [0, {cursor}) but split has "
            f"{split_total} images — incomplete tiling"
        )

        # ---- Check 3: independent label checksum vs ground truth ----
        gt = ground_truth_labels(dataset_name, split, args.data_root)
        assert len(gt) == split_total, (
            f"Check 3 FAILED ({split}): dataset has {len(gt)} images, chunks "
            f"claim split_total={split_total}"
        )
        for c in split_chunks:
            chunk_lbls = np.asarray(c["labels"]).reshape(-1).astype(np.int64)
            assert np.array_equal(chunk_lbls, gt[c["start"]:c["end"]]), (
                f"Check 3 FAILED ({split}): labels in {c['path'].name} do not match "
                f"ground-truth labels[{c['start']}:{c['end']}] — chunk is mis-ordered "
                f"or mis-positioned"
            )

        feats = np.concatenate([c["features"] for c in split_chunks], axis=0)
        lbls = np.concatenate([np.asarray(c["labels"]).reshape(-1) for c in split_chunks], axis=0)
        assert feats.shape[0] == split_total == lbls.shape[0]
        merged[f"{split}_features"] = feats
        merged[f"{split}_labels"] = lbls
        counts[split] = split_total
        print(f"Checks 2+3 OK ({split}): {len(split_chunks)} chunk(s) tile [0,{split_total}); "
              f"labels match ground truth. shape {feats.shape}")

    # same metadata as a single run
    metadata = {k: ref[k] for k in IDENTITY_KEYS}
    metadata["channel_kernel_map"] = ref["channel_kernel_map"]
    metadata["created_at"] = datetime.now().isoformat()
    metadata["train_samples"] = counts["train"]
    metadata["val_samples"] = counts["val"]
    metadata["test_samples"] = counts["test"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / generate_output_filename(metadata)

    np.savez_compressed(
        out_path,
        train_features=merged["train_features"], train_labels=merged["train_labels"],
        val_features=merged["val_features"], val_labels=merged["val_labels"],
        test_features=merged["test_features"], test_labels=merged["test_labels"],
        metadata=json.dumps(metadata),
    )
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll checks passed. Merged cache written to:\n  {out_path}")
    print(f"  train={counts['train']}, val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()
