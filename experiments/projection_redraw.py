"""Re-run the capacity-sweep projection cells (random_*, rff_*) with the random
draw tied to the training seed.

The published sweep uses one draw (RANDOM_FILTER_SEED = 42) for every head and
seed, so its seed spread contains no projection noise. Here seed s drives the
draw, the head initialisation and the shuffling; the best hyperparameters of
each cell are reused (no new search). The head is seeded before it is built,
which the published sweep did not do, so compare per-seed draws against
--frozen-seed 42 runs of this script, never against the published values.

One JSON per (head, source, seed); existing ones are skipped.
--save-predictions also writes the per-example test scores.

    python experiments/projection_redraw.py --dataset breast_mnist \
        --cells outputs/paper_results/capacity_sweep_std/breast_mnist \
        --out outputs/camera_ready/a12/breast_mnist_varying --save-predictions
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_capacity_sweep as hcs

# sources with a random draw (poly-2 and the quantum kernels have none)
PROJECTION_SOURCES = ["random_9", "random_45", "random_180", "rff_45", "rff_180"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_mnist")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--cells", required=True,
                    help="existing sweep dir holding cells/<head>__<src>/summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", default=PROJECTION_SOURCES)
    ap.add_argument("--heads", nargs="+",
                    default=["linear", "mlp1", "mlp2", "cnn_small", "cnn_large"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--frozen-seed", type=int, default=None,
                    help="use this one draw for every training seed (the published protocol)")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip z-scoring (the published sweep is standardised)")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    hcs.DATASET = args.dataset
    hcs.NUM_CLASSES = args.num_classes
    hcs.BATCH_SIZE = args.batch_size
    hcs.EPOCHS = args.epochs
    hcs.FEATURE_SOURCES = hcs.build_feature_sources(args.dataset)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    mode = "frozen" if args.frozen_seed is not None else "per-seed"
    print(f"{args.dataset}: {len(args.sources)} sources x {len(args.heads)} heads "
          f"x {len(args.seeds)} seeds, projection draw = {mode}", flush=True)

    built_draw, built = None, None

    for source in args.sources:
        spec = hcs.FEATURE_SOURCES[source]
        built_draw, built = None, None
        best = {}
        for head in args.heads:
            summary = Path(args.cells) / "cells" / f"{head}__{source}" / "summary.json"
            if not summary.exists():
                print(f"  SKIP {head}__{source}: no summary.json at {summary}", flush=True)
                continue
            best[head] = json.load(open(summary))["best_params"]
        if not best:
            continue

        for seed in args.seeds:
            draw = args.frozen_seed if args.frozen_seed is not None else seed
            todo = [h for h in best
                    if not (out_root / f"{h}__{source}" / f"seed_{seed}.json").exists()]
            if not todo:
                continue

            # features are rebuilt only when the draw changes (once in frozen mode)
            if draw != built_draw:
                train, val, test, in_ch = hcs.load_features(source, {**spec, "seed": draw})
                if not args.no_standardize:
                    train, val, test = hcs.standardize_features(train, val, test)
                built_draw, built = draw, (train, val, test, in_ch)
            train, val, test, in_ch = built
            print(f"  {source} draw={draw} seed={seed}: features {tuple(train[0].shape)}",
                  flush=True)

            for head_name in todo:
                builder, use_scheduler = hcs._head_factory(head_name)
                hp = best[head_name]
                torch.manual_seed(seed)
                np.random.seed(seed)
                head = builder(args.num_classes, hp["dropout"], hcs.ACTIVATION, in_ch)
                metrics = hcs.train_head(
                    head, train, val, test,
                    lr=hp["lr"], weight_decay=hp["wd"],
                    use_scheduler=use_scheduler, device=args.device, seed=seed,
                    return_scores=args.save_predictions,
                )
                cell_dir = out_root / f"{head_name}__{source}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if args.save_predictions:
                    np.savez_compressed(cell_dir / f"seed_{seed}_preds.npz",
                                        scores=metrics.pop("test_scores"),
                                        labels=metrics.pop("test_labels"))
                record = {"seed": seed, "projection_draw": draw, "head": head_name,
                          "source": source, "in_channels": int(in_ch),
                          "params": hp, **metrics}
                with open(cell_dir / f"seed_{seed}.json", "w") as fh:
                    json.dump(record, fh, indent=2)
                print(f"    {head_name:<10} val {metrics['val_acc']:.4f}  "
                      f"test AUC {metrics['test_auc']:.4f}", flush=True)

    print(f"done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
