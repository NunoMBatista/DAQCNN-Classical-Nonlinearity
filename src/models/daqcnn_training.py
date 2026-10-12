"""End-to-end training of one seed (used by the HP search for trials and validation)."""

import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.daqcnn import DAQCNN
from src.utils.data import get_dataloaders
from src.utils.evaluate import accuracy, evaluate
from src.utils.plotting import plot_confusion_matrix, plot_loss_curves, plot_roc_curve
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset, load_cached_quantum_dataset
from src.utils.training_utils import resolve_device


def get_model_metadata(model):
    """Parameter counts and a description of the head layers (stored with each result)."""
    metadata = {
        "total_params": 0,
        "trainable_params": 0,
        "head_structure": [],
        "quantum_kernel_size": None,
        "quantum_kernel_topologies": [],
    }

    if hasattr(model, "quantum_convolutional_layer"):
        q_layer = model.quantum_convolutional_layer
        metadata["quantum_kernel_size"] = q_layer.kernel_size
        metadata["quantum_kernel_topologies"] = list(q_layer.quantum_kernel.kernel_topology_names)

    for param in model.parameters():
        metadata["total_params"] += param.numel()
        if param.requires_grad:
            metadata["trainable_params"] += param.numel()

    for i, layer in enumerate(model.head):
        layer_info = {"index": i, "type": layer.__class__.__name__}
        if isinstance(layer, nn.Conv2d):
            layer_info.update({
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding,
            })
        elif isinstance(layer, nn.Linear):
            layer_info["in_features"] = layer.in_features
            layer_info["out_features"] = layer.out_features
        elif isinstance(layer, nn.BatchNorm2d):
            layer_info["num_features"] = layer.num_features
        elif isinstance(layer, nn.Dropout):
            layer_info["p"] = layer.p
        elif isinstance(layer, nn.MaxPool2d):
            layer_info["kernel_size"] = layer.kernel_size
            layer_info["stride"] = layer.stride
        metadata["head_structure"].append(layer_info)

    return metadata


def train_one_epoch(model, loader, optim, device, grad_clip):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optim.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        total_count += bs

    return total_loss / total_count, total_acc / total_count


def fit_and_evaluate(model, train_loader, val_loader, test_loader, cfg, seed, output_dir,
                     num_classes, device, verbose=True):
    """Train with Adam (optional cosine schedule) and early stopping on validation
    loss, save both checkpoints and the per-seed plots, return the metrics.

    best_model_state holds references to the live tensors, not a copy, so the
    "restore best checkpoint" step below is a no-op and every reported number
    (and best_model_seed_k.pt) is from the last epoch. This is the protocol of
    the published runs: do not change it to a copy.
    """
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    epochs = optim_cfg.get("epochs", 1)
    grad_clip = optim_cfg.get("grad_clip", 0.0)
    patience = optim_cfg.get("patience", None)

    # frozen random filters (fixed_random_filters) stay out of the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )

    scheduler = None
    if optim_cfg.get("use_scheduler", False):
        T_max = optim_cfg.get("scheduler_T_max", epochs)
        eta_min = float(optim_cfg.get("scheduler_eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in tqdm(range(1, epochs + 1), desc=f"Seed {seed}"):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, device, split_name="Val")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if verbose:
            tqdm.write(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            if patience is not None:
                epochs_without_improvement = 0
        elif patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping after {epoch} epochs (patience={patience})")
                break

        if scheduler is not None:
            scheduler.step()

    model_metadata = get_model_metadata(model)
    if verbose:
        print(f"\nModel parameters: {model_metadata['total_params']:,} total, "
              f"{model_metadata['trainable_params']:,} trainable")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "num_classes": num_classes,
        },
        os.path.join(output_dir, f"final_model_seed_{seed}.pt"),
    )
    if best_model_state is not None:
        torch.save(
            {
                "model_state_dict": best_model_state,
                "config": cfg,
                "seed": seed,
                "num_classes": num_classes,
                "best_val_loss": best_val_loss,
            },
            os.path.join(output_dir, f"best_model_seed_{seed}.pt"),
        )
        model.load_state_dict(best_model_state)  # no-op, see docstring

    val_metrics = evaluate(model, val_loader, device, split_name="Val", compute_full_metrics=True)
    test_metrics = evaluate(model, test_loader, device, split_name="Test", compute_full_metrics=True)

    if verbose:
        print(f"Test (seed={seed}): loss {test_metrics['loss']:.4f}  acc {test_metrics['accuracy']:.4f}  "
              f"AUC {test_metrics['auc']:.4f}  F1 {test_metrics['f1']:.4f}  recall {test_metrics['recall']:.4f}")

    plot_loss_curves(train_losses, val_losses, os.path.join(output_dir, f"loss_curve_seed_{seed}.png"))
    plot_roc_curve(test_metrics["labels"], test_metrics["probs"],
                   os.path.join(output_dir, f"roc_curve_seed_{seed}.png"), num_classes=num_classes)
    plot_confusion_matrix(test_metrics["confusion_matrix"],
                          os.path.join(output_dir, f"confusion_matrix_seed_{seed}.png"))

    return {
        "seed": seed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "val_acc": val_metrics["accuracy"],
        "val_auc": val_metrics["auc"],
        "val_loss": val_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_recall": test_metrics["recall"],
        "test_probs": test_metrics["probs"].tolist(),
        "test_labels": test_metrics["labels"].tolist(),
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "num_classes": num_classes,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "model_metadata": model_metadata,
    }


def run_single_seed(cfg, seed, output_dir, verbose=True, set_seed_fn=None):
    """Train and evaluate DAQCNN on the cached features matching cfg (images if no cache)."""
    if set_seed_fn:
        set_seed_fn(seed)

    device = resolve_device(cfg, verbose=verbose)
    model_cfg = cfg.get("model", {})

    cached_path = find_cached_quantum_dataset(cfg)
    if cached_path is not None:
        if verbose:
            print(f"Using cached features: {cached_path}")
        train_loader, val_loader, test_loader, n_classes, cache_meta = load_cached_quantum_dataset(
            cached_path,
            cfg.get("dataset", {}).get("batch_size", 32),
            cfg.get("dataset", {}).get("num_workers", 2),
            requested_kernels=model_cfg.get("kernel_topology_names", None),
        )
    else:
        if verbose:
            print("No cached features found, the quantum layer runs on every batch.")
        train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)
        cache_meta = None

    num_classes = model_cfg.get("num_classes", n_classes)
    model_kwargs = {
        "num_classes": num_classes,
        "kernel_size": model_cfg.get("kernel_size", 3),
        "stride": model_cfg.get("stride", 1),
        "kernel_topology_names": model_cfg.get("kernel_topology_names", None),
        "scaling_factor": model_cfg.get("scaling_factor", 1.0),
        "evolution_time": model_cfg.get("evolution_time", 0.2),
        "mode": model_cfg.get("mode", "trotter"),
        "dropout": model_cfg.get("dropout", 0.1),
        "activation": model_cfg.get("activation", "relu"),
        "quantum_device": model_cfg.get("quantum_device", "default.qubit"),
        "quantum_device_kwargs": model_cfg.get("quantum_device_kwargs", None),
        "classical_device": device,
        "interface": model_cfg.get("interface", "torch"),
        "include_correlators": model_cfg.get("include_correlators", False),
        "head_hidden_channels": model_cfg.get("head_hidden_channels", 64),
    }
    if cache_meta:
        model_kwargs["override_quantum_out_channels"] = cache_meta.get("out_channels")

    model = DAQCNN(**model_kwargs)
    if cache_meta is not None:
        model.bypass_quantum = True
    model.to(device)

    return fit_and_evaluate(model, train_loader, val_loader, test_loader, cfg, seed, output_dir,
                            num_classes, device, verbose)
