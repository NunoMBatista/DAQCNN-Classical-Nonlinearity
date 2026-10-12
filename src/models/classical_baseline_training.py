"""End-to-end training of one seed of the classical baselines (random, trainable, raw pixels)."""

from src.models.classical_baseline import ClassicalBaselineCNN
from src.models.daqcnn_training import fit_and_evaluate
from src.utils.data import get_dataloaders
from src.utils.training_utils import resolve_device


def run_classical_baseline(cfg, seed, output_dir, verbose=True, set_seed_fn=None):
    if set_seed_fn:
        set_seed_fn(seed)

    device = resolve_device(cfg, verbose=verbose)
    train_loader, val_loader, test_loader, n_classes = get_dataloaders(cfg)

    model_cfg = cfg.get("model", {})
    num_classes = model_cfg.get("num_classes", n_classes)
    model = ClassicalBaselineCNN(
        num_classes=num_classes,
        in_channels=model_cfg.get("in_channels", 1),
        kernel_size=model_cfg.get("kernel_size", 3),
        stride=model_cfg.get("stride", 3),
        out_channels=model_cfg.get("out_channels", 180),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        head_hidden_channels=model_cfg.get("head_hidden_channels", 64),
        fixed_random_filters=model_cfg.get("fixed_random_filters", False),
        raw_features=model_cfg.get("raw_features", False),
    )
    model.to(device)

    return fit_and_evaluate(model, train_loader, val_loader, test_loader, cfg, seed, output_dir,
                            num_classes, device, verbose)
