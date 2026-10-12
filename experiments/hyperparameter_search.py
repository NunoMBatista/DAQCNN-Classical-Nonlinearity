"""Optuna (TPE) hyperparameter search for the end-to-end models, with optional
multi-seed validation of the best trials (this produced Table III).

The base config fixes the model; model.architecture is "original" (DAQCNN on
cached features, also used for the poly-2 / RFF caches) or "classical_baseline".
The search config has a `settings` block (metric, direction, study_name) and a
`search_space` block mapping dotted config keys to distributions:

    optim.lr: {type: loguniform, low: 1.0e-8, high: 5.0e-3}

Types: uniform (optional step), loguniform, int (step or log), categorical,
fixed (not searched, just overrides the base config).

Each trial trains one seed (--seed). --validate-top-k K re-trains the K best
trials on --validation-seeds and writes validation/validation_summary.json.
DAQCNN runs need the matching feature cache (experiments/create_quantum_dataset.py).

    python experiments/hyperparameter_search.py \\
        --config configs/breast_mnist/original/1_kernel_3x3_digital_zz.yml \\
        --search-config configs/breast_mnist/hp_search/digital_zz_single_kernel_search.yml \\
        --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import torch
import yaml

# list-valued categorical choices (topology lists) are intended
warnings.filterwarnings(
    "ignore",
    message=r"Choices for a categorical distribution should be a tuple.*persistent storage.*",
    category=UserWarning,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.classical_baseline_training import run_classical_baseline  # noqa: E402
from src.models.daqcnn_training import run_single_seed  # noqa: E402
from src.utils.quantum_dataset_cache import find_cached_quantum_dataset  # noqa: E402

METRIC_KEYS = ("test_acc", "test_auc", "test_f1", "test_loss", "test_recall")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """d["optim"]["lr"] = value for dotted_key "optim.lr"."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _deep_get(d: dict, dotted_key: str, default: Any = None) -> Any:
    keys = dotted_key.split(".")
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def _sample_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    dist_type = spec["type"].lower().strip()

    if dist_type in ("uniform", "float"):
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        if step is not None:
            return trial.suggest_float(name, low, high, step=float(step))
        return trial.suggest_float(name, low, high)

    elif dist_type in ("loguniform", "log_uniform", "logfloat", "log_float"):
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)

    elif dist_type in ("int", "integer"):
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        if bool(spec.get("log", False)):
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)

    elif dist_type in ("categorical", "cat", "choice"):
        return trial.suggest_categorical(name, spec["choices"])

    elif dist_type == "fixed":
        return spec["value"]

    else:
        raise ValueError(
            f"Unknown distribution type '{dist_type}' for parameter '{name}'. "
            f"Supported: uniform, loguniform, int, categorical, fixed."
        )


def build_trial_config(base_cfg: dict, search_space: dict, trial: optuna.Trial) -> dict:
    """A copy of base_cfg with every search-space entry sampled (or fixed) in."""
    cfg = copy.deepcopy(base_cfg)
    for dotted_key, spec in search_space.items():
        value = _sample_param(trial, dotted_key, spec)
        if value == "null" or value == "None":
            value = None
        _deep_set(cfg, dotted_key, value)
    return cfg


def _run_original(cfg, seed, output_dir):
    return run_single_seed(cfg, seed, output_dir, verbose=False, set_seed_fn=set_seed)


def _run_classical_baseline(cfg, seed, output_dir):
    return run_classical_baseline(cfg, seed, output_dir, verbose=False, set_seed_fn=set_seed)


_RUNNERS = {
    "original": _run_original,
    "classical_baseline": _run_classical_baseline,
}


class HPSearchObjective:
    """Optuna objective: build the trial config, train one seed, return the metric."""

    def __init__(self, base_cfg: dict, search_space: dict, architecture: str, seed: int,
                 output_root: str, metric: str = "test_acc"):
        if architecture not in _RUNNERS:
            raise ValueError(f"Unknown architecture '{architecture}'. Supported: {list(_RUNNERS)}")
        self.base_cfg = base_cfg
        self.search_space = search_space
        self.architecture = architecture
        self.seed = seed
        self.output_root = output_root
        self.metric = metric

    def __call__(self, trial: optuna.Trial) -> float:
        t_start = time.time()
        cfg = build_trial_config(self.base_cfg, self.search_space, trial)

        # without a cache the quantum layer would be simulated on every batch
        if self.architecture == "original" and find_cached_quantum_dataset(cfg) is None:
            sys.exit(
                f"No cached features match trial {trial.number}'s config. Build the cache first "
                f"with experiments/create_quantum_dataset.py (model.kernel_topology_names = "
                f"{cfg['model'].get('kernel_topology_names')})."
            )

        trial_dir = os.path.join(self.output_root, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        s = self.seed
        seed_dir = os.path.join(trial_dir, f"seed_{s}")
        os.makedirs(seed_dir, exist_ok=True)
        cfg_s = copy.deepcopy(cfg)
        cfg_s.setdefault("misc", {})["seed"] = s
        try:
            set_seed(s)
            result = _RUNNERS[self.architecture](cfg_s, s, seed_dir)
        except Exception as e:
            with open(os.path.join(seed_dir, "error.txt"), "w") as ef:
                ef.write(f"Trial {trial.number} seed {s} failed:\n")
                ef.write(traceback.format_exc())
            warnings.warn(f"Trial {trial.number} seed {s} failed: {e}")
            raise optuna.TrialPruned("All seeds failed for this trial")

        if result.get(self.metric) is None:
            warnings.warn(f"Trial {trial.number}: metric '{self.metric}' is None.")
            raise optuna.TrialPruned(f"Metric '{self.metric}' is None")
        obj_value = float(result[self.metric])

        # the "_std" entries are 0.0 (one seed); kept so the files keep their format
        for key in METRIC_KEYS:
            if result.get(key) is not None:
                trial.set_user_attr(key, float(result[key]))
                trial.set_user_attr(f"{key}_std", 0.0)
        trial.set_user_attr(f"{self.metric}_std", 0.0)
        trial.set_user_attr("n_seeds_completed", 1)
        trial.set_user_attr("trial_seeds", [s])
        trial.set_user_attr("duration_s", time.time() - t_start)
        if result.get("train_losses"):
            trial.set_user_attr("train_losses", result["train_losses"])
        if result.get("val_losses"):
            trial.set_user_attr("val_losses", result["val_losses"])

        agg_summary: dict = {
            "trial_number": trial.number,
            "seeds": [s],
            "n_seeds_completed": 1,
            f"{self.metric}_mean": obj_value,
            f"{self.metric}_std": 0.0,
        }
        for key in METRIC_KEYS:
            if result.get(key) is not None:
                agg_summary[f"{key}_mean"] = float(result[key])
                agg_summary[f"{key}_std"] = 0.0
        with open(os.path.join(trial_dir, "result.json"), "w") as f:
            json.dump(agg_summary, f, indent=2, default=str)

        print(
            f"  Trial {trial.number:4d} | "
            f"{self.metric}={obj_value:.4f}±0.0000 | "
            f"acc={float(result.get('test_acc') or 0.0):.4f}±0.0000 "
            f"auc={float(result.get('test_auc') or 0.0):.4f} "
            f"f1={float(result.get('test_f1') or 0.0):.4f} | "
            f"1/1 seeds | {time.time() - t_start:.1f}s"
        )
        return obj_value


def validate_top_k(
    study: optuna.Study,
    base_cfg: dict,
    search_space_keys: List[str],
    architecture: str,
    output_dir: str,
    *,
    top_k: int = 5,
    seeds: Optional[List[int]] = None,
    search_space: Optional[dict] = None,
) -> List[dict]:
    """Re-train the top_k completed trials on each validation seed; one aggregate dict per trial."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    runner = _RUNNERS[architecture]

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(trials, key=lambda t: t.value or -float("inf"), reverse=True)
    else:
        sorted_trials = sorted(trials, key=lambda t: t.value or float("inf"))
    top_trials = sorted_trials[:top_k]

    print(f"\n{'=' * 60}")
    print(f"Validating Top-{min(top_k, len(top_trials))} Configurations")
    print(f"Seeds: {seeds}")
    print(f"{'=' * 60}\n")

    validation_results = []
    for rank, trial in enumerate(top_trials, 1):
        cfg = copy.deepcopy(base_cfg)
        # `fixed` entries never appear in trial.params, so re-apply them here
        # (validation configs used to fall back to the base config for them)
        for key, spec in (search_space or {}).items():
            if isinstance(spec, dict) and spec.get("type") == "fixed":
                _deep_set(cfg, key, spec["value"])
        for key in search_space_keys:
            if key in trial.params:
                _deep_set(cfg, key, trial.params[key])

        val_dir = os.path.join(output_dir, f"validation_rank{rank}_trial{trial.number}")
        os.makedirs(val_dir, exist_ok=True)
        with open(os.path.join(val_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(f"  Rank {rank} (trial {trial.number}, search {study.direction.name}={trial.value:.4f})")
        print(f"  Params: {trial.params}")

        seed_results = []
        for seed in seeds:
            seed_dir = os.path.join(val_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            try:
                set_seed(seed)
                result = runner(cfg, seed, seed_dir)
                seed_results.append(result)
                print(f"    seed={seed}: acc={result.get('test_acc', 0):.4f} "
                      f"auc={result.get('test_auc', 0):.4f}")
            except Exception as e:
                print(f"    seed={seed}: FAILED ({e})")

        if not seed_results:
            continue

        agg: Dict[str, Any] = {
            "rank": rank,
            "trial_number": trial.number,
            "search_value": trial.value,
            "params": trial.params,
        }
        for key in ("test_acc", "test_auc", "test_f1", "test_recall", "test_loss"):
            vals = [r[key] for r in seed_results if r.get(key) is not None]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

        with open(os.path.join(val_dir, "validation_aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2, default=str)
        validation_results.append(agg)

        print(f"  -> Validated: acc={agg.get('test_acc', {}).get('mean', 0):.4f} "
              f"+/- {agg.get('test_acc', {}).get('std', 0):.4f}\n")

    return validation_results


def export_best_config(study: optuna.Study, base_cfg: dict, search_space: dict, output_path: str,
                       *, validation_seeds: Optional[List[int]] = None) -> str:
    """Write the best trial's config (searched values filled in) as YAML."""
    best = study.best_trial
    cfg = copy.deepcopy(base_cfg)
    for key in search_space:
        if key in best.params:
            _deep_set(cfg, key, best.params[key])

    if validation_seeds:
        cfg.setdefault("misc", {})["seed"] = validation_seeds
    else:
        cfg.setdefault("misc", {})["seed"] = _deep_get(base_cfg, "misc.seed", [0, 1, 2])

    with open(output_path, "w") as f:
        f.write("# Auto-generated by hyperparameter_search.py\n")
        f.write(f"# Best trial: {best.number} (value={best.value:.6f})\n")
        f.write(f"# Tuned params: {best.params}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the end-to-end models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True, help="Base YAML config file.")
    parser.add_argument("--search-config", type=str, required=True, help="Search space YAML file.")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="New Optuna trials to run (default: 50; 0 = only export/validate).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed of every trial, also the TPE sampler seed (default: 42).")
    parser.add_argument("--metric", type=str, default=None,
                        help="Metric to optimise (default: settings.metric of the search config, else test_acc).")
    parser.add_argument("--direction", type=str, default=None, choices=["maximize", "minimize"],
                        help="Default: settings.direction of the search config, else maximize.")
    parser.add_argument("--resume", action="store_true", help="Continue the study in <output-dir>/study.db.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: outputs/hp_search_<architecture>_<dataset>_<timestamp>.")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Default: settings.study_name of the search config.")
    parser.add_argument("--validate-top-k", type=int, default=0,
                        help="After the search, re-train the top K trials on the validation seeds.")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None,
                        help="Seeds for the validation (default: 0 1 2 3 4).")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    search_cfg = load_yaml(args.search_config)
    search_space = search_cfg.get("search_space", {})
    if not search_space:
        parser.error("search_space section is empty or missing in search config.")

    search_settings = search_cfg.get("settings", {})
    metric = args.metric or search_settings.get("metric", "test_acc")
    direction = args.direction or search_settings.get("direction", "maximize")
    n_trials = args.n_trials
    seed = args.seed

    architecture = base_cfg.get("model", {}).get("architecture", "original")
    dataset_name = base_cfg.get("dataset", {}).get("name", "unknown")
    dataset_slug = dataset_name.replace("_", "")
    study_name = args.study_name or search_settings.get("study_name") or f"hp_{architecture}_{dataset_slug}"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", f"hp_search_{architecture}_{dataset_slug}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "study.db")
    storage = f"sqlite:///{os.path.abspath(db_path)}"

    print("\n" + "=" * 70)
    print(f"  Architecture:    {architecture}")
    print(f"  Base config:     {args.config}")
    print(f"  Search config:   {args.search_config}")
    print(f"  Study name:      {study_name}")
    print(f"  Metric:          {metric} ({direction})")
    print(f"  Trials:          {n_trials}  (seed {seed})")
    print(f"  Output:          {output_dir}")
    for key, spec in search_space.items():
        print(f"    {key:40s} {spec}  (base: {_deep_get(base_cfg, key)})")
    print("=" * 70 + "\n")

    with open(os.path.join(output_dir, "base_config.yml"), "w") as f:
        yaml.dump(base_cfg, f, default_flow_style=False)
    with open(os.path.join(output_dir, "search_config.yml"), "w") as f:
        yaml.dump(search_cfg, f, default_flow_style=False)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=args.resume,
    )
    existing_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if existing_trials > 0:
        print(f"Resuming study with {existing_trials} existing completed trials.\n")

    trials_dir = os.path.join(output_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)
    objective = HPSearchObjective(base_cfg=base_cfg, search_space=search_space, architecture=architecture,
                                  seed=seed, output_root=trials_dir, metric=metric)

    t_search_start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))
    search_time = time.time() - t_search_start

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 70)
    print(f"  Search time:    {search_time:.1f}s")
    print(f"  Completed:      {len(completed)} / {len(study.trials)} trials")
    print(f"  Failed/pruned:  {len(failed)}")
    if completed:
        best = study.best_trial
        print(f"\n  Best trial:     #{best.number}")
        print(f"  Best {metric}:  {best.value:.6f}")
        for k, v in best.params.items():
            print(f"    {k:40s} = {v:.6g}" if isinstance(v, float) else f"    {k:40s} = {v}")
        for k in METRIC_KEYS:
            if best.user_attrs.get(k) is not None:
                print(f"    {k:40s} = {best.user_attrs[k]:.6f}")
    print(f"{'=' * 70}\n")

    if completed:
        best_cfg_path = os.path.join(output_dir, "best_config.yml")
        export_best_config(study, base_cfg, search_space, best_cfg_path,
                           validation_seeds=args.validation_seeds or [0, 1, 2, 3, 4])
        print(f"Best config exported to: {best_cfg_path}")

    if args.validate_top_k > 0 and completed:
        val_dir = os.path.join(output_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        val_results = validate_top_k(
            study,
            base_cfg,
            list(search_space.keys()),
            architecture,
            val_dir,
            top_k=args.validate_top_k,
            seeds=args.validation_seeds or [0, 1, 2, 3, 4],
            search_space=search_space,
        )
        val_summary_path = os.path.join(val_dir, "validation_summary.json")
        with open(val_summary_path, "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        print(f"Validation summary saved to: {val_summary_path}")

    best = study.best_trial if completed else None
    summary = {
        "study_name": study_name,
        "architecture": architecture,
        "dataset": dataset_name,
        "metric": metric,
        "direction": direction,
        "n_trials_requested": n_trials,
        "n_trials_completed": len(completed),
        "n_trials_failed": len(failed),
        "search_time_s": search_time,
        "best_trial": best.number if best is not None else None,
        "best_value": best.value if best is not None else None,
        "best_params": best.params if best is not None else None,
        "db_path": db_path,
    }
    with open(os.path.join(output_dir, "search_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
