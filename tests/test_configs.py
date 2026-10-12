"""Every config parses and is consistent with the code that reads it."""

import glob
import os

import optuna
import pytest
import yaml

import hyperparameter_search as hs
from create_quantum_dataset import generate_output_filename, load_params
from src.models.daqcnn import DAQCNN
from src.physics.kernel_topologies import get_3x3_kernel_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yml"), recursive=True))
SEARCH = [p for p in CONFIGS if "hp_search" in p]
BASE = [p for p in CONFIGS if p not in SEARCH]
TOPOLOGIES = set(get_3x3_kernel_set())


def test_configs_exist():
    assert len(CONFIGS) == 88


@pytest.mark.parametrize("path", BASE, ids=lambda p: os.path.relpath(p, ROOT))
def test_base_config(path):
    cfg = yaml.safe_load(open(path))
    assert cfg["dataset"]["name"] in ("breast_mnist", "pneumonia_mnist", "tissue_mnist")
    assert cfg["dataset"]["color_space"] == "GRAYSCALE"
    model = cfg["model"]
    if model.get("architecture", "original") == "original":
        assert model["kernel_size"] == 3
        assert set(model["kernel_topology_names"]) <= TOPOLOGIES
        if "num_classes" in model:  # end-to-end and capacity configs, not cache-generation ones
            DAQCNN(num_classes=model["num_classes"], kernel_size=3, stride=model["stride"],
                   kernel_topology_names=model["kernel_topology_names"],
                   evolution_time=model["evolution_time"], include_correlators=model["include_correlators"])


@pytest.mark.parametrize("path", SEARCH, ids=lambda p: os.path.relpath(p, ROOT))
def test_search_config_samples(path):
    scfg = yaml.safe_load(open(path))
    assert scfg["settings"]["metric"] == "val_acc"   # never a test metric
    trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42)).ask()
    cfg = hs.build_trial_config({}, scfg["search_space"], trial)
    for key, spec in scfg["search_space"].items():
        if spec["type"] == "fixed":
            assert hs._deep_get(cfg, key) == spec["value"]


def test_cache_file_name():
    p = load_params(os.path.join(ROOT, "configs", "tsweep", "breast_mnist__analog_T0.2.yml"))
    assert generate_output_filename(p) == "breast_mnist__k3_s3_tkin-cro-sta_ev0.20_sc1_gray_zz_analog.npz"
    p = load_params(os.path.join(ROOT, "configs", "pneumonia_mnist", "cache_generation", "digital_z.yml"))
    assert generate_output_filename(p) == \
        "pneumonia_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray.npz"


def test_validation_reapplies_fixed_entries(tmp_path):
    # `fixed` entries are not trial params; validation must still apply them
    space = {"model.fixed_random_filters": {"type": "fixed", "value": True},
             "optim.lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    hs.build_trial_config({}, space, trial)
    study.tell(trial, 0.5)

    seen = []

    def fake_runner(cfg, seed, output_dir):
        seen.append(cfg)
        return {"test_acc": 0.5}

    monkey = pytest.MonkeyPatch()
    monkey.setitem(hs._RUNNERS, "classical_baseline", fake_runner)
    try:
        hs.validate_top_k(study, {"model": {}}, list(space), "classical_baseline", str(tmp_path),
                          top_k=1, seeds=[0], search_space=space)
    finally:
        monkey.undo()
    assert seen[0]["model"]["fixed_random_filters"] is True
