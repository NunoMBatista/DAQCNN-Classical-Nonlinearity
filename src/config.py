"""Project paths and the MedMNIST datasets the code supports (28x28 grayscale)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

# DAQCNN_QUANTUM_DATASETS_DIR points the cache lookup at another directory
# (e.g. a one-off cache with a different RFF draw).
QUANTUM_DATASETS_DIR = Path(
    os.environ.get("DAQCNN_QUANTUM_DATASETS_DIR", DATA_DIR / "quantum_datasets")
)

# dataset name -> (medmnist flag, medmnist class name)
DATASET_REGISTRY = {
    "pneumonia_mnist": ("pneumoniamnist", "PneumoniaMNIST"),
    "breast_mnist": ("breastmnist", "BreastMNIST"),
    "tissue_mnist": ("tissuemnist", "TissueMNIST"),
    "oct_mnist": ("octmnist", "OCTMNIST"),
    "organ_a_mnist": ("organamnist", "OrganAMNIST"),
    "organ_c_mnist": ("organcmnist", "OrganCMNIST"),
    "organ_s_mnist": ("organsmnist", "OrganSMNIST"),
}
