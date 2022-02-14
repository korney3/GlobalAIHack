import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent
package_path = get_project_root()

DATA_PATH = os.path.join(package_path, "./data")
PREDICTIONS_PATH = os.path.join(package_path, "./predictions")
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

SMILES_COLUMN = "Smiles"
TARGET_COLUMN = "Active"

THREADS = os.cpu_count()

SCORING="f1"
