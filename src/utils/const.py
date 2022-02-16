import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


package_path = get_project_root()

DATA_PATH = os.path.join(package_path, "./data")
PREDICTIONS_PATH = os.path.join(package_path, "./predictions")
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

SMILES_COLUMN = "Smiles"
TARGET_COLUMN = "Active"


@dataclass
class Fingerprints:
    ECFP4 = partial(AllChem.GetMorganFingerprintAsBitVect, radius=2, nBits=2048)
    TOPOTORSION = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect
    MACCS = MACCSkeys.GenMACCSKeys
    RDKitFP = Chem.RDKFingerprint
    PATTERN = Chem.PatternFingerprint
    ATOMPAIR = AllChem.GetHashedAtomPairFingerprintAsBitVect


class CVSplitters(Enum):
    Stratified_CV = "stratified_cv"
    Scaffold_CV = "scaffold_cv"
