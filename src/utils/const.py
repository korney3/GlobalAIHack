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
MODELS_PATH = os.path.join(package_path, "./models")
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TARGETPDB_FILE = "target.pdb"

SMILES_COLUMN = "Smiles"
TARGET_COLUMN = "Active"

THREADS = os.cpu_count()

SCORING = "f1"

class ImbalanceStrategies(Enum):
    UNDERSAMPLE = "udersample"
    OVERSAMPLE = "oversample"

class FingerprintsNames(Enum):
    TOPOTORSION = "topological_torsion"
    MACCS = "MACCSkeys"
    RDKitFP = "RDKFingerprint"
    PATTERN = "PatternFingerprint"
    ATOMPAIR = "AtomPairFingerprint"
    ECFP4 = "morgan_2_2048"


FINGERPRINTS_METHODS = {FingerprintsNames.TOPOTORSION: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect,
                        FingerprintsNames.MACCS: MACCSkeys.GenMACCSKeys,
                        FingerprintsNames.RDKitFP: Chem.RDKFingerprint,
                        FingerprintsNames.PATTERN: Chem.PatternFingerprint,
                        FingerprintsNames.ATOMPAIR: AllChem.GetHashedAtomPairFingerprintAsBitVect,
                        FingerprintsNames.ECFP4: partial(AllChem.GetMorganFingerprintAsBitVect, radius=2, nBits=2048)
                        }


class CVSplitters(Enum):
    Stratified_CV = "stratified_cv"
    Scaffold_CV = "scaffold_cv"
