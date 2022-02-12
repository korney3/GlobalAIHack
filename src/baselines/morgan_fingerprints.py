from rdkit import Chem
from rdkit.Chem import AllChem


def get_morgan_fingerprint(smiles: str, radius=3, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return (list(morgan_fp))