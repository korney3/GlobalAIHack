from rdkit import Chem
from rdkit.Chem import AllChem


def get_morgan_fingerprint(smiles: str, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return (list(morgan_fp))