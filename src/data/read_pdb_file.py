import os

from rdkit.Chem import MolFromPDBFile, MolToSmiles

from src.utils.const import DATA_PATH, TARGETPDB_FILE


def get_targetpdb_smiles():
    mol = MolFromPDBFile(os.path.join(DATA_PATH, TARGETPDB_FILE))
    smiles = MolToSmiles(mol)
    return smiles
