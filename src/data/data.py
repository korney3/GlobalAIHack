import os
from typing import List

import pandas as pd
from rdkit import Chem

from src.utils.const import DATA_PATH, TRAIN_FILE, SMILES_COLUMN, TARGET_COLUMN
from rdkit.Chem.SaltRemover import SaltRemover


class Data():
    def __init__(self, data_dir: str = DATA_PATH, filename: str = TRAIN_FILE):
        self.data_dir = data_dir
        self.filename = filename
        self.smiles_column = SMILES_COLUMN
        self.y_column = TARGET_COLUMN

        self.data = None
        self.smiles = None
        self.targets = None
        self.indices = None

        self.target_map = {True: 1, False: 0}

    def __getitem__(self, item):
        return self.smiles[item], self.indices[item], self.targets[item]

    def get_processed_smiles_and_targets(self):
        self.data = self.load_train_data()
        list_of_smiles = self.data[self.smiles_column]
        processed_list_of_smiles = self.process_list_of_smiles(list_of_smiles)
        self.smiles = processed_list_of_smiles
        if self.y_column in list(self.data):
            self.targets = self.change_str_target_to_int(self.data[self.y_column])
        self.indices = list(range(len(self.smiles)))
        return self.smiles, self.targets

    def load_train_data(self):
        path = os.path.join(self.data_dir, self.filename)
        data = pd.read_csv(path)
        return data

    def process_list_of_smiles(self, list_of_smiles: List[str]):
        processed_list_of_smiles = list(map(lambda x: self.remove_salts_and_canonicalized(x), list_of_smiles))
        return processed_list_of_smiles

    def remove_salts_and_canonicalized(self, smiles: str):
        remover = SaltRemover(defnData="[Cl,Br]")
        mol = Chem.MolFromSmiles(smiles)
        res = remover.StripMol(mol)
        processed_smiles = Chem.MolToSmiles(mol)
        return processed_smiles

    def change_str_target_to_int(self, targets:  pd.Series):
        processed_targets = targets.map(self.target_map)
        return processed_targets.values
