#!/usr/bin/env python3

'''
Use this script to load and run saved models
'''

from src.data import Data
from src.data.get_fingerprints import get_np_array_of_fps
from src.data.save_predictions import save_prediction
from src.utils.const import TEST_FILE, FINGERPRINTS_METHODS, SMILES_COLUMN, FingerprintsNames
from typing import Optional
from xgboost import XGBClassifier
import argparse
import os


def _load_model(path: str):
    if not os.path.exists(path):
        raise Exception(f"No such file: {path}")

    model = XGBClassifier()
    model.load_model(path)
    return model


def _get_fingerprints(data: Data, type_name: str):
    if not type_name in FingerprintsNames._member_names_:
        raise Exception(
            f"No such fingerprint method: {type_name}\n"
            f"Available methods:\n{FingerprintsNames._member_names_}"
        )

    method = FINGERPRINTS_METHODS[FingerprintsNames[type_name]]
    smiles, _ = data.get_processed_smiles_and_targets()

    test_fps = get_np_array_of_fps(fp_type=method, smiles=smiles)
    return test_fps


def main(model_path: str, fp_type: str, output: Optional[str]):
    model = _load_model(model_path)

    test_data = Data(filename=TEST_FILE)
    test_fp = _get_fingerprints(test_data, fp_type)

    test_predictions = model.predict(test_fp)

    if output is None:
        output = f"{os.path.basename(model_path)}.csv"
    save_prediction(test_data.data[SMILES_COLUMN], test_predictions, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="path to file with model", required=True)
    parser.add_argument("-f", "--fingerprints", help="type of fingerprints", required=True)
    parser.add_argument("-o", "--output", help="name of output file", required=False)
    args = parser.parse_args()

    try:
        main(args.model, args.fingerprints, args.output)
    except Exception as e:
        print(f"Got error:\n{e}")
