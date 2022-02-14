import os
from typing import List

import pandas as pd

from src.utils.const import SMILES_COLUMN, TARGET_COLUMN, PREDICTIONS_PATH


def save_prediction(test_smiles: List[str], predictions: List[int], filename: str,
                    predictions_dir: str = PREDICTIONS_PATH):
    test_submission = pd.DataFrame(columns=[SMILES_COLUMN, TARGET_COLUMN])
    test_submission[SMILES_COLUMN] = test_smiles
    test_submission[TARGET_COLUMN] = predictions

    if not os.path.exists(PREDICTIONS_PATH):
        os.mkdir(PREDICTIONS_PATH)

    test_submission.to_csv(os.path.join(predictions_dir, filename))
    return test_submission
