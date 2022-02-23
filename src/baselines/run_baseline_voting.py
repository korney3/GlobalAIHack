from xgboost import XGBClassifier
from src.data import Data
from src.data.get_fingerprints import get_np_array_of_fps
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, FINGERPRINTS_METHODS, FingerprintsNames
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import numpy as np


def main():
    np.random.seed(42)

    print("Loading and preprocessing data\n")

    train_data = Data(filename=TRAIN_FILE)
    test_data = Data(filename=TEST_FILE)

    smiles_train, y_train = train_data.get_processed_smiles_and_targets()
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    fingerprint_type_name = FingerprintsNames.MACCS
    fingerprint_type_method = FINGERPRINTS_METHODS[fingerprint_type_name]

    train_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_train)
    test_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_test)

    xgb = XGBClassifier(
        subsample=1.0,
        scale_pos_weight=35,
        n_estimators=1250,
        max_depth=100,
        gamma=0.5,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        nthread=-1,
        verbose=3,
    )

    svm = SVC(
        kernel="poly",
        degree=3,
        C=5,
        random_state=42,
    )

    votint_clf = VotingClassifier(
        estimators=[("xgb", xgb), ("svm", svm)],
        voting="hard",
        n_jobs=-1,
        verbose=True
    )

    votint_clf.fit(train_fp, y_train)

    test_predictions = votint_clf.predict(test_fp)

    save_prediction(
        test_data.data[SMILES_COLUMN],
        test_predictions,
        f"{fingerprint_type_name.value}_vote.csv"
    )


if __name__ == "__main__":
    main()
