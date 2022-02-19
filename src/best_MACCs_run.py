import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.data.get_balanced_data import get_balanced_data
from src.data.get_cv_splitter import get_cv_splitter
from src.data.get_fingerprints import get_np_array_of_fps
from src.data import Data
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, CVSplitters, FingerprintsNames, \
    FINGERPRINTS_METHODS, ImbalanceStrategies


def main():
    print('\n Loading and preprocess data')

    train_data = Data(filename=TRAIN_FILE)
    imb_strategy = None
    smiles_train, y_train = train_data.get_processed_smiles_and_targets(imb_strategy=imb_strategy)

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    fingerprint_type_name = FingerprintsNames.MACCS
    fingerprint_type_method = FINGERPRINTS_METHODS[fingerprint_type_name]

    cv_type = CVSplitters.Scaffold_CV

    print(f'\n Get fingerprints {fingerprint_type_name}')
    train_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_train)
    test_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_test)

    folds = 4

    xgb = XGBClassifier(subsample=1, scale_pos_weight=35, reg_lambda=20, n_estimators=2000, min_child_weight=0.5,
                        max_depth=100, learning_rate=0.2, gamma=0.5, colsample_bytree=0.7, colsample_bylevel=0.7,
                        use_label_encoder=False, nthread=-1)
    print("\n Start training")
    xgb.fit(train_fp, y_train)
    print("\n Start evaluation")
    test_predictions = xgb.predict(test_fp)
    test_predictions_df = save_prediction(test_data.data[SMILES_COLUMN], test_predictions,
                                          f"{fingerprint_type_name.value}_{CVSplitters.Scaffold_CV.value}_best_test_submission.csv")


if __name__ == "__main__":
    main()
