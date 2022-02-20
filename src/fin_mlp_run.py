import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.get_balanced_data import get_balanced_data
from src.data.get_cv_splitter import get_cv_splitter
from src.data.get_fingerprints import get_np_array_of_fps
from src.data import Data
from src.data.read_pdb_file import get_targetpdb_smiles
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, CVSplitters, FingerprintsNames, \
    FINGERPRINTS_METHODS, ImbalanceStrategies


def main():
    print('\n Loading and preprocess data')

    train_data = Data(filename=TRAIN_FILE)
    imb_strategy = ImbalanceStrategies.OVERSAMPLE
    smiles_train, y_train = train_data.get_processed_smiles_and_targets(imb_strategy=imb_strategy)

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    fingerprint_type_name = FingerprintsNames.ECFP4
    fingerprint_type_method = FINGERPRINTS_METHODS[fingerprint_type_name]

    cv_type = CVSplitters.Scaffold_CV

    pdb_smiles = get_targetpdb_smiles()

    print(f'\n Get fingerprints {fingerprint_type_name}')
    pdb_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=[pdb_smiles])

    train_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_train)
    test_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_test)

    train_pdb_fp = np.tile(pdb_fp, (len(train_fp), 1))
    train_fp = np.hstack([train_fp, train_pdb_fp])

    test_pdb_fp = np.tile(pdb_fp, (len(test_fp), 1))
    test_fp = np.hstack([test_fp, test_pdb_fp])

    folds = 4

    sc = StandardScaler()
    train_fp_sc = sc.fit_transform(train_fp)
    test_fp_sc = sc.fit_transform(test_fp)

    model = MLPClassifier(max_iter=100, hidden_layer_sizes=(2048, 1024, 512, 256, 128, 2),
                          n_iter_no_change=5, activation='relu', solver='adam', batch_size=256,
                          random_state=69, verbose=True, early_stopping=False, alpha=1e-4, learning_rate_init=1e-5,
                          tol=1e-4)
    print("\n Start training")
    model.fit(train_fp_sc, y_train)
    print("\n Start evaluation")
    test_predictions = model.predict(test_fp_sc)
    test_predictions_df = save_prediction(test_data.data[SMILES_COLUMN], test_predictions,
                                          f"{fingerprint_type_name.value}_{CVSplitters.Scaffold_CV.value}_mlp_best_test_submission.csv")


if __name__ == "__main__":
    main()
