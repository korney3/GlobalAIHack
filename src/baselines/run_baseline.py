from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from src.data import Data
from src.data.get_fingerprints import get_np_array_of_fps
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, FINGERPRINTS_METHODS, FingerprintsNames
from src.baselines.hyperparams_search import Algorithm, Params, HyperparamsSearch


def main():
    print("Loading and preprocessing data\n")

    train_data = Data(filename=TRAIN_FILE)
    test_data = Data(filename=TEST_FILE)

    smiles_train, y_train = train_data.get_processed_smiles_and_targets()
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    fingerprint_type_name = FingerprintsNames.MACCS
    fingerprint_type_method = FINGERPRINTS_METHODS[fingerprint_type_name]

    train_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_train)
    test_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_test)

    model = XGBClassifier(verbose=3)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1001)

    search = HyperparamsSearch(
        Algorithm.RANDOM_SEARCH,
        model,
        Params.XGBOOST,
        cv=skf.split(train_fp, y_train),
        verbose=3
    )

    print("Searching hyperparams\n")
    search.fit(train_fp, y_train)

    print(search)

    test_predictions = search.best_estimator().predict(test_fp)
    save_prediction(
        test_data.data[SMILES_COLUMN],
        test_predictions,
        "MACCSkeys_xgboost.csv"
    )

    search.save_best(f"{fingerprint_type_name.value}_xgboost")


if __name__ == "__main__":
    main()
