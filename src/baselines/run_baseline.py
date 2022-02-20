from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.data.get_fingerprints import get_np_array_of_fps
from src.data import Data
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, Fingerprints
from src.baselines.hyperparams_search import Algorithm, Params, HyperparamsSearch


def main():
    # Preprocessing
    print('\n Loading and preprocess data')
    train_data = Data(filename=TRAIN_FILE)
    smiles_train, y_train = train_data.get_processed_smiles_and_targets()

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    train_morgan_fp = get_np_array_of_fps(fp_type=Fingerprints.ECFP4, smiles=smiles_train)
    test_morgan_fp = get_np_array_of_fps(fp_type=Fingerprints.ECFP4, smiles=smiles_test)

    # Setting up model
    model = XGBClassifier()
    # model = CatBoostClassifier()

    folds = 4

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    search = HyperparamsSearch(
        Algorithm.RANDOM_SEARCH,
        model,
        Params.XGBOOST,
        cv=skf.split(train_morgan_fp, y_train),
        verbose=3
    )
    print("Start search\n")
    search.fit(train_morgan_fp, y_train)

    print(search)

    test_predictions = search.best_estimator().predict(test_morgan_fp)
    test_predictions_df = save_prediction(test_data.data[SMILES_COLUMN], test_predictions,
                                          "morgan_fp_2_2048_test_submission.csv")
    ## Use this method to save model:
    # search.save_best("model_name")
    return


if __name__ == "__main__":
    main()