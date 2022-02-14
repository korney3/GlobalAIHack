from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.baselines.morgan_fingerprints import get_morgan_fingerprint
from src.data import Data
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN
from src.baselines.hyperparams_search import Params, HyperparamsSearch


def main():
    print('\n Loading and preprocess data')
    train_data = Data(filename=TRAIN_FILE)
    smiles_train, y_train = train_data.get_processed_smiles_and_targets()

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    train_morgan_fp = list(map(lambda x: get_morgan_fingerprint(x), smiles_train))
    test_morgan_fp = list(map(lambda x: get_morgan_fingerprint(x), smiles_test))

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, nthread=1)
    folds = 4

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    search = HyperparamsSearch(
        xgb,
        params=Params.XGBOOST_LIGHT,
        cv=skf.split(train_morgan_fp, y_train),
        verbose=3
    )
    print('\n Start Grid Search')
    search.fit(train_morgan_fp, y_train)

    print('\n All results:')
    print(search.cv_results())
    print('\n Best estimator:')
    print(search.best_estimator())
    print('\n Best normalized score for %d-fold search with:' % (folds))
    print(search.best_score())
    print('\n Best hyperparameters:')
    print(search.best_params())

    test_predictions = search.best_estimator().predict(test_morgan_fp)
    test_predictions_df = save_prediction(test_data.data[SMILES_COLUMN], test_predictions,
                                          "morgan_fp_2_2048_test_submission.csv")
    return


if __name__ == "__main__":
    main()
