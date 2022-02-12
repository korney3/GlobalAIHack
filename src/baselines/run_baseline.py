import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from src.baselines.morgan_fingerprints import get_morgan_fingerprint
from src.data import Data
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN


def main():
    print('\n Loading and preprocess data')
    train_data = Data(filename=TRAIN_FILE)
    smiles_train, y_train = train_data.get_processed_smiles_and_targets()

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    train_morgan_fp = list(map(lambda x: get_morgan_fingerprint(x), smiles_train))
    test_morgan_fp = list(map(lambda x: get_morgan_fingerprint(x), smiles_test))


    params = {
        # 'min_child_weight': [1, 5, 10],
        # 'gamma': [0.5, 1, 1.5, 2, 5],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [50, 100, 200, 300, 500, 600, 900],
        'n_estimators': [300, 400, 500, 1000]
    }
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, nthread=1)
    folds = 4

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    grid_search = GridSearchCV(xgb, param_grid=params, scoring='f1', n_jobs=4,
                               cv=skf.split(train_morgan_fp, y_train), verbose=3)
    print('\n Start Grid Search')
    grid_search.fit(train_morgan_fp, y_train)
    print('\n All results:')
    print(grid_search.cv_results_)
    print('\n Best estimator:')
    print(grid_search.best_estimator_)
    print('\n Best normalized score for %d-fold search with:' % (folds))
    print(grid_search.best_score_)
    print('\n Best hyperparameters:')
    print(grid_search.best_params_)

    test_predictions = grid_search.best_estimator_.predict(test_morgan_fp)
    test_predictions_df = save_prediction(test_data.data[SMILES_COLUMN], test_predictions,
                                          "morgan_fp_2_2048_test_submission.csv")
    return


if __name__ == "__main__":
    main()
