from dgllife.utils import ScaffoldSplitter
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.data.get_cv_splitter import get_cv_splitter
from src.data.get_fingerprints import get_np_array_of_fps
from src.data import Data
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, Fingerprints, CVSplitters


def main():
    print('\n Loading and preprocess data')
    train_data = Data(filename=TRAIN_FILE)
    smiles_train, y_train = train_data.get_processed_smiles_and_targets()

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    print('\n Get fingerprints')
    train_morgan_fp = get_np_array_of_fps(fp_type=Fingerprints.ECFP4, smiles=smiles_train)
    test_morgan_fp = get_np_array_of_fps(fp_type=Fingerprints.ECFP4, smiles=smiles_test)

    print('\n Get CV splits')

    folds = 4
    cv_splits = get_cv_splitter(CVSplitters.Scaffold_CV, train_data, y_train, k_folds=folds, random_state=69)
    params = {
        # 'alpha': [0, 1, 2, 5],
        # 'gamma': [0, 1, 2, 5],
        'max_depth': [50, 200, 500, 900],
        'n_estimators': [500, 1000]
    }
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, nthread=1, use_label_encoder=False)

    grid_search = GridSearchCV(xgb, param_grid=params, scoring='f1', n_jobs=4,
                               cv=cv_splits, verbose=1000)
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
