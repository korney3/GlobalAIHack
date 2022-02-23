import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline, make_pipeline

from src.data import Data
from src.data.get_cv_splitter import get_cv_splitter
from src.data.get_fingerprints import get_np_array_of_fps
from src.data.read_pdb_file import get_targetpdb_smiles
from src.data.save_predictions import save_prediction
from src.utils.const import TRAIN_FILE, TEST_FILE, SMILES_COLUMN, CVSplitters, FingerprintsNames, \
    FINGERPRINTS_METHODS, ImbalanceStrategies


def main():
    print('\n Loading and preprocess data')

    train_data = Data(filename=TRAIN_FILE)
    imb_strategy = ImbalanceStrategies.OVERSAMPLE
    smiles_train, y_train = train_data.get_processed_smiles_and_targets(imb_strategy=None)

    test_data = Data(filename=TEST_FILE)
    smiles_test, _ = test_data.get_processed_smiles_and_targets()

    fingerprint_type_name = FingerprintsNames.MACCS
    fingerprint_type_method = FINGERPRINTS_METHODS[fingerprint_type_name]

    cv_type = CVSplitters.Scaffold_CV

    pdb_smiles = get_targetpdb_smiles()

    print(f'\n Get fingerprints {fingerprint_type_name}')
    pdb_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=[pdb_smiles])

    train_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_train)
    test_fp = get_np_array_of_fps(fp_type=fingerprint_type_method, smiles=smiles_test)

    # train_pdb_fp = np.tile(pdb_fp, (len(train_fp), 1))
    # train_fp = np.hstack([train_fp, train_pdb_fp])
    #
    # test_pdb_fp = np.tile(pdb_fp, (len(test_fp), 1))
    # test_fp = np.hstack([test_fp, test_pdb_fp])

    print(f'\n Get CV splits {cv_type.value}')

    folds = 4
    cv_splits = get_cv_splitter(cv_type, train_data, y_train, k_folds=folds, random_state=69)

    params = {
        'mlpclassifier__max_iter': [50, 100, 300],
        'mlpclassifier__alpha': [1e-4, 1e-3, 1e-2],
        'mlpclassifier__learning_rate_init': [1e-5, 1e-4, 1e-3]}
    print(f'\n Scale data')

    # lb = preprocessing.LabelBinarizer()
    # y_train_ohe = lb.fit_transform(y_train.reshape(-1, 1))
    sc = StandardScaler()
    train_fp_sc = sc.fit_transform(train_fp)
    test_fp_sc = sc.fit_transform(test_fp)

    model = MLPClassifier(tol=1e-4, hidden_layer_sizes=(128, 128, 64, 32, 2),
                          n_iter_no_change=5, activation='relu', solver='adam', batch_size=256,
                          random_state=69, verbose=True, early_stopping=False)
    imba_pipeline = make_pipeline(RandomOverSampler(sampling_strategy="all", random_state=69),
                                  model)
    grid_search = GridSearchCV(imba_pipeline, param_grid=params, scoring='f1', n_jobs=4,
                               cv=cv_splits, verbose=1000)
    print('\n Start Grid Search')
    grid_search.fit(train_fp_sc, y_train)
    print('\n All results:')
    print(grid_search.cv_results_)
    print('\n Best estimator:')
    print(grid_search.best_estimator_)
    print('\n Best normalized score for %d-fold search with:' % (folds))
    print(grid_search.best_score_)
    print('\n Best hyperparameters:')
    print(grid_search.best_params_)

    test_predictions = grid_search.best_estimator_.predict(test_fp_sc)
    test_predictions_df = save_prediction(test_data.data[SMILES_COLUMN], test_predictions,
                                          f"{fingerprint_type_name.value}_{CVSplitters.Scaffold_CV.value}_mlp_with_pdb_fp_submission.csv")


if __name__ == "__main__":
    main()
