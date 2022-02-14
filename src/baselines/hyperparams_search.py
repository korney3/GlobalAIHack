from dataclasses import dataclass
from typing import Any, Dict
from sklearn.model_selection import GridSearchCV
from src.utils.const import THREADS, SCORING


@dataclass
class Params:
    XGBOOST_LIGHT = {
        'max_depth': [5, 10],
        'n_estimators': [30, 40],
    }
    XGBOOST = {
        "max_depth": [50, 100, 200, 300, 500, 600, 900],
        "n_estimators": [300, 400, 500, 1000],
    }
    CATBOOST = {
        "cbr__iterations": [100, 500, 1000, 2000],
        "cbr__depth": [3, 5, 10],
    }


class HyperparamsSearch:
    def __init__(self, model: Any, params: Dict, cv: Any, verbose: int) -> None:
        self.search = GridSearchCV(model, param_grid=params, scoring=SCORING, n_jobs=THREADS, cv=cv, verbose=verbose)

    def fit(self, train_morgan_fp, y_train) -> None:
        self.search.fit(train_morgan_fp, y_train)

    def best_estimator(self) -> Any:
        return self.search.best_estimator_

    def best_score(self) -> Any:
        return self.search.best_score_

    def best_params(self) -> Any:
        return self.search.best_params_

    def cv_results(self) -> Any:
        return self.search.cv_results_
