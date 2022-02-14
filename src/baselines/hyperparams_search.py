from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.utils.const import THREADS, SCORING


class Algorithm(Enum):
    GRID_SEARCH = "Grid search"
    RANDOM_SEARCH = "Random search"


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
    def __init__(self, algorithm: Algorithm, estimator: Any, params: Dict, cv: Any, verbose: int = 0) -> None:
        if algorithm == Algorithm.GRID_SEARCH:
            self.search = GridSearchCV(
                estimator, params, scoring=SCORING, n_jobs=THREADS, cv=cv, verbose=verbose
            )
        elif algorithm == Algorithm.RANDOM_SEARCH:
            self.search = RandomizedSearchCV(
                estimator, params, scoring=SCORING, n_jobs=THREADS, cv=cv, verbose=verbose
            )
        else:
            raise Exception("No such search algorithm")

    def fit(self, X, y) -> None:
        self.search.fit(X, y)

    def best_estimator(self) -> Any:
        return self.search.best_estimator_

    def best_score(self) -> Any:
        return self.search.best_score_

    def best_params(self) -> Any:
        return self.search.best_params_

    def cv_results(self) -> Any:
        return self.search.cv_results_
