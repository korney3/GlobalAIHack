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
        "max_depth": [5, 10, 20],
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 25, 50],
    }
    XGBOOST = {
        "max_depth": [10, 20, 50, 100, 200],
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "subsample": [0.5, 0.75, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "colsample_bylevel": [0.4, 0.6, 0.8, 1.0],
        "min_child_weight": [0.5, 1.0, 3.0, 5.0],
        "gamma": [0, 0.25, 0.5, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 20.0, 100.0],
        "n_estimators": [10, 100, 200, 500],
    }
    CATBOOST = {
        "depth": [3, 4, 5, 8, 10],
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 25, 50],
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
    
    def best_mean(self) -> float:
        return self.search.cv_results_["mean_test_score"][self.search.best_index_]

    def best_std(self) -> float:
        return self.search.cv_results_["std_test_score"][self.search.best_index_]
