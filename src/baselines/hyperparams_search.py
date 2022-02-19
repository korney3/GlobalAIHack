from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.utils.const import THREADS, SCORING


class Algorithm(Enum):
    GRID_SEARCH = "Grid search"
    RANDOM_SEARCH = "Random search"
    HYPEROPT = "Hyperopt"


@dataclass
class Params:
    BOOSTING_SMOKE_TEST = {
        "max_depth": [5, 10],
        "n_estimators": [10, 25],
    }
    XGBOOST_LIGHT = {
        "max_depth": [5, 10, 20],
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 25, 50],
    }
    XGBOOST = {
        "max_depth": [4, 5, 6, 7, 10, 20, 50, 100],
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "subsample": [0.5, 0.65, 0.75, 0.85, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.7, 0.8, 1.0],
        "colsample_bylevel": [0.4, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0],
        "gamma": [0, 0.01, 0.1, 0.25, 0.5, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 20.0, 100.0],
        "n_estimators": [100, 200, 500, 600, 750, 1000, 1250, 1500, 1750, 2000],
        "scale_pos_weight": [10, 15, 20, 25, 30, 35],
    }
    XGBOOST_MORGAN_0_3864 = {
        "subsample": [1.0],
        "scale_pos_weight": [35],
        "reg_lambda": [20.0],
        "n_estimators": [2000],
        "min_child_weight": [0.5],
        "max_depth": [100],
        "learning_rate": [0.2],
        "gamma": [0.5],
        "colsample_bytree": [0.7],
        "colsample_bylevel": [0.7],
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
        elif algorithm == Algorithm.HYPEROPT:
            raise Exception("Not implemented yet")
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

    def save_best(self, filename: str) -> None:
        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)
        path = os.path.join(MODELS_PATH, filename)
        self.search.best_estimator_.save_model(path)
