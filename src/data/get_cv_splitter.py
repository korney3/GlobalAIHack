from dgllife.utils import ScaffoldSplitter
from sklearn.model_selection import StratifiedKFold

from src.utils.const import CVSplitters


class ScaffoldCVSklearn:
    def __init__(self, data, k_folds):
        self.scaffold_splits = ScaffoldSplitter.k_fold_split(data, k=k_folds)

    def split(self):
        indices_splits = []
        for train_data, val_data in self.scaffold_splits:
            train_indices = self.convert_data_to_indices(train_data)
            val_indices = self.convert_data_to_indices(val_data)
            indices_splits.append((train_indices, val_indices))
        return indices_splits

    def convert_data_to_indices(self, dataset):
        indices = [item[1] for item in dataset]
        return indices


def get_cv_splitter(splitter_type: CVSplitters, X, y, k_folds=5, random_state=69):
    if splitter_type == CVSplitters.Stratified_CV:
        cv_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        return cv_splitter.split(X, y)
    elif splitter_type == CVSplitters.Scaffold_CV:
        cv_splitter = ScaffoldCVSklearn(X, k_folds)
        return cv_splitter.split()
