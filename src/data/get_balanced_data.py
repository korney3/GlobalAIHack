import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.utils.const import ImbalanceStrategies


def get_balanced_data(imbalance_strategy: ImbalanceStrategies, smiles, y):
    smiles = np.array(smiles).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    if imbalance_strategy == ImbalanceStrategies.UNDERSAMPLE:
        under_sampler = RandomUnderSampler(sampling_strategy="all")
        smiles_balanced, y_balanced = under_sampler.fit_resample(smiles, y)
    elif imbalance_strategy == ImbalanceStrategies.OVERSAMPLE:
        over_sampler = RandomOverSampler(sampling_strategy="all")
        smiles_balanced, y_balanced = over_sampler.fit_resample(smiles, y)
    return smiles_balanced.reshape(-1).tolist(), y_balanced.reshape(-1).tolist()
