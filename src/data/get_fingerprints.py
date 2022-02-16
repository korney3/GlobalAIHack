from typing import List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.DataStructs import ExplicitBitVect

from src.utils.const import Fingerprints


def bit_vectors_to_numpy_arrays(fps: List[ExplicitBitVect]) -> np.array:
    output_arrays = [np.zeros((1,)) for i in range(len(fps))]
    _ = list(
        map(lambda fp_output_array: DataStructs.ConvertToNumpyArray(fp_output_array[0], fp_output_array[1]),
            zip(fps, output_arrays)))
    return np.asarray(output_arrays)


def get_np_array_of_fps(fp_type: Fingerprints.ECFP4, smiles: List[str]):
    # Calculate the morgan fingerprint
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fp = list(map(fp_type, mols))
    return bit_vectors_to_numpy_arrays(fp)
