"""Calculates persistence diagrams"""
import os

import pandas as pd
import dvc.api
from dataclasses import dataclass

from utils import get_repo_path, get_metadata
from modurec.features.feature import calculate_feature, FeatureData
from joblib import Memory
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Optional

# Setting random seed
# TODO: To refactor when needed
# params = params_show()['diagrams']
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Prepare caching
CACHEDIR = get_repo_path() / '.cache'
memory = Memory(CACHEDIR, verbose=0)

@dataclass
class Params:
    input: str
    output: str
    restrict: int
    columns: dict
    params: list

# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['prepare']['output']}, **params['diagrams']}
params = Params(**params_dict)

# Global data frame
df = pd.DataFrame()


# Generate FeatureData out of the parameters
def generate_feature_data() -> list[FeatureData]:
    """Generates FeatureData out of the parameters in global variable params"""

    return list(map(lambda values: FeatureData(name='diagram', params=values),
                    params.params))


class WindowEncoder:

    def __init__(self):
        self._encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, y):
        y = y.flatten().reshape(-1, 1)
        self._encoder.fit(y)
        return self

    def transform(self, y):

        def _encode(x):
            zero_by_zero_matrix = np.array(x).reshape(-1, 1)
            encoded_label = self._encoder.transform(zero_by_zero_matrix)
            # return encoded_label.toarray()  # toarray needed because encoder returns sparse matrix
            return encoded_label

        tensor3d = np.apply_along_axis(_encode, 1, y)
        window_size = tensor3d.shape[1]
        return np.sum(tensor3d/window_size, axis=1)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return self._encoder.inverse_transform(y)


def encode_data(data):
    """Returns decoded version of y"""
    enc = WindowEncoder()
    y = np.array(data.apply(pd.Series))
    result = enc.fit_transform(y)
    result = enc.inverse_transform(result)
    result = pd.Series(result.flatten())
    result.index = data.index

    return result


@memory.cache
def calculate_feature_cached(hash_value: str, feature_data: FeatureData) -> pd.Series:
    """Cached version of calculate_feature"""

    global df

    return calculate_feature(df=df, feature_data=feature_data)


def restrict_data(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    """Restricts data so that each class has maximum n samples"""

    if n is None:
        return df

    return df.groupby(meta.label_column, group_keys=False).apply(
        lambda g: g.sample(n=min(n, len(g)))
    )


def compute_diagrams(original_df: pd.DataFrame, key: str, columns: list[str, str]) -> pd.DataFrame:
    """Computes diagrams from the given columns. Uses global dataframe df."""

    global df

    df = original_df.copy()

    # Create point cloud
    df['point_cloud'] = df[columns].apply(lambda x: np.stack((x.iloc[0], x.iloc[1]), axis=-1),
                                          axis=1)

    results = []
    for case, feature_data in enumerate(generate_feature_data()):
        print(f'Calculating {case+1}/{len(generate_feature_data())}. Feature data: {feature_data}')
        print('------------------------')
        results.append(calculate_feature_cached(hash_value=key, feature_data=feature_data))

    result_df = pd.concat(results, axis=1)

    # Add label column to the result_df
    result_df = result_df.join(df[meta.label_column])

    # Add patient column to the result_df
    result_df = result_df.join(df[meta.patient_column])

    return result_df


def compute_required_diagrams(original_df: pd.DataFrame) -> dict:
    # Encode labels
    original_df[meta.label_column] = encode_data(original_df[meta.label_column])

    # Restrict data
    original_df = restrict_data(original_df, params.restrict)

    print('Calculating diagrams...')
    diagrams = {}
    for key, value in params.columns.items():
        print(f'Calculating diagrams for {key}')
        print('===============================')
        result_df = compute_diagrams(original_df, key, value)
        diagrams[key] = result_df

    return diagrams


def main():

    # Quick fix for not changing the diagrams
    # Perform copy
    os.system('cp ../temp/merged_diagrams.pkl data/diagrams.pkl')
    return

    # Read data
    original_df = pd.read_pickle(get_repo_path() / data_dir / params.input)

    diagrams = compute_required_diagrams(original_df)

    # Save results
    with open(get_repo_path() / data_dir / params.output, 'wb') as f:
        pd.to_pickle(diagrams, f)

    # Clean cache
    memory.clear(warn=False)


if __name__ == '__main__':
    main()