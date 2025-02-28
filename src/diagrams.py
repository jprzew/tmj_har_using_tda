"""Calculates persistence diagrams"""

import config as cfg
import pandas as pd
from utils import get_repo_path
from modurec.features.feature import calculate_feature
from joblib import Memory
from dvc.api import params_show
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

# Global data frame
df = None


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
def calculate_feature_cached(feature_data):
    """Cached version of calculate_feature"""

    global df

    return calculate_feature(df=df, feature_data=feature_data)


def restrict_data(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    """Restricts data so that each class has maximum n samples"""

    if n is None:
        return df

    return df.groupby(cfg.label_column).sample(n)


def main():

    global df

    # Read data
    df = pd.read_pickle(get_repo_path() / cfg.data_dir / cfg.prepare_target)

    # Encode labels
    df[cfg.label_column] = encode_data(df[cfg.label_column])

    # Restrict data
    df = restrict_data(df, cfg.restrict)

    # Prepare data to computation
    df['point_cloud'] = df[cfg.columns].apply(lambda x: np.stack((x.iloc[0], x.iloc[1]), axis=-1),
                                              axis=1)

    # Calculate diagrams
    print('Calculating diagrams...')
    results = []
    for case, feature_data in enumerate(cfg.to_calculate):
        print(f'Calculating {case+1}/{len(cfg.to_calculate)}. Feature data: {feature_data}')
        print('------------------------')
        results.append(calculate_feature_cached(feature_data=feature_data))

    result_df = pd.concat(results, axis=1)

    # Add label column to the result_df
    result_df = result_df.join(df[cfg.label_column])

    # Add patient column to the result_df
    result_df = result_df.join(df[cfg.patient_column])

    # Save results
    result_df.to_pickle(get_repo_path() / cfg.data_dir/ cfg.diagrams_target)

    # Clean cache
    memory.clear(warn=False)


if __name__ == '__main__':
    main()