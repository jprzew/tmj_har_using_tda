"""Calculates features out of persistence diagrams"""

from dataclasses import dataclass
import pandas as pd
import dvc.api
import warnings
from utils import get_repo_path, get_metadata
from feature_list import all_features
from modurec.features.feature import calculate_feature


@dataclass
class Params:
    input: str
    output: str

# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['diagrams']['output']}, **params['featurize']}
params = Params(**params_dict)


def compute_features(key: str, diagrams: pd.DataFrame) -> pd.DataFrame:

    results = []
    for case, feature_data in enumerate(all_features):
        print(f'Calculating {case+1}/{len(all_features)}. Key: {key}, Feature data: {feature_data}')
        print('------------------------')
        with warnings.catch_warnings(record=True) as w:
            results.append(calculate_feature(df=diagrams, feature_data=feature_data))
            if (len(w) > 0) and (w[0].category == pd.errors.PerformanceWarning):  # In case of dataframe fragmentation
                diagrams = diagrams.copy()

    # Create results df with label column
    results_df = pd.concat(results, axis=1)
    results_df = results_df.join(diagrams[meta.label_column])

    # Add patient column to the results_df
    results_df = results_df.join(diagrams[meta.patient_column])

    return results_df


def main():

    # Reading the diagrams
    diagrams = pd.read_pickle(get_repo_path() / data_dir / params.input)

    # Calculate features
    print('Calculating features...')

    results = {}
    for key, value in diagrams.items():
        print(f'Calculating features for {key}')
        print('===============================')
        results[key] = compute_features(key, value)

    # Save to file
    with open(get_repo_path() / data_dir / params.output, 'wb') as f:
        pd.to_pickle(results, f)


if __name__ == '__main__':
    main()
