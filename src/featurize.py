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


def extract_invariant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Utility function to extract columns which do not depend on highest level of the MultiIndex
    Args:
        df: DataFrame with MultiIndex columns each stratum has the same column names

    Returns:
        A pair of dataframes: the first contains invariant columns,
        the second contains df with invariant columns removed
    """

    # Compare columns to see which of them do not depend on stratum
    comparisons = []
    for column1, column2 in zip(df.columns.levels[0][:-1], df.columns.levels[0][1:]):
        comparisons.append((df[column1] == df[column2]).all())

    if not comparisons:
        raise ValueError("No columns to compare")

    is_column_invariant = pd.concat(comparisons, axis=1).all(axis=1)  # Series with True for invariant columns
    invariant_columns = is_column_invariant[is_column_invariant].index

    return df[column1][invariant_columns], df.drop(invariant_columns, axis=1, level=1)


def extract_dataframes(data_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat(data_dict, axis=1)

    invariant_df, filtered_df = extract_invariant_columns(df)

    # Rename columns to remove MultiIndex
    filtered_df.columns = filtered_df.columns.map(lambda x: f"{x[1]}_{x[0]}")

    return invariant_df, filtered_df


def tabularize_data(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Utility function to tabularize data from a dictionary of dataframes
    Args:
        data_dict: Dictionary of dataframes with some columns
                   which are the same for all dataframes (invariant columns)
    Returns:
        A dataframe with invariant columns and filtered columns
    """

    invariant_df, filtered_df = extract_dataframes(data_dict)
    return pd.concat([invariant_df, filtered_df], axis=1)


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

    # Tabularize data
    df = tabularize_data(results)

    # Save to file
    df.to_pickle(get_repo_path() / data_dir / params.output)


if __name__ == '__main__':
    main()
