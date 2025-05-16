"""This is part of a pipeline - calculates best features using multiple RFE"""
# Standard library imports
import pickle
from functools import wraps
import warnings
from dataclasses import dataclass

# Third party imports
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.feature_selection import RFECV
import pandas as pd
import dvc.api

# Local imports
from ml_utils import prepare_dataset, get_model, make_pipeline, TrainingData
from utils import get_repo_path, get_metadata

# Parameters

@dataclass
class Params:
    input: str
    output: str
    rfecv_output: str
    cycles_to_run: int
    rfe_groups: bool
    random_seed: int
    restrict: int|bool

# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['featurize']['output']}, **params['rfe_reduce']}
params = Params(**params_dict)

# Set random seed
np.random.seed(params.random_seed)


def ignore_user_warning(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            return func(*args, **kwargs)

    return wrapper


@ignore_user_warning
def persistent_elimination(training_data: TrainingData, base_estimator, groups=False, max_steps=30):

    X = training_data.X
    results = []
    for _ in tqdm(range(max_steps)):

        # TODO: Consider using weights
        # base_estimator = RandomForestClassifier(n_estimators=N_ESTIMATORS)

        if groups:
            crossvalidator = GroupKFold(n_splits=5)
        else:
            crossvalidator = KFold(n_splits=10)

        rfecv = RFECV(step=1,
                      min_features_to_select=1,
                      n_jobs=1,
                      estimator=base_estimator,
                      cv=crossvalidator,
                      scoring='accuracy',
                      importance_getter='named_steps.model.feature_importances_')

        rfecv.fit(X, training_data.y, groups=training_data.groups)

        results.append(rfecv)

        if rfecv.n_features_ == 1:
            break

        X = rfecv.transform(X)

    return results


def shuffle_data(training_data: TrainingData) -> TrainingData:
    # Shuffle data
    indices = np.arange(training_data.X.shape[0])
    np.random.shuffle(indices)

    return TrainingData(X=training_data.X[indices],
                        y=training_data.y[indices],
                        feature_names=training_data.feature_names,
                        groups=training_data.groups[indices])


def get_feature_sequence(cycle, feature_names: list[str]) -> list[list[str]]:
    """Extracts feature lists from the estimators resulting from the RFECV cycle."""
    fn = feature_names
    selected = []
    for estimator in cycle:
        fn = estimator.get_feature_names_out(fn)
        selected.append(fn)

    return selected


def compute_feature_ranking(results, feature_names: list[str]) -> pd.Series:

    def _compute_feature_counts(x):
        return pd.Series(np.concatenate(get_feature_sequence(x, feature_names))).value_counts()

    results_df = pd.concat(map(_compute_feature_counts, results), axis=1)
    return results_df.fillna(0).apply(sum, axis=1).sort_values(ascending=False)


def main():

    # Read the input file
    df = pd.read_pickle(get_repo_path() / data_dir / params.input)

    # Sample df for testing purposes
    if params.restrict:
        df = df.sample(frac=params.restrict, random_state=params.random_seed)

    training_data = prepare_dataset(df,
                                    non_feature_cols=meta.scalar_columns,
                                    target_col=meta.label_column,
                                    group_col=meta.patient_column)

    training_data = shuffle_data(training_data)
    model = make_pipeline(get_model('rf'))  # TODO: Parametrize this

    results = []
    for cycle in range(params.cycles_to_run):
        print('Cycle:', cycle)
        results.append(persistent_elimination(training_data, base_estimator=model, groups=params.rfe_groups))

    # Compute feature ranking
    feature_ranking = compute_feature_ranking(results, training_data.feature_names)

    # Save feature ranking
    feature_ranking.to_pickle(get_repo_path() / data_dir / params.output)

    # Save RFECV results
    with open(get_repo_path() / data_dir / params.rfecv_output, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()