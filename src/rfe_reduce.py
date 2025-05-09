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
CYCLES_TO_RUN = 1
RANDOM_SEED = 0  # previous 42
RESULTS_FILE = 'metrics/rfe_results.pkl'

# Hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 30

# Set random seed
np.random.seed(RANDOM_SEED)


@dataclass
class Params:
    input: str
    rfe_groups: bool

# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['featurize']['output']}, **params['rfe_reduce']}
params = Params(**params_dict)

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


def get_protein_ranking(results, feature_names):
    def get_protein_ids(cycle):
        fn = feature_names

        protein_ids = []
        for estimator in cycle:
            fn = estimator.get_feature_names_out(fn)
            protein_ids.append(fn)

        return protein_ids

    def count_protein_ids(cycle):
        return pd.Series(np.concatenate(get_protein_ids(cycle))).value_counts()

    results_df = pd.concat(map(count_protein_ids, results), axis=1)

    # Create ranking
    return results_df.fillna(0).apply(sum, axis=1).sort_values(ascending=False)


def shuffle_data(training_data: TrainingData) -> TrainingData:
    # Shuffle data
    indices = np.arange(training_data.X.shape[0])
    np.random.shuffle(indices)

    return TrainingData(X=training_data.X[indices],
                        y=training_data.y[indices],
                        feature_names=training_data.feature_names,
                        groups=training_data.groups[indices])


# TODO: This function will be used to prepare the output ranking

# def get_protein_sequence(cycle, feature_names):
#     fn = feature_names
#     selected = []
#     for estimator in cycle:
#         fn = estimator.get_feature_names_out(fn)
#         selected.append(fn)
#
#     return selected

#
# # %%
# get_protein_sequence(results[0])
#
# # %% [markdown]
# # ## Create results_df
#
# # %%
# f = lambda x: pd.Series(np.concatenate(get_protein_sequence(x))).value_counts()
#
# results_df = pd.concat(map(f, results), axis=1)


# TODO: Read params + metadata, load dataset & finish this script

def main():

    # Read input data

    training_data = load_dataset()

    training_data = shuffle_data(training_data)


    model = make_pipeline(get_model(['rf']))  # TODO: Parametrize this

    results = []
    for cycle in range(CYCLES_TO_RUN):
        print('Cycle:', cycle)
        results.append(persistent_elimination(training_data, base_estimator=model, groups=params.rfe_groups))

    # Save RFECV results
    with open(get_repo_path() / RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)




if __name__ == '__main__':
    main()