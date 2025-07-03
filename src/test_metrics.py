"""This model computes test metrics."""
# Standard library imports
from dataclasses import dataclass
import pickle

# Third party imports
import numpy as np
from dvc.parsing import METRICS_KWD
from prometheus_client.metrics_core import METRIC_NAME_RE
from sklearn import model_selection
from sklearn.metrics import balanced_accuracy_score, recall_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, KFold
import pandas as pd
import dvc.api

# Local imports
from ml_utils import prepare_dataset, get_model, make_pipeline, TrainingData
from utils import get_repo_path, get_metadata, wrapped_partial, labels_to_events
from crossvalidate import get_scorers

from utils import labels_to_events

# Parameters
@dataclass
class Params:
    input: str
    input_ranking: str
    output: str
    cv: int|str
    output_categories: str|list
    features: int|str
    use_pipeline: bool
    random_seed: int


# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Directories
data_dir = params['directories']['data']
MODELS_DIR = 'models'
METRICS_DIR = 'metrics'
OUTPUT_JSON = 'test_metrics.json'

# Model file name
MODEL_FILE = 'model.pkl'

# Training data
TEST_DATA_FILE = 'test_features.pkl'




# Stage parameters
params_dict = {**{'input': params['train_test_split']['training_output'],
                  'input_ranking': params['rfe_reduce']['output']},
               **params['crossvalidate']}
params = Params(**params_dict)


# Set random seed
np.random.seed(params.random_seed)


def filter_data(training_data: TrainingData, rank_df: pd.DataFrame) -> TrainingData:
    # Get the features
    if params.features == 'all':
        feature_names = training_data.feature_names
        X = training_data.X
        groups = training_data.groups
        y = training_data.y
    elif isinstance(params.features, int):
        feature_names = np.array(training_data.feature_names)

        # Use the top features
        top_features = pd.Series(rank_df.index)[:params.features]
        to_take_idx = np.isin(feature_names, top_features)
        X = training_data.X[:, to_take_idx]
        feature_names = feature_names[to_take_idx]

        groups = training_data.groups
        y = training_data.y
    else:
        raise ValueError(f'Invalid value for features: {params.features}')

    if isinstance(params.output_categories, list):
        # Get the indices of the categories
        to_take_idx = np.isin(y, params.output_categories)
        X = X[to_take_idx, :]
        y = y[to_take_idx]
        groups = groups[to_take_idx]
    elif params.output_categories == 'all':
        pass
    else:
        raise ValueError(f'Invalid value for output_categories: {params.output_categories}')

    return TrainingData(X, y, feature_names, groups)


def main():
    # Load the dataset
    df = pd.read_pickle(get_repo_path() / data_dir / TEST_DATA_FILE)
    rank_df = pd.read_pickle(get_repo_path() / data_dir / params.input_ranking)

    # Prepare the dataset
    training_data = prepare_dataset(df,
                                    non_feature_cols=meta.scalar_columns,
                                    target_col=meta.label_column,
                                    group_col=meta.patient_column)

    # Filter the data
    test_data = filter_data(training_data, rank_df)

    scorers = get_scorers(labels_to_events)


    # Save the model to pickle file
    model_path = get_repo_path() / MODELS_DIR / MODEL_FILE
    with open(model_path, 'rb') as f:
        model = pickle.load(f)


    metrics = pd.Series(scorers).apply(lambda s: s(model, test_data.X, test_data.y))

    metrics.to_json(get_repo_path() / METRICS_DIR / OUTPUT_JSON)


if __name__ == '__main__':
    main()