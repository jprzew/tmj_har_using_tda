"""This is part of a pipeline - calculates best features using multiple RFE"""
# Standard library imports
import warnings
from dataclasses import dataclass
import json

# Third party imports
import numpy as np
from sklearn import model_selection
from sklearn.metrics import balanced_accuracy_score, recall_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, KFold
import pandas as pd
import dvc.api

# Local imports
from ml_utils import prepare_dataset, get_model, make_pipeline, TrainingData
from utils import get_repo_path, get_metadata, wrapped_partial, labels_to_events

# Parameters

@dataclass
class Params:
    input: str
    input_ranking: str
    output: str
    cv: int|str
    output_categories: str|list
    features: int|str
    random_seed: int


# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Directories
data_dir = params['directories']['data']
metrics_dir = params['directories']['metrics']

# Model parameters
model_name = params['model']['name']
model_params = {key: value for key, value in params['model'].items() if key != 'name'}

# Stage parameters
params_dict = {**{'input': params['train_test_split']['training_output'],
                  'input_ranking': params['rfe_reduce']['output']},
               **params['crossvalidate']}
params = Params(**params_dict)

# Set random seed
np.random.seed(params.random_seed)


def get_scorers(labels_dict):

    def labeled_matthews(y_test, y_pred, value):
        return matthews_corrcoef(y_test == value, y_pred == value)

    def labeled_recall(y_test, y_pred, value):
        return recall_score(y_test == value, y_pred == value, zero_division=0)

    # general scorers
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)

    scorers = {'balanced_accuracy': balanced_accuracy_scorer,
               'matthews': matthews_corrcoef_scorer}

    # matthews correlation scorers
    for label, event in labels_dict.items():
        scoring_function = wrapped_partial(labeled_matthews, value=label)
        scorers[f'matthews_{event}'] = make_scorer(scoring_function)

    # recall scorers
    for label, event in labels_dict.items():
        scoring_function = wrapped_partial(labeled_recall, value=label)
        scorers[f'recall_{event}'] = make_scorer(scoring_function)

    return scorers


def cross_val_evaluate_model(X, y, model, cv, groups=None):

    scorers = get_scorers(labels_to_events)
    scores = model_selection.cross_validate(model, X, y,
                                            scoring=scorers,
                                            cv=cv, groups=groups, n_jobs=-1, )

    return pd.Series(scores)


def robust_cross_val_evaluate_model(X, y, model, cv, groups=None):
    """Evaluates a model and try to trap errors and hide warnings"""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = cross_val_evaluate_model(X, y, model, cv, groups=groups)
    except Exception as e:
        scores = None
    return scores


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
    df = pd.read_pickle(get_repo_path() / data_dir / params.input)
    rank_df = pd.read_pickle(get_repo_path() / data_dir / params.input_ranking)

    # Prepare the dataset
    training_data = prepare_dataset(df,
                                    non_feature_cols=meta.scalar_columns,
                                    target_col=meta.label_column,
                                    group_col=meta.patient_column)

    # Filter the data
    training_data = filter_data(training_data, rank_df)

    # Get crossvalidator
    if isinstance(params.cv, int):
        cv = KFold(n_splits=params.cv, shuffle=True, random_state=params.random_seed)
    elif params.cv == 'logo':
        cv = LeaveOneGroupOut()
    else:
        raise ValueError(f"Invalid value for cv: {params.cv}")

    # Create a model
    model = get_model(model_name, params=model_params, random_seed=params.random_seed)

    # Create a pipeline
    pipeline = make_pipeline(model)

    result = robust_cross_val_evaluate_model(training_data.X,
                                             training_data.y,
                                             pipeline,
                                             cv=cv,
                                             groups=training_data.groups)
    result_dict = result.apply(lambda x: np.mean(x)).to_dict()

    # Save the results
    with open(get_repo_path() / metrics_dir / params.output, 'w') as f:
        f.write(json.dumps(result_dict))


if __name__ == '__main__':
    main()