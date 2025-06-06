"""OPTUNA"""
# Standard library imports
import logging
from dataclasses import dataclass
import json

# Third party imports
import optuna
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold
import pandas as pd
import dvc.api

# Local imports
from ml_utils import prepare_dataset, get_model, make_pipeline, TrainingData
from utils import get_repo_path, get_metadata, wrapped_partial, labels_to_events
import crossvalidate
from crossvalidate import filter_data, robust_cross_val_evaluate_model

# Configure logging
logging.basicConfig(
    filename=str(get_repo_path() / 'optuna.log'),
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='%(asctime)s %(levelname)s:%(message)s'
)

logging.info('Started Optuna optimization.')

# Load metadata
meta = get_metadata()

# # Get the DVC parameters
params = dvc.api.params_show()

# params = crossvalidate.params

# Directories
data_dir = params['directories']['data']
metrics_dir = params['directories']['metrics']

# Model parameters
model_name = params['model']['name']
# model_params = {key: value for key, value in params['model'].items() if key != 'name'}

# Stage parameters
# params_dict = {**{'input': params['train_test_split']['training_output'],
#                   'input_ranking': params['rfe_reduce']['output']},
#                **params['crossvalidate']}
params = crossvalidate.params

# Set random seed
np.random.seed(params.random_seed)


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


def objective(trial):

    model_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.4),
    }

    model = get_model(model_name, params=model_params, random_seed=params.random_seed)

    # Create a pipeline
    if params.use_pipeline:
        model = make_pipeline(model)

    result = robust_cross_val_evaluate_model(training_data.X,
                                             training_data.y,
                                             model,
                                             cv=cv,
                                             groups=training_data.groups)

    return result['test_balanced_accuracy'].mean()


def main():

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)



    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print('  Params:')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
    print('  User attrs:')
    for key, value in study.best_trial.user_attrs.items():
        print(f'    {key}: {value}')

    # Save the study results
    with open(get_repo_path() / metrics_dir / 'optuna_results.json', 'w') as f:
        json.dump({**{'metric': study.best_trial.value}, **study.best_trial.params}, f)


if __name__ == '__main__':
    main()