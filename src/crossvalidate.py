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
    input_ranking: str
    random_seed: int


# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['train_test_split']['training_output'],
                  'input_ranking': params['rfe_reduce']['output']},
               **params['crossvalidate']}
params = Params(**params_dict)

# Set random seed
np.random.seed(params.random_seed)

print(params)
print(meta)