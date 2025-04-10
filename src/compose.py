"""Reads data from the database to create one dataset

This script is run as a DVC-stage. Inputs, outputs and parameters defined in: dvc.yaml
"""
# Standard library imports

# Third-party imports
import dvc.api

# Local imports
from utils import get_repo_path
import datasets


# Get the DVC parameters
params = dvc.api.params_show()

# Stage parameters
output = params['compose']['output']
dataset = params['compose']['dataset']

# Data dir
data_dir = params['directories']['data']

# Datasets
datasets_dict = datasets.datasets

# script parameters
target_path = get_repo_path() / data_dir / output

df = datasets_dict[dataset].get_dataset()

# Writing to the target file
df.to_pickle(target_path)
