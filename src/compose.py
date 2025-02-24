"""Reads data from the database to create one dataset

This script is run as a DVC-stage. Inputs, outputs and parameters defined in: dvc.yaml
"""
# Standard library imports
import os

# Third-party imports

# Local imports
import config as cfg
from utils import get_repo_path
import datasets

datasets_dict = datasets.datasets

# script parameters
target_path = get_repo_path() / cfg.data_dir / cfg.compose_target

df = datasets_dict[cfg.dataset].get_dataset()

# Writing to the target file
df.to_pickle(target_path)
