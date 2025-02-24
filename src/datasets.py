# Standard library imports
from abc import ABC, abstractmethod

# Local imports
from utils import get_repo_path
import config as cfg

# Third-party imports
import pandas as pd
import numpy as np


class Dataset(ABC):

    @abstractmethod
    def get_dataset(self):
        pass


class LaboratoryData(Dataset):

    def get_dataset(self):

        df_acc = pd.read_pickle(get_repo_path() / 'datasets/data_raw.pkl')

        df_acc[cfg.label_column] = df_acc[cfg.label_column].apply(lambda x: x.astype(int))

        return df_acc

datasets = {'lab_data': LaboratoryData()}