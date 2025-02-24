# Standard library imports
from abc import ABC, abstractmethod
from datetime import timedelta
import os

# Local imports
# from datadownloader.indexer import recordings_file, get_acc_data
# from tools.utils import timestamp_to_time, get_repo_path
# from config import Patients
# import config

# Third-party imports
# import pandas as pd
# import numpy as np


class Dataset(ABC):

    @abstractmethod
    def get_dataset(self):
        pass


class LaboratoryData(Dataset):

    def get_dataset(self):

        df_acc = pd.read_pickle(get_repo_path() / 'datasets/data_raw.pkl')

        df_acc[config.General.label_column] = df_acc[config.General.label_column].apply(lambda x: x.astype(int))

        return df_acc