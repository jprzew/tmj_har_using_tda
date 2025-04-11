# Standard library imports
from abc import ABC, abstractmethod

# Local imports
from utils import get_repo_path, get_metadata

# Third-party imports
import pandas as pd
import numpy as np

# Load metadata
meta = get_metadata()


class Dataset(ABC):

    @abstractmethod
    def get_dataset(self):
        pass


class LaboratoryData(Dataset):

    def get_dataset(self):

        df_acc = pd.read_pickle(get_repo_path() / 'datasets/data_raw.pkl')

        df_acc[meta.label_column] = df_acc[meta.label_column].apply(lambda x: x.astype(int))

        return df_acc

datasets = {'lab_data': LaboratoryData()}