import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from utils import get_repo_path, get_metadata
import dvc.api


@dataclass
class Params:
    input: str
    training_output: str
    test_output: str
    test_proportion: float


# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['featurize']['output']}, **params['train_test_split']}
params = Params(**params_dict)


def data_split(input_data: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing subsets.

    Args:
        input_data (pd.DataFrame): The input DataFrame to split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple (train_df, test_df) where train_df and test_df are DataFrames for training and testing.
    """
    train_indices, test_indices = train_test_split(input_data.index,
                                                   test_size=test_size,
                                                   random_state=random_state)

    train_df = input_data.loc[train_indices]
    test_df = input_data.loc[test_indices]
    return train_df, test_df


def main():

    # Read input data
    input_data = pd.read_pickle(get_repo_path() / data_dir / params.input)

    # Split data
    train_df, test_df = data_split(input_data, test_size=params.test_proportion)

    # Write output data
    train_df.to_pickle(get_repo_path() / data_dir / params.training_output)
    test_df.to_pickle(get_repo_path() / data_dir / params.test_output)


if __name__ == "__main__":
    main()
