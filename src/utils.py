import os
from pathlib import Path
import dvc.api
from dataclasses import dataclass


@dataclass
class Metadata:
    label_column: str
    timestamp_column: str
    correct_labels: list
    timestamp_multiplier: float
    signal_columns: list
    scalar_columns: list
    datetime_column: str
    patient_column: str
    measure_column: str
    augment_column: str


def get_repo_path():
    current_path = Path(os.path.abspath('.'))
    path = current_path / Path(__file__)

    # Find .git folder in superdirectory
    while not os.path.exists(path / '.git'):
        path = path.parent
        if path == Path('/'):
            raise FileNotFoundError('Probably not inside git repository.')

    return path


def get_metadata():
    """Load metadata from dvc.yaml file"""
    # Get the DVC parameters
    params = dvc.api.params_show()

    # Metadata
    metadata = params['metadata']
    metadata = Metadata(**metadata)

    metadata.signal_columns = [metadata.signal_columns['acc'], metadata.signal_columns['gyro']]
    return metadata
