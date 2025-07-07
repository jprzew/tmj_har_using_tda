import sys

sys.path.append('/home/jprzew/work/repos/eda_topology/src/')

from utils import get_repo_path, get_metadata
from featurize import compute_all_features, tabularize_data

import pandas as pd


with open(get_repo_path() / 'stream_data' / 'stream_diagrams.pkl', 'rb') as f:
    diagrams = pd.read_pickle(f)

results = compute_all_features(diagrams)

df = tabularize_data(results)

df.to_pickle(get_repo_path() / 'stream_data' / 'stream_features.pkl')


