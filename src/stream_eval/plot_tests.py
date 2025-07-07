import sys
sys.path.append('/home/jprzew/work/repos/eda_topology/src/')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_repo_path

from collections import defaultdict


# from tools.dimension_reduction import print_tsne
#
# from tools.data_transformer import df_chopper
# from tools.utils import calculate_timestamp
#
# import config as cfg


labels_to_colours = {0: 'tab:blue',
                     1: 'tab:orange',
                     2: 'tab:red',
                     4: 'tab:purple',
                     8: 'tab:green'}

labels_to_events = {0: 'nothing',
                    1: 'speech',
                    2: 'smoking',
                    4: 'drinking',
                    8: 'eating'}

labels_to_events = defaultdict(lambda: 'wrong label', labels_to_events)
labels_to_colours = defaultdict(lambda: 'black', labels_to_colours)


def df_chopper(df, category, scalar_columns):
    """Cuts long time series contained in df into chunks defined by labels

        Args:
            df - dataframe with signals as columns
            category - column to chop dataframe by
            scalar_columns - columns with scalars
            columns - columns to take, if 'all' takes all the columns

        Returns:
            Dataframe with signals cut into chunks
    """

    def create_row(df):
        """Converts df to one-row-dataframe with columns of df converted to numpy-arrays"""

        new_df = pd.DataFrame({column: [np.array(df[column])] for column in columns})

        scalars = df[df.columns[df.columns.isin(scalar_columns)]].iloc[0]  # filter empty scalars
        new_df[df.columns[df.columns.isin(scalar_columns)]] = scalars
        return new_df

    columns = df.columns.drop([scalar_columns], errors='ignore')

    # Detecting indices where labels change
    encoded_labels = df[category].astype('category').cat.codes.reset_index(drop=True)
    segments = encoded_labels[encoded_labels.diff() != 0].index.tolist()

    splitted = np.split(df, segments)[1:]  # we remove the first element as it is an empty df
    return pd.concat(map(create_row, splitted), ignore_index=True)


# %%
df = pd.read_pickle(get_repo_path() / 'stream_data/stream_predictions.pkl')
df['acc_gyro_timestamp'] = pd.Series(range(len(df)))


# %%
def plot_whole(df, signal_col, event_col='acc_gyro_event', prediction_col='predicted_event',
               time_col='acc_gyro_timestamp', figsize=(10, 5), title=None, bias=6000):
    def concatenate_signals(sig_series):
        """Concatenates series of signals with np.NaN in between"""
        n = len(sig_series)
        to_concat = [np.array([np.nan])] * (2 * n - 1)
        to_concat[::2] = list(sig_series)

        return np.concatenate(to_concat)

    def prepare_plot(event_col, bias=0):
        signals_chopped = df_chopper(df[[signal_col, event_col, time_col]], event_col, [event_col])

        grouped = signals_chopped.groupby(event_col).apply(lambda x: (concatenate_signals(x[time_col]),
                                                                      concatenate_signals(x[signal_col] + bias),
                                                                      x[event_col].iloc[0]))

        grouped.apply(lambda x: plt.plot((x[0] - min_timestamp) * timestamp_multiplier,  # normalised timestamp
                                         x[1],  # data from data_col
                                         label=labels_to_events[x[2]],
                                         color=labels_to_colours[x[2]]))


    timestamp_multiplier = 1



    min_timestamp = min(df[time_col])

    fig = plt.figure(figsize=figsize)

    prepare_plot(event_col)

    if bias != 0:
        prepare_plot(prediction_col, bias=bias)

    fig.legend()
    fig.suptitle(title, fontsize=20)
    return fig


fig = plot_whole(df, 'acc_x', event_col='event', prediction_col='predictions', bias=5000)

plt.savefig(get_repo_path() / 'stream_data/stream_plot.png')

