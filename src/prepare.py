"""Prepares data for learning. Cuts signals into time-windows

This script is run as a DVC-stage. Inputs, outputs and parameters defined in: dvc.yaml
"""
# Standard library imports
import gc
from collections import Counter
from numpy.random import seed as random_seed


# Third-party imports
import pandas as pd
import numpy as np
from dataclasses import dataclass
import dvc.api

# Local imports
from utils import get_repo_path, get_metadata
from data_transformer import df_windower


@dataclass
class Params:
    input: str
    output: str
    window_size: int
    window_step: int
    random_seed: int
    augment_percent: int
    augment_params: dict
    augmenter: str
    partial_windows: bool
    subsample: bool

# Load metadata
meta = get_metadata()

# Get the DVC parameters
params = dvc.api.params_show()

# Data dir
data_dir = params['directories']['data']

# Stage parameters
params_dict = {**{'input': params['compose']['output']}, **params['prepare']}
params = Params(**params_dict)


class Prepare:

    # ---------------------------
    # Data specific functions
    # ---------------------------

    @staticmethod
    def prepare_preprocess(df):
        # correcting mislabeled events
        Prepare.set_event(df, 16745605820553, '4')  # drinking '2023-01-05 11:20:06.839897'
        Prepare.set_event(df, 16745593471746, '1')  # speech '2023-01-12 12:19:54.042704'
        Prepare.set_event(df, 16745593472779, '0')  # nothing '2023-01-12 12:19:54.042704'
        Prepare.set_event(df, 16745593561236, '0')  # nothing '2023-01-12 12:20:03.041402'
        Prepare.set_event(df, 16745593561869, '0')  # nothing '2023-01-12 12:20:03.041402'
        Prepare.set_event(df, 16742336422623, '1')  # speech '2023-01-24 11:09:12.219610',
        Prepare.set_event(df, 16742336423556, '0')  # nothing '2023-01-24 11:09:12.219610',
        Prepare.set_event(df, 16742336590719, '0')  # nothing '2023-01-24 11:09:29.050752'
        Prepare.set_event(df, 16742336591286, '0')  # nothing '2023-01-24 11:09:29.050752'

        df = df[~df.measure_id.isin(['712', '718'])]  # removing erroneous measurements

        return df

    @staticmethod
    def prepare_postprocess(df):
        return df

    # -------------------
    # General Functions
    # -------------------

    @staticmethod
    def set_event(df, timestamp, new_event):
        """Sets the whole event with a given timestamp to new_event

        Args:
            df: dataframe in the form of data_raw.pkl
            timestamp: timestamp of one sample in the
            new_event: value of the new event

        """

        def _replace_event_in_row(x):
            """Takes as an input single row of data_raw (i.e. x is a single recording)
               Then replaces the whole event with a given timestamp with a new event
               (the whole event, not the single sample)"""
            event_index = np.where(x[meta.timestamp_column] == timestamp)[0]
            if event_index.size == 0:
                return

            event = x[meta.label_column][event_index]
            other_events = np.where(x[meta.label_column] != event)[0]

            infimum = max([index for index in other_events if index < event_index])
            supremum = min([index for index in other_events if index > event_index])

            x[meta.label_column][infimum+1:supremum] = new_event

        df.apply(_replace_event_in_row, axis=1)

    @staticmethod
    def count_events(df):
        """
        Counts how many samples are associated with the events. Then divide number of event samples by 700 (number of
        samples in window). Returns dataframe: column names are the events and values are rounded number of events.
        """
        events = Counter()

        def _count_add_row(x, events=events):
            events += Counter(x)

        df[meta.label_column].apply(_count_add_row)
        dict_events = {key: value / 700 for (key, value) in dict(events).items()}
        return pd.DataFrame([dict_events]).round(2)

    @staticmethod
    def remove_partial_window(df):
        new_df = df[df[meta.label_column].apply(lambda x: len(np.unique(x)) == 1)]
        return new_df.reset_index(drop=True)

    @staticmethod
    def remove_wrong_labels(df):
        return df[df[meta.label_column].apply(lambda x: set(x) <= set(meta.correct_labels))]

    # @staticmethod
    # def data_augmenter(df, percent, augmenter, seed=42, **kwargs):
    #     df[cfg.augment_column] = False  # TODO: Here I got warning
    #     if augmenter is None:
    #         return df
    #     augmented_rows = df.sample(frac=percent/100, replace=True, random_state=seed)
    #
    #     random_seed(seed)
    #     augmented_rows[signal_columns] = \
    #         augmented_rows[signal_columns].apply(lambda x: augmenter(x, **params.Prepare.augment_params), axis=1)
    #
    #     augmented_rows[config.General.augment_column] = True
    #     df_new = pd.concat([df, augmented_rows])
    #     return df_new.reset_index(drop=True)

    @staticmethod
    def prepare(df):
        df = Prepare.prepare_preprocess(df)

        df = df_windower(df,
                         window_size=params.window_size,
                         window_step=params.window_step,
                         scalar_columns=meta.scalar_columns)
        df = Prepare.remove_wrong_labels(df)

        if not params.partial_windows:
            df = Prepare.remove_partial_window(df)

        # df = Prepare.data_augmenter(df,
        #                             percent=params.Prepare.augment_percent,
        #                             augmenter=params.Prepare.augmenter,
        #                             seed=params.Prepare.random_seed)

        counts = Prepare.count_events(df)
        print("\n##############################################")
        print("DATASET EVENTS NUMBER AFTER AUGUMENTATION AND MAYBE REMOVAL")
        print(counts)
        print("\n##############################################")
        df = Prepare.prepare_postprocess(df)
        return df


def main():

    data_path = get_repo_path() / data_dir / params.input
    target_path = get_repo_path() / data_dir / params.output

    df = pd.read_pickle(data_path)
    new_df = Prepare.prepare(df)

    del df
    gc.collect()

    if params.subsample:
        new_df.sample(params.subsample).reset_index(drop=True).to_pickle(target_path)
    else:
        new_df.reset_index(drop=True).to_pickle(target_path)


if __name__ == '__main__':
    main()
