# Repackage
import repackage

# Add the parent directory to sys.path
repackage.add_path('..')

# Standard library imports
import pickle
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm


import sys
print(sys.path)


# Local imports
# import config
# import params
# from tools import get_signal_names
from utils import get_repo_path, get_metadata
from data_transformer import df_windower
from diagrams import compute_required_diagrams
# from tools import data_transformer

# Load metadata
meta = get_metadata()

STEP = 30

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
    def prepare(df, do_not_remove_wrong_labels=False, window_size=700, window_step=100):
        df = Prepare.prepare_preprocess(df)

        df = df_windower(df,
                         window_size=window_size,
                         window_step=window_step,
                         scalar_columns=meta.scalar_columns)

        if not do_not_remove_wrong_labels:
            df = Prepare.remove_wrong_labels(df)

        # df = Prepare.remove_partial_window(df)



        counts = Prepare.count_events(df)
        print("\n##############################################")
        print("DATASET EVENTS NUMBER AFTER AUGUMENTATION AND MAYBE REMOVAL")
        print(counts)
        print("\n##############################################")
        df = Prepare.prepare_postprocess(df)
        return df



def rolling_predict(X_test, model):
    predictions = []
    for i in tqdm(range(X_test.shape[0])):
        p = model.predict(X_test[i, :].reshape(1, -1))
        predictions.append(p)

    return pd.Series(np.concatenate(predictions))


# TODO: This function needs to be tested
# def generate_ml_predictions(df,
#                             model,
#                             signal_columns,
#                             features,
#                             predictors,
#                             window_step=params.EvaluateModel.window_step,
#                             window_size=params.EvaluateModel.window_size):
#
#     df_featurized = compute_features(df, signal_columns, features, window_step, window_size)
#     result = rolling_predict(np.array(df_featurized[predictors]), model)
#     df[params.EvaluateModel.prediction_column] = result.loc[result.index.repeat(window_step)].reset_index(drop=True)
#     return df


def read_test_data():
    df = pd.read_pickle(get_repo_path() / 'data' / 'data_raw_test.pkl')
    return df.query('patient_id == "5"')


def main():
    df = read_test_data()


    prepared_df = Prepare.prepare(df, do_not_remove_wrong_labels=True, window_step=STEP)
    prepared_df[meta.label_column] = prepared_df[meta.label_column].apply(lambda x: x.astype(int))

    diagrams = compute_required_diagrams(prepared_df)

    # Save the diagrams to a file
    with open(get_repo_path() / 'stream_data' / 'stream_diagrams.pkl', 'wb') as f:
        pd.to_pickle(diagrams, f)


if __name__ == "main":
    main()


# # Script parameters
# repo_path = get_repo_path()
# data_path = repo_path / config.EvaluateModel.data
# target_path = repo_path / config.EvaluateModel.target
#
# model_file = repo_path / config.EvaluateModel.model
# predictors_file = repo_path / config.EvaluateModel.predictor_index
# features_file = repo_path / config.EvaluateModel.features
#
# model_ann_file = repo_path / config.EvaluateModel.model_ann
# encoder_file = repo_path / config.EvaluateModel.encoder
# scaler_file = repo_path / config.EvaluateModel.scaler
#
# signal_columns = get_signal_names()
# model_ann = tf.keras.models.load_model(model_ann_file)
#
# df = pd.read_pickle(data_path)
# df = df[df[config.General.measure_column] == str(params.EvaluateModel.measure_id)]
#
# if params.EvaluateModel.model_type == "ann":
#     with open(encoder_file, 'rb') as f:
#         encoder = pickle.load(f)
#     with open(scaler_file, 'rb') as f:
#         scaler = pickle.load(f)
#     model = tf.keras.models.load_model(model_ann_file)
#
#     df_predictions = generate_ann_predictions(df, model, encoder, signal_columns, scaler)
# elif params.EvaluateModel.model_type == "skl":
#     with open(model_file, 'rb') as f:
#         model = pickle.load(f)
#     with open(predictors_file, 'rb') as f:
#         predictors = pickle.load(f)
#     with open(features_file, 'rb') as f:
#         features = pickle.load(f)
#
#     df_predictions = generate_ml_predictions(df, model, signal_columns, features, predictors)
# else:
#     Exception("Specify correct model type in params.EvaluateModel!")
#
# df_predictions.to_pickle(target_path)