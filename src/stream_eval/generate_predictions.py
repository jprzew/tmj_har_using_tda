import sys
import pickle
import tqdm
import numpy as np

sys.path.append('/home/jprzew/work/repos/eda_topology/src/')

from utils import get_repo_path, get_metadata
from featurize import compute_all_features, tabularize_data
from crossvalidate import prepare_dataset, filter_data

import pandas as pd

from generate_diagrams import read_test_data

STEP = 30
NO_FEAT = 100
repo_path = get_repo_path()
model_file = repo_path / 'models/model.pkl'

# Load metadata
meta = get_metadata()


#
# # Script parameters
# repo_path = get_repo_path()
# data_path = repo_path / config.EvaluateModel.data
# target_path = repo_path / config.EvaluateModel.target
#
#
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


# def rolling_predict(X_test, model):
#     predictions = []
#     for i in tqdm(range(X_test.shape[0])):
#         p = model.predict(X_test[i, :].reshape(1, -1))
#         predictions.append(p)
#
#     return pd.Series(np.concatenate(predictions))
#
#
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


df_test = read_test_data()
df_features = pd.read_pickle(repo_path / 'stream_data' / 'stream_features.pkl')
rank_df = pd.read_pickle(repo_path / 'data' / 'features_rank.pkl')



with open(model_file, 'rb') as f:
    model = pickle.load(f)


# Prepare the dataset
training_data = prepare_dataset(df_features,
                                non_feature_cols=meta.scalar_columns,
                                target_col=meta.label_column,
                                group_col=meta.patient_column)

# Filter the data
training_data = filter_data(training_data, rank_df)


X_df = df_features[training_data.feature_names]

y = model.predict(X_df)
y_duplicated = np.repeat(y, STEP)


signal = df_test.acc_x.iloc[0]
event = df_test.acc_gyro_event.iloc[0]


results_df =pd.DataFrame({'acc_x': pd.Series(signal), 'event': pd.Series(event),
                          'predictions': pd.Series(y_duplicated)})
# results_df['acc_x'] = df_test.acc_x

results_df['event'] = results_df['event'].astype(float)

results_df.to_pickle(repo_path / 'stream_data' / 'stream_predictions.pkl')

