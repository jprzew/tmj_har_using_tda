# Standard library imports
from dataclasses import dataclass


# Third party imports
# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np


def _params_or_default(params: dict, default: dict) -> dict:
    """Returns params if params is not none, otherwise returns default."""
    return params if params is not None else default


def get_model(model_name: str, params=None):

    match model_name:
        case name if name.startswith('ridge-'):  # e.g. ridge-0.1
            return RidgeClassifier(**_params_or_default(params, {'alpha': name.split('-')[1]}))
        case 'logistic':
            return LogisticRegression(**_params_or_default(params, {}))
        case 'sgd':
            return SGDClassifier(**_params_or_default(params, {'max_iter': 1000, 'tol': 1e-3}))
        case 'pa':
            return PassiveAggressiveClassifier(**_params_or_default(params, {'max_iter': 1000, 'tol': 1e-3}))
        case 'lda':
            return LinearDiscriminantAnalysis(**_params_or_default(params, {}))
        case 'cart':
            return DecisionTreeClassifier(**_params_or_default(params, {}))
        case 'extra':
            return ExtraTreeClassifier(**_params_or_default(params, {}))
        case name if name.startswith('knn-'):
            return KNeighborsClassifier(**_params_or_default(params, {'n_neighbors': int(name.split('-')[1])}))
        case name if name.startswith('svmr'):  # e.g. svmr0.1
            return SVC(**_params_or_default(params, {'C': float(name.split('svmr')[1])}))
        case 'svml':
            return SVC(**_params_or_default(params, {'kernel': 'linear'}))
        case 'svmp':
            return SVC(**_params_or_default(params, {'kernel': 'poly'}))
        case 'bayes':
            return GaussianNB(**_params_or_default(params, {}))
        case 'ada':
            return AdaBoostClassifier(**_params_or_default(params, {'n_estimators': 100}))
        case 'bag':
            return BaggingClassifier(**_params_or_default(params, {'n_estimators': 100}))
        case 'rf':
            return RandomForestClassifier(**_params_or_default(params, {'n_estimators': 100}))
        case 'et':
            return ExtraTreesClassifier(**_params_or_default(params, {'n_estimators': 100}))
        case 'gbm':
            return GradientBoostingClassifier(**_params_or_default(params, {'n_estimators': 100}))
        case _:
            raise ValueError(f"Unknown model: {model_name}")


@dataclass
class TrainingData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    groups: np.ndarray


# load the dataset, returns X and y elements
def prepare_dataset(df: pd.DataFrame, non_feature_cols: list[str], target_col: str, group_col: str) -> TrainingData:

    # Target
    y = np.array(df[target_col])

    # Features
    features_df = df.drop(columns=non_feature_cols + [target_col, group_col], errors='ignore')
    X = np.array(features_df)

    # Groups & feature names
    groups = np.array(df[group_col])
    feature_names = list(features_df.columns)

    return TrainingData(X, y, feature_names, groups)


# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create a pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline
