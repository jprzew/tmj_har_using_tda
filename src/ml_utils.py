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
import matplotlib.pyplot as plt
import seaborn as sns

def _params_or_default(params: dict, default: dict) -> dict:
    """Returns params if params is not none, otherwise returns default."""
    return params if params is not None else default


def get_model(model_name: str, params=None, random_seed=42):

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
            return RandomForestClassifier(**_params_or_default(params, {'n_estimators': 100,
                                                                               'random_state': random_seed}))
        case 'et':
            return ExtraTreesClassifier(**_params_or_default(params, {'n_estimators': 100,
                                                                             'random_state': random_seed}))
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

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    """
    This function will make a pretty plot of a sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    percent:       If True shows percentages. Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for _ in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        perc_matrix = (cf.T / np.sum(cf, axis=1)).T
        group_percentages = ["{0:.0%}".format(value)
                             for value in perc_matrix.flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels,
                fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

