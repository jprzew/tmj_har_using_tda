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
