from pyexpat import features
from sklearn.model_selection import ParameterGrid
import subprocess
import random


param_grid = {'n_estimators': [50, 100, 200, 500, 1000],
              'min_samples_leaf': [1, 5, 10, 20],
              'min_samples_split': [2, 5, 10, 30],
              'features': [10, 20, 40, 80, 100, 200],
              'max_depth': [3, 5, 10, 20, None]}

grid = ParameterGrid(param_grid)


for params in grid:
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"model=random_forest",
                    "--set-param", f"model.n_estimators={params['n_estimators']}",
                    "--set-param", f"model.min_samples_leaf={params['min_samples_leaf']}",
                    "--set-param", f"model.min_samples_split={params['min_samples_split']}",
                    "--set-param", f"model.max_depth={params['max_depth']}",
                    "--set-param", f"crossvalidate.features={params['features']}"])
