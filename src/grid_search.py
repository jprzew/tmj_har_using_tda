from sklearn.model_selection import ParameterGrid
import subprocess
from utils import get_repo_path
from tqdm import tqdm

param_grid = {'n_estimators': [200, 500],
              'min_samples_leaf': [1, 2],
              'features': [100, 200],
              'max_depth': [10, 25],
              'class_weight': ['null', 'balanced']}

grid = ParameterGrid(param_grid)


for params in tqdm(grid):
    command = ["dvc", "exp", "run", "--queue",
               "--set-param", f"model=random_forest",
               "--set-param", f"model.n_estimators={params['n_estimators']}",
               "--set-param", f"model.min_samples_leaf={params['min_samples_leaf']}",
               "--set-param", f"model.max_depth={params['max_depth']}",
               "--set-param", f"crossvalidate.features={params['features']}",
               "--set-param", f"model.class_weight={params['class_weight']}"]

    subprocess.run(command, cwd=str(get_repo_path()))

# To run: dvc exp run --run-all
# To show: dvc exp show --drop '.*' --keep 'Experiment|.*accuracy|.*estimators|.*leaf'
# To csv: dvc exp show --csv --drop '.*' --keep 'Experiment|State|.*accuracy|.*estimators|.*leaf|.*depth|.*features'
