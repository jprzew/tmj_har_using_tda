# EDA via TDA

```
pip install pip-tools
```


Best hyperparameters so far:
Best trial:
  Value: 0.8125736438225987
  Params:
    n_estimators: 174
    max_depth: 46
    min_samples_split: 2
    min_samples_leaf: 1
    max_features: sqrt
  User attrs:

crossvalidate:
  random_seed: 42
  features: 200
  cv: 5
  output_categories: all
  use_pipeline: false
  output: crossval_results.json