# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python(venv)
#     language: python
#     name: venv
# ---

# %%
import pandas as pd

# %%
df = pd.read_csv('results.csv')


# %%
def show_boxplot(df, var):
    df[['metric', var]].boxplot(by=var)


# %%
show_boxplot(df, 'crossvalidate.cv')

# %%
show_boxplot(df, 'crossvalidate.use_pipeline')


# %%
show_boxplot(df, 'crossvalidate.features')


# %%
df.columns

# %%

# %%
df = pd.read_csv('feature_crossval.csv')

# %%
'crossvalidate.features' in df.columns

# %%
df[['test_balanced_accuracy', 'crossvalidate.features']].boxplot(by='crossvalidate.features')

# %%
df[['test_balanced_accuracy', 'crossvalidate.features']]

# %%
df[['test_balanced_accuracy', 'crossvalidate.features']].query('`crossvalidate.features`== 120')

# %%
