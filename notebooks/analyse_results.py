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
df = pd.read_csv('results2.csv')

# %%

# %%
df

# %%
# df['State'] = df.State.fillna('Success')

# %%
df = df.query('State == "Success"')

# %%
df[['test_balanced_accuracy', 'model.n_estimators']].boxplot(by='model.n_estimators')

# %%
df[['test_balanced_accuracy', 'model.max_depth']].boxplot(by='model.max_depth')

# %%
df[['test_balanced_accuracy', 'model.max_leaf_nodes']].value_counts()

# %%
df.columns

# %%
df[['test_balanced_accuracy', 'model.min_samples_leaf']].boxplot(by='model.min_samples_leaf')

# %%
df[['test_balanced_accuracy', 'model.min_weight_fraction_leaf']].boxplot(by='model.min_weight_fraction_leaf')

# %%
df['model.min_weight_fraction_leaf'].value_counts()

# %%

# %%

df[['test_balanced_accuracy', 'crossvalidate.features']].boxplot(by='crossvalidate.features')

# %%
df['model.class_weight'].value_counts()

# %%
df.sort_values(by='test_balanced_accuracy', ascending=False).head(n=10)

# %% [markdown]
# **Results**
#
# number of estimators: 200-500; does not really matter much right now
#
# max depth: the more the better, trend up to 20; did not check more; but looks bit like saturate in the range 10-20; failed with "none"
#
# min samples leaf: looks that it is the smaller the better; optimum around 2; maybe 1 is better; who knows
#
# number of features looks to be around 100
#

# %% [markdown]
# **New experiment**
#
# Maybe fix number of features to 100; 200
#
# min samples leaf: 1 or 2
#
# check number of estimators 200 - 500
#
# max depth 20 30 infty?
#
# consider multiple values of max features sqrt
#
#

# %%
df.query('`model.n_estimators` == 1000')

# %%

# %%
df[['test_balanced_accuracy', 'model.class_weight']].fillna('na').boxplot(by='model.class_weight')

# %%
df.sort_values(by='test_balanced_accuracy', ascending=False)

# %%
df[['test_balanced_accuracy', 'crossvalidate.features']].fillna('na').boxplot(by='crossvalidate.features')

# %%
