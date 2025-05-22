# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import path

# +
import pandas as pd
import pickle
from utils import get_repo_path
import config as cfg

import matplotlib.pyplot as plt

all_data = pd.option_context('display.max_rows', None,
                             'display.max_columns', None)
# -

results = pd.read_pickle('../metrics/rfe_results.pkl')

df = pd.read_pickle(get_repo_path() / cfg.data_dir / cfg.features_target)

feature_names = df.columns[:-1]

# %%
# Load computation results
with open('../metrics/rfe_results.pkl', 'rb') as f:
    results = pickle.load(f)

results


# +
def get_selected_proteins(cycle):
    fn = feature_names
    
    for estimator in cycle:
        fn = estimator.get_feature_names_out(fn)
    
    return fn


# %%
for cycle in results:
    print(get_selected_proteins(cycle))
# -



cycle[0]

# +
rfecv = cycle[0]

cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()
# -



X



# +




# # %%
# # Get data
# df = get_and_prepare_data('final_data')

# # Prepare samples data
# samples_df = get_samples_data()

# # Take only the samples wchich satisfy the entry criterion and add 'Class' column
# samples_df = classify_samples(samples_df, classifier_name='predefined_groups')

# # %%
# df = df[df.index != 'iRT-Kit_WR_fusion'].copy()


# %%
# def secure_name(x):
#     try:
#         return get_protein_name(x)
#     except:
#         return 'Error'

# names = pd.Series(df.index).apply(secure_name)

# # %%
# cls_input = prepare_classifier_input(df, samples_df['Class'])
# X = cls_input.X
# y = cls_input.y
# weights = cls_input.weights
# feature_names = cls_input.feature_names
# groups = cls_input.groupings


# %% [markdown]
# ## Selected proteins

# %%



# %% [markdown]
# ## Protein rankings

# %%
def get_protein_sequence(cycle):
    fn = feature_names
    selected = []
    for estimator in cycle:
        fn = estimator.get_feature_names_out(fn)
        selected.append(fn)
    
    return selected


# %%
get_protein_sequence(results[0])

# %% [markdown]
# ## Create results_df

# %%
f = lambda x: pd.Series(np.concatenate(get_protein_sequence(x))).value_counts()

results_df = pd.concat(map(f, results), axis=1)


# %%
with all_data:
    display(results_df.fillna(0))

# %% [markdown]
# ## Display ranking

# %%
with all_data:
    display(results_df.fillna(0).apply(sum, axis=1).sort_values(ascending=False))

# %%
ranking = results_df.fillna(0).apply(sum, axis=1).sort_values(ascending=False)

# %%
names = pd.concat([ranking,
                   pd.Series(ranking.index, index=ranking.index).apply(secure_name)],
                  axis=1)
# -





df[['mean__n_0__kind_abs__fil_star', 'acc_gyro_event']].boxplot(by='acc_gyro_event')

red_df = df[['mean__n_0__dim_10', 'mean__n_0__kind_abs__fil_star',
 'mean__n_0__kind_phi__fil_star', 'no__n_0__kind_abs__fil_star',
 'var__n_1__dim_4__step_30', 'entropy__n_1__dim_3', 'entropy__n_0__dim_10',
 'entropy__n_1__dim_10', 'entropy__n_0__dim_10__kind_abs',
 'entropy__n_1__dim_10__kind_abs', 'entropy__n_0__dim_10__kind_phi',
 'entropy__n_0__kind_abs__fil_star', 'entropy__n_0__dim_10__input_deaths',
 'entropy__n_1__kind_abs__input_births',
 'entropy__n_1__dim_10__kind_abs__input_births',
 'wasser_ampl__p_2__n_1__dim_4__step_30']]

red_df.corr()

['mean__n_0__dim_10' 'mean__n_0__dim_10__kind_abs'
 'mean__n_0__kind_abs__fil_star' 'mean__n_0__kind_phi__fil_star'
 'no__n_0__kind_abs__fil_star' 'no__n_0__kind_phi__fil_star'
 'var__n_1__dim_4__step_30' 'entropy__n_1__dim_3' 'entropy__n_0__dim_10'
 'entropy__n_1__dim_10' 'entropy__n_0__dim_10__kind_abs'
 'entropy__n_1__dim_10__kind_abs' 'entropy__n_0__dim_10__kind_phi'
 'entropy__n_0__kind_abs__fil_star' 'entropy__n_0__dim_10__input_deaths'
 'entropy__n_1__kind_abs__input_births'
 'entropy__n_1__dim_10__kind_abs__input_births'
 'wasser_ampl__p_2__n_1__dim_4__step_30']
