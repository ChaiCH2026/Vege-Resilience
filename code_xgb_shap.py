# %% [markdown]
# 
# xgboost + shap
# %%
import pandas as pd
import numpy as np
import shap 
import xgboost as xgb

import time
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.inspection import permutation_importance

import optuna

plt.rc('font',family='Times New Roman',size = 15)

# %% [markdown]
# ## 1 read data

# %%
vod_trait_df = pd.read_csv(r'result_data/vod_trait_PC_fi.csv')
vod_trait_df.head()

# %%
gimms_trait_df = pd.read_csv(r'result_data/gimms_trait_PC_fi.csv')
gimms_trait_df.head()

# %%
x_vod_ac1_trait = vod_trait_df.loc[: ,['ldmc', 'lnc',  'sla', 'lnpr', 'root_depth', 'rplant', 'height']]
x_vod_ac1_trait = np.log(x_vod_ac1_trait)

x_vod_ac1_clim_pc = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav','ai', 'pc1', 'pc2']]
x_vod_ac1_clim = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav','ai']]

x_vod_var_trait = vod_trait_df.loc[: ,['ldmc', 'lnc', 'sla', 'lnpr', 'root_depth', 'rplant', 'height']]
x_vod_var_trait = np.log(x_vod_var_trait)

x_vod_var_clim_pc = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav', 'ai', 'pc1', 'pc2']]
x_vod_var_clim = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav', 'ai']]

y_vod_ac1 = vod_trait_df['ac1']
y_vod_var = vod_trait_df['VAR']

# %%
x_gimms_ac1_trait = gimms_trait_df.loc[: ,['ldmc', 'lnc',  'sla', 'lnpr', 'root_depth', 'rplant', 'height']]
x_gimms_ac1_trait = np.log(x_gimms_ac1_trait)

x_gimms_ac1_clim_pc = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai', 'pc1', 'pc2']]
x_gimms_ac1_clim = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai']]

x_gimms_var_trait = gimms_trait_df.loc[: ,['ldmc', 'lnc',  'sla', 'lnpr', 'root_depth', 'rplant', 'height']]
x_gimms_var_trait = np.log(x_gimms_var_trait)

x_gimms_var_clim_pc = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai', 'pc1', 'pc2']]
x_gimms_var_clim = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai']]

y_gimms_ac1 = gimms_trait_df['ac1']
y_gimms_var = gimms_trait_df['VAR']

# %%
x_vod_ac1_trait.hist(bins = 30, figsize=[12,10])

# %%
x_gimms_ac1_trait.hist(bins = 30, figsize=[12,10])

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif(df):
    """input：DataFrame"""
    X = add_constant(df)  
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data.drop(index=0)  

# %%
calculate_vif(x_vod_ac1_trait)

# %%
calculate_vif(x_gimms_var_clim_pc)

# %%
x_vod_ac1_trait.head()

# %%
vod_trait_df['landcover'].shape

# %%
gimms_trait_df['landcover'].shape

# %%
vod_trait_df['landcover'].to_csv(r'result_data/vod_trait_landcover.csv', index = False)
gimms_trait_df['landcover'].to_csv(r'result_data/gimms_trait_landcover.csv', index = False)

# %%
def prepare_data_split(df_dic, size_val= 0.3):
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(df_dic['X'], df_dic['y'], test_size=size_val, random_state=1412)

    return {'X_train': X_train, 'X_val': X_val,  'y_train': y_train, 'y_val': y_val}

# %%
def model_fit(dataset, best_params):
    finalmodel = xgb.XGBRegressor(**best_params)
    finalmodel.fit(dataset['X_train'], dataset['y_train'])

    print( 'train r2:' ,finalmodel.score(dataset['X_train'], dataset['y_train']),
           'val r2:' ,finalmodel.score(dataset['X_val'], dataset['y_val']))
    
    return finalmodel

# %% [markdown]
# ## 2 gimms

# %% [markdown]
# ### 2.1 gimms ac1 trait

# %%
sets_gimms_ac1_trait = prepare_data_split({'X': x_gimms_ac1_trait, 'y': y_gimms_ac1})
sets_gimms_ac1_trait

# %%

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # cross_val
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait['X_train'], sets_gimms_ac1_trait['y_train'], scoring='r2', cv=cv).min()

    return score  

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_gimms_ac1_trait = study.best_params

# %%
best_params_gimms_ac1_trait

# %%
'''
pars     
{'n_estimators': 750,
 'learning_rate': 0.01809368849165838,
 'max_depth': 7,
 'subsample': 0.6189656148308222,
 'colsample_bytree': 0.9664128275938754,
 'min_child_weight': 2.4949459705247357,
 'gamma': 1.0187712871854602,
 'reg_alpha': 2.3247145047672455,
 'reg_lambda': 3.2893951884326835}
'''

# %%
finalmodel_gimms_ac1_trait = model_fit(sets_gimms_ac1_trait, best_params_gimms_ac1_trait)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait)
shap_explainer_gimms_ac1_trait = shap_explainer(x_gimms_ac1_trait)   

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait,max_display=20)

# %%
x_gimms_ac1_trait.columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : x_gimms_ac1_trait.columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
import pickle
### save
with open(r'result_data/shap_explainer_gimms_ac1_trait.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait, file)

# %% [markdown]
# ### 2.2 gimms var trait

# %%
sets_gimms_var_trait = prepare_data_split({'X': x_gimms_var_trait, 'y': y_gimms_var})
sets_gimms_var_trait

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait['X_train'], sets_gimms_var_trait['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800) 

print("Best trial:")
print(study.best_trial)

best_params_gimms_var_trait = study.best_params

# %%
best_params_gimms_var_trait

# %%
'''
{'n_estimators': 500,
 'learning_rate': 0.013009044653017809,
 'max_depth': 8,
 'subsample': 0.5583035108144657,
 'colsample_bytree': 0.9693380425365601,
 'min_child_weight': 2.7786313628812547,
 'gamma': 1.0031250406155576,
 'reg_alpha': 1.3615994009375114,
 'reg_lambda': 3.0392854449463615}
'''

# %%
finalmodel_gimms_var_trait = model_fit(sets_gimms_var_trait, best_params_gimms_var_trait)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait)
shap_explainer_gimms_var_trait = shap_explainer(x_gimms_var_trait)   

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait,max_display=20)

# %%
var_names_sorted =pd.DataFrame({'vars' : x_gimms_var_trait.columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait, file)

# %% [markdown]
# ## 3 vod

# %% [markdown]
# ### 3.1 vod ac1 trait

# %%
sets_vod_ac1_trait = prepare_data_split({'X': x_vod_ac1_trait, 'y': y_vod_ac1})
sets_vod_ac1_trait

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait['X_train'], sets_vod_ac1_trait['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_vod_ac1_trait = study.best_params

# %%
best_params_vod_ac1_trait

# %%
'''
{'n_estimators': 700,
 'learning_rate': 0.021686589772488447,
 'max_depth': 8,
 'subsample': 0.5660946459506793,
 'colsample_bytree': 0.9259077096176975,
 'min_child_weight': 1.4094456997523306,
 'gamma': 1.0482915004533124,
 'reg_alpha': 4.054343399700015,
 'reg_lambda': 4.569494905272867}
'''

# %%
finalmodel_vod_ac1_trait = model_fit(sets_vod_ac1_trait, best_params_vod_ac1_trait)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait)
shap_explainer_vod_ac1_trait = shap_explainer(x_vod_ac1_trait)   

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait,max_display=20)

# %%
x_vod_ac1_trait.columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : x_vod_ac1_trait.columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait, file)

# %% [markdown]
# ### 3.2 vod var trait

# %%
sets_vod_var_trait = prepare_data_split({'X': x_vod_var_trait, 'y': y_vod_var})
sets_vod_var_trait

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait['X_train'], sets_vod_var_trait['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_vod_var_trait = study.best_params

# %%
best_params_vod_var_trait

# %%
'''
{'n_estimators': 950,
 'learning_rate': 0.11828171427780378,
 'max_depth': 8,
 'subsample': 0.7997027688007137,
 'colsample_bytree': 0.7842407655556669,
 'min_child_weight': 2.700990382484118,
 'gamma': 1.0412951863922237,
 'reg_alpha': 1.363576106865044,
 'reg_lambda': 2.6557337978319135}
'''

# %%
finalmodel_vod_var_trait = model_fit(sets_vod_var_trait, best_params_vod_var_trait)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait)
shap_explainer_vod_var_trait = shap_explainer(x_vod_var_trait)   

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait,max_display=20)

# %%
var_names_sorted =pd.DataFrame({'vars' : x_vod_var_trait.columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait, file)

# %% [markdown]
# ## 4 landcover

# %%
vod_trait_df['landcover'].value_counts()

# %%
gimms_trait_df['landcover'].value_counts()

# %%
ld_use = [10,9,7,2,8,5,1,4]

# %%
def get_ld_data(ld_n):
    vod_trait_df_ld = vod_trait_df[vod_trait_df['landcover'] == ld_n]
    x_vod_ac1_trait = vod_trait_df.loc[: ,['ldmc', 'lnc',  'sla', 'lnpr','root_depth', 'rplant', 'height']]
    x_vod_ac1_trait = np.log(x_vod_ac1_trait)

    x_vod_ac1_clim_pc = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav','ai', 'pc1', 'pc2']]
    x_vod_ac1_clim = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav','ai']]

    x_vod_var_trait = vod_trait_df.loc[: ,['ldmc', 'lnc',  'sla', 'lnpr', 'root_depth', 'rplant', 'height']]
    x_vod_var_trait = np.log(x_vod_var_trait)

    x_vod_var_clim_pc = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav', 'ai', 'pc1', 'pc2']]
    x_vod_var_clim = vod_trait_df.loc[:, ['temp', 'prec_mean', 'prec_iav', 'ai']]

    y_vod_ac1 = vod_trait_df['ac1']
    y_vod_var = vod_trait_df['VAR']

    gimms_trait_df_ld = gimms_trait_df[gimms_trait_df['landcover'] == ld_n]
    x_gimms_ac1_trait = gimms_trait_df.loc[: ,['ldmc', 'lnc',  'sla', 'lnpr','root_depth', 'rplant', 'height']]
    x_gimms_ac1_trait = np.log(x_gimms_ac1_trait)

    x_gimms_ac1_clim_pc = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai', 'pc1', 'pc2']]
    x_gimms_ac1_clim = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai']]

    x_gimms_var_trait = gimms_trait_df.loc[: ,['ldmc', 'lnc', 'sla', 'lnpr','root_depth', 'rplant', 'height']]
    x_gimms_var_trait = np.log(x_gimms_var_trait)

    x_gimms_var_clim_pc = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai', 'pc1', 'pc2']]
    x_gimms_var_clim = gimms_trait_df.loc[:, ['temp', 'prec_mean','prec_iav', 'ai']]

    y_gimms_ac1 = gimms_trait_df['ac1']
    y_gimms_var = gimms_trait_df['VAR']

    vod_dic = {'x_ac1_trait': x_vod_ac1_trait,
               'x_ac1_clim_pc': x_vod_ac1_clim_pc,
               'x_ac1_clim': x_vod_ac1_clim,
               'x_var_trait': x_vod_var_trait,
               'x_var_clim_pc': x_vod_var_clim_pc,
               'x_var_clim': x_vod_var_clim,
               'y_ac1': y_vod_ac1,
               'y_var': y_vod_var}

    gimms_dic = {'x_ac1_trait': x_gimms_ac1_trait,  
               'x_ac1_clim_pc': x_gimms_ac1_clim_pc,       
               'x_ac1_clim': x_gimms_ac1_clim,
               'x_var_trait': x_gimms_var_trait,
               'x_var_clim_pc': x_gimms_var_clim_pc,
               'x_var_clim': x_gimms_var_clim,
               'y_ac1': y_gimms_ac1,
               'y_var': y_gimms_var}

    return vod_dic, gimms_dic

# %%
vod_dic_n, gimms_dic_n = get_ld_data(10)

# %%
vod_dic_n

# %% [markdown]
# ### 4.1 ld_10

# %%
ld_use

# %%
vod_dic_ld10, gimms_dic_ld10 = get_ld_data(10)

# %% [markdown]
# #### 4.1.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld10 = prepare_data_split({'X': gimms_dic_ld10['x_ac1_trait'], 'y': gimms_dic_ld10['y_ac1']})
sets_gimms_ac1_trait_ld10

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld10['X_train'], sets_gimms_ac1_trait_ld10['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_gimms_ac1_trait_ld10 = study.best_params

# %%
best_params_gimms_ac1_trait_ld10

# %%
'''
{'n_estimators': 900,
 'learning_rate': 0.05436433566513693,
 'max_depth': 8,
 'subsample': 0.636421193324808,
 'colsample_bytree': 0.8766431741625842,
 'min_child_weight': 4.171061551722894,
 'gamma': 1.1160613591663167,
 'reg_alpha': 2.770896760615811,
 'reg_lambda': 4.184285691103045}
'''

# %%
finalmodel_gimms_ac1_trait_ld10 = model_fit(sets_gimms_ac1_trait_ld10, best_params_gimms_ac1_trait_ld10)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld10)
shap_explainer_gimms_ac1_trait_ld10 = shap_explainer(gimms_dic_ld10['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld10,max_display=20)

# %%
gimms_dic_ld10['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld10['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld10.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld10.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld10, file)

# %% [markdown]
# #### 4.1.2  gimms var trait

# %%
sets_gimms_var_trait_ld10 = prepare_data_split({'X': gimms_dic_ld10['x_var_trait'], 'y': gimms_dic_ld10['y_var']})
sets_gimms_var_trait_ld10

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld10['X_train'], sets_gimms_var_trait_ld10['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_gimms_var_trait_ld10 = study.best_params

# %%
best_params_gimms_var_trait_ld10

# %%
'''
{'n_estimators': 950,
 'learning_rate': 0.014101898332916259,
 'max_depth': 8,
 'subsample': 0.6557471848569718,
 'colsample_bytree': 0.7582068734563229,
 'min_child_weight': 1.1121515856524202,
 'gamma': 1.2133515950737532,
 'reg_alpha': 1.9496284331142495,
 'reg_lambda': 1.3872794495907634}
'''

# %%
finalmodel_gimms_var_trait_ld10 = model_fit(sets_gimms_var_trait_ld10, best_params_gimms_var_trait_ld10)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld10)
shap_explainer_gimms_var_trait_ld10 = shap_explainer(gimms_dic_ld10['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld10,max_display=20)

# %%
gimms_dic_ld10['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld10['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld10.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld10.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld10, file)

# %% [markdown]
# #### 4.1.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld10 = prepare_data_split({'X': vod_dic_ld10['x_ac1_trait'], 'y': vod_dic_ld10['y_ac1']})
sets_vod_ac1_trait_ld10

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld10['X_train'], sets_vod_ac1_trait_ld10['y_train'], scoring='r2', cv=cv).min()

    return score

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_vod_ac1_trait_ld10 = study.best_params

# %%
best_params_vod_ac1_trait_ld10

# %%
'''
{'n_estimators': 950,
 'learning_rate': 0.012248888897812678,
 'max_depth': 8,
 'subsample': 0.7868993563326732,
 'colsample_bytree': 0.8171093801707239,
 'min_child_weight': 3.015830490514084,
 'gamma': 1.0092343269462805,
 'reg_alpha': 1.0108977630983875,
 'reg_lambda': 1.3128445693390502}
'''

# %%
finalmodel_vod_ac1_trait_ld10 = model_fit(sets_vod_ac1_trait_ld10, best_params_vod_ac1_trait_ld10)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld10)
shap_explainer_vod_ac1_trait_ld10 = shap_explainer(vod_dic_ld10['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld10,max_display=20)

# %%
vod_dic_ld10['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld10['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld10.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld10.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld10, file)

# %% [markdown]
# #### 4.1.4 vod var trait

# %%
sets_vod_var_trait_ld10 = prepare_data_split({'X': vod_dic_ld10['x_var_trait'], 'y': vod_dic_ld10['y_var']})
sets_vod_var_trait_ld10

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld10['X_train'], sets_vod_var_trait_ld10['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_vod_var_trait_ld10 = study.best_params

# %%
best_params_vod_var_trait_ld10

# %%
'''
{'n_estimators': 850,
 'learning_rate': 0.035397713293680945,
 'max_depth': 8,
 'subsample': 0.5756657445038094,
 'colsample_bytree': 0.8721448885185383,
 'min_child_weight': 4.072088583025047,
 'gamma': 1.069013166286584,
 'reg_alpha': 1.7776780128368095,
 'reg_lambda': 3.8208141924871883}
'''

# %%
finalmodel_vod_var_trait_ld10 = model_fit(sets_vod_var_trait_ld10, best_params_vod_var_trait_ld10)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld10)
shap_explainer_vod_var_trait_ld10 = shap_explainer(vod_dic_ld10['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld10,max_display=20)

# %%
vod_dic_ld10['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld10['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld10.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld10.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld10, file)

# %% [markdown]
# ### 4.2 ld_9

# %%
ld_use

# %%
vod_dic_ld9, gimms_dic_ld9 = get_ld_data(9)

# %% [markdown]
# #### 4.2.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld9 = prepare_data_split({'X': gimms_dic_ld9['x_ac1_trait'], 'y': gimms_dic_ld9['y_ac1']})
sets_gimms_ac1_trait_ld9

# %%
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld9['X_train'], sets_gimms_ac1_trait_ld9['y_train'], scoring='r2', cv=cv).min()

    return score 

study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  

print("Best trial:")
print(study.best_trial)

best_params_gimms_ac1_trait_ld9 = study.best_params

# %%
best_params_gimms_ac1_trait_ld9

# %%
'''
{'n_estimators': 400,
 'learning_rate': 0.028232075163984568,
 'max_depth': 8,
 'subsample': 0.754132006516745,
 'colsample_bytree': 0.9565455526771866,
 'min_child_weight': 2.885406426989132,
 'gamma': 1.1836958655569407,
 'reg_alpha': 4.176187837484831,
 'reg_lambda': 1.1645701019163845}
'''

# %%
finalmodel_gimms_ac1_trait_ld9 = model_fit(sets_gimms_ac1_trait_ld9, best_params_gimms_ac1_trait_ld9)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld9)
shap_explainer_gimms_ac1_trait_ld9 = shap_explainer(gimms_dic_ld9['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld9,max_display=20)

# %%
gimms_dic_ld9['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld9['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld9.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld9.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld9, file)

# %% [markdown]
# #### 4.2.2  gimms var trait

# %%
sets_gimms_var_trait_ld9 = prepare_data_split({'X': gimms_dic_ld9['x_var_trait'], 'y': gimms_dic_ld9['y_var']})
sets_gimms_var_trait_ld9

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld9['X_train'], sets_gimms_var_trait_ld9['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld9 = study.best_params

# %%
best_params_gimms_var_trait_ld9

# %%
'''  
{'n_estimators': 950,
 'learning_rate': 0.07768175756432812,
 'max_depth': 8,
 'subsample': 0.7204961756010304,
 'colsample_bytree': 0.9645574865174288,
 'min_child_weight': 3.5304119817566137,
 'gamma': 1.0124142423164788,
 'reg_alpha': 1.0402300939010076,
 'reg_lambda': 4.1803394626323875}
'''

# %%
finalmodel_gimms_var_trait_ld9 = model_fit(sets_gimms_var_trait_ld9, best_params_gimms_var_trait_ld9)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld9)
shap_explainer_gimms_var_trait_ld9 = shap_explainer(gimms_dic_ld9['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld9,max_display=20)

# %%
gimms_dic_ld9['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld9['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld9.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld9.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld9, file)

# %% [markdown]
# #### 4.2.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld9 = prepare_data_split({'X': vod_dic_ld9['x_ac1_trait'], 'y': vod_dic_ld9['y_ac1']})
sets_vod_ac1_trait_ld9

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld9['X_train'], sets_vod_ac1_trait_ld9['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld9 = study.best_params

# %%
best_params_vod_ac1_trait_ld9

# %%
'''
         
{'n_estimators': 350,
 'learning_rate': 0.054313279348326096,
 'max_depth': 8,
 'subsample': 0.7821331158562083,
 'colsample_bytree': 0.8366488852318389,
 'min_child_weight': 4.9593635261503195,
 'gamma': 1.2011813766405777,
 'reg_alpha': 3.1295367937459515,
 'reg_lambda': 4.384958871325772}
'''

# %%
finalmodel_vod_ac1_trait_ld9 = model_fit(sets_vod_ac1_trait_ld9, best_params_vod_ac1_trait_ld9)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld9)
shap_explainer_vod_ac1_trait_ld9 = shap_explainer(vod_dic_ld9['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld9,max_display=20)

# %%
vod_dic_ld9['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld9['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld9.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld9.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld9, file)

# %% [markdown]
# #### 4.2.4 vod var trait

# %%
sets_vod_var_trait_ld9 = prepare_data_split({'X': vod_dic_ld9['x_var_trait'], 'y': vod_dic_ld9['y_var']})
sets_vod_var_trait_ld9

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld9['X_train'], sets_vod_var_trait_ld9['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld9 = study.best_params

# %%
best_params_vod_var_trait_ld9

# %%
'''
         
{'n_estimators': 850,
 'learning_rate': 0.01629772509625054,
 'max_depth': 8,
 'subsample': 0.6576412875845943,
 'colsample_bytree': 0.7602126642279464,
 'min_child_weight': 1.7923452788318581,
 'gamma': 1.002559637142319,
 'reg_alpha': 4.0880397653053,
 'reg_lambda': 4.6617084201689405}
'''

# %%
finalmodel_vod_var_trait_ld9 = model_fit(sets_vod_var_trait_ld9, best_params_vod_var_trait_ld9)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld9)
shap_explainer_vod_var_trait_ld9 = shap_explainer(vod_dic_ld9['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld9,max_display=20)

# %%
vod_dic_ld9['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld9['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld9.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld9.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld9, file)

# %% [markdown]
# ### 4.3 ld_7

# %%
ld_use

# %%
vod_dic_ld7, gimms_dic_ld7 = get_ld_data(7)

# %% [markdown]
# #### 4.3.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld7 = prepare_data_split({'X': gimms_dic_ld7['x_ac1_trait'], 'y': gimms_dic_ld7['y_ac1']})
sets_gimms_ac1_trait_ld7

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld7['X_train'], sets_gimms_ac1_trait_ld7['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_ac1_trait_ld7 = study.best_params

# %%
best_params_gimms_ac1_trait_ld7

# %%
'''
         
{'n_estimators': 800,
 'learning_rate': 0.10356721965725432,
 'max_depth': 8,
 'subsample': 0.7714430224906771,
 'colsample_bytree': 0.931365969683221,
 'min_child_weight': 4.803495012737521,
 'gamma': 1.2359054159837322,
 'reg_alpha': 1.3770728678605908,
 'reg_lambda': 3.370765873785518}
'''

# %%
finalmodel_gimms_ac1_trait_ld7 = model_fit(sets_gimms_ac1_trait_ld7, best_params_gimms_ac1_trait_ld7)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld7)
shap_explainer_gimms_ac1_trait_ld7 = shap_explainer(gimms_dic_ld7['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld7,max_display=20)

# %%
gimms_dic_ld7['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld7['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld7.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld7.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld7, file)

# %% [markdown]
# #### 4.3.2  gimms var trait

# %%
sets_gimms_var_trait_ld7 = prepare_data_split({'X': gimms_dic_ld7['x_var_trait'], 'y': gimms_dic_ld7['y_var']})
sets_gimms_var_trait_ld7

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld7['X_train'], sets_gimms_var_trait_ld7['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld7 = study.best_params

# %%
best_params_gimms_var_trait_ld7

# %%
'''
         
{'n_estimators': 750,
 'learning_rate': 0.02020331457569499,
 'max_depth': 8,
 'subsample': 0.658274386175599,
 'colsample_bytree': 0.9349685470974849,
 'min_child_weight': 2.3277444011057105,
 'gamma': 1.0684406654807481,
 'reg_alpha': 2.286494994944669,
 'reg_lambda': 3.051742900326345}
'''

# %%
finalmodel_gimms_var_trait_ld7 = model_fit(sets_gimms_var_trait_ld7, best_params_gimms_var_trait_ld7)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld7)
shap_explainer_gimms_var_trait_ld7 = shap_explainer(gimms_dic_ld7['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld7,max_display=20)

# %%
gimms_dic_ld7['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld7['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld7.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld7.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld7, file)

# %% [markdown]
# #### 4.3.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld7 = prepare_data_split({'X': vod_dic_ld7['x_ac1_trait'], 'y': vod_dic_ld7['y_ac1']})
sets_vod_ac1_trait_ld7

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld7['X_train'], sets_vod_ac1_trait_ld7['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld7 = study.best_params

# %%
best_params_vod_ac1_trait_ld7

# %%
'''
         
{'n_estimators': 700,
 'learning_rate': 0.04218550158305073,
 'max_depth': 8,
 'subsample': 0.7416005044540704,
 'colsample_bytree': 0.7229084021390174,
 'min_child_weight': 2.011805025321214,
 'gamma': 1.1757620655828236,
 'reg_alpha': 1.3223075358003789,
 'reg_lambda': 1.8043530628589943}
'''

# %%
finalmodel_vod_ac1_trait_ld7 = model_fit(sets_vod_ac1_trait_ld7, best_params_vod_ac1_trait_ld7)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld7)
shap_explainer_vod_ac1_trait_ld7 = shap_explainer(vod_dic_ld7['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld7,max_display=20)

# %%
vod_dic_ld7['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld7['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld7.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld7.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld7, file)

# %% [markdown]
# #### 4.3.4 vod var trait

# %%
sets_vod_var_trait_ld7 = prepare_data_split({'X': vod_dic_ld7['x_var_trait'], 'y': vod_dic_ld7['y_var']})
sets_vod_var_trait_ld7

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld7['X_train'], sets_vod_var_trait_ld7['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld7 = study.best_params

# %%
best_params_vod_var_trait_ld7

# %%
'''
         
{'n_estimators': 950,
 'learning_rate': 0.010034833593469319,
 'max_depth': 8,
 'subsample': 0.6033312669201581,
 'colsample_bytree': 0.7646821407955898,
 'min_child_weight': 2.501121817856065,
 'gamma': 1.1738046385397323,
 'reg_alpha': 2.3596040891672967,
 'reg_lambda': 3.0061948997342394}
'''

# %%
finalmodel_vod_var_trait_ld7 = model_fit(sets_vod_var_trait_ld7, best_params_vod_var_trait_ld7)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld7)
shap_explainer_vod_var_trait_ld7 = shap_explainer(vod_dic_ld7['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld7,max_display=20)

# %%
vod_dic_ld7['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld7['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld7.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld7.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld7, file)

# %% [markdown]
# ### 4.4 ld_2

# %%
ld_use

# %%
vod_dic_ld2, gimms_dic_ld2 = get_ld_data(2)

# %% [markdown]
# #### 4.4.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld2 = prepare_data_split({'X': gimms_dic_ld2['x_ac1_trait'], 'y': gimms_dic_ld2['y_ac1']})
sets_gimms_ac1_trait_ld2

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld2['X_train'], sets_gimms_ac1_trait_ld2['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_ac1_trait_ld2 = study.best_params

# %%
best_params_gimms_ac1_trait_ld2

# %%
'''
         
{'n_estimators': 650,
 'learning_rate': 0.016801494135580155,
 'max_depth': 8,
 'subsample': 0.7567475259755564,
 'colsample_bytree': 0.9919312917576065,
 'min_child_weight': 3.441440200197119,
 'gamma': 1.0791075915079462,
 'reg_alpha': 4.097809355360556,
 'reg_lambda': 1.0030116045148771}
'''

# %%
finalmodel_gimms_ac1_trait_ld2 = model_fit(sets_gimms_ac1_trait_ld2, best_params_gimms_ac1_trait_ld2)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld2)
shap_explainer_gimms_ac1_trait_ld2 = shap_explainer(gimms_dic_ld2['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld2,max_display=20)

# %%
gimms_dic_ld2['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld2['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld2.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld2.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld2, file)

# %% [markdown]
# #### 4.4.2  gimms var trait

# %%
sets_gimms_var_trait_ld2 = prepare_data_split({'X': gimms_dic_ld2['x_var_trait'], 'y': gimms_dic_ld2['y_var']})
sets_gimms_var_trait_ld2

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld2['X_train'], sets_gimms_var_trait_ld2['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld2 = study.best_params

# %%
best_params_gimms_var_trait_ld2

# %%
'''
         
{'n_estimators': 500,
 'learning_rate': 0.14261575416643116,
 'max_depth': 8,
 'subsample': 0.6569086738300851,
 'colsample_bytree': 0.9419015597056035,
 'min_child_weight': 4.212407938886788,
 'gamma': 1.0275246483560898,
 'reg_alpha': 3.062859057476099,
 'reg_lambda': 2.582704255974212}
'''

# %%
finalmodel_gimms_var_trait_ld2 = model_fit(sets_gimms_var_trait_ld2, best_params_gimms_var_trait_ld2)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld2)
shap_explainer_gimms_var_trait_ld2 = shap_explainer(gimms_dic_ld2['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld2,max_display=20)

# %%
gimms_dic_ld2['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld2['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld2.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld2.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld2, file)

# %% [markdown]
# #### 4.4.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld2 = prepare_data_split({'X': vod_dic_ld2['x_ac1_trait'], 'y': vod_dic_ld2['y_ac1']})
sets_vod_ac1_trait_ld2

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld2['X_train'], sets_vod_ac1_trait_ld2['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld2 = study.best_params

# %%
best_params_vod_ac1_trait_ld2

# %%
'''
         
{'n_estimators': 900,
 'learning_rate': 0.06654092445728665,
 'max_depth': 7,
 'subsample': 0.7998843993131054,
 'colsample_bytree': 0.7218083589893313,
 'min_child_weight': 1.0156095817050608,
 'gamma': 1.0681269894376573,
 'reg_alpha': 2.757629053115652,
 'reg_lambda': 4.979569265815545}
'''

# %%
finalmodel_vod_ac1_trait_ld2 = model_fit(sets_vod_ac1_trait_ld2, best_params_vod_ac1_trait_ld2)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld2)
shap_explainer_vod_ac1_trait_ld2 = shap_explainer(vod_dic_ld2['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld2,max_display=20)

# %%
vod_dic_ld2['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld2['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld2.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld2.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld2, file)

# %% [markdown]
# #### 4.4.4 vod var trait

# %%
sets_vod_var_trait_ld2 = prepare_data_split({'X': vod_dic_ld2['x_var_trait'], 'y': vod_dic_ld2['y_var']})
sets_vod_var_trait_ld2

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld2['X_train'], sets_vod_var_trait_ld2['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld2 = study.best_params

# %%
best_params_vod_var_trait_ld2

# %%
'''
         
{'n_estimators': 550,
 'learning_rate': 0.024189640738500093,
 'max_depth': 8,
 'subsample': 0.6918915524902104,
 'colsample_bytree': 0.9755723449686666,
 'min_child_weight': 1.5526661881980757,
 'gamma': 1.0043950600897817,
 'reg_alpha': 4.07676318159378,
 'reg_lambda': 3.843467660569099}
'''

# %%
finalmodel_vod_var_trait_ld2 = model_fit(sets_vod_var_trait_ld2, best_params_vod_var_trait_ld2)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld2)
shap_explainer_vod_var_trait_ld2 = shap_explainer(vod_dic_ld2['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld2,max_display=20)

# %%
vod_dic_ld2['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld2['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld2.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld2.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld2, file)

# %% [markdown]
# ### 4.5 ld_8

# %%
ld_use

# %%
vod_dic_ld8, gimms_dic_ld8 = get_ld_data(8)

# %% [markdown]
# #### 4.5.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld8 = prepare_data_split({'X': gimms_dic_ld8['x_ac1_trait'], 'y': gimms_dic_ld8['y_ac1']})
sets_gimms_ac1_trait_ld8

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld8['X_train'], sets_gimms_ac1_trait_ld8['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_ac1_trait_ld8 = study.best_params

# %%
best_params_gimms_ac1_trait_ld8

# %%
'''
         
{'n_estimators': 850,
 'learning_rate': 0.03741876530778624,
 'max_depth': 8,
 'subsample': 0.6311092975966383,
 'colsample_bytree': 0.953129619246927,
 'min_child_weight': 4.533629979875666,
 'gamma': 1.0037188920984546,
 'reg_alpha': 4.713878751958739,
 'reg_lambda': 4.524720593983151}
'''

# %%
finalmodel_gimms_ac1_trait_ld8 = model_fit(sets_gimms_ac1_trait_ld8, best_params_gimms_ac1_trait_ld8)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld8)
shap_explainer_gimms_ac1_trait_ld8 = shap_explainer(gimms_dic_ld8['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld8,max_display=20)

# %%
gimms_dic_ld8['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld8['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld8.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld8.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld8, file)

# %% [markdown]
# #### 4.5.2  gimms var trait

# %%
sets_gimms_var_trait_ld8 = prepare_data_split({'X': gimms_dic_ld8['x_var_trait'], 'y': gimms_dic_ld8['y_var']})
sets_gimms_var_trait_ld8

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld8['X_train'], sets_gimms_var_trait_ld8['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld8 = study.best_params

# %%
best_params_gimms_var_trait_ld8

# %%
'''
         
{'n_estimators': 650,
 'learning_rate': 0.12980020399978345,
 'max_depth': 8,
 'subsample': 0.6792104766077738,
 'colsample_bytree': 0.9303860540390669,
 'min_child_weight': 4.241280497108415,
 'gamma': 1.0013511140044307,
 'reg_alpha': 4.52129335200169,
 'reg_lambda': 4.023912710045696}
'''

# %%
finalmodel_gimms_var_trait_ld8 = model_fit(sets_gimms_var_trait_ld8, best_params_gimms_var_trait_ld8)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld8)
shap_explainer_gimms_var_trait_ld8 = shap_explainer(gimms_dic_ld8['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld8,max_display=20)

# %%
gimms_dic_ld8['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld8['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld8.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld8.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld8, file)

# %% [markdown]
# #### 4.5.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld8 = prepare_data_split({'X': vod_dic_ld8['x_ac1_trait'], 'y': vod_dic_ld8['y_ac1']})
sets_vod_ac1_trait_ld8

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld8['X_train'], sets_vod_ac1_trait_ld8['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld8 = study.best_params

# %%
best_params_vod_ac1_trait_ld8

# %%
'''
         
{'n_estimators': 600,
 'learning_rate': 0.15872110788183239,
 'max_depth': 8,
 'subsample': 0.7569906206552636,
 'colsample_bytree': 0.9190506746548526,
 'min_child_weight': 3.5228737085863067,
 'gamma': 1.0357228502150548,
 'reg_alpha': 1.2271371693886972,
 'reg_lambda': 2.1830954851313686}
'''

# %%
finalmodel_vod_ac1_trait_ld8 = model_fit(sets_vod_ac1_trait_ld8, best_params_vod_ac1_trait_ld8)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld8)
shap_explainer_vod_ac1_trait_ld8 = shap_explainer(vod_dic_ld8['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld8,max_display=20)

# %%
vod_dic_ld8['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld8['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld8.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld8.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld8, file)

# %% [markdown]
# #### 4.5.4 vod var trait

# %%
sets_vod_var_trait_ld8 = prepare_data_split({'X': vod_dic_ld8['x_var_trait'], 'y': vod_dic_ld8['y_var']})
sets_vod_var_trait_ld8

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld8['X_train'], sets_vod_var_trait_ld8['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld8 = study.best_params

# %%
best_params_vod_var_trait_ld8

# %%
'''
         
{'n_estimators': 700,
 'learning_rate': 0.01359587388518824,
 'max_depth': 8,
 'subsample': 0.6968446363134974,
 'colsample_bytree': 0.9314624508466132,
 'min_child_weight': 4.491240588495853,
 'gamma': 1.1908148891206265,
 'reg_alpha': 3.52870696192549,
 'reg_lambda': 4.878771764841784}
'''

# %%
finalmodel_vod_var_trait_ld8 = model_fit(sets_vod_var_trait_ld8, best_params_vod_var_trait_ld8)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld8)
shap_explainer_vod_var_trait_ld8 = shap_explainer(vod_dic_ld8['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld8,max_display=20)

# %%
vod_dic_ld8['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld8['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld8.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld8.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld8, file)

# %% [markdown]
# ### 4.6 ld_5

# %%
ld_use

# %%
vod_dic_ld5, gimms_dic_ld5 = get_ld_data(5)

# %% [markdown]
# #### 4.6.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld5 = prepare_data_split({'X': gimms_dic_ld5['x_ac1_trait'], 'y': gimms_dic_ld5['y_ac1']})
sets_gimms_ac1_trait_ld5

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld5['X_train'], sets_gimms_ac1_trait_ld5['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_ac1_trait_ld5 = study.best_params

# %%
best_params_gimms_ac1_trait_ld5

# %%
'''
         
{'n_estimators': 550,
 'learning_rate': 0.022074797162573448,
 'max_depth': 8,
 'subsample': 0.7701179696361712,
 'colsample_bytree': 0.9116399593271739,
 'min_child_weight': 3.319559484912862,
 'gamma': 1.014570839232323,
 'reg_alpha': 1.122572585482402,
 'reg_lambda': 1.520248787210269}
'''

# %%
finalmodel_gimms_ac1_trait_ld5 = model_fit(sets_gimms_ac1_trait_ld5, best_params_gimms_ac1_trait_ld5)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld5)
shap_explainer_gimms_ac1_trait_ld5 = shap_explainer(gimms_dic_ld5['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld5,max_display=20)

# %%
gimms_dic_ld5['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld5['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld5.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld5.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld5, file)

# %% [markdown]
# #### 4.6.2  gimms var trait

# %%
sets_gimms_var_trait_ld5 = prepare_data_split({'X': gimms_dic_ld5['x_var_trait'], 'y': gimms_dic_ld5['y_var']})
sets_gimms_var_trait_ld5

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld5['X_train'], sets_gimms_var_trait_ld5['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld5 = study.best_params

# %%
best_params_gimms_var_trait_ld5

# %%
'''
         
{'n_estimators': 550,
 'learning_rate': 0.012012642833349046,
 'max_depth': 8,
 'subsample': 0.6657832277550096,
 'colsample_bytree': 0.9114513906087189,
 'min_child_weight': 1.7637849386140632,
 'gamma': 1.206705230668737,
 'reg_alpha': 1.2638489187257187,
 'reg_lambda': 3.0563373969020824}
'''

# %%
finalmodel_gimms_var_trait_ld5 = model_fit(sets_gimms_var_trait_ld5, best_params_gimms_var_trait_ld5)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld5)
shap_explainer_gimms_var_trait_ld5 = shap_explainer(gimms_dic_ld5['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld5,max_display=20)

# %%
gimms_dic_ld5['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld5['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld5.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld5.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld5, file)

# %% [markdown]
# #### 4.6.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld5 = prepare_data_split({'X': vod_dic_ld5['x_ac1_trait'], 'y': vod_dic_ld5['y_ac1']})
sets_vod_ac1_trait_ld5

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld5['X_train'], sets_vod_ac1_trait_ld5['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld5 = study.best_params

# %%
best_params_vod_ac1_trait_ld5

# %%
'''
         
{'n_estimators': 850,
 'learning_rate': 0.02069656460516067,
 'max_depth': 8,
 'subsample': 0.5729293531051338,
 'colsample_bytree': 0.9577869612576199,
 'min_child_weight': 2.308964389567073,
 'gamma': 1.0162175580146582,
 'reg_alpha': 4.492559681472831,
 'reg_lambda': 1.3865404626116093}
'''

# %%
finalmodel_vod_ac1_trait_ld5 = model_fit(sets_vod_ac1_trait_ld5, best_params_vod_ac1_trait_ld5)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld5)
shap_explainer_vod_ac1_trait_ld5 = shap_explainer(vod_dic_ld5['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld5,max_display=20)

# %%
vod_dic_ld5['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld5['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld5.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld5.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld5, file)

# %% [markdown]
# #### 4.6.4 vod var trait

# %%
sets_vod_var_trait_ld5 = prepare_data_split({'X': vod_dic_ld5['x_var_trait'], 'y': vod_dic_ld5['y_var']})
sets_vod_var_trait_ld5

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld5['X_train'], sets_vod_var_trait_ld5['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld5 = study.best_params

# %%
best_params_vod_var_trait_ld5

# %%
'''
         
{'n_estimators': 300,
 'learning_rate': 0.044377239656075484,
 'max_depth': 8,
 'subsample': 0.705417537434144,
 'colsample_bytree': 0.8523359575187959,
 'min_child_weight': 4.9412241528638,
 'gamma': 1.2125667369184112,
 'reg_alpha': 1.881764718092179,
 'reg_lambda': 2.8112417272683325}
'''

# %%
finalmodel_vod_var_trait_ld5 = model_fit(sets_vod_var_trait_ld5, best_params_vod_var_trait_ld5)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld5)
shap_explainer_vod_var_trait_ld5 = shap_explainer(vod_dic_ld5['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld5,max_display=20)

# %%
vod_dic_ld5['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld5['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld5.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld5.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld5, file)

# %% [markdown]
# ### 4.7 ld_1

# %%
ld_use

# %%
vod_dic_ld1, gimms_dic_ld1 = get_ld_data(1)

# %% [markdown]
# #### 4.7.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld1 = prepare_data_split({'X': gimms_dic_ld1['x_ac1_trait'], 'y': gimms_dic_ld1['y_ac1']})
sets_gimms_ac1_trait_ld1

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld1['X_train'], sets_gimms_ac1_trait_ld1['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_ac1_trait_ld1 = study.best_params

# %%
best_params_gimms_ac1_trait_ld1

# %%
'''
         
{'n_estimators': 800,
 'learning_rate': 0.01979154329716012,
 'max_depth': 8,
 'subsample': 0.7168208229334767,
 'colsample_bytree': 0.8146474620168496,
 'min_child_weight': 1.6419350668804524,
 'gamma': 1.1199111460374398,
 'reg_alpha': 1.0105738782422784,
 'reg_lambda': 2.0739193997486827}
'''

# %%
finalmodel_gimms_ac1_trait_ld1 = model_fit(sets_gimms_ac1_trait_ld1, best_params_gimms_ac1_trait_ld1)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld1)
shap_explainer_gimms_ac1_trait_ld1 = shap_explainer(gimms_dic_ld1['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld1,max_display=20)

# %%
gimms_dic_ld1['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld1['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld1.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld1.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld1, file)

# %% [markdown]
# #### 4.7.2  gimms var trait

# %%
sets_gimms_var_trait_ld1 = prepare_data_split({'X': gimms_dic_ld1['x_var_trait'], 'y': gimms_dic_ld1['y_var']})
sets_gimms_var_trait_ld1

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld1['X_train'], sets_gimms_var_trait_ld1['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld1 = study.best_params

# %%
best_params_gimms_var_trait_ld1

# %%
'''
         
{'n_estimators': 950,
 'learning_rate': 0.013242999756557032,
 'max_depth': 8,
 'subsample': 0.6046124303649943,
 'colsample_bytree': 0.9978569592284405,
 'min_child_weight': 2.039703757248994,
 'gamma': 1.2318507723402965,
 'reg_alpha': 1.2027856941142856,
 'reg_lambda': 1.598612676606543}
'''

# %%
finalmodel_gimms_var_trait_ld1 = model_fit(sets_gimms_var_trait_ld1, best_params_gimms_var_trait_ld1)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld1)
shap_explainer_gimms_var_trait_ld1 = shap_explainer(gimms_dic_ld1['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld1,max_display=20)

# %%
gimms_dic_ld1['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld1['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld1.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld1.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld1, file)

# %% [markdown]
# #### 4.7.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld1 = prepare_data_split({'X': vod_dic_ld1['x_ac1_trait'], 'y': vod_dic_ld1['y_ac1']})
sets_vod_ac1_trait_ld1

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld1['X_train'], sets_vod_ac1_trait_ld1['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld1 = study.best_params

# %%
best_params_vod_ac1_trait_ld1

# %%
'''
         
{'n_estimators': 350,
 'learning_rate': 0.020617467273432766,
 'max_depth': 8,
 'subsample': 0.5867849860859955,
 'colsample_bytree': 0.8664947000778976,
 'min_child_weight': 1.9892193965867944,
 'gamma': 1.0059895288624914,
 'reg_alpha': 2.004032845230798,
 'reg_lambda': 4.632279499625723}
'''

# %%
finalmodel_vod_ac1_trait_ld1 = model_fit(sets_vod_ac1_trait_ld1, best_params_vod_ac1_trait_ld1)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld1)
shap_explainer_vod_ac1_trait_ld1 = shap_explainer(vod_dic_ld1['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld1,max_display=20)

# %%
vod_dic_ld1['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld1['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld1.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld1.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld1, file)

# %% [markdown]
# #### 4.7.4 vod var trait

# %%
sets_vod_var_trait_ld1 = prepare_data_split({'X': vod_dic_ld1['x_var_trait'], 'y': vod_dic_ld1['y_var']})
sets_vod_var_trait_ld1

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld1['X_train'], sets_vod_var_trait_ld1['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld1 = study.best_params

# %%
best_params_vod_var_trait_ld1

# %%
'''
         
{'n_estimators': 250,
 'learning_rate': 0.16584545471385356,
 'max_depth': 8,
 'subsample': 0.7993422623450802,
 'colsample_bytree': 0.8612012805563082,
 'min_child_weight': 3.399665039720519,
 'gamma': 1.0032204771565092,
 'reg_alpha': 2.0203877777603223,
 'reg_lambda': 3.8661162315336637}
'''

# %%
finalmodel_vod_var_trait_ld1 = model_fit(sets_vod_var_trait_ld1, best_params_vod_var_trait_ld1)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld1)
shap_explainer_vod_var_trait_ld1 = shap_explainer(vod_dic_ld1['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld1,max_display=20)

# %%
vod_dic_ld1['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld1['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld1.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld1.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld1, file)

# %% [markdown]
# ### 4.8  ld_4

# %%
ld_use

# %%
vod_dic_ld4, gimms_dic_ld4 = get_ld_data(4)

# %% [markdown]
# #### 4.8.1  gimms ac1 trait

# %%
sets_gimms_ac1_trait_ld4 = prepare_data_split({'X': gimms_dic_ld4['x_ac1_trait'], 'y': gimms_dic_ld4['y_ac1']})
sets_gimms_ac1_trait_ld4

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_ac1_trait_ld4['X_train'], sets_gimms_ac1_trait_ld4['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_ac1_trait_ld4 = study.best_params

# %%
best_params_gimms_ac1_trait_ld4

# %%
'''
         
{'n_estimators': 850,
 'learning_rate': 0.08913519508091684,
 'max_depth': 8,
 'subsample': 0.6297090634244358,
 'colsample_bytree': 0.7292200936023359,
 'min_child_weight': 4.1199497345688245,
 'gamma': 1.0080767928459908,
 'reg_alpha': 1.3942935539134071,
 'reg_lambda': 2.8577896269749616}
'''

# %%
finalmodel_gimms_ac1_trait_ld4 = model_fit(sets_gimms_ac1_trait_ld4, best_params_gimms_ac1_trait_ld4)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_ac1_trait_ld4)
shap_explainer_gimms_ac1_trait_ld4 = shap_explainer(gimms_dic_ld4['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_ac1_trait_ld4,max_display=20)

# %%
gimms_dic_ld4['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld4['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_ac1_trait_ld4.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_ac1_trait_ld4.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_ac1_trait_ld4, file)

# %% [markdown]
# #### 4.8.2  gimms var trait

# %%
sets_gimms_var_trait_ld4 = prepare_data_split({'X': gimms_dic_ld4['x_var_trait'], 'y': gimms_dic_ld4['y_var']})
sets_gimms_var_trait_ld4

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_gimms_var_trait_ld4['X_train'], sets_gimms_var_trait_ld4['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_gimms_var_trait_ld4 = study.best_params

# %%
best_params_gimms_var_trait_ld4

# %%
'''
         
{'n_estimators': 950,
 'learning_rate': 0.0925770544940363,
 'max_depth': 8,
 'subsample': 0.6037155676689855,
 'colsample_bytree': 0.8392883756195529,
 'min_child_weight': 4.25594688468228,
 'gamma': 1.029595445602148,
 'reg_alpha': 1.3590317120380146,
 'reg_lambda': 2.2651610440905623}
'''

# %%
finalmodel_gimms_var_trait_ld4 = model_fit(sets_gimms_var_trait_ld4, best_params_gimms_var_trait_ld4)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_gimms_var_trait_ld4)
shap_explainer_gimms_var_trait_ld4 = shap_explainer(gimms_dic_ld4['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_gimms_var_trait_ld4,max_display=20)

# %%
gimms_dic_ld4['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : gimms_dic_ld4['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_gimms_var_trait_ld4.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_gimms_var_trait_ld4.pickle', 'wb') as file:
    pickle.dump(shap_explainer_gimms_var_trait_ld4, file)

# %% [markdown]
# #### 4.8.3 vod ac1 trait

# %%
sets_vod_ac1_trait_ld4 = prepare_data_split({'X': vod_dic_ld4['x_ac1_trait'], 'y': vod_dic_ld4['y_ac1']})
sets_vod_ac1_trait_ld4

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_ac1_trait_ld4['X_train'], sets_vod_ac1_trait_ld4['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_ac1_trait_ld4 = study.best_params

# %%
best_params_vod_ac1_trait_ld4

# %%
'''
     20260408
{'n_estimators': 850,
 'learning_rate': 0.050835888682020894,
 'max_depth': 8,
 'subsample': 0.726790551262595,
 'colsample_bytree': 0.7352976893923898,
 'min_child_weight': 3.3300232537514223,
 'gamma': 1.2251340848064032,
 'reg_alpha': 2.7629866803080656,
 'reg_lambda': 3.1935223020754884}
'''

# %%
finalmodel_vod_ac1_trait_ld4 = model_fit(sets_vod_ac1_trait_ld4, best_params_vod_ac1_trait_ld4)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_ac1_trait_ld4)
shap_explainer_vod_ac1_trait_ld4 = shap_explainer(vod_dic_ld4['x_ac1_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_ac1_trait_ld4,max_display=20)

# %%
vod_dic_ld4['x_ac1_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld4['x_ac1_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_ac1_trait_ld4.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_ac1_trait_ld4.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_ac1_trait_ld4, file)

# %% [markdown]
# #### 4.8.4 vod var trait

# %%
sets_vod_var_trait_ld4 = prepare_data_split({'X': vod_dic_ld4['x_var_trait'], 'y': vod_dic_ld4['y_var']})
sets_vod_var_trait_ld4

# %%
##  
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,5),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    #   
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, sets_vod_var_trait_ld4['X_train'], sets_vod_var_trait_ld4['y_train'], scoring='r2', cv=cv).min()

    return score  #    

#     
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  #     

#     
print("Best trial:")
print(study.best_trial)

#     
best_params_vod_var_trait_ld4 = study.best_params

# %%
best_params_vod_var_trait_ld4

# %%
'''
         
{'n_estimators': 750,
 'learning_rate': 0.02584496069107582,
 'max_depth': 8,
 'subsample': 0.714002317506031,
 'colsample_bytree': 0.6257582795738764,
 'min_child_weight': 3.780702110533783,
 'gamma': 1.0061902251062116,
 'reg_alpha': 2.5116051513134687,
 'reg_lambda': 1.013050979986293}
'''

# %%
finalmodel_vod_var_trait_ld4 = model_fit(sets_vod_var_trait_ld4, best_params_vod_var_trait_ld4)

# %%
shap_explainer = shap.TreeExplainer(finalmodel_vod_var_trait_ld4)
shap_explainer_vod_var_trait_ld4 = shap_explainer(vod_dic_ld4['x_var_trait'])  

# %%
shap.plots.beeswarm(shap_explainer_vod_var_trait_ld4,max_display=20)

# %%
vod_dic_ld4['x_var_trait'].columns.values

# %%
var_names_sorted =pd.DataFrame({'vars' : vod_dic_ld4['x_var_trait'].columns.values, 
              'shap_val' : np.abs(shap_explainer_vod_var_trait_ld4.values).mean(axis=0),
              'var_name': ['LDMC','LNC','SLA','LNPR','RD','HC','HEI']}).sort_values(by = 'shap_val', ascending = False)
var_names_sorted

# %%
fig, ax  = plt.subplots(figsize = [8,5])
var_names_sorted = var_names_sorted.sort_values(by = 'shap_val', ascending = True)
ax.barh(var_names_sorted['vars'], var_names_sorted['shap_val'], color = 'skyblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('')
ax.set_yticklabels(var_names_sorted['var_name'])
plt.tight_layout()

# %%
with open(r'result_data/shap_explainer_vod_var_trait_ld4.pickle', 'wb') as file:
    pickle.dump(shap_explainer_vod_var_trait_ld4, file)

# %% [markdown]
# ## change log
# 1. 2026.04.09 用所有的 性状 包括 LNC  LNPR跑了一遍

# %%



