import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
import os
import joblib
import optuna
import json

import xgboost as xgb

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(split_agent_features):
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    ruleset_names = df['GameRulesetName'].to_pandas()
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    if split_agent_features:
        for col in AGENT_COLS:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")
        
        df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in AGENT_COLS])            
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    
    print(f'Data shape: {df.shape}')
    
    return ruleset_names, df.to_pandas()

def TrainModels(ruleset_names, train_test_df, xgb_params, early_stopping_round_count, fold_count):
    models = []
    rmse_scores = []

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    group_kfold = GroupKFold(n_splits=fold_count)
    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        model = xgb.XGBRegressor(
            early_stopping_rounds=early_stopping_round_count,
            enable_categorical=True,
            **xgb_params)
        model.fit(
            train_x,
            train_y,
            eval_set=[(test_x, test_y)],
            verbose=False)

        models.append(model) 
        rmse_scores.append(np.mean(model.evals_result()['validation_0']['rmse']))

    print('Fold RMSEs:', rmse_scores)
    
    mean_score = np.mean(rmse_scores)
    print(f'Average RMSE: {mean_score:.4f}')

    return models, mean_score

def Objective(trial, ruleset_names, train_test_df, fold_count):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 8192, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1e-2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 1e-6, log=True),
        "device": "cuda",
    }
    
    early_stopping_round_count = 50
    
    _, mean_score = TrainModels(
        ruleset_names,
        train_test_df,
        xgb_params,
        early_stopping_round_count,
        fold_count
    )
    
    return mean_score

def GetOptimalConfig(trial_count):
    ruleset_names, train_test_df = GetPreprocessedData(split_agent_features = True)

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: Objective(trial, ruleset_names, train_test_df, fold_count=5), 
        n_trials=trial_count
    )

    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=2))

    best_score = study.best_trial.value
    output_filepath = f'configs/xgb_{int(best_score * 100000)}.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

def SaveModels(trained_models, test_rmse):
    output_directory_path = f'models/xgb_{int(test_rmse*100000)}_{len(trained_models)}'
    os.makedirs(output_directory_path)

    for fold_index, model in enumerate(trained_models):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump(model, output_filepath)

def CreateEnsemble(xgb_params, early_stopping_round_count, fold_count):
    ruleset_names, train_test_df = GetPreprocessedData(split_agent_features = True)
    trained_models, test_rmse = TrainModels(
        ruleset_names,
        train_test_df, 
        xgb_params, 
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count
    )

    SaveModels(trained_models, test_rmse)

if __name__ == '__main__':
    # GetOptimalConfig(trial_count = 100)

    CreateEnsemble(
        xgb_params = {
            'n_estimators': 3068, 
            'learning_rate': 0.016808506487839252, 
            'max_depth': 11, 
            'min_child_weight': 1, 
            'subsample': 0.9134366488212371, 
            'colsample_bytree': 0.9798193136743985, 
            'colsample_bynode': 0.6129533063581413, 
            'reg_alpha': 0.0008556951413159803, 
            'reg_lambda': 5.070747853658675e-07,
            "device": "cuda",
        },
        early_stopping_round_count = 100,
        fold_count = 10,
    )