import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
import os
import joblib
import optuna
import json
import glob

from catboost import CatBoostRegressor, Pool

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(include_lud_stats):
    df = pl.read_csv('data/games_to_utility_stats.csv')

    ruleset_names = df['GameRulesetName'].to_pandas()
    english_rules = df['EnglishRules'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
        
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    df = df.to_pandas()

    # ADD TEXT FEATURES.
    if include_lud_stats:
        preexisting_column_count = df.shape[1]
        df.insert(preexisting_column_count, "EnglishRulesLength", [len(rule) for rule in english_rules])
        df.insert(preexisting_column_count + 1, "LudRulesLength", [len(rule) for rule in lud_rules])
        df.insert(preexisting_column_count + 2, "LudIfStatementCount", [rule.count('if') for rule in english_rules])
        df.insert(preexisting_column_count + 3, "LudArrayCount", [rule.count('array') for rule in english_rules])
        df.insert(preexisting_column_count + 4, "LudVariableCount", [rule.count('var') for rule in english_rules])
        df.insert(preexisting_column_count + 5, "LudToCount", [rule.count('to') for rule in english_rules])


    print(f'Data shape: {df.shape}')
    
    return ruleset_names, df

def TrainModels(ruleset_names, train_test_df, cat_params, early_stopping_round_count, fold_count, target_column):
    models = []
    rmse_scores = []

    X = train_test_df.drop([target_column], axis=1)
    y = train_test_df[target_column]

    categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']

    group_kfold = GroupKFold(n_splits=fold_count)
    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        train_pool = Pool(
            train_x, 
            train_y, 
            cat_features=categorical_features)
        test_pool = Pool(
            test_x, 
            test_y, 
            cat_features=categorical_features)

        model = CatBoostRegressor(**cat_params)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=early_stopping_round_count,
            verbose=False
        )

        models.append(model) 
        rmse_scores.append(model.best_score_['validation']['RMSE'])

    print('Fold RMSEs:', rmse_scores)
    
    mean_score = np.mean(rmse_scores)
    print(f'Average RMSE: {mean_score:.4f}')

    return models, mean_score

def Objective(trial, ruleset_names, train_test_df, fold_count, target_column):
    cat_params = {
        'iterations': trial.suggest_int('iterations', 1000, 12000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 2e-1, log=True),
        'depth': trial.suggest_int('depth', 5, 15),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-7, 10, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 8),
        "task_type": "GPU"
    }
    
    early_stopping_round_count = 50
    
    _, mean_score = TrainModels(
        ruleset_names,
        train_test_df,
        cat_params,
        early_stopping_round_count,
        fold_count,
        target_column
    )
    
    return mean_score

def GetOptimalConfig(trial_count, target_column):
    ruleset_names, train_test_df = GetPreprocessedData(include_lud_stats = True)

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: Objective(trial, ruleset_names, train_test_df, target_column, fold_count=5), 
        n_trials=trial_count
    )

    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=2))

    best_score = study.best_trial.value
    output_filepath = f'configs/catboost_{int(best_score * 100000)}.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

def SaveModels(trained_models, test_rmse, output_directory_suffix = ''):
    output_directory_path = f'models/utilitystats_catboost_{int(test_rmse*100000)}_{len(trained_models)}{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, model in enumerate(trained_models):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump(model, output_filepath)

def CreateEnsemble(cat_params, early_stopping_round_count, fold_count, target_column, output_directory_suffix):
    ruleset_names, train_test_df = GetPreprocessedData(include_lud_stats = True)
    trained_models, test_rmse = TrainModels(
        ruleset_names,
        train_test_df, 
        cat_params, 
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
        target_column = target_column
    )

    if output_directory_suffix is not None:
        SaveModels(trained_models, test_rmse, output_directory_suffix)

if __name__ == '__main__':
    DROPPED_COLUMNS += ['NumTopSites']
    # GetOptimalConfig(trial_count = 100)

    CreateEnsemble(
        cat_params = {
            "iterations": 10219, 
            "learning_rate": 0.010964241393786744, 
            "depth": 10, 
            "l2_leaf_reg": 0.0012480029901784353, 
            "grow_policy": "SymmetricTree", 
            "max_ctr_complexity": 6,
            "task_type": "GPU"
        },
        early_stopping_round_count = 100,
        fold_count = 10,
        # output_directory_suffix = None
        target_column = 'mean_agent1_utilities',
        output_directory_suffix = 'raw_test',
        # target_column = 'mean_absolute_agent1_utilities',
        # output_directory_suffix = 'absolute_test',
    )