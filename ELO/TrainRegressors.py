import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import joblib
import optuna
import json
import glob
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,QuantileTransformer

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

COMPONENT_NAMES_TO_OPTIONS = {
    'selection': ['UCB1', 'UCB1GRAVE', 'ProgressiveHistory', 'UCB1Tuned'],
    'exploration_const': ['0.1', '0.6', '1.41421356237'],
    'playout': ['Random200', 'MAST', 'NST'],
    'score_bounds': ['true', 'false']
}

def GetPreprocessedData(games_filepath, target_component_name):
    # LOAD DATA.
    df = pd.read_csv(games_filepath)

    # CATEGORIZE LABEL COLUMNS.
    valid_label_cols = []
    invalid_label_cols = []
    for component_name, options in COMPONENT_NAMES_TO_OPTIONS.items():
        for option in options:
            label_name = f'{component_name}_{option}_elo'

            if component_name == target_component_name:
                valid_label_cols.append(label_name)
            else:
                invalid_label_cols.append(label_name)

    # DROP COLUMNS.
    df = df.drop(invalid_label_cols, axis=1)

    # DROP ROWS WITH NULL LABELS.
    df = df.dropna(subset=valid_label_cols)

    # EXTRACT LUD RULES.
    lud_rules = df['LudRules'].values
    df = df.drop('LudRules', axis=1)

    return df, valid_label_cols, lud_rules

def TrainModels(component_name, cat_params, fold_count, run_id):
    rmse_scores = []
    models = []

    # LOAD ORGANIZER DATA.
    organizer_df, valid_label_cols, lud_rules = GetPreprocessedData(
        f'ELO/CSV/labeled_organizer_data_{run_id}.csv',
        component_name
    )

    # SPLIT DATA.
    X = organizer_df.drop(valid_label_cols, axis=1)
    y = organizer_df[valid_label_cols]

    # TRAIN REGRESSORS.
    random_seed = run_id*1000
    kf = KFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # TRAIN CATBOOST.
        model = CatBoostRegressor(
            **cat_params,
            loss_function='MultiRMSE',
            boosting_type='Plain',
            task_type='GPU',
            random_seed=random_seed
        )
        model.fit(
            X_train, 
            y_train, 
            eval_set=(X_test, y_test), 
            verbose=False,
            early_stopping_rounds=50
        )

        # REFIT WITH ONLY MOST IMPORTANT FEATURES.
        feature_importances = model.get_feature_importance()
        important_feature_indices = np.argsort(feature_importances)[::-1][:400]
        important_feature_names = X_train.columns[important_feature_indices]

        model = CatBoostRegressor(
            **cat_params,
            loss_function='MultiRMSE',
            boosting_type='Plain',
            task_type='GPU',
            random_seed=random_seed
        )
        model.fit(
            X_train[important_feature_names], 
            y_train, 
            eval_set=(X_test[important_feature_names], y_test), 
            verbose=False,
            early_stopping_rounds=50
        )

        rmse = model.best_score_['validation']['MultiRMSE']
        rmse_scores.append(rmse)

        models.append({
            'model': model,
            'selected_feature_names': important_feature_names,
            'training_luds': lud_rules[train_idx]
        })

    print(f'{component_name} RMSE:', np.mean(rmse_scores))

    return np.mean(rmse_scores), models

if __name__ == '__main__':
    # TRAIN MODELS.
    cat_params = {
        'iterations': 2000,
        'learning_rate': 0.02,
        'depth': 6
    }
    FOLD_COUNT = 5
    all_run_rmses = []
    for run_id in range(3):
        rmse_scores = []
        component_names_to_models = {}
        for component_name in COMPONENT_NAMES_TO_OPTIONS:
            rmse, models = TrainModels(component_name, cat_params, FOLD_COUNT, run_id)
            rmse_scores.append(rmse)
            component_names_to_models[component_name] = models

        mean_rmse = np.mean(rmse_scores)
        all_run_rmses.append(mean_rmse)
        print(f'Run {run_id} RMSE: {mean_rmse}')

        output_filepath = f'ELO/models/run_{run_id}_{int(mean_rmse * 10000)}.pkl'
        joblib.dump(component_names_to_models, output_filepath)

    print('All run RMSEs:', all_run_rmses)
    print('Mean RMSE:', np.mean(all_run_rmses))