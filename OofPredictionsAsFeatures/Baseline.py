import pandas as pd
import polars as pl
import numpy as np
from sklearn.utils import shuffle
import os
import joblib
import optuna
import json
import glob
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

import sys
sys.path.append('./')
from GroupKFoldShuffle import GroupKFoldShuffle
from ColumnNames import IRRELEVANT_COLS, OUTCOME_COUNT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(
        games_csv_path, 
        starting_evals_json_paths,
        dropped_features_path):
    df = pl.read_csv(games_csv_path)

    ruleset_names = df['GameRulesetName'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    
    RULE_TEXT_COLS = ['EnglishRules', 'LudRules']
    df = df.drop(filter(lambda x: x in df.columns, RULE_TEXT_COLS))
    df = df.drop(filter(lambda x: x in df.columns, IRRELEVANT_COLS))
    df = df.drop(filter(lambda x: x in df.columns, OUTCOME_COUNT_COLS))

    df = df.to_pandas()
    df['agent1'] = df['agent1'].str.replace('-random-', '-Random200-')
    df['agent2'] = df['agent2'].str.replace('-random-', '-Random200-')
    df = pl.DataFrame(df)

    CATEGORICAL_COLS = ['GameRulesetName', 'agent1', 'agent2']
    df = df.with_columns([
        pl.col(col).cast(pl.Categorical) 
        for col in df.columns 
        if (col in CATEGORICAL_COLS)
    ])
    df = df.with_columns([
        pl.col(col).cast(pl.Float32) 
        for col in df.columns 
        if not (col in CATEGORICAL_COLS)
    ])
    
    df = df.to_pandas()

    # ADD MCTS EVALUATION FEATURES.
    for starting_evals_json_path in starting_evals_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)

        feature_name = starting_evals_json_path.split('/')[-1].replace('.json', '')
        df.insert(
            df.shape[1], 
            feature_name,
            [luds_to_mcts_evals[lud] for lud in lud_rules]
        )

    # DROP FEATURES.
    dropped_features_df = pd.read_csv(dropped_features_path)
    dropped_feature_names = dropped_features_df['drop_features'].tolist()

    df = df.drop(columns=dropped_feature_names)

    print(f'Data shape: {df.shape}')
    return ruleset_names, df

def TrainAndTestModel(
        ruleset_names,
        train_test_df,
        model_params,
        model_type,
        early_stopping_round_count,
        fold_count, 
        oof_predictions_df = None,
        random_seed = None):
    models = []
    oof_predictions = np.empty(train_test_df.shape[0])
    rmse_scores = []

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    if oof_predictions_df is not None:
        X = pd.concat([X, oof_predictions_df], axis=1)

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in tqdm(enumerate(folds), total=fold_count):
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        if model_type == 'CatBoost':
            categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']

            train_pool = Pool(
                data=train_x,
                label=train_y,
                cat_features=categorical_features
            )
            test_pool = Pool(
                data=test_x,
                label=test_y,
                cat_features=categorical_features
            )

            model = CatBoostRegressor(**model_params)
            model.fit(
                train_pool,
                eval_set=(test_pool),
                early_stopping_rounds=early_stopping_round_count,
                verbose=False
            )
        elif model_type == 'LGBM':
            model = lgb.LGBMRegressor(**model_params)
            model.fit(
                train_x, train_y,
                eval_set=[(test_x, test_y)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(early_stopping_round_count)]
            )
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        predictions = model.predict(test_x)
        oof_predictions[test_index] = predictions

        models.append(model)
        rmse_scores.append(root_mean_squared_error(test_y, predictions))

    return models, oof_predictions, np.mean(rmse_scores)

# Without OOF (LGBM, cat): (0.43133153721195694, 0.4262014719101982)
# With OOF (LGBM, cat): (0.3987, 0.3997492665056025)
if __name__ == '__main__':
    ruleset_names, train_test_df = GetPreprocessedData(
        games_csv_path='/mnt/data01/data/TreeSearch/data/from_organizers/train.csv',
        starting_evals_json_paths=glob.glob('StartingPositionEvaluation/Evaluations/FromKaggle/organizer_UCB1Tuned-1.41421356237-random-false_60s.json'),
        dropped_features_path='OofPredictionsAsFeatures/baseline_dropped_features.csv'
    )

    lgbm_params = {
        'objective': 'regression',
        'min_child_samples': 24,
        'n_estimators': 20000,
        'learning_rate': 0.07,
        'extra_trees': True,
        'reg_lambda': 0.8,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'device': 'cpu',
        'max_depth': 24,
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }
    cat_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'num_trees': 20000,
        'random_state': 42,
        'task_type': 'CPU',
        'reg_lambda': 0.8,
        'depth': 8
    }

    # # TRAIN WITHOUT OOF FEATURES.
    lgbm_1_models, lgbm_1_oof_predictions, rmse = TrainAndTestModel(
        ruleset_names,
        train_test_df,
        lgbm_params,
        'LGBM',
        early_stopping_round_count=500,
        fold_count=5,
        random_seed=None
    )
    print('LGBM RMSE:', rmse)

    cat_1_models, cat_1_oof_predictions, rmse = TrainAndTestModel(
        ruleset_names,
        train_test_df,
        cat_params,
        'CatBoost',
        early_stopping_round_count=500,
        fold_count=5,
        random_seed=None
    )
    print('CatBoost RMSE:', rmse)

    # TRAIN WITH OOF FEATURES.
    oof_predictions_df = pd.DataFrame({
        'lgbm_1': lgbm_1_oof_predictions,
        'cat_1': cat_1_oof_predictions
    })

    lgbm_2_models, lgbm_2_oof_predictions, lgbm_rmse = TrainAndTestModel(
        ruleset_names,
        train_test_df,
        lgbm_params,
        'LGBM',
        early_stopping_round_count=500,
        fold_count=5,
        oof_predictions_df=oof_predictions_df,
        random_seed=None
    )
    print('LGBM RMSE with OOF features:', lgbm_rmse)

    cat_2_models, cat_2_oof_predictions, cat_rmse = TrainAndTestModel(
        ruleset_names,
        train_test_df,
        cat_params,
        'CatBoost',
        early_stopping_round_count=500,
        fold_count=5,
        oof_predictions_df=oof_predictions_df,
        random_seed=None
    )
    print('CatBoost RMSE with OOF features:', cat_rmse)

    # SAVE MODELS.
    os.makedirs('OofPredictionsAsFeatures/models_and_predictions', exist_ok=True)
    models_and_predictions = {
        'lgbm_1': lgbm_1_models,
        'cat_1': cat_1_models,
        'lgbm_2': lgbm_2_models,
        'cat_2': cat_2_models,
        'lgbm_1_oof_predictions': lgbm_1_oof_predictions,
        'cat_1_oof_predictions': cat_1_oof_predictions,
        'lgbm_2_oof_predictions': lgbm_2_oof_predictions,
        'cat_2_oof_predictions': cat_2_oof_predictions
    }
    joblib.dump(models_and_predictions, f'OofPredictionsAsFeatures/models_and_predictions/baseline_{int(lgbm_rmse*10000)}_{int(cat_rmse*10000)}.p')