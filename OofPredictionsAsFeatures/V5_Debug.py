import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
import os
import joblib
import optuna
import json
import glob
from catboost import CatBoostRegressor, Pool

import sys
sys.path.append('./')
from GroupKFoldShuffle import GroupKFoldShuffle
from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(
        games_csv_path, 
        starting_evals_json_paths):
    df = pl.read_csv(games_csv_path)

    ruleset_names = df['GameRulesetName'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    df = df.to_pandas()
    df['agent1'] = df['agent1'].str.replace('-random-', '-Random200-')
    df['agent2'] = df['agent2'].str.replace('-random-', '-Random200-')
    df = pl.DataFrame(df)

    for col in AGENT_COLS:
        df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")

    df = df.with_columns([
        pl.col(col).cast(pl.Categorical) 
        for col in df.columns 
        if (col[:6] in AGENT_COLS) and ('elo' not in col)
    ])
    df = df.with_columns([
        pl.col(col).cast(pl.Float32) 
        for col in df.columns 
        if not ((col[:6] in AGENT_COLS) and ('elo' not in col))
    ])
    
    df = df.to_pandas()

    # ADD MCTS EVALUATION FEATURES.
    luds_to_all_mcts_evals = {}
    for starting_evals_json_path in starting_evals_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)

        for lud, eval in luds_to_mcts_evals.items():
            if lud not in luds_to_all_mcts_evals:
                luds_to_all_mcts_evals[lud] = []

            if eval is None:
                eval = 0

            luds_to_all_mcts_evals[lud].append(eval)

    luds_to_mcts_evals = {
        lud: np.mean(evals)
        for lud, evals in luds_to_all_mcts_evals.items()
    }

    df.insert(
        df.shape[1], 
        'mcts_eval',
        [luds_to_mcts_evals[lud] for lud in lud_rules]
    )

    print(f'Data shape: {df.shape}')
    return ruleset_names, df

def TrainModels(
        ruleset_names, 
        train_test_df, 
        extra_train_df,
        cat_params, 
        early_stopping_round_count, 
        fold_count):
    models = []
    rmse_scores = []

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    if extra_train_df is not None:
        extra_train_X = extra_train_df.drop(['utility_agent1'], axis=1)
        extra_train_y = extra_train_df['utility_agent1']

    categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']


    group_kfold_shuffle = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=RANDOM_SEED)
    folds = list(group_kfold_shuffle.split(X, y, groups=ruleset_names))
    
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        # SPLIT INTO TRAIN AND TEST.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        if extra_train_df is not None:
            if extra_train_X.columns.tolist() != train_x.columns.tolist():
                extraneous_columns = extra_train_X.columns.difference(train_x.columns)
                missing_columns = train_x.columns.difference(extra_train_X.columns)
                # print(f'WARNING! Extra train has different columns than train!')
                # print(f'Extraneous columns: {extraneous_columns}')
                # print(f'Missing columns: {missing_columns}')

                extra_train_X = extra_train_X[train_x.columns]

            train_x = pd.concat([train_x, extra_train_X], ignore_index=True)
            train_y = pd.concat([train_y, extra_train_y], ignore_index=True)

        # TRAIN MODEL.
        train_pool = Pool(
            train_x, 
            train_y,
            cat_features=categorical_features)
        test_pool = Pool(
            test_x, 
            test_y, 
            cat_features=categorical_features)

        model = CatBoostRegressor(**cat_params, random_seed=RANDOM_SEED)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=early_stopping_round_count,
            verbose=False
        )

        # RECORD RESULT.
        models.append(model) 
        rmse_scores.append(model.best_score_['validation']['RMSE'])

    print('Fold RMSEs:', rmse_scores)
    
    mean_score = np.mean(rmse_scores)
    print(f'Average RMSE: {mean_score:.4f}')

    return models, mean_score

def SaveModels(trained_models, test_rmse, output_directory_suffix = ''):
    output_directory_path = f'models/catboost_{int(test_rmse*100000)}_{len(trained_models)}_{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, model in enumerate(trained_models):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump(model, output_filepath)

def CreateEnsemble(
        cat_params, 
        early_stopping_round_count, 
        fold_count, 
        starting_evals_json_paths,
        extra_train_paths = None,
        output_directory_suffix = ''):
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
        starting_evals_json_paths = starting_evals_json_paths,
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
            starting_evals_json_paths = extra_train_paths['starting_position_evals_json_paths'],
        )

    trained_models, test_rmse = TrainModels(
        ruleset_names,
        train_test_df, 
        extra_train_df,
        cat_params, 
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
    )

    if output_directory_suffix is not None:
        SaveModels(trained_models, test_rmse, output_directory_suffix)

    return test_rmse

if __name__ == '__main__':
    # extra_train_paths = {
    #     'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
    #     'starting_position_evals_json_paths': [
    #         'StartingPositionEvaluation/Evaluations/FromKaggle/extra_UCB1Tuned-1.41421356237-random-false_60s.json',
    #     ]
    # }
    extra_train_paths = None

    scores = []
    for seed in [3333]:
        RANDOM_SEED = seed
        print(f'Seed: {RANDOM_SEED}')

        rmse = CreateEnsemble(
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
            starting_evals_json_paths = [
                'StartingPositionEvaluation/Evaluations/FromKaggle/organizer_UCB1Tuned-1.41421356237-random-false_60s.json',
                # 'StartingPositionEvaluation/Evaluations/FromKaggle/organizer_UCB1Tuned-1.41421356237-random-false_30s.json',
                # 'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json',
                # 'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-0.6-NST-true-16s.json',
                # 'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-750ms.json',
            ],
            extra_train_paths = extra_train_paths,
            output_directory_suffix = None
        )

        scores.append(rmse)

    print('Scores:', scores)
    print('Mean:', np.mean(scores))