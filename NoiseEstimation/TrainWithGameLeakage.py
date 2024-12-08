import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import shuffle
import os
import joblib
import optuna
import json
import glob
from tqdm import tqdm

from catboost import CatBoostRegressor, Pool

import sys
sys.path.append('./')
from GroupKFoldShuffle import GroupKFoldShuffle
from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

def LoadAllQualityMetrics():
    luds_to_quality_metrics_filepaths = glob.glob('GAVEL/organizer_lud_preprocessing/*_metrics.json')

    quality_metric_values = []
    all_luds_to_quality_metrics = {}
    for filepath in luds_to_quality_metrics_filepaths:
        with open(filepath, 'r') as f:
            luds_to_quality_metrics = json.load(f)

        for lud, quality in luds_to_quality_metrics.items():
            quality_metric_values.append(quality)
            all_luds_to_quality_metrics[lud] = quality

    return all_luds_to_quality_metrics

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(
        games_csv_path, 
        starting_evals_json_paths,
        split_agent_features, 
        include_english_lsa, 
        include_lud_lsa,
        oof_predictions_path = None):
    df = pl.read_csv(games_csv_path)

    ruleset_names = df['GameRulesetName'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    try:
        english_rules = df['EnglishRules'].to_pandas()
    except:
        english_rules = []
    
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    df = df.to_pandas()
    df['agent1'] = df['agent1'].str.replace('-random-', '-Random200-')
    df['agent2'] = df['agent2'].str.replace('-random-', '-Random200-')
    df = pl.DataFrame(df)

    if split_agent_features:
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

    # MAYBE ADD ENGLISH FEATURES.
    if include_english_lsa:
        with open('data/lsa/english_rules_to_selected_features.json') as f:
            english_rules_to_features = json.load(f)

        preexisting_column_count = df.shape[1]
        added_feature_count = len(list(english_rules_to_features.values())[0])
        for i in range(added_feature_count):
            df.insert(preexisting_column_count + i, f"english_lsa_{i}", [english_rules_to_features[rule][i] for rule in english_rules])
    
    # MAYBE ADD LUD FEATURES.
    if include_lud_lsa:
        with open('data/lsa/lud_rules_to_selected_features.json') as f:
            lud_rules_to_features = json.load(f)

        preexisting_column_count = df.shape[1]
        added_feature_count = len(list(lud_rules_to_features.values())[0])
        for i in range(added_feature_count):
            df.insert(preexisting_column_count + i, f"lud_lsa_{i}", [lud_rules_to_features[rule][i] for rule in lud_rules])

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

    # ADD OOF PREDICTIONS.
    if oof_predictions_path is not None:
        oof_predictions = pd.read_csv(oof_predictions_path)['prediction'].values
        df.insert(df.shape[1], 'oof_predictions', oof_predictions)

    # LOAD QUALITY METRIC VALUES.
    all_luds_to_quality_metrics = LoadAllQualityMetrics()
    try:
        quality_metric_values = np.array([
            all_luds_to_quality_metrics[lud]
            for lud in lud_rules
        ])
    except:
        quality_metric_values = None

    print(f'Data shape: {df.shape}')
    return ruleset_names, df, quality_metric_values

def TrainModels(
        train_test_df, 
        extra_train_df,
        game_fitness_scores,
        cat_params, 
        early_stopping_round_count, 
        fold_count,
        down_weight_negative_fitness = False):
    models = []
    rmse_scores = []

    extra_train_df = extra_train_df[train_test_df.columns]
    all_data = pd.concat([train_test_df, extra_train_df], ignore_index=True)

    X = all_data.drop(['utility_agent1'], axis=1)
    y = all_data['utility_agent1']
    all_predictions = np.empty(len(y))


    categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']

    kf = KFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_SEED)
    folds = list(kf.split(X, y))
    for fold_index, (train_index, test_index) in enumerate(tqdm(folds)):
        # SPLIT INTO TRAIN AND TEST.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        # SET SAMPLE WEIGHTS.
        if down_weight_negative_fitness:
            train_fitness_scores = game_fitness_scores[train_index]
            train_weights = [1 if score > 0 else 0.5 for score in train_fitness_scores]
            test_fitness_scores = game_fitness_scores[test_index]
            test_weights = [1 if score > 0 else 0.5 for score in test_fitness_scores]
        else:
            train_weights = None
            test_weights = None

        # TRAIN MODEL.
        train_pool = Pool(
            train_x, 
            train_y,
            weight=train_weights, 
            cat_features=categorical_features)
        test_pool = Pool(
            test_x, 
            test_y, 
            weight=test_weights,
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

        # RECORD PREDICTIONS.
        all_predictions[test_index] = model.predict(test_x)

    # CALCULATE TEST RMSE.
    print('Fold RMSEs:', rmse_scores)
    
    mean_score = np.mean(rmse_scores)
    print(f'Average RMSE: {mean_score:.4f}')

    # SPLIT PREDICTIONS.
    train_test_predictions = all_predictions[:len(train_test_df)]
    extra_train_predictions = all_predictions[len(train_test_df):]

    return models, mean_score, train_test_predictions, extra_train_predictions

def SaveModels(trained_models, test_rmse, output_directory_suffix = ''):
    output_directory_path = f'NoiseEstimation/models/catboost_{int(test_rmse*100000)}_{len(trained_models)}_{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, model in enumerate(trained_models):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump(model, output_filepath)

def CreateEnsembleAndPredictions(
        cat_params, 
        early_stopping_round_count, 
        fold_count, 
        starting_evals_json_paths,
        include_english_lsa = False,
        include_lud_lsa = False,
        oof_predictions_path = None,
        extra_train_paths = None,
        output_directory_suffix = None,
        predictions_filename_prefix = 'predictions_'):
    # LOAD DATA.
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, train_test_df, game_fitness_scores = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
        # games_csv_path = f'ELO/CSV/organizer_oof_pred_deltas_{RANDOM_SEED//1111 - 3}.csv', 
        starting_evals_json_paths = starting_evals_json_paths,
        split_agent_features = True,
        include_english_lsa = include_english_lsa,
        include_lud_lsa = include_lud_lsa,
        oof_predictions_path = oof_predictions_path,
    )

    extra_ruleset_names, extra_train_df, _ = GetPreprocessedData(
        games_csv_path = extra_train_paths['games_csv_path'],
        starting_evals_json_paths = extra_train_paths['starting_position_evals_json_paths'],
        split_agent_features = True,
        include_english_lsa = include_english_lsa,
        include_lud_lsa = include_lud_lsa,
        oof_predictions_path = None
    )

    # TRAINING & INFERENCE.
    trained_models, test_rmse, train_test_predictions, extra_train_predictions = TrainModels(
        train_test_df, 
        extra_train_df,
        game_fitness_scores,
        cat_params, 
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
    )

    # SAVE MODELS.
    if output_directory_suffix is not None:
        SaveModels(trained_models, test_rmse, output_directory_suffix)

    # SAVE PREDICTIONS.
    organizer_predictions = pd.DataFrame({
        'prediction': train_test_predictions
    })
    organizer_predictions.to_csv(f'NoiseEstimation/predictions/{predictions_filename_prefix}organizer_seed{RANDOM_SEED}.csv', index=False)

    extra_train_predictions = pd.DataFrame({
        'prediction': extra_train_predictions
    })
    extra_train_predictions.to_csv(f'NoiseEstimation/predictions/{predictions_filename_prefix}extra_seed{RANDOM_SEED}.csv', index=False)

    return test_rmse

def CreateNestedEnsemblePredictions(
        cat_params, 
        early_stopping_round_count, 
        inner_fold_count, 
        outer_fold_count,
        starting_evals_json_paths,
        include_english_lsa = False,
        include_lud_lsa = False,
        oof_predictions_path = None,
        extra_train_paths = None,
        predictions_filename_prefix = 'predictions_'):
    # LOAD DATA.
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, train_test_df, game_fitness_scores = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
        # games_csv_path = f'ELO/CSV/organizer_oof_pred_deltas_{RANDOM_SEED//1111 - 3}.csv', 
        starting_evals_json_paths = starting_evals_json_paths,
        split_agent_features = True,
        include_english_lsa = include_english_lsa,
        include_lud_lsa = include_lud_lsa,
        oof_predictions_path = oof_predictions_path,
    )

    extra_ruleset_names, extra_train_df, _ = GetPreprocessedData(
        games_csv_path = extra_train_paths['games_csv_path'],
        starting_evals_json_paths = extra_train_paths['starting_position_evals_json_paths'],
        split_agent_features = True,
        include_english_lsa = include_english_lsa,
        include_lud_lsa = include_lud_lsa,
        oof_predictions_path = None
    )

    # PROCESS OUTER FOLDS.
    group_kfold_shuffle = GroupKFoldShuffle(
        n_splits=outer_fold_count, 
        shuffle=True, 
        random_state=0)
    folds = list(group_kfold_shuffle.split(train_test_df, train_test_df['utility_agent1'], groups=ruleset_names))
    test_rmses = []
    for fold_index, (train_index, test_index) in enumerate(tqdm(folds, desc='Outer folds')):
        filtered_train_test_df = train_test_df.iloc[train_index]

        # TRAINING & INFERENCE.
        trained_models, test_rmse, train_test_predictions, extra_train_predictions = TrainModels(
            filtered_train_test_df, 
            extra_train_df,
            game_fitness_scores,
            cat_params, 
            early_stopping_round_count = early_stopping_round_count, 
            fold_count = inner_fold_count,
        )
        test_rmses.append(test_rmse)

        # SAVE PREDICTIONS.
        organizer_predictions = pd.DataFrame({
            'prediction': train_test_predictions
        })
        organizer_predictions.to_csv(f'NoiseEstimation/predictions/{predictions_filename_prefix}organizer_seed{RANDOM_SEED}_fold{fold_index}.csv', index=False)

        extra_train_predictions = pd.DataFrame({
            'prediction': extra_train_predictions
        })
        extra_train_predictions.to_csv(f'NoiseEstimation/predictions/{predictions_filename_prefix}extra_seed{RANDOM_SEED}_fold{fold_index}.csv', index=False)

    return np.mean(test_rmses)

if __name__ == '__main__':
    extra_train_paths = {
        'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
        'starting_position_evals_json_paths': [
            'GAVEL/generated_csvs/complete_datasets/233/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json',
            'GAVEL/generated_csvs/complete_datasets/252/MCTS-UCB1Tuned-0.6-NST-true-16s.json',
        ]
    }

    scores = []
    for seed in [0, 1, 2]:
        RANDOM_SEED = seed
        print(f'Seed: {RANDOM_SEED}')

        rmse = CreateNestedEnsemblePredictions(
            cat_params = {
                "iterations": 10219, 
                "learning_rate": 0.010964241393786744, 
                "depth": 10, 
                "l2_leaf_reg": 0.0012480029901784353, 
                "grow_policy": "SymmetricTree", 
                "max_ctr_complexity": 6,
                "task_type": "GPU"
            },
            # cat_params = {
            #     "iterations": 1000, 
            #     "learning_rate": 0.1,
            #     "task_type": "GPU"
            # },
            early_stopping_round_count = 100,
            inner_fold_count = 8,
            outer_fold_count = 10,
            starting_evals_json_paths = [
                'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json',
                'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-0.6-NST-true-16s.json',
            ],
            include_english_lsa = False,
            include_lud_lsa = False,
            oof_predictions_path = None,
            extra_train_paths = extra_train_paths,
            predictions_filename_prefix = 'nested_debug_v4_'
        )

        scores.append(rmse)

    print('Scores:', scores)
    print('Mean:', np.mean(scores))