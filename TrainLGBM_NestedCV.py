import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
from GroupKFoldShuffle import GroupKFoldShuffle
import os
import joblib
import optuna
import json
from sklearn.metrics import root_mean_squared_error
from cir_model import CenteredIsotonicRegression
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(games_csv_path, starting_evals_json_paths):
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
        if (col[:6] in AGENT_COLS)
    ])
    df = df.with_columns([
        pl.col(col).cast(pl.Float32) 
        for col in df.columns 
        if not (col[:6] in AGENT_COLS)
    ])
    
    df = df.to_pandas()

    # ADD MCTS EVALUATION FEATURES.
    '''
    for starting_evals_json_path in starting_evals_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)

        feature_name = starting_evals_json_path.split('/')[-1].replace('.json', '')
        df.insert(
            df.shape[1], 
            feature_name,
            [luds_to_mcts_evals[lud] for lud in lud_rules]
        )
    '''

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

def TrainIsotonicModels(oof_predictions, targets, ruleset_names, fold_count, random_seed):
    clipped_oof_predictions = np.clip(oof_predictions, -1, 1)
    isotonic_oof_predictions = np.empty_like(clipped_oof_predictions)

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    models = []
    rmses = []
    for fold_index, (train_index, test_index) in enumerate(group_kfold.split(clipped_oof_predictions, targets, groups=ruleset_names)):
        train_predictions = clipped_oof_predictions[train_index]
        train_targets = targets.to_numpy()[train_index]
        test_predictions = clipped_oof_predictions[test_index]
        test_targets = targets.to_numpy()[test_index]

        # model = CenteredIsotonicRegression()
        model = IsotonicRegression(out_of_bounds = 'clip')
        model.fit(train_predictions, train_targets)
        models.append(model)

        predictions = model.predict(test_predictions)
        isotonic_oof_predictions[test_index] = predictions

    return models, isotonic_oof_predictions

def TrainModels(
        ruleset_names, 
        train_test_df, 
        extra_ruleset_names,
        extra_train_test_df,
        feature_importances_dir,
        dropped_feature_count,
        lgb_params, 
        early_stopping_round_count, 
        fold_count, 
        random_seed = None):
    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    if extra_train_test_df is not None:
        extra_X = extra_train_test_df.drop(['utility_agent1'], axis=1)
        extra_y = extra_train_test_df['utility_agent1']

        if extra_X.columns.tolist() != X.columns.tolist():
            extra_X = extra_X[X.columns]

        X = pd.concat([X, extra_X], ignore_index=True)
        y = pd.concat([y, extra_y], ignore_index=True)
        ruleset_names = pd.concat([ruleset_names, extra_ruleset_names], ignore_index=True)


    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    models = []
    base_oof_predictions = np.empty(X.shape[0])
    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        # SPLIT INTO TRAIN AND TEST.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        # DROP FEATURES.
        if feature_importances_dir is not None:
            feature_importances_df = pd.read_csv(f'{feature_importances_dir}/{fold_index}.csv')
            feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
            dropped_feature_names = feature_importances_df['feature'].tolist()[-dropped_feature_count:]

            train_x = train_x.drop(columns=dropped_feature_names)
            test_x = test_x.drop(columns=dropped_feature_names)

        # TRAIN MODEL.
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            train_x,
            train_y,
            eval_set=[(test_x, test_y)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_round_count)])

        models.append(model) 

        # GENERATE OOF PREDICTIONS.
        predictions = model.predict(test_x)
        base_oof_predictions[test_index] = predictions

    base_rmse = root_mean_squared_error(y, base_oof_predictions)
    isotonic_models, isotonic_oof_predictions = TrainIsotonicModels(
        base_oof_predictions, 
        y,
        ruleset_names, 
        fold_count, 
        random_seed)
    isotonic_rmse = root_mean_squared_error(y, isotonic_oof_predictions)

    print(f'Average RMSE (base, isotonic): {base_rmse:.5f}, {isotonic_rmse:.5f}')

    oof_predictions = isotonic_oof_predictions[:train_test_df.shape[0]]
    extra_oof_predictions = isotonic_oof_predictions[train_test_df.shape[0]:]

    return models, isotonic_models, oof_predictions, extra_oof_predictions

def Objective(trial, ruleset_names, train_test_df, fold_count):
    lgb_params = {
        'num_iterations': trial.suggest_categorical('num_iterations', [5_000, 10_000, 20_000, 40_000]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 384),
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1e-4, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1e-4, log=True),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
    }
    
    early_stopping_round_count = 50
    
    _, mean_score = TrainModels(
        ruleset_names,
        train_test_df,
        lgb_params,
        early_stopping_round_count,
        fold_count,
        random_seed = RANDOM_SEED
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
    output_filepath = f'configs/lgbm_{int(best_score * 100000)}.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

def SaveModels(gbdt_models, isotonic_models, base_rmse, isotonic_rmse, output_directory_suffix = ''):
    output_directory_path = f'models/lgbm_iso_{int(base_rmse*100000)}_{int(isotonic_rmse*100000)}_{len(gbdt_models)}{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, (gbdt_model, isotonic_model) in enumerate(zip(gbdt_models, isotonic_models)):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump({
            'gbdt_model': gbdt_model,
            'isotonic_model': isotonic_model
        }, output_filepath)

def CreateNestedEnsemblePredictions(
        lgb_params, 
        early_stopping_round_count, 
        inner_fold_count, 
        outer_fold_count, 
        starting_evals_json_paths,
        extra_train_paths,
        feature_importances_dir,
        dropped_feature_count,
        predictions_filename_prefix):
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
        # games_csv_path = f'ELO/CSV/organizer_oof_pred_deltas_{RANDOM_SEED//1111 - 3}.csv', 
        starting_evals_json_paths = starting_evals_json_paths,
    )

    extra_ruleset_names, extra_train_df = GetPreprocessedData(
        games_csv_path = extra_train_paths['games_csv_path'],
        starting_evals_json_paths = extra_train_paths['starting_position_evals_json_paths']
    )

    group_kfold_shuffle = GroupKFoldShuffle(
        n_splits=outer_fold_count, 
        shuffle=True, 
        random_state=RANDOM_SEED)
    folds = list(group_kfold_shuffle.split(train_test_df, train_test_df['utility_agent1'], groups=ruleset_names))
    isotonic_rmses = []
    for fold_index, (train_index, test_index) in enumerate(tqdm(folds, desc='Outer folds')):
        filtered_ruleset_names = ruleset_names.iloc[train_index]
        filtered_train_test_df = train_test_df.iloc[train_index]

        # TRAINING & INNER CV.
        base_models, isotonic_models, organizer_isotonic_oof_predictions, extra_isotonic_oof_predictions = TrainModels(
            filtered_ruleset_names,
            filtered_train_test_df, 
            extra_ruleset_names,
            extra_train_df,
            feature_importances_dir,
            dropped_feature_count,
            lgb_params, 
            early_stopping_round_count = early_stopping_round_count, 
            fold_count = inner_fold_count,
            random_seed = RANDOM_SEED
        )

        # OUTER CV.
        test_df = train_test_df.iloc[test_index]
        base_predictions = np.mean([
            model.predict(test_df.drop(['utility_agent1'], axis=1))
            for model in base_models
        ], axis=0)
        isotonic_predictions = np.mean([
            model.predict(base_predictions)
            for model in isotonic_models
        ], axis=0)

        full_organizer_oof_predictions = np.empty(train_test_df.shape[0])
        full_organizer_oof_predictions[train_index] = organizer_isotonic_oof_predictions
        full_organizer_oof_predictions[test_index] = isotonic_predictions

        # SAVE PREDICTIONS.
        organizer_predictions_df = pd.DataFrame({
            'prediction': full_organizer_oof_predictions
        })
        organizer_predictions_df.to_csv(f'predictions/{predictions_filename_prefix}organizer_seed{RANDOM_SEED}_fold{fold_index}.csv', index=False)

        extra_predictions_df = pd.DataFrame({
            'prediction': extra_isotonic_oof_predictions
        })
        extra_predictions_df.to_csv(f'predictions/{predictions_filename_prefix}extra_seed{RANDOM_SEED}_fold{fold_index}.csv', index=False)

        # EVALUATE.
        rmse = root_mean_squared_error(
            train_test_df['utility_agent1'].iloc[train_index], 
            organizer_isotonic_oof_predictions)
        isotonic_rmses.append(rmse)

    return np.mean(isotonic_rmses)

'''
Without dropping features (base, isotonic): 0.37256, 0.37083
350 dropped features (base, isotonic): 0.37454, 0.37318
300 dropped features (base, isotonic): 0.37270, 0.37102
200 dropped features (base, isotonic): 0.37253, 0.37085
100 dropped features (base, isotonic): 0.37219, 0.37054
50 dropped features (base, isotonic): 0.37303, 0.37140
0 dropped features (base, isotonic): 0.37200, 0.37025
'''
if __name__ == '__main__':
    extra_train_paths = {
        'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
        'starting_position_evals_json_paths': [
            # 'StartingPositionEvaluation/Evaluations/FromKaggle/extra_UCB1Tuned-1.41421356237-random-false_60s.json',
            # 'StartingPositionEvaluation/Evaluations/FromKaggle/extra_UCB1Tuned-1.41421356237-random-false_30s.json',
            'GAVEL/generated_csvs/complete_datasets/233/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json',
            # 'GAVEL/generated_csvs/complete_datasets/252/MCTS-UCB1Tuned-0.6-NST-true-16s.json',
            # 'GAVEL/generated_csvs/complete_datasets/252/MCTS-UCB1Tuned-1.41421356237-random-false-750ms.json',
        ]
    }

    seeds_to_importance_dirs = {
        3333: 'data/feature_importances/lgbm_iso_37609_37430_10_et_v4_rand_16s_seed3333',
        4444: 'data/feature_importances/lgbm_iso_37105_36915_10_et_v4_rand_16s_seed4444',
        5555: 'data/feature_importances/lgbm_iso_37053_36902_10_et_v4_rand_16s_seed5555'
    }

    base_scores = []
    isotonic_scores = []
    # for seed in [3333, 4444, 5555]:
    for seed in [0000, 1111, 2222]:
        RANDOM_SEED = seed
        print(f'Seed: {RANDOM_SEED}')

        isotonic_rmse = CreateNestedEnsemblePredictions(
            lgb_params = {
                "n_estimators": 10000,
                "learning_rate": 0.03184567466358953,
                "num_leaves": 196,
                "max_depth": 17,
                "min_child_samples": 94,
                "subsample": 0.8854325308371437,
                "colsample_bytree": 0.9612980174610098,
                "colsample_bynode": 0.6867101420064379,
                "reg_alpha": 1.593152807295967e-05,
                "reg_lambda": 4.8636580199114866e-08,
                "extra_trees": False,
                "verbose": -1
            },
            early_stopping_round_count = 100,
            inner_fold_count = 10,
            outer_fold_count = 5,
            starting_evals_json_paths = [
                # 'StartingPositionEvaluation/Evaluations/FromKaggle/organizer_UCB1Tuned-1.41421356237-random-false_60s.json',
                # 'StartingPositionEvaluation/Evaluations/FromKaggle/organizer_UCB1Tuned-1.41421356237-random-false_30s.json',
                'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json',
                # 'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-0.6-NST-true-16s.json',
                # 'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-750ms.json',
            ],
            extra_train_paths = extra_train_paths,
            # feature_importances_dir = seeds_to_importance_dirs[RANDOM_SEED],
            # dropped_feature_count = droped_feature_count,
            feature_importances_dir = None,
            dropped_feature_count = 0,
            predictions_filename_prefix = f'nested_5outer_full_'
        )

        isotonic_scores.append(isotonic_rmse)

    print(f'\nIsotonic RMSE: {np.mean(isotonic_scores):.5f} +/- {np.std(isotonic_scores):.5f}')