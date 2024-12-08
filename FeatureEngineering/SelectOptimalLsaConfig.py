import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
import os
import joblib
import optuna
import json
from sklearn.metrics import root_mean_squared_error
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression
import pickle
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from LsaPreprocessor import LsaPreprocessor

import sys
sys.path.append('./')
from ColumnNames import DROPPED_COLUMNS, AGENT_COLS
from GroupKFoldShuffle import GroupKFoldShuffle

# global
RANDOM_SEED = None

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(games_csv_path):
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

    # print('WARNING: Using experimental added feature, possibly revert!')
    # df['DurationToComplexityRatio'] = df['DurationActions'] / (df['StateTreeComplexity'] + 1e-15)

    print(f'Data shape: {df.shape}')
    return ruleset_names, lud_rules, df

def AddStartingEvalFeatures(starting_eval_json_paths, train_test_df, lud_rules, random_seed, global_average_weight = 0.33):
    all_luds_to_mcts_evals = []
    for starting_evals_json_path in starting_eval_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)
            all_luds_to_mcts_evals.append(luds_to_mcts_evals)

    np.random.seed(random_seed)
    multi_eval = isinstance(list(all_luds_to_mcts_evals[0].values())[0], list)
    if multi_eval:
        mcts_feature_count = len(list(all_luds_to_mcts_evals[0].values())[0])
        luds_to_mean_evals = {
            lud: [
                np.mean([luds_to_mcts_evals[lud][i] for luds_to_mcts_evals in all_luds_to_mcts_evals]) 
                for i in range(mcts_feature_count)
            ]
            for lud in lud_rules
        }

        for i in range(mcts_feature_count):
            feature_name = f'mcts_eval_{i}'
            train_test_df.insert(
                train_test_df.shape[1], 
                feature_name,
                [
                    np.average(
                        [np.random.choice(all_luds_to_mcts_evals)[lud][i], luds_to_mean_evals[lud][i]], 
                        weights=[1-global_average_weight, global_average_weight])
                    for lud in lud_rules
                ]
            )
    else:
        train_test_df.insert(
            train_test_df.shape[1], 
            'mcts_eval',
            [np.random.choice(all_luds_to_mcts_evals)[lud] for lud in lud_rules]
        )

    return train_test_df

def AddLsaFeatures(lsa_config, train_df, train_luds, test_df, test_luds, train_targets):
    lsa_preprocessor = LsaPreprocessor(**lsa_config)
    train_df = lsa_preprocessor.fit_transform(train_luds, train_df, train_targets)
    test_df = lsa_preprocessor.transform(test_luds, test_df)

    return lsa_preprocessor, train_df, test_df

def TrainIsotonicModels(oof_predictions, targets, ruleset_names, fold_count, random_seed):
    clipped_oof_predictions = np.clip(oof_predictions, -1, 1)

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    models = []
    rmses = []
    isotonic_oof_preds = np.empty(len(oof_predictions))
    for fold_index, (train_index, test_index) in enumerate(group_kfold.split(clipped_oof_predictions, targets, groups=ruleset_names)):
        train_predictions = clipped_oof_predictions[train_index]
        train_targets = targets[train_index]
        test_predictions = clipped_oof_predictions[test_index]
        test_targets = targets[test_index]

        # print(f'Train range: {min(train_targets):.5f} - {max(train_targets):.5f}')
        if min(train_targets) == -1 and max(train_targets) == 1:
            # print('Using centered isotonic regression...')
            model = CenteredIsotonicRegression()
        else:
            # print('Using isotonic regression...')
            model = IsotonicRegression(out_of_bounds='clip')
        model.fit(train_predictions, train_targets)
        models.append(model)

        # print(f'Test range: {min(test_targets):.5f} - {max(test_targets):.5f}')
        predictions = model.predict(test_predictions)
        # print('NAN prediction count:', np.sum(np.isnan(predictions)))
        
        nan_mask = np.isnan(predictions)
        predictions[nan_mask] = test_predictions[nan_mask]

        rmse = root_mean_squared_error(test_targets, predictions)
        rmses.append(rmse)

        isotonic_oof_preds[test_index] = predictions

    return models, np.mean(rmses), isotonic_oof_preds

def TrainModels(
        # Competition organizer data.
        ruleset_names, 
        train_test_df, 
        starting_eval_json_paths,
        lud_rules,
        # Extra data.
        extra_train_df,
        extra_starting_eval_json_paths,
        extra_lud_rules,
        # Everything else.
        feature_importances_dir,
        dropped_feature_count,
        lgb_params, 
        lsa_params,
        early_stopping_round_count, 
        fold_count, 
        random_seed = None):
    models = []
    lsa_preprocessors = []
    oof_predictions = np.empty(train_test_df.shape[0])

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    if extra_train_df is not None:
        extra_train_X = extra_train_df.drop(['utility_agent1'], axis=1)
        extra_train_y = extra_train_df['utility_agent1']

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        # SPLIT INTO TRAIN AND TEST.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        heldout_run_index = fold_index%len(starting_eval_json_paths)
        training_run_eval_paths = starting_eval_json_paths[:heldout_run_index] + starting_eval_json_paths[heldout_run_index+1:]
        train_x = AddStartingEvalFeatures(training_run_eval_paths, train_x, lud_rules.iloc[train_index], random_seed)
        test_x = AddStartingEvalFeatures([starting_eval_json_paths[heldout_run_index]], test_x, lud_rules.iloc[test_index], random_seed)

        if extra_train_df is not None:
            if extra_train_X.columns.tolist() != train_x.columns.tolist():
                extra_training_run_eval_paths = extra_starting_eval_json_paths[:heldout_run_index] + extra_starting_eval_json_paths[heldout_run_index+1:]
                extra_train_X = AddStartingEvalFeatures(extra_training_run_eval_paths, extra_train_X, extra_lud_rules, random_seed)
                extra_train_X = extra_train_X[train_x.columns]

            train_x = pd.concat([train_x, extra_train_X], ignore_index=True)
            train_y = pd.concat([train_y, extra_train_y], ignore_index=True)

        # DROP FEATURES.
        if feature_importances_dir is not None:
            feature_importances_df = pd.read_csv(f'{feature_importances_dir}/{fold_index}.csv')
            feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
            dropped_feature_names = feature_importances_df['feature'].tolist()[-dropped_feature_count:]

            train_x = train_x.drop(columns=dropped_feature_names)
            test_x = test_x.drop(columns=dropped_feature_names)

        # ADD LSA FEATURES.
        lsa_preprocessor = None
        if lsa_params is not None:
            lsa_preprocessor, train_x, test_x = AddLsaFeatures(
                lsa_params, 
                train_x, 
                pd.concat([lud_rules.iloc[train_index], extra_lud_rules], ignore_index = True) if (extra_lud_rules is not None) else lud_rules.iloc[train_index], 
                test_x, 
                lud_rules.iloc[test_index],
                train_y
            )

        lsa_preprocessors.append(lsa_preprocessor)

        # TRAIN MODEL.
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            train_x,
            train_y,
            eval_set=[(test_x, test_y)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_round_count)] if 'dart' not in lgb_params.values() else None,
        )

        models.append(model) 

        # GENERATE OOF PREDICTIONS.
        predictions = model.predict(test_x)
        oof_predictions[test_index] = predictions

    base_rmse = root_mean_squared_error(y, oof_predictions)
    isotonic_models, isotonic_rmse, isotonic_oof_preds = TrainIsotonicModels(oof_predictions, y, ruleset_names, fold_count, random_seed)

    print(f'Average RMSE (base, isotonic): {base_rmse:.5f}, {isotonic_rmse:.5f}')

    return lsa_preprocessors, models, isotonic_models, base_rmse, isotonic_rmse, oof_predictions, isotonic_oof_preds

def Objective(trial, fold_count, extra_train_paths, starting_eval_json_paths):
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
        )

    analyzer = trial.suggest_categorical("analyzer", ['word', 'char'])
    if analyzer == 'word':
        min_ngram_range = trial.suggest_int("min_ngram_range", 1, 4)
        max_ngram_range = trial.suggest_int("max_ngram_range", min_ngram_range, 4)
    else:
        min_ngram_range = trial.suggest_int("min_ngram_range", 1, 7)
        max_ngram_range = trial.suggest_int("max_ngram_range", min_ngram_range, 7)
    lsa_params = {
        "n_components": trial.suggest_int("n_components", 50, 100),
        "analyzer": analyzer,
        "ngram_range": (min_ngram_range, max_ngram_range),
        "max_df": trial.suggest_float("max_df", 0.75, 1.0),
        "min_df": trial.suggest_float("min_df", 0.0, 0.25),
        "kept_feature_count": trial.suggest_int("kept_feature_count", 1, 50),
    }
    
    lgb_params = {
        "n_estimators": 19246,
        "learning_rate": 0.0028224515150795885,
        "num_leaves": 365,
        "max_depth": 14,
        "min_child_samples": 55,
        "colsample_bytree": 0.9886746573495085,
        "colsample_bynode": 0.9557863173491425,
        "reg_alpha": 4.530707764807948e-07,
        "reg_lambda": 2.5292981163243776e-05,
        "extra_trees": True,
        "verbose": -1
    }

    early_stopping_round_count = 50
    lsa_preprocessors, lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
        # Competition organizer data.
        ruleset_names,
        train_test_df, 
        starting_eval_json_paths,
        lud_rules,
        # Extra data.
        extra_train_df,
        extra_train_paths['starting_position_evals_json_paths'],
        extra_lud_rules,
        # Everything else.
        feature_importances_dir=None,
        dropped_feature_count=0,
        lgb_params = lgb_params, 
        lsa_params = lsa_params,
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
        random_seed=RANDOM_SEED
    )
    
    return isotonic_rmse

def GetOptimalConfig(trial_count, extra_train_paths, starting_eval_json_paths):
    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: Objective(
            trial, 
            fold_count=5, 
            extra_train_paths=extra_train_paths, 
            starting_eval_json_paths=starting_eval_json_paths), 
        n_trials=trial_count
    )

    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=2))

    best_score = study.best_trial.value
    output_filepath = f'configs/lgbm_{int(best_score * 100000)}.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

def SaveModels(lsa_preprocessors, gbdt_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix = ''):
    output_directory_path = f'models/lgbm_iso_{int(base_rmse*100000)}_{int(isotonic_rmse*100000)}_{len(gbdt_models)}{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, (lsa_preprocessor, gbdt_model, isotonic_model) in enumerate(zip(lsa_preprocessors, gbdt_models, isotonic_models)):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump({
            'lsa_preprocessor': lsa_preprocessor,
            'gbdt_model': gbdt_model,
            'isotonic_model': isotonic_model,
            'base_oof_preds': base_oof_preds,
            'isotonic_oof_preds': isotonic_oof_preds
        }, output_filepath)

        # with open(output_filepath, 'wb') as f:
        #     pickle.dump({
        #         'gbdt_model': gbdt_model,
        #         'isotonic_model': isotonic_model,
        #         'base_oof_preds': base_oof_preds,
        #         'isotonic_oof_preds': isotonic_oof_preds
        #     }, f)

# TODO: Maybe update to receive LSA params.
def CreateEnsemble(
        lgb_params,
        lsa_params,
        early_stopping_round_count,
        fold_count,
        starting_eval_json_paths,
        extra_train_paths,
        # Everything else.
        feature_importances_dir,
        dropped_feature_count,
        output_directory_suffix = ''):
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path']
        )

    lsa_preprocessors, lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
        # Competition organizer data.
        ruleset_names,
        train_test_df, 
        starting_eval_json_paths,
        lud_rules,
        # Extra data.
        extra_train_df,
        extra_train_paths['starting_position_evals_json_paths'],
        extra_lud_rules,
        # Everything else.
        feature_importances_dir,
        dropped_feature_count,
        lgb_params, 
        lsa_params,
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
        random_seed = RANDOM_SEED
    )

    if output_directory_suffix is not None:
        SaveModels(lsa_preprocessors, lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix)

    return base_rmse, isotonic_rmse

'''
Trial 1 finished with value: 0.3705929446754424 and parameters: {'analyzer': 'word', 'min_ngram_range': 4, 'max_ngram_range': 4, 'n_components': 100, 'max_df': 0.8705023665904886, 'min_df': 0.05016684092193893, 'kept_feature_count': 20}. Best is trial 1 with value: 0.3705929446754424.
'''
if __name__ == '__main__':
    # extra_train_paths = {
    #     'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
    #     'starting_position_evals_json_paths': [
    #         f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
    #         for i in range(10)
    #     ]
    # }
    # starting_eval_json_paths = [
    #     f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
    #     for i in range(10)
    # ]

    # RANDOM_SEED = 4732497
    # GetOptimalConfig(30, extra_train_paths, starting_eval_json_paths)

    #'''
    LGBM_CONFIGS = [
        { # Baseline, tuned 200 iters without extra data, MCTS features, or isotonic regression.
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
        { # Retuned 64 iters with v4 extra data, MCTS features, and isotonic regression.
            "n_estimators": 19246,
            "learning_rate": 0.0028224515150795885,
            "num_leaves": 365,
            "max_depth": 14,
            "min_child_samples": 55,
            # subsample does nothing because forgot to set bagging_freq.
            "subsample": 0.585026292470321,
            "colsample_bytree": 0.9886746573495085,
            "colsample_bynode": 0.9557863173491425,
            "reg_alpha": 4.530707764807948e-07,
            "reg_lambda": 2.5292981163243776e-05,
            "extra_trees": True,
            "verbose": -1
        },
        { # DART, 7 iter.
            "n_estimators": 9648,
            "learning_rate": 0.08036455460239479,
            "num_leaves": 99,
            "max_depth": 24,
            "min_child_samples": 8,
            "bagging_freq": 0,
            "subsample": 0.9845338171021742,
            "colsample_bytree": 0.8357840074927564,
            "colsample_bynode": 0.9035812734166077,
            "reg_alpha": 9.817738265024515e-05,
            "reg_lambda": 3.582329308097412e-06,
            "extra_trees": True,
            'boosting': 'dart',
            "verbose": -1
        },
        { # GOSS, 25 iter.
            "n_estimators": 5586,
            "learning_rate": 0.009858459423497792,
            "num_leaves": 160,
            "max_depth": 23,
            "min_child_samples": 47,
            "colsample_bytree": 0.9004502117117108,
            "colsample_bynode": 0.7467639442223327,
            "reg_alpha": 1.0096561472394783e-08,
            "reg_lambda": 6.197582361106275e-06,
            'data_sample_strategy': 'goss',
            "extra_trees": True,
            "verbose": -1
        }
    ]

    LSA_CONFIG = {
        'analyzer': 'word', 
        'ngram_range': (4, 4),
        'n_components': 100, 
        'max_df': 0.8705023665904886, 
        'min_df': 0.05016684092193893, 
        'kept_feature_count': 20
    }

    MCTS_CONFIG_NAMES = [
        # '1.41421356237-random-true',
        '1.41421356237-random-false',
        # '1.41421356237-nst-true',
        # '1.41421356237-nst-false',
        # '0.6-random-true',
        # '0.6-random-false',
        # '0.6-nst-true',
        # '0.6-nst-false',
    ]
    MCTS_RUNTIMES_SEC = [15]
    for mcts_config_name in MCTS_CONFIG_NAMES:
        for mcts_runtime_sec in MCTS_RUNTIMES_SEC:
            for lgbm_config_index, lgbm_params in enumerate(LGBM_CONFIGS):
                if lgbm_config_index not in [2]:
                    continue

                extra_train_paths = {
                    'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
                    'starting_position_evals_json_paths': [
                        f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                        for i in range(10)
                    ]
                }
                starting_eval_json_paths = [
                    f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                    for i in range(10)
                ]

                base_rmses, isotonic_rmses = [], []
                for RANDOM_SEED in [3333, 4444, 5555]:
                    base_rmse, isotonic_rmse = CreateEnsemble(
                        lgb_params = lgbm_params,
                        lsa_params = LSA_CONFIG,
                        early_stopping_round_count = 100,
                        fold_count = 10,
                        starting_eval_json_paths = starting_eval_json_paths,
                        extra_train_paths = extra_train_paths,
                        feature_importances_dir = None,
                        dropped_feature_count = 0,
                        output_directory_suffix = f'_lsa_et_v4_{mcts_config_name}_{mcts_runtime_sec}s_cfg{lgbm_config_index}_seed{RANDOM_SEED}_r1-10_aug_gaw33'
                    )
                    base_rmses.append(base_rmse)
                    isotonic_rmses.append(isotonic_rmse)

                    print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, lgbm config = {lgbm_config_index}')
                    print(f'Base RMSE: {base_rmse:.5f}, Isotonic RMSE: {isotonic_rmse:.5f}\n')

                print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, lgbm config = {lgbm_config_index}')
                print(f'Base RMSEs: {base_rmses}, (mean = {np.mean(base_rmses):.5f})')
                print(f'Isotonic RMSEs: {isotonic_rmses}, (mean = {np.mean(isotonic_rmses):.5f})\n')
    #'''