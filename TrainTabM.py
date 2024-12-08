import math
import random
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold, train_test_split
from GroupKFoldShuffle import GroupKFoldShuffle
from sklearn.metrics import root_mean_squared_error
import torch
import joblib
import optuna
import json
import os
import pickle

from tabm.tabm_reference import Model, make_parameter_groups
import rtdl_num_embeddings
import scipy.special

from TabMRegressor import TabMRegressor

from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

RANDOM_SEED = 0
DROPPED_COLUMNS += ['MancalaBoard', 'MancalaStores', 'MancalaTwoRows', 'MancalaThreeRows', 'MancalaFourRows', 'MancalaSixRows', 'MancalaCircular', 'AlquerqueBoard', 'AlquerqueBoardWithOneTriangle', 'AlquerqueBoardWithTwoTriangles', 'AlquerqueBoardWithFourTriangles', 'AlquerqueBoardWithEightTriangles', 'ThreeMensMorrisBoard', 'ThreeMensMorrisBoardWithTwoTriangles', 'NineMensMorrisBoard', 'StarBoard', 'CrossBoard', 'KintsBoard', 'PachisiBoard', 'FortyStonesWithFourGapsBoard', 'Sow', 'SowCW', 'SowCCW', 'GraphStyle', 'ChessStyle', 'GoStyle', 'MancalaStyle', 'PenAndPaperStyle', 'ShibumiStyle', 'BackgammonStyle', 'JanggiStyle', 'XiangqiStyle', 'ShogiStyle', 'TableStyle', 'SurakartaStyle', 'TaflStyle', 'NoBoard', 'MarkerComponent', 'StackType', 'Stack', 'Symbols', 'ShowPieceValue', 'ShowPieceState']

def GetPreprocessedData(games_csv_path: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pl.read_csv(games_csv_path)

    ruleset_names = df['GameRulesetName'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    
    df = df.drop([col for col in DROPPED_COLUMNS if col in df.columns])

    df = df.to_pandas()
    df['agent1'] = df['agent1'].str.replace('-random-', '-Random200-')
    df['agent2'] = df['agent2'].str.replace('-random-', '-Random200-')
    df = pl.DataFrame(df)

    for col in AGENT_COLS:
        df = df.with_columns(
            pl.col(col)
            .str.split(by="-")
            .list.to_struct(fields=lambda idx: f"{col}_{idx}")
        ).unnest(col).drop(f"{col}_0")

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

    print(f'Data shape: {df.shape}')
    return ruleset_names, lud_rules, df

def AddStartingEvalFeatures(starting_eval_json_paths, train_test_df, lud_rules, global_average_weight = 0.0):
    all_luds_to_mcts_evals = []
    for starting_evals_json_path in starting_eval_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)
            all_luds_to_mcts_evals.append(luds_to_mcts_evals)

    np.random.seed(RANDOM_SEED)
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

def TrainIsotonicModels(oof_predictions: np.ndarray, targets: np.ndarray, ruleset_names: pd.Series, fold_count: int, random_seed: Optional[int]) -> Tuple[List, float, np.ndarray]:
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

        if min(train_targets) == -1 and max(train_targets) == 1:
            model = CenteredIsotonicRegression()
        else:
            model = IsotonicRegression(out_of_bounds='clip')
        model.fit(train_predictions, train_targets)
        models.append(model)

        predictions = model.predict(test_predictions)

        nan_mask = np.isnan(predictions)
        predictions[nan_mask] = test_predictions[nan_mask]

        rmse = root_mean_squared_error(test_targets, predictions)
        rmses.append(rmse)

        isotonic_oof_preds[test_index] = predictions

    return models, np.mean(rmses), isotonic_oof_preds

def TrainModels(
        ruleset_names: pd.Series, 
        train_test_df: pd.DataFrame, 
        starting_eval_json_paths: List[str],
        lud_rules: pd.Series,
        extra_train_df: Optional[pd.DataFrame],
        extra_starting_eval_json_paths: Optional[List[str]],
        extra_lud_rules: Optional[pd.Series],
        extra_data_weight: Optional[float],
        reannotated_features_path: Optional[str],
        feature_importances_dir: Optional[str],
        dropped_feature_count: int,
        global_average_weight: float,
        tabm_params: Dict,
        fold_count: int,
        random_seed: Optional[int] = None
    ) -> Tuple[List[TabMRegressor], List, float, float, np.ndarray, np.ndarray]:
    models = []
    oof_predictions = np.empty(train_test_df.shape[0])

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1'].values

    if extra_train_df is not None:
        extra_train_X = extra_train_df.drop(['utility_agent1'], axis=1)
        extra_train_y = extra_train_df['utility_agent1'].values

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        # SPLIT INTO TRAIN AND TEST.
        train_x = X.iloc[train_index].copy()
        train_y = y[train_index].copy()

        test_x = X.iloc[test_index].copy()
        test_y = y[test_index].copy()

        # ADD STARTING EVAL FEATURES.
        heldout_run_index = fold_index % len(starting_eval_json_paths)
        training_run_eval_paths = starting_eval_json_paths[:heldout_run_index] + starting_eval_json_paths[heldout_run_index+1:]
        train_x = AddStartingEvalFeatures(training_run_eval_paths, train_x, lud_rules.iloc[train_index], global_average_weight)
        test_x = AddStartingEvalFeatures([starting_eval_json_paths[heldout_run_index]], test_x, lud_rules.iloc[test_index], global_average_weight)

        # ADD EXTRA TRAINING DATA.
        train_luds = lud_rules.iloc[train_index]
        sample_weights = np.ones(len(train_x))
        if extra_train_df is not None:
            if extra_train_X.columns.tolist() != train_x.columns.tolist():
                extra_training_run_eval_paths = extra_starting_eval_json_paths[:heldout_run_index] + extra_starting_eval_json_paths[heldout_run_index+1:]
                extra_train_X = AddStartingEvalFeatures(extra_training_run_eval_paths, extra_train_X, extra_lud_rules, global_average_weight)
                extra_train_X = extra_train_X[train_x.columns]

            train_x = pd.concat([train_x, extra_train_X], ignore_index=True)
            train_y = np.concatenate([train_y, extra_train_y])
            train_luds = pd.concat([train_luds, extra_lud_rules], ignore_index=True)
            sample_weights = np.concatenate([sample_weights, np.ones(len(extra_train_X)) * extra_data_weight])

        # DROP FEATURES.
        if feature_importances_dir is not None and dropped_feature_count > 0:
            feature_importances_df = pd.read_csv(f'{feature_importances_dir}/{fold_index}.csv')
            feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
            dropped_feature_names = feature_importances_df['feature'].tolist()[-dropped_feature_count:]

            train_x = train_x.drop(columns=dropped_feature_names)
            test_x = test_x.drop(columns=dropped_feature_names)

        # AUGMENT NONDETERMINISTIC FEATURES.
        if reannotated_features_path is not None:
            with open(reannotated_features_path, 'r') as f:
                luds_to_features_to_estimates = json.load(f)

            for feature_name in luds_to_features_to_estimates[train_luds[0]].keys():
                original_feature_values = train_x[feature_name]
                feature_estimates = np.array([
                    luds_to_features_to_estimates.get(lud, {}).get(feature_name, original_feature_values[lud_index])
                    for lud_index, lud in enumerate(train_luds)
                ])
                interpolated_feature_values = [
                    np.random.uniform(min(orig_val, reann_val), max(orig_val, reann_val))
                    for orig_val, reann_val
                    in zip(original_feature_values, feature_estimates)
                ]

                train_x.loc[:, feature_name] = interpolated_feature_values

        # INITIALIZE AND TRAIN TABM REGRESSOR
        regressor = TabMRegressor(**tabm_params, random_state=RANDOM_SEED)
        regressor.fit(
            X=train_x,
            y=train_y,
            sample_weight=sample_weights,
            eval_set=(test_x, test_y)
        )
        models.append(regressor)

        # GENERATE OOF PREDICTIONS.
        predictions = regressor.predict(test_x)
        oof_predictions[test_index] = predictions

    # CALCULATE BASE RMSE
    base_rmse = root_mean_squared_error(y, oof_predictions)

    # APPLY ISOTONIC REGRESSION
    isotonic_models, isotonic_rmse, isotonic_oof_preds = TrainIsotonicModels(
        oof_predictions, y, ruleset_names, fold_count, random_seed
    )

    print(f'Average RMSE (base, isotonic): {base_rmse:.5f}, {isotonic_rmse:.5f}')

    return models, isotonic_models, base_rmse, isotonic_rmse, oof_predictions, isotonic_oof_preds

def Objective(trial, fold_count: int, extra_train_paths: Dict, starting_eval_json_paths: List[str]) -> float:
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
    )

    extra_train_df = None
    extra_lud_rules = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
        )

    # tabm_params = {
    #     'arch_type': 'tabm',
    #     'backbone': {
    #         "type": "MLP",
    #         'n_blocks': trial.suggest_int('n_blocks', 1, 7),
    #         'd_block': trial.suggest_int('d_block', 64, 1024),
    #         'dropout': trial.suggest_float('dropout', 0.0, 0.5)
    #     },
    #     'k': 32,
    #     'learning_rate': trial.suggest_float('learning_rate', 5e-6, 5e-4, log=True),
    #     'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
    #     'clip_grad_norm': trial.suggest_categorical('clip_grad_norm', [True, False]),
    #     'max_epochs': 2000,
    #     'patience': 16,
    #     'batch_size': trial.suggest_int('batch_size', 48, 256, log=True),
    #     'compile_model': True,
    #     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    #     'verbose': False
    # }

    tabm_params = {
        # 'arch_type': 'tabm-mini',
        'arch_type': 'tabm',
        'backbone': {
            "type": "MLP",
            'n_blocks': trial.suggest_int('n_blocks', 2, 7),
            'd_block': trial.suggest_int('d_block', 256, 1280),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5)
        },
        'use_embeddings': True,
        'k': 32,
        'd_embedding': trial.suggest_int('d_embedding', 4, 48),
        'bin_count': trial.suggest_int('bin_count', 8, 64),
        'learning_rate': trial.suggest_float('learning_rate', 5e-6, 2e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 5e-6, 1e-1, log=True),
        'clip_grad_norm': True,
        'max_epochs': 2000,
        'patience': 16,
        'batch_size': trial.suggest_int('batch_size', 32, 256, log=True),
        'compile_model': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbose': False
    }

    models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
        # Organizer data.
        ruleset_names=ruleset_names,
        train_test_df=train_test_df, 
        starting_eval_json_paths=starting_eval_json_paths,
        lud_rules=lud_rules,
        # Extra data.
        extra_train_df=extra_train_df,
        extra_starting_eval_json_paths=extra_train_paths['starting_position_evals_json_paths'],
        extra_lud_rules=extra_lud_rules,
        # Everything else.
        feature_importances_dir=None,
        global_average_weight=0.5,
        dropped_feature_count=0,
        tabm_params=tabm_params, 
        fold_count=fold_count,
        random_seed=RANDOM_SEED
    )
    
    return isotonic_rmse

def SafeObjective(**kwargs):
    try:
        return Objective(**kwargs)
    except Exception as e:
        print(f'Error: {e}')
        return 1.0

def GetOptimalConfig(trial_count: int, extra_train_paths: Dict, starting_eval_json_paths: List[str]):
    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: SafeObjective(
            trial=trial, 
            fold_count=5, 
            extra_train_paths=extra_train_paths, 
            starting_eval_json_paths=starting_eval_json_paths
        ), 
        n_trials=trial_count
    )

    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=2))

    best_score = study.best_trial.value
    output_filepath = f'configs/tabm_{int(best_score * 100000)}.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

def GetOptimalLr(extra_train_paths: Dict, starting_eval_json_paths: List[str]):
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
    )

    extra_train_df = None
    extra_lud_rules = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
        )

    best_lr = None
    best_rmse = 99999
    for lr in [0.00011/3, 0.00011/1.7, 0.00011, 0.00011*1.7, 0.00011*3]:
        tabm_config = { # similar to cfg 2, but scaled down ~25% along multiple dimensions.
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                # 'n_blocks': 4,
                # 'd_block': 882,
                'n_blocks': 3,
                'd_block': 662,
                'dropout': 0.10232426156025509
            },
            'k': 32,
            # 'd_embedding': 9,
            'd_embedding': 7,
            'bin_count': 42,
            # 'learning_rate': 0.00011358084690023735,
            'learning_rate': lr,
            'weight_decay': 0.014363806955838766,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            # 'batch_size': 63,
            'batch_size': 64,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        }

        models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
            # Organizer data.
            ruleset_names=ruleset_names,
            train_test_df=train_test_df, 
            starting_eval_json_paths=starting_eval_json_paths,
            lud_rules=lud_rules,
            # Extra data.
            extra_train_df=extra_train_df,
            extra_starting_eval_json_paths=extra_train_paths['starting_position_evals_json_paths'],
            extra_lud_rules=extra_lud_rules,
            # Everything else.
            reannotated_features_path=None,
            feature_importances_dir=None,
            global_average_weight=0.5,
            dropped_feature_count=0,
            tabm_params=tabm_config, 
            fold_count=5,
            random_seed=RANDOM_SEED
        )

        print(f'LR: {lr}, Isotonic RMSE: {isotonic_rmse:.5f}')

        if isotonic_rmse < best_rmse:
            best_rmse = isotonic_rmse
            best_lr = lr
            print(f'New best LR: {best_lr}, Isotonic RMSE: {best_rmse:.5f}')

    print(f'Best LR: {best_lr}, Isotonic RMSE: {best_rmse:.5f}')

def SaveModels(tabm_models: List[TabMRegressor], isotonic_models: List, base_rmse: float, isotonic_rmse: float, base_oof_preds: np.ndarray, isotonic_oof_preds: np.ndarray, output_directory_suffix: str = ''):
    output_directory_path = f'models/tabm_iso_{int(base_rmse*100000)}_{int(isotonic_rmse*100000)}_{len(tabm_models)}{output_directory_suffix}'
    os.makedirs(output_directory_path, exist_ok=True)

    for fold_index, (tabm_model, isotonic_model) in enumerate(zip(tabm_models, isotonic_models)):
        tabm_model.device = 'cpu'
        tabm_model.model = tabm_model.model._orig_mod
        tabm_model.model = tabm_model.model.cpu()
        
        output_filepath = f'{output_directory_path}/{fold_index}.pkl'
        joblib.dump({
            'tabm_model': tabm_model,
            'isotonic_model': isotonic_model,
            'base_oof_preds': base_oof_preds,
            'isotonic_oof_preds': isotonic_oof_preds
        }, output_filepath)

def CreateEnsemble(
        tabm_params: Dict,
        fold_count: int,
        starting_eval_json_paths: List[str],
        extra_train_paths: Dict,
        extra_data_weight: float,
        reannotated_features_path: Optional[str],
        feature_importances_dir: Optional[str],
        dropped_feature_count: int,
        global_average_weight: float,
        output_directory_suffix: str = ''
    ) -> Tuple[float, float]:
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
    )

    extra_train_df = None
    extra_lud_rules = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path']
        )

    models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
        ruleset_names=ruleset_names,
        train_test_df=train_test_df, 
        starting_eval_json_paths=starting_eval_json_paths,
        lud_rules=lud_rules,
        extra_train_df=extra_train_df,
        extra_starting_eval_json_paths=extra_train_paths.get('starting_position_evals_json_paths', []),
        extra_lud_rules=extra_lud_rules,
        extra_data_weight=extra_data_weight,
        reannotated_features_path=reannotated_features_path,
        feature_importances_dir=feature_importances_dir,
        dropped_feature_count=dropped_feature_count,
        global_average_weight=global_average_weight,
        tabm_params=tabm_params, 
        fold_count=fold_count,
        random_seed=RANDOM_SEED
    )

    if output_directory_suffix is not None:
        SaveModels(models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix)

    return base_rmse, isotonic_rmse

'''
Results with R1-5:
    Baseline tabm:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 0
        Base RMSEs: [0.3952465291506045, 0.39416805034142677, 0.39202455301373557], (mean = 0.39381)
        Isotonic RMSEs: [0.3926164579954306, 0.39202315535044635, 0.38994180597329375], (mean = 0.39153)
    Baseline tabm-mini:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 0
        Base RMSEs: [0.4218808610786466, 0.40054372054993126, 0.3971259264816967], (mean = 0.40652)
        Isotonic RMSEs: [0.41500302049378424, 0.39550804025097147, 0.39517464742929986], (mean = 0.40190)
    Tuned (v1) tabm-mini:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 0
        Base RMSEs: [0.3662038285225992, 0.3700594384008505, 0.3656227598431819], (mean = 0.36730)
        Isotonic RMSEs: [0.36545008924421185, 0.36958424017505126, 0.36462465120108584], (mean = 0.36655)
    Tuned (v1) tabm:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 1
        Base RMSEs: [0.3882227197199079, 0.3843758645733568, 0.38378703309020534], (mean = 0.38546)
        Isotonic RMSEs: [0.3833816550691668, 0.3798465376569892, 0.38064230358917833], (mean = 0.38129)
    Tuned v1.1-1 tabm-mini:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2
        Base RMSEs: [0.35877757686085643, 0.3589976145320932, 0.35500040267707306], (mean = 0.35759)
        Isotonic RMSEs: [0.3585575143700912, 0.35876251210878246, 0.35484585166127147], (mean = 0.35739)
    Tuned v1.1-2 tabm-mini:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 3
        Base RMSEs: [0.36543920487855963, 0.3641945565650167, 0.3648133128459294], (mean = 0.36482)
        Isotonic RMSEs: [0.3620872921516898, 0.36067637410635867, 0.3607305142192521], (mean = 0.36116)
    Tuned v1.1 tabm:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 4
        Base RMSEs: [0.3713997651396825, 0.3709853165282678, 0.36982749242160184], (mean = 0.37074)
        Isotonic RMSEs: [0.3707587685698848, 0.37101223380445203, 0.3700326554945027], (mean = 0.37060)
Results with R1-10:
    Tuned v1.1-1 tabm-mini:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2
        Base RMSEs: [0.35981467100288655, 0.360947707850575, 0.3566114285443291], (mean = 0.35912)
        Isotonic RMSEs: [0.3598415157264444, 0.361063693952001, 0.3566180860577831], (mean = 0.35917)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2, gaw = 0.33
        Base RMSEs: [0.35786527219078335, 0.363638537967497, 0.358647823120711], (mean = 0.36005)
        Isotonic RMSEs: [0.3578334619007061, 0.3634050575711567, 0.35879657283461847], (mean = 0.36001)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2, gaw = 0.67
        Base RMSEs: [0.36026225653104244, 0.3617958615578252, 0.3567940525928649], (mean = 0.35962)
        Isotonic RMSEs: [0.3603273197503404, 0.3617390846482284, 0.3571626490406722], (mean = 0.35974)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2, gaw = 1
        Base RMSEs: [0.36026421152776805, 0.3606975734295167, 0.3547455199471198], (mean = 0.35857)
        Isotonic RMSEs: [0.3600843924864068, 0.36037483143267696, 0.3548518971729733], (mean = 0.35844)
    Tuned v2.0 tabm-mini:
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 5, gaw = 0
        Base RMSEs: [0.3577122915956563, 0.3577949127538691, 0.354103903299097], (mean = 0.35654)
        Isotonic RMSEs: [0.35730815981480507, 0.3576230125117356, 0.35405117206099185], (mean = 0.35633)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 5, gaw = 0.33
        Base RMSEs: [0.3549925001429751, 0.35805392632445415, 0.3542459639164629], (mean = 0.35576)
        Isotonic RMSEs: [0.35480776523318586, 0.35799554618508533, 0.35435581887670453], (mean = 0.35572)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 5, gaw = 0.67
        Base RMSEs: [0.35396879114076846, 0.359014748181839, 0.35246219179868926], (mean = 0.35515)
        Isotonic RMSEs: [0.3540193080616604, 0.3585380203553805, 0.3524649198149069], (mean = 0.35501)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 5, gaw = 1
        Base RMSEs: [0.3569629389768478, 0.35702380056205274, 0.3533843749508434], (mean = 0.35579)
        Isotonic RMSEs: [0.3566080656664331, 0.35691313451485684, 0.35332045655977456], (mean = 0.35561)
    Tuned v3.0 tabm-mini (cfg 6):
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 6, gaw = 0.33
        Base RMSEs: [0.3561829906744009, 0.3585290111477561, 0.35480388339854374], (mean = 0.35651)
        Isotonic RMSEs: [0.35592856030338144, 0.3585428322118935, 0.3549206294036884], (mean = 0.35646)
    Tuned v3.1 tabm (cfg 7):
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 7, gaw = 0.33
        Base RMSEs: [0.35568821226270153, 0.3598259900094788, 0.354922593072769], (mean = 0.35681)
        Isotonic RMSEs: [0.35562739599809234, 0.35983490738251905, 0.3552540676621877], (mean = 0.35691)
    Tuned v3.2 tabm (cfg 8):
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 8, gaw = 0.33
        Base RMSEs: [0.35448800806770864, 0.3566149205478659, 0.3533287753420433], (mean = 0.35481)
        Isotonic RMSEs: [0.35426342813975575, 0.356372488416494, 0.35281298266750083], (mean = 0.35448)
    Downscaled (cfg 9)
        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 9, gaw = 0.33
        Base RMSEs: [0.35975221422205217, 0.3599758689036973, 0.3561915790679184], (mean = 0.35864)
        Isotonic RMSEs: [0.35926905942070275, 0.3598813474983292, 0.356126301955238], (mean = 0.35843)

        MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 9, gaw = 0.33
        Base RMSEs: [0.35975221422205217, 0.3599758689036973, 0.3561915790679184], (mean = 0.35864)
        Isotonic RMSEs: [0.35926905942070275, 0.3598813474983292, 0.356126301955238], (mean = 0.35843)
    
0.6-random-true:
    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 2, gaw = 0.33
    Base RMSEs: [0.3588719048586135, 0.3594718857386933, 0.3566925632422761], (mean = 0.35835)
    Isotonic RMSEs: [0.35875377852276275, 0.3590232905314526, 0.35659565233787777], (mean = 0.35812)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 5, gaw = 0.33
    Base RMSEs: [0.35775821201902885, 0.3567170197388978, 0.35553624509446846], (mean = 0.35667)
    Isotonic RMSEs: [0.3574853998937234, 0.35651575960423254, 0.3555630791600889], (mean = 0.35652)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 6, gaw = 0.33
    Base RMSEs: [0.3560140340800971, 0.35835085161527175, 0.3535821053360394], (mean = 0.35598)
    Isotonic RMSEs: [0.35585988062441887, 0.35822526821177936, 0.3537254151577893], (mean = 0.35594)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 7, gaw = 0.33
    Base RMSEs: [0.35629024343916965, 0.3553884879475882, 0.3542897227968059], (mean = 0.35532)
    Isotonic RMSEs: [0.35617910506623046, 0.3553933693526324, 0.354221241457067], (mean = 0.35526)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.35401420889696317, 0.3549222821958026, 0.3545418650231083], (mean = 0.35449)
    Isotonic RMSEs: [0.3538046782413894, 0.3546646642207474, 0.35457328392336096], (mean = 0.35435)

v4 extra, drop:
    MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2, gaw = 0.33
    Base RMSEs: [0.36057840374633804, 0.36110327959473193, 0.3564407469835267], (mean = 0.35937)
    Isotonic RMSEs: [0.3605374614639323, 0.3610051988293322, 0.35655983394119933], (mean = 0.35937)

    MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.3551724116382445, 0.35521014610839424, 0.3509144579737477], (mean = 0.35377)
    Isotonic RMSEs: [0.35454790045226686, 0.35492298150382934, 0.35095421205927585], (mean = 0.35348)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 2, gaw = 0.33
    Base RMSEs: [0.35784366664092565, 0.35743826213831276, 0.35554083407305787], (mean = 0.35694)
    Isotonic RMSEs: [0.35759007291677214, 0.3573221944245141, 0.35560277213674535], (mean = 0.35684)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.35250073788594405, 0.3538610405894692, 0.3504457867765034], (mean = 0.35227)
    Isotonic RMSEs: [0.3521882532767669, 0.3537848072048682, 0.35036787274012093], (mean = 0.35211)

v6 extra:
    MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 2, gaw = 0.33
    Base RMSEs: [0.358957230566543, 0.3607156393805924, 0.355946752594644], (mean = 0.35854)
    Isotonic RMSEs: [0.35889740227293, 0.3606288178240423, 0.35588811869075593], (mean = 0.35847)

    MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.3537845584536741, 0.35596410091427605, 0.3515889381353037], (mean = 0.35378)
    Isotonic RMSEs: [0.35341826924732755, 0.35547096917588916, 0.3515200929764892], (mean = 0.35347)

    MCTS config = 1.41421356237-random-false, runtime = 15s, tabm config = 9, gaw = 0.33
    Base RMSEs: [0.3574966216574463, 0.3602423684193523, 0.35505538469167314], (mean = 0.35760)
    Isotonic RMSEs: [0.3574117405358195, 0.3599757954042374, 0.3552019446400898], (mean = 0.35753)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 2, gaw = 0.33
    Base RMSEs: [0.3592807726541724, 0.3565321703961481, 0.35797476469317513], (mean = 0.35793)
    Isotonic RMSEs: [0.3596039889554804, 0.35649124093640827, 0.35824460087236115], (mean = 0.35811)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.3561584209244662, 0.35350716082875167, 0.34990764279646064], (mean = 0.35319)
    Isotonic RMSEs: [0.3557642461631852, 0.353384533460651, 0.3498314728118129], (mean = 0.35299)

    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 9, gaw = 0.33
    Base RMSEs: [0.3577119799061156, 0.3567738084396536, 0.35348895221494764], (mean = 0.35599)
    Isotonic RMSEs: [0.35765205071828693, 0.3565715055763425, 0.35317803909848855], (mean = 0.35580)

v6 extra, drop:
    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.35435538227547214, 0.35200016145173185, 0.35076934002632015], (mean = 0.35237)
    Isotonic RMSEs: [0.354133879768583, 0.35178966911397663, 0.3506435679319484], (mean = 0.35219)

v6 extra, drop, rann:
    ???
    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33
    Base RMSEs: [0.35367921409860265, 0.35214161653517323, 0.3510429460803356], (mean = 0.35229)
    Isotonic RMSEs: [0.35308394161842427, 0.35197021401068695, 0.35089588787729753], (mean = 0.35198)  

    extra weight 0.25:
        MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33, w25
        Base RMSEs: [0.35286839841277146, 0.3524143113946882, 0.3504319466947389], (mean = 0.35190)
        Isotonic RMSEs: [0.3527019378544505, 0.35230339988843495, 0.3500660463198945], (mean = 0.35169)
    extra weight 0.5:
        MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33, w50
        Base RMSEs: [0.3526643205122711, 0.3528565734066571, 0.3491910930618002], (mean = 0.35157)
        Isotonic RMSEs: [0.3522877608231361, 0.3526600311367996, 0.3491454070663348], (mean = 0.35136) <---- Current best!
    extra weight 0.75:
        MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33, w75
        Base RMSEs: [0.3542358857798285, 0.35189149648156653, 0.3495675935651545], (mean = 0.35190)
        Isotonic RMSEs: [0.35379203265714737, 0.35189125342841276, 0.349389667967975], (mean = 0.35169) 
    extra weight 1.0:
        MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33, w100
        Base RMSEs: [0.3529996458589043, 0.3539409462587774, 0.34974038461542095], (mean = 0.35223)
        Isotonic RMSEs: [0.3529014071117783, 0.3535982399725788, 0.34974229312442795], (mean = 0.35208)

v6 extra (weight 0.5), drop, reann (v2):
    MCTS config = 0.6-random-true, runtime = 15s, tabm config = 8, gaw = 0.33, w50, seed = 5555
    Base RMSE: 0.35086, Isotonic RMSE: 0.35088
'''
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    #'''
    TABM_CONFIGS = [
        { # Tuned 50 iter, v1
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                'n_blocks': 4,
                'd_block': 647,
                'dropout': 0.08085360103412836
            },
            'k': 32,
            'd_embedding': 21,
            'bin_count': 25,
            'learning_rate': 2.147334304120347e-05,
            'weight_decay': 0.013576153216684277,
            'clip_grad_norm': False,
            'max_epochs': 1000,
            'patience': 16,
            'batch_size': 121,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Tuned 50 iter, v1
            'arch_type': 'tabm',
            'backbone': {
                "type": "MLP",
                'n_blocks': 5,
                'd_block': 505,
                'dropout': 0.30155489826373166
            },
            'k': 32,
            'learning_rate': 2.147334304120347e-05,
            'weight_decay': 6.199086510189844e-05,
            'clip_grad_norm': False,
            'max_epochs': 1000,
            'patience': 16,
            'batch_size': 118,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned, v1.1
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                'n_blocks': 4,
                'd_block': 882,
                'dropout': 0.10232426156025509
            },
            'k': 32,
            'd_embedding': 9,
            'bin_count': 42,
            'learning_rate': 0.00011358084690023735,
            'weight_decay': 0.014363806955838766,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 63,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned, v1.1
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                'n_blocks': 2,
                'd_block': 540,
                'dropout': 0.0842408932694782
            },
            'k': 32,
            'd_embedding': 22,
            'bin_count': 16,
            'learning_rate': 7.034652762523713e-06,
            'weight_decay': 0.005235037527585988,
            'clip_grad_norm': False,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 52,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned v1.1
            'arch_type': 'tabm',
            'backbone': {
                "type": "MLP",
                'n_blocks': 3,
                'd_block': 950,
                'dropout': 0.025798488014042095
            },
            'k': 32,
            'learning_rate': 0.00010790653724821627,
            'weight_decay': 0.00015343173903032935,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 57,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned, v2.0
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                'n_blocks': 5,
                'd_block': 927,
                'dropout': 0.13358105651120605,
            },
            'k': 32,
            'd_embedding': 24,
            'bin_count': 17,
            'learning_rate': 3.5570080146315535e-05,
            'weight_decay': 5.043402866987622e-05,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 51,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned, v3.0
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                'n_blocks': 6,
                'd_block': 1217,
                'dropout': 0.10172392466648286,
            },
            'k': 32,
            'd_embedding': 21,
            'bin_count': 60,
            'learning_rate': 2.172546790276922e-05,
            'weight_decay': 0.012554079985897347,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 58,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned, v3.1
            'arch_type': 'tabm',
            'backbone': {
                "type": "MLP",
                'n_blocks': 5,
                'd_block': 1246,
                'dropout': 0.22858793394852134,
            },
            'k': 32,
            'd_embedding': 11,
            'bin_count': 48,
            'learning_rate': 2.1781230928710606e-05,
            'weight_decay': 7.74289466114024e-06,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 32,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # Retuned, v3.2
            'arch_type': 'tabm',
            'backbone': {
                "type": "MLP",
                'n_blocks': 4,
                'd_block': 1171,
                'dropout': 0.20853880769549843,
            },
            # 'k': 32,
            'k': 64,
            'd_embedding': 11,
            'bin_count': 16,
            'learning_rate': 4.2334241624676516e-05,
            'weight_decay': 1.267098894123156e-05,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 35,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        },
        { # similar to cfg 2, but scaled down ~25% along multiple dimensions.
            'arch_type': 'tabm-mini',
            'backbone': {
                "type": "MLP",
                'n_blocks': 3,
                'd_block': 662,
                'dropout': 0.10232426156025509
            },
            'k': 32,
            'd_embedding': 7,
            'bin_count': 42,
            'learning_rate': 6.47e-5,
            'weight_decay': 0.014363806955838766,
            'clip_grad_norm': True,
            'max_epochs': 2000,
            'patience': 16,
            'batch_size': 64,
            'compile_model': True,
            'device': 'cuda',
            'verbose': False
        }
    ]

    MCTS_CONFIG_NAMES = [
        # '1.41421356237-random-false',
        '0.6-random-true',
    ]
    MCTS_RUNTIMES_SEC = [15]
    GLOBAL_AVERAGE_WEIGHTS = [0.33]
    for mcts_config_name in MCTS_CONFIG_NAMES:
        for mcts_runtime_sec in MCTS_RUNTIMES_SEC:
            for global_average_weight in GLOBAL_AVERAGE_WEIGHTS:
                for tabm_config_index, tabm_params in enumerate(TABM_CONFIGS):
                    for extra_data_weight in [0.5]:
                        if tabm_config_index not in [8]:
                            continue

                        updated_extra_train_paths = {
                            # 'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv', # V4
                            # 'starting_position_evals_json_paths': [
                            #     f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                            #     for i in range(10)
                            # ]
                            'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-11-25_21-41-25.csv', # V6
                            'starting_position_evals_json_paths': [
                                f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/merged_extra_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                                for i in range(10)
                            ]
                        }
                        updated_starting_eval_json_paths = [
                            f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                            for i in range(10)
                        ]
                        reannotated_features_path = 'data/reannotation/lud_to_features_to_estimates_v2.0.json'

                        base_rmses, isotonic_rmses = [], []
                        # for RANDOM_SEED in [3333, 4444, 5555]:
                        # for RANDOM_SEED in [4444, 3333]:
                        for RANDOM_SEED in [5555]:
                            base_rmse, isotonic_rmse = CreateEnsemble(
                                tabm_params = tabm_params,
                                fold_count = 10,
                                starting_eval_json_paths = updated_starting_eval_json_paths,
                                extra_train_paths = updated_extra_train_paths,
                                extra_data_weight = extra_data_weight,
                                reannotated_features_path = reannotated_features_path,
                                feature_importances_dir = None,
                                dropped_feature_count = 0,
                                global_average_weight = global_average_weight,
                                # output_directory_suffix = f'_et_v6_w{int(extra_data_weight*100)}_{mcts_config_name}_{mcts_runtime_sec}s_cfg{tabm_config_index}_seed{RANDOM_SEED}_r1-10_gaw{global_average_weight:.3f}_reann_drop'.replace('.', '')
                                output_directory_suffix = f'_et_v6_w{int(extra_data_weight*100)}_{mcts_config_name}_{mcts_runtime_sec}s_cfg{tabm_config_index}_k64_seed{RANDOM_SEED}_r1-10_gaw{global_average_weight:.3f}_reann_drop'.replace('.', '')
                            )
                            base_rmses.append(base_rmse)
                            isotonic_rmses.append(isotonic_rmse)

                            print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, tabm config = {tabm_config_index}, gaw = {global_average_weight}, w{int(extra_data_weight*100)}, seed = {RANDOM_SEED}')
                            print(f'Base RMSE: {base_rmse:.5f}, Isotonic RMSE: {isotonic_rmse:.5f}\n')

                        print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, tabm config = {tabm_config_index}, gaw = {global_average_weight}, w{int(extra_data_weight*100)}')
                        print(f'Base RMSEs: {base_rmses}, (mean = {np.mean(base_rmses):.5f})')
                        print(f'Isotonic RMSEs: {isotonic_rmses}, (mean = {np.mean(isotonic_rmses):.5f})\n')
    #'''