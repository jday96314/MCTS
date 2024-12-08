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
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression
import pickle

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

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
        reannotated_features_path,
        feature_importances_dir,
        dropped_feature_count,
        lgb_params, 
        early_stopping_round_count, 
        fold_count, 
        extra_data_weight = 0.75,
        random_seed = None):
    models = []
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

        train_luds = lud_rules.iloc[train_index]
        sample_weights = np.ones(len(train_x))
        if extra_train_df is not None:
            if extra_train_X.columns.tolist() != train_x.columns.tolist():
                extra_training_run_eval_paths = extra_starting_eval_json_paths[:heldout_run_index] + extra_starting_eval_json_paths[heldout_run_index+1:]
                extra_train_X = AddStartingEvalFeatures(extra_training_run_eval_paths, extra_train_X, extra_lud_rules, random_seed)
                extra_train_X = extra_train_X[train_x.columns]

            train_x = pd.concat([train_x, extra_train_X], ignore_index=True)
            train_y = pd.concat([train_y, extra_train_y], ignore_index=True)
            train_luds = pd.concat([train_luds, extra_lud_rules], ignore_index=True)
            sample_weights = np.concatenate([sample_weights, np.ones(len(extra_train_X)) * extra_data_weight])

        # DROP FEATURES.
        if feature_importances_dir is not None:
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

        # TRAIN MODEL.
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            train_x,
            train_y,
            sample_weight=sample_weights,
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

    return models, isotonic_models, base_rmse, isotonic_rmse, oof_predictions, isotonic_oof_preds

def Objective(trial, fold_count, extra_train_paths, starting_eval_json_paths):
    GAMES_FILEPATH = 'data/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAMES_FILEPATH, 
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
        )

    # lgb_params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 3000, 40000, log=True),
    #     'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True), # TODO: Lower upper bound to 1e-1, lower lower bound to 5e-4
    #     'num_leaves': trial.suggest_int('num_leaves', 10, 384), # TODO: Try increasing upper & lower bounds (64 - 1024)!
    #     'max_depth': trial.suggest_int('max_depth', 10, 25),
    #     'min_child_samples': trial.suggest_int('min_child_samples', 5, 200, log=True),
    #     'subsample': trial.suggest_float('subsample', 0.25, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # TODO: Try locking to 1.0!
    #     'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0), # TODO: Try locking to 1.0!
    #     'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1e-2, log=True), # TODO: Try locking to 4e-7!
    #     'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1e-4, log=True), # TODO: Try locking to 7e-5!
    #     'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
    #     "verbose": -1
    # }

    lgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 3000, 35000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 2e-3, 2e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 4000, log=True),
        'max_depth': trial.suggest_int('max_depth', 10, 35),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 200, log=True),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 10, log=True),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
        'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2, log=True),
        'max_drop': trial.suggest_int('max_drop', 30, 70),
        'skip_drop': trial.suggest_float('skip_drop', 0.25, 0.75),
        'boosting': 'dart',
        "verbose": -1
    }
    
    # lgb_params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 3000, 35000, log=True),
    #     'learning_rate': trial.suggest_float('learning_rate', 5e-4, 4e-1, log=True),
    #     'num_leaves': trial.suggest_int('num_leaves', 64, 40000, log=True),
    #     'max_depth': trial.suggest_int('max_depth', 10, 25),
    #     'min_child_samples': trial.suggest_int('min_child_samples', 5, 200, log=True),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
    #     'colsample_bynode': trial.suggest_float('colsample_bynode', 0.7, 1.0),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1e-1, log=True),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1e-1, log=True),
    #     'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
    #     'data_sample_strategy': 'goss',
    #     'verbose': -1,
    # }

    early_stopping_round_count = 50
    lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
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
        reannotated_features_path = 'data/reannotation/lud_to_features_to_estimates_v1.json',
        feature_importances_dir=None,
        dropped_feature_count=0,
        lgb_params = lgb_params, 
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

def SaveModels(gbdt_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix = ''):
    output_directory_path = f'models/lgbm_iso_{int(base_rmse*100000)}_{int(isotonic_rmse*100000)}_{len(gbdt_models)}{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, (gbdt_model, isotonic_model) in enumerate(zip(gbdt_models, isotonic_models)):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump({
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

def CreateEnsemble(
        lgb_params,
        early_stopping_round_count,
        fold_count,
        starting_eval_json_paths,
        extra_train_paths,
        extra_data_weight,
        reannotated_features_path,
        feature_importances_dir,
        dropped_feature_count,
        output_directory_suffix = ''):
    GAMES_FILEPATH = 'data/train.csv'
    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAMES_FILEPATH, 
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path']
        )

    lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
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
        reannotated_features_path,
        feature_importances_dir,
        dropped_feature_count,
        lgb_params, 
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
        random_seed = RANDOM_SEED,
        extra_data_weight = extra_data_weight
    )

    if output_directory_suffix is not None:
        SaveModels(lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix)

    return base_rmse, isotonic_rmse

'''
Without dropping features (base, isotonic): 0.37256, 0.37083
350 dropped features (base, isotonic): 0.37454, 0.37318
300 dropped features (base, isotonic): 0.37270, 0.37102
200 dropped features (base, isotonic): 0.37253, 0.37085
100 dropped features (base, isotonic): 0.37219, 0.37054
50 dropped features (base, isotonic): 0.37303, 0.37140
0 dropped features (base, isotonic): 0.37200, 0.37025

5-fold tuning:
    Trial 63 finished with value: 0.37342094017745264 and parameters: {'n_estimators': 19246, 'learning_rate': 0.0028224515150795885, 'num_leaves': 365, 'max_depth': 14, 'min_child_samples': 55, 'subsample': 0.585026292470321, 'colsample_bytree': 0.9886746573495085, 'colsample_bynode': 0.9557863173491425, 'reg_alpha': 4.530707764807948e-07, 'reg_lambda': 2.5292981163243776e-05, 'extra_trees': True}. Best is trial 63 with value: 0.37342094017745264.

Seed 6666, 10-fold, 16s local:
    Baseline hparams: 0.3704707462865228 0.36840620035544325
    V1 tuned: 0.3701773450000877 0.3676097397782373
Seed 6666, 10-fold, 15s kaggle:
    Baseline hparams: 0.3734164637546649 0.37142235635296106
    V1 tuned: 0.3721755577025514 0.36964330045367577
    V2 tuned: 0.3733772796191778 0.3703289209034208

DART: Trial 0 finished with value: 0.37356262055946937 and parameters: 
{'n_estimators': 9648, 'learning_rate': 0.08036455460239479, 'num_leaves': 99, 'max_depth': 2
4, 'min_child_samples': 8, 'bagging_freq': 0, 'subsample': 0.9845338171021742, 'colsample_byt
ree': 0.8357840074927564, 'colsample_bynode': 0.9035812734166077, 'reg_alpha': 9.817738265024
515e-05, 'reg_lambda': 3.582329308097412e-06, 'extra_trees': True}


MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 1
Base RMSEs: [0.37496335799404334, 0.37229433518531063, 0.37349463002097866], (mean = 0.37358)
Isotonic RMSEs: [0.3725412360499901, 0.3701664870976773, 0.37116337119048665], (mean = 0.37129)

MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
Base RMSEs: [0.3704139585068627, 0.3699010184736979, 0.3686159730017891], (mean = 0.36964)
Isotonic RMSEs: [0.36838183814233233, 0.3681249631924698, 0.3669188195923442], (mean = 0.36781)


MCTS config = 1.41421356237-random-false, runtime = 30s, lgbm config = 1
Base RMSEs: [0.37371857908209755, 0.370801215526227, 0.37139587051715256], (mean = 0.37197)
Isotonic RMSEs: [0.3714132765205723, 0.3685964423054781, 0.36923798238078764], (mean = 0.36975)

MCTS config = 1.41421356237-random-false, runtime = 30s, lgbm config = 2
Base RMSEs: [0.3683094629488465, 0.36715675992046937, 0.36612930464508037], (mean = 0.36720)
Isotonic RMSEs: [0.3662888402017566, 0.3652308906807269, 0.3642693040387789], (mean = 0.36526)

Baseline features, dart cfg 2:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2 (v2 data, R1 - 5, no leak)
    Base RMSEs: [0.36639664721275506, 0.36550164395783297, 0.3643335701201239], (mean = 0.36541)
    Isotonic RMSEs: [0.3639285640333119, 0.3630308462619219, 0.36196385765452455], (mean = 0.36297)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2  (v2 data, R1 - 10, no leak, global average weight 0.0)
    Base RMSEs: [0.3664237016037175, 0.3682086409024809, 0.3644426502529867], (mean = 0.36636)
    Isotonic RMSEs: [0.363718017259187, 0.36614680756900064, 0.3618976628525915], (mean = 0.36392)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2   (v2 data, R1 - 10, no leak, global average weight 0.33)
    Base RMSEs: [0.36496660579778334, 0.36702187541984377, 0.3639442469972603], (mean = 0.36531)
    Isotonic RMSEs: [0.3623454299942668, 0.3649168171232435, 0.3617535464592936], (mean = 0.36301)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2 (v2 data, R1 - 10, no leak, global average weight 0.5)
    Base RMSEs: [0.3655078829291414, 0.3675059915550361, 0.3628213893673742], (mean = 0.36528)
    Isotonic RMSEs: [0.36299815276819264, 0.36549700044716615, 0.36060617170187687], (mean = 0.36303)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2 (v2 data, R1 - 10, no leak, global average weight 0.67)
    Base RMSEs: [0.36494771239121715, 0.3672585551615869, 0.3630780776394755], (mean = 0.36509)
    Isotonic RMSEs: [0.3624188062298531, 0.3653259561549105, 0.36090118785898223], (mean = 0.36288)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2 (v2 data, R1 - 10, no leak, global average weight 0.75)
    Base RMSEs: [0.3641598218137354, 0.3683509206382285, 0.36261287336521214], (mean = 0.36504)
    Isotonic RMSEs: [0.3617382103720705, 0.366631331918356, 0.3604643960966257], (mean = 0.36294)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2 (v2 data, R1 - 10, no leak, global average weight 1.0)
    Base RMSEs: [0.36475376608411564, 0.3681099486200342, 0.36343935431154634], (mean = 0.36543)
    Isotonic RMSEs: [0.3625672424589828, 0.3663454780091032, 0.3615399679526178], (mean = 0.36348)
Baseline features, baseline data (v4 et), no rean, dart cfg 4:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 4 (gaw 0.33)
    Base RMSEs: [0.36497813791630423, 0.36706505535399814, 0.3622879347296005], (mean = 0.36478)
    Isotonic RMSEs: [0.36191733317298763, 0.3646905357986075, 0.3594557911540868], (mean = 0.36202)

    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 4
    Base RMSEs: [0.3648652318658653, 0.36361545690356845, 0.36176562576850235], (mean = 0.36342)
    Isotonic RMSEs: [0.36198975880872086, 0.36099761239549427, 0.35869105348701413], (mean = 0.36056)
Baseline features, baseline data (v4 et), no rean, dart cfg 5:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 5
    Base RMSEs: [0.364319714638409, 0.36561248876389174, 0.3618237440693743)], (mean = 0.36392)
    Isotonic RMSEs: [0.3613057458826853, 0.362579359865667, 0.35900648652168643)], (mean = 0.36096)

    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 5
    Base RMSEs: [0.3634955889109299, 0.36268236190013853, 0.35976112419120126)], (mean = 0.36198)
    Isotonic RMSEs: [0.3607500668221658, 0.3597284009354456, 0.35688975687727903)], (mean = 0.35912)

Baseline features, non-dart:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 1
    Base RMSEs: [0.36987318794373564, 0.3702228858023979, 0.3677219868501488], (mean = 0.36927)
    Isotonic RMSEs: [0.36673163177021934, 0.3676524900260456, 0.3648264311897084], (mean = 0.36640)
Added DurationToComplexityRatio feature, dart:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2 (v2 data, R1 - 10, no leak, global average weight 0.33)
    Base RMSEs: [0.3652696856504791, 0.3680033260705, 0.3638217332356744], (mean = 0.36570)
    Isotonic RMSEs: [0.3626786143733254, 0.36600209546501405, 0.3615255756857543], (mean = 0.36340)
LSA, 20 features:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.3672096365999978, 0.3692311861482785, 0.3653783637582157], (mean = 0.36727)
    Isotonic RMSEs: [0.3644188429713605, 0.36692588021082273, 0.36287743708599146], (mean = 0.36474)

15 vs. 30s comparison:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 1
    Base RMSEs: [0.3699922681471779, 0.3701040695733734, 0.36756362520603747], (mean = 0.36922)
    Isotonic RMSEs: [0.3668127338197088, 0.36753987744356664, 0.36468390139908735], (mean = 0.36635)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.36597428706039514, 0.36823436656443986, 0.3640805775552879], (mean = 0.36610)
    Isotonic RMSEs: [0.3633958734791879, 0.36628595657808904, 0.3618485466112621], (mean = 0.36384)

    MCTS config = 1.41421356237-random-false, runtime = 30s, lgbm config = 1
    Base RMSEs: [0.3687436558136597, 0.365663558400244, 0.36668084286713004], (mean = 0.36703)
    Isotonic RMSEs: [0.3659745506706996, 0.3630201481090075, 0.36395090181756806], (mean = 0.36432)

    MCTS config = 1.41421356237-random-false, runtime = 30s, lgbm config = 2
    Base RMSEs: [0.3654577660393142, 0.3638242598394948, 0.36163990911496746], (mean = 0.36364)
    Isotonic RMSEs: [0.36344473055322746, 0.36179699100619617, 0.359438458550506], (mean = 0.36156)

No reannotation aug:
    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.36566829579769766, 0.36541880175042285, 0.3628668500061629], (mean = 0.36465)
    Isotonic RMSEs: [0.36324668113858116, 0.3632873217694533, 0.3604525090991003], (mean = 0.36233)

Reannotation aug:
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 1
    Base RMSEs: [0.3707886172472522, 0.37148955982274623, 0.3687055764591082], (mean = 0.37033)
    Isotonic RMSEs: [0.36693943697462617, 0.36825740492955666, 0.3648757001422403], (mean = 0.36669)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.3653042873219851, 0.36685055542205214, 0.3627259106943674], (mean = 0.36496)
    Isotonic RMSEs: [0.3622992557015859, 0.36438489599858503, 0.35966142518199545], (mean = 0.36212)

    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 1
    Base RMSEs: [0.3702732229520228, 0.36813140001392414, 0.3685067774024226], (mean = 0.36897)
    Isotonic RMSEs: [0.36666976538154356, 0.36471439101667114, 0.3648082007777208], (mean = 0.36540)

Reannotation aug & v6 extra data (weight 1.0) <-- Polluted with drop!
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.3619663767500515, 0.36500381410080657, 0.3591007883350011], (mean = 0.36202)
    Isotonic RMSEs: [0.3592531707609297, 0.3628061623656233, 0.35643484868393627], (mean = 0.35950)
    
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 4
    Base RMSEs: [0.3608590470793635, 0.36383881262626233, 0.3586138891140533], (mean = 0.36110)
    Isotonic RMSEs: [0.35765289085660473, 0.3609685157748907, 0.35535834679516093], (mean = 0.35799)

Reannotation aug & v6 extra data (weight 1.0) (no drop):
    1.4-rand-false:
        MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
        Base RMSEs: [0.36222835467108205, 0.3663644338436198, 0.3605417636163132], (mean = 0.36304)
        Isotonic RMSEs: [0.3595120754788619, 0.36423633351626605, 0.35782908569566335], (mean = 0.36053)

        MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 4
        Base RMSEs: [0.3618380349875508, 0.3648825761770343, 0.35952001670074707], (mean = 0.36208)
        Isotonic RMSEs: [0.3586034541658844, 0.36222197258094657, 0.3563854327985287], (mean = 0.35907)

        MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 5
        Base RMSEs: [0.3607519027805396, 0.3641526015102385, 0.35809327104945726], (mean = 0.36100)
        Isotonic RMSEs: [0.35760964615323615, 0.3610759058592135, 0.3548156783995254], (mean = 0.35783)
    0.6-random-true:
        MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 2
        Base RMSEs: [0.364091578586942, 0.3628148670294087, 0.35888410389399383)], (mean = 0.36193)
        Isotonic RMSEs: [0.36160801339139564, 0.3605721057060049, 0.3559041251218561)], (mean = 0.35936)

        MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 4
        Base RMSEs: [0.3625369802863454, 0.36218603113087466, 0.3592436741586572)], (mean = 0.36132)
        Isotonic RMSEs: [0.3594477423306159, 0.35947924636613415, 0.35582063094599115)], (mean = 0.35825)

        MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 5
        Base RMSEs: [0.3609592526689469, 0.3616421344438106, 0.35749636808907315)], (mean = 0.36003)
        Isotonic RMSEs: [0.35791963657794873, 0.35857211745263745, 0.3541215643819819)], (mean = 0.35687)

Reannotation aug & v6 extra data (weight 0.75):
    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 1
    Base RMSEs: [0.3681700011611952, 0.3699693181215685, 0.36683722587412954], (mean = 0.36833)
    Isotonic RMSEs: [0.36467397008731234, 0.36688373454881035, 0.3633080904920859], (mean = 0.36496)

    MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.362221829728131, 0.36654672617108863, 0.36060256560633447], (mean = 0.36312)
    Isotonic RMSEs: [0.35951377354639347, 0.3643722914984525, 0.3578849659621126], (mean = 0.36059)

    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 1
    Base RMSEs: [0.3676609445450944, 0.36714400985046036, 0.36643949872109666], (mean = 0.36708)
    Isotonic RMSEs: [0.3643062566767433, 0.36397652670401026, 0.36294799615559753], (mean = 0.36374)

    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 2
    Base RMSEs: [0.36357482090683746, 0.36244182031158667, 0.35872607993579214], (mean = 0.36158)
    Isotonic RMSEs: [0.3611149803538588, 0.36017960065254356, 0.3557106798936691], (mean = 0.35900)

Drop features (v4 data, no rean)
    1.4-rand-false:
        MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 2
        Base RMSEs: [0.3645698563906922, 0.36714361807947793, 0.3620125218130986], (mean = 0.36458)
        Isotonic RMSEs: [0.361906957434545, 0.36519669343115546, 0.3594823012357228], (mean = 0.36220)

        MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 4
        Base RMSEs: [0.36462380452082327, 0.3668008544879338, 0.36131998969836543], (mean = 0.36425)
        Isotonic RMSEs: [0.3615599863055162, 0.3643386909678779, 0.3584205904553101], (mean = 0.36144)

        MCTS config = 1.41421356237-random-false, runtime = 15s, lgbm config = 5
        Base RMSEs: [0.3640606118508367, 0.3654666022046223, 0.36169829716100443], (mean = 0.36374)
        Isotonic RMSEs: [0.36098330372306664, 0.36246174954667876, 0.3589147402315958], (mean = 0.36079)
    0.6-rand-true:
        MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 2
        Base RMSEs: [0.36507624731396954, 0.36392397442889657, 0.362229585288514)], (mean = 0.36374)
        Isotonic RMSEs: [0.3627293897943781, 0.36155070780099635, 0.3596156107276247)], (mean = 0.36130)

        MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 4
        Base RMSEs: [0.36373634184260006, 0.36351480744550285, 0.36156882935355494)], (mean = 0.36294)
        Isotonic RMSEs: [0.36084447285679355, 0.3608118955518223, 0.35846459652732965)], (mean = 0.36004)

        MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 5
        Base RMSEs: [0.3631382009931156, 0.3623829239201405, 0.35978228933844986)], (mean = 0.36177)
        Isotonic RMSEs: [0.36021682469101207, 0.3592974619324204, 0.3569056582635824)], (mean = 0.35881)

v6 w100, v2 rean, no drop:
    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 5
    Base RMSEs: [0.3613769950441511, 0.362208002007767, 0.359359696268668], (mean = 0.36098)
    Isotonic RMSEs: [0.3583588596636742, 0.3591379294643321, 0.356099913975488], (mean = 0.35787)

v6 w100, v2 rean, drop:
    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 5
    Base RMSEs: [0.361026115892951, 0.3612545290153748, 0.3588999418527611)], (mean = 0.36039)
    Isotonic RMSEs: [0.35790491431194343, 0.3581264250220767, 0.35562902713447986)], (mean = 0.35722)

v6 w100, drop, no reann:
    MCTS config = 0.6-random-true, runtime = 15s, lgbm config = 5
    Base RMSEs: [0.36094447945688224, 0.3604770640553028, 0.358477369018568)], (mean = 0.35997)
    Isotonic RMSEs: [0.3581806184826899, 0.3575723688635487, 0.3557628102611211)], (mean = 0.35717)     <-- Current best!
'''

# Trial 5 finished with value: 0.3626934091400635 and parameters: {'n_estimators': 17154, 'learning_rate': 0.01076762389148406, 'num_leaves': 135, 'max_depth': 35, 'min_child_samples': 1, 'bagging_freq': 1, 'subsample': 0.710350970545652, 'colsample_bytree': 0.9633806028912855, 'colsample_bynode': 0.7927949738988681, 'reg_alpha': 1.2531297021026286e-07, 'reg_lambda': 0.06939652139576243, 'extra_trees': False, 'drop_rate': 0.15224913962956735, 'max_drop': 51, 'skip_drop': 0.5391263431327021}. Best is trial 5 with value: 0.3626934091400635.

# Trial 26 finished with value: 0.36658608689532535 and parameters: {'n_estimators': 6632, 'learning_rate': 0.045809796256298725, 'num_leaves': 74, 'max_depth': 12, 'min_child_samples': 1, 'bagging_freq': 0, 'subsample': 0.8376938414998899, 'colsample_bytree': 0.9962873138360554, 'colsample_bynode': 0.8809473711574799, 'reg_alpha': 3.610729042499901e-05, 'reg_lambda': 0.000421651169623802, 'extra_trees': True, 'drop_rate': 0.14590114126109052, 'max_drop': 43, 'skip_drop': 0.7497039946828294}. Best is trial 26 with value: 0.36658608689532535
if __name__ == '__main__':
    # Running 9950x
    '''
    extra_train_paths = {
        'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
        'starting_position_evals_json_paths': [
            f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
            for i in range(10)
        ]
    }
    starting_eval_json_paths = [
        f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
        for i in range(10)
    ]

    RANDOM_SEED = 574393
    GetOptimalConfig(50, extra_train_paths, starting_eval_json_paths)
    '''

    # Running 5950x
    '''
    extra_train_paths = {
        'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
        'starting_position_evals_json_paths': [
            f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-0.6-random-true_15s_v2_r{i+1}.json'
            for i in range(10)
        ]
    }
    starting_eval_json_paths = [
        f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-0.6-random-true_15s_v2_r{i+1}.json'
        for i in range(10)
    ]

    RANDOM_SEED = 1634938
    GetOptimalConfig(50, extra_train_paths, starting_eval_json_paths)
    '''
    
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
        },
        { # DART, tuned 50 iter with 0.6-random-true & v4 extra data.
            "n_estimators": 9388,
            "learning_rate": 0.027388674932049363,
            "num_leaves": 90,
            "max_depth": 13,
            "min_child_samples": 6,
            "bagging_freq": 0,
            "subsample": 0.7857351408252482,
            "colsample_bytree": 0.9937794180043592,
            "colsample_bynode": 0.8486085912576853,
            "reg_alpha": 0.0002720767697111974,
            "reg_lambda": 3.476655628939002e-05,
            "extra_trees": True,
            "drop_rate": 0.16079296905977924,
            "max_drop": 43,
            "skip_drop": 0.6454760462576317,
            'boosting': 'dart',
            "verbose": -1
        },
        {# DART, tuned 50 iter with 1.41421356237-random-false & v4 extra data.
            "n_estimators": 26412,
            "learning_rate": 0.029365895179364134,
            "num_leaves": 62,
            "max_depth": 21,
            "min_child_samples": 77,
            "bagging_freq": 1,
            "subsample": 0.8747212422432125,
            "colsample_bytree": 0.7942284999947743,
            "colsample_bynode": 0.7206669383813662,
            "reg_alpha": 0.036143589209288174,
            "reg_lambda": 0.699040096201788,
            "extra_trees": False,
            "drop_rate": 0.056734962273416595,
            "max_drop": 51,
            "skip_drop": 0.44869355337268213,
            'boosting': 'dart',
            "verbose": -1
        }
    ]

    DROPPED_COLUMNS += ['MancalaBoard', 'MancalaStores', 'MancalaTwoRows', 'MancalaThreeRows', 'MancalaFourRows', 'MancalaSixRows', 'MancalaCircular', 'AlquerqueBoard', 'AlquerqueBoardWithOneTriangle', 'AlquerqueBoardWithTwoTriangles', 'AlquerqueBoardWithFourTriangles', 'AlquerqueBoardWithEightTriangles', 'ThreeMensMorrisBoard', 'ThreeMensMorrisBoardWithTwoTriangles', 'NineMensMorrisBoard', 'StarBoard', 'CrossBoard', 'KintsBoard', 'PachisiBoard', 'FortyStonesWithFourGapsBoard', 'Sow', 'SowCW', 'SowCCW', 'GraphStyle', 'ChessStyle', 'GoStyle', 'MancalaStyle', 'PenAndPaperStyle', 'ShibumiStyle', 'BackgammonStyle', 'JanggiStyle', 'XiangqiStyle', 'ShogiStyle', 'TableStyle', 'SurakartaStyle', 'TaflStyle', 'NoBoard', 'MarkerComponent', 'StackType', 'Stack', 'Symbols', 'ShowPieceValue', 'ShowPieceState']

    MCTS_CONFIG_NAMES = [
        # '1.41421356237-random-false',
        '0.6-random-true',
    ]
    MCTS_RUNTIMES_SEC = [15]
    for mcts_config_name in MCTS_CONFIG_NAMES:
        for mcts_runtime_sec in MCTS_RUNTIMES_SEC:
            for lgbm_config_index, lgbm_params in enumerate(LGBM_CONFIGS):
                for use_reann_aug in [False]:
                    if lgbm_config_index not in [5]:
                        continue

                    extra_train_paths = {
                        'games_csv_path': 'data/ExtraAnnotatedGames_v6.csv',
                        'starting_position_evals_json_paths': [
                            f'data/StartingPositionEvals/StartingPositionEvals/merged_extra_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                            for i in range(10)
                        ]
                    }
                    starting_eval_json_paths = [
                        f'data/StartingPositionEvals/StartingPositionEvals/organizer_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                        for i in range(10)
                    ]

                    if use_reann_aug:
                        reannotated_features_path = 'data/RecomputedFeatureEstimates.json'
                    else:
                        reannotated_features_path = None

                    base_rmses, isotonic_rmses = [], []
                    for RANDOM_SEED in [3333, 4444, 5555]:
                        base_rmse, isotonic_rmse = CreateEnsemble(
                            lgb_params = lgbm_params,
                            early_stopping_round_count = 100,
                            fold_count = 10,
                            starting_eval_json_paths = starting_eval_json_paths,
                            extra_train_paths = extra_train_paths,
                            # extra_data_weight = 0.75,
                            extra_data_weight = 1,
                            reannotated_features_path = reannotated_features_path,
                            feature_importances_dir = None,
                            dropped_feature_count = 0,
                            # output_directory_suffix = f'_et_v6_w100_{mcts_config_name}_{mcts_runtime_sec}s_cfg{lgbm_config_index}_seed{RANDOM_SEED}_r1-10_aug_gaw33_reann-v2_nd'
                            output_directory_suffix = f'_et_v6_w100_{mcts_config_name}_{mcts_runtime_sec}s_cfg{lgbm_config_index}_seed{RANDOM_SEED}_r1-10_aug_gaw33{"_reann-v2" if use_reann_aug else ""}_drop_nd'
                        )
                        base_rmses.append(base_rmse)
                        isotonic_rmses.append(isotonic_rmse)

                        print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, lgbm config = {lgbm_config_index}')
                        print(f'Base RMSE: {base_rmse:.5f}, Isotonic RMSE: {isotonic_rmse:.5f}\n')

                    print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, lgbm config = {lgbm_config_index}')
                    print(f'Base RMSEs: {base_rmses}, (mean = {np.mean(base_rmses):.5f})')
                    print(f'Isotonic RMSEs: {isotonic_rmses}, (mean = {np.mean(isotonic_rmses):.5f})\n')
    #'''