import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import GroupKFold
from GroupKFoldShuffle import GroupKFoldShuffle
from sklearn.utils import shuffle
import os
import joblib
import optuna
import json
import glob
from sklearn.metrics import root_mean_squared_error
from cir_model import CenteredIsotonicRegression
from sklearn.isotonic import IsotonicRegression
import socket
from FeatureEngineering.LsaPreprocessor import LsaPreprocessor

from catboost import CatBoostRegressor, Pool

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
        split_agent_features, 
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

    # ADD DURATION TO COMPLEXITY RATIO.
    # print('WARNING: Using experimental added feature, possibly revert!')
    # df['DurationToComplexityRatio'] = df['DurationActions'] / (df['StateTreeComplexity'] + 1e-15)

    print(f'Data shape: {df.shape}')
    return ruleset_names, lud_rules, df, quality_metric_values

def AddStartingEvalFeatures(starting_eval_json_paths, train_test_df, lud_rules, global_average_weight = 0.33):
    '''
    for starting_evals_json_path in starting_eval_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)

        feature_name = starting_evals_json_path.split('/')[-1].replace('.json', '')
        df.insert(
            df.shape[1], 
            feature_name,
            [luds_to_mcts_evals[lud] for lud in lud_rules]
        )
    '''

    '''
    for starting_evals_json_path in starting_eval_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)

        multi_eval = isinstance(list(luds_to_mcts_evals.values())[0], list)
        if multi_eval:
            for i in range(len(list(luds_to_mcts_evals.values())[0])):
                feature_name = f'mcts_eval_{i}'
                df.insert(
                    df.shape[1], 
                    feature_name,
                    [luds_to_mcts_evals[lud][i] for lud in lud_rules]
                )
        else:
            df.insert(
                df.shape[1], 
                'mcts_eval',
                [luds_to_mcts_evals[lud] for lud in lud_rules]
            )
    '''

    all_luds_to_mcts_evals = []
    for starting_evals_json_path in starting_eval_json_paths:
        with open(starting_evals_json_path) as f:
            luds_to_mcts_evals = json.load(f)

            # print('WARNING: Using experimental feature engineering. Possibly revert!')
            # for lud in luds_to_mcts_evals.keys():
            #     raw_evals = luds_to_mcts_evals[lud]
            #     luds_to_mcts_evals[lud] = [
            #         raw_evals[0],
            #         raw_evals[1],
            #         raw_evals[1]/(raw_evals[2] + 1e-2),
            #         raw_evals[2],
            #     ]

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

    '''
    luds_to_all_mcts_evals = {}
    for starting_evals_json_path in starting_eval_json_paths:
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
    '''

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

        '''
        # model = CenteredIsotonicRegression()
        model = IsotonicRegression(out_of_bounds = 'clip')
        model.fit(train_predictions, train_targets)
        models.append(model)

        predictions = model.predict(test_predictions)
        '''

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
        extra_data_weight,
        oof_prediction_path_prefix,
        game_fitness_scores,
        cat_params, 
        lsa_params,
        early_stopping_round_count, 
        fold_count,
        reannotated_features_path = None,
        down_weight_negative_fitness = False):
    models = []
    lsa_preprocessors = []
    oof_predictions = np.empty(train_test_df.shape[0])

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    if extra_train_df is not None:
        extra_train_X = extra_train_df.drop(['utility_agent1'], axis=1)
        extra_train_y = extra_train_df['utility_agent1']

    categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']

    EMBEDDING_COLUMN_NAMES = [
        'PcaUtilities',
        'BothPlayersClusters',
        'Player1Clusters',
        'Player2Clusters'
    ]
    embedding_features = [col for col in EMBEDDING_COLUMN_NAMES if col in X.columns]

    group_kfold_shuffle = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=RANDOM_SEED)
    folds = list(group_kfold_shuffle.split(X, y, groups=ruleset_names))
    
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{fold_count}...')

        # LOAD OOF PREDICTIONS.
        baseline_train_predictions = None
        baseline_test_predictions = None
        if oof_prediction_path_prefix is not None:
            organizer_oof_predictions_path = f'{oof_prediction_path_prefix}_organizer_seed{RANDOM_SEED}_fold{fold_index}.csv'
            oof_predictions = pd.read_csv(organizer_oof_predictions_path)['prediction'].values / 2
            baseline_train_predictions = oof_predictions[train_index]
            baseline_test_predictions = oof_predictions[test_index]

            if extra_train_df is not None:
                extra_oof_predictions_path = f'{oof_prediction_path_prefix}_extra_seed{RANDOM_SEED}_fold{fold_index}.csv'
                extra_oof_predictions = pd.read_csv(extra_oof_predictions_path)['prediction'].values / 2
                baseline_train_predictions = np.concatenate([baseline_train_predictions, extra_oof_predictions])

        # SPLIT INTO TRAIN AND TEST.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        heldout_run_index = fold_index%len(starting_eval_json_paths)
        training_run_eval_paths = starting_eval_json_paths[:heldout_run_index] + starting_eval_json_paths[heldout_run_index+1:]
        train_x = AddStartingEvalFeatures(training_run_eval_paths, train_x, lud_rules.iloc[train_index])
        # train_x = AddStartingEvalFeatures([starting_eval_json_paths[heldout_run_index]], train_x, lud_rules.iloc[train_index]) # Intentional leakage
        # train_x = AddStartingEvalFeatures([starting_eval_json_paths[(heldout_run_index + 1) % len(starting_eval_json_paths)]], train_x, lud_rules.iloc[train_index]) # Crippled augmentation
        test_x = AddStartingEvalFeatures([starting_eval_json_paths[heldout_run_index]], test_x, lud_rules.iloc[test_index])

        train_luds = lud_rules.iloc[train_index]
        if extra_train_df is not None:
            if extra_train_X.columns.tolist() != train_x.columns.tolist():
                extra_training_run_eval_paths = extra_starting_eval_json_paths[:heldout_run_index] + extra_starting_eval_json_paths[heldout_run_index+1:]
                extra_train_X = AddStartingEvalFeatures(extra_training_run_eval_paths, extra_train_X, extra_lud_rules)
                # extra_train_X = AddStartingEvalFeatures([extra_starting_eval_json_paths[heldout_run_index]], extra_train_X, extra_lud_rules) # Intentional leakage
                # extra_train_X = AddStartingEvalFeatures([extra_starting_eval_json_paths[(heldout_run_index + 1) % len(extra_starting_eval_json_paths)]], extra_train_X, extra_lud_rules) # Cripple augmentation
                extra_train_X = extra_train_X[train_x.columns]

            train_x = pd.concat([train_x, extra_train_X], ignore_index=True)
            train_y = pd.concat([train_y, extra_train_y], ignore_index=True)
            train_luds = pd.concat([train_luds, extra_lud_rules], ignore_index=True)

        # SET SAMPLE WEIGHTS.
        # if down_weight_negative_fitness:
        #     train_fitness_scores = game_fitness_scores[train_index]
        #     train_weights = [1 if score > 0 else 0.5 for score in train_fitness_scores]
        #     test_fitness_scores = game_fitness_scores[test_index]
        #     test_weights = [1 if score > 0 else 0.5 for score in test_fitness_scores]
        # else:
        #     train_weights = None
        #     test_weights = None

        train_weights = None
        test_weights = None
        if extra_data_weight != 1.0:
            train_weights = [1.0] * len(train_x)
            for i, lud in enumerate(train_luds):
                if lud in extra_lud_rules:
                    train_weights[i] = extra_data_weight

        # AUGMENT NONDETERMINISTIC FEATURES.
        if reannotated_features_path is not None:
            with open(reannotated_features_path, 'r') as f:
                luds_to_features_to_estimates = json.load(f)

            for feature_name in luds_to_features_to_estimates[train_luds[0]].keys():
                original_feature_values = train_x[feature_name].to_numpy()
                feature_estimates = np.array([
                    luds_to_features_to_estimates.get(lud, {}).get(feature_name, original_feature_values[lud_index])
                    for lud_index, lud in enumerate(train_luds)
                ])
                # interpolated_feature_values = np.average([original_feature_values, feature_estimates], axis=0, weights=[0.6, 0.4]).astype(np.float32)
                interpolated_feature_values = [
                    np.random.uniform(min(orig_val, reann_val), max(orig_val, reann_val))
                    # (np.random.uniform(min(orig_val, reann_val), max(orig_val, reann_val))*0.9) + (orig_val*0.1)
                    for orig_val, reann_val
                    in zip(original_feature_values, feature_estimates)
                ]

                train_x.loc[:, feature_name] = interpolated_feature_values

            # train_x.loc[:, 'PlayoutsPerSecond'] *= np.random.normal(1, 0.073, len(train_x))
            # train_x.loc[:, 'MovesPerSecond'] *= np.random.normal(1, 0.073, len(train_x))

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
        # train_x = train_x.assign(baseline_prediction=baseline_train_predictions)
        # test_x = test_x.assign(baseline_prediction=baseline_test_predictions)
        train_pool = Pool(
            train_x, 
            train_y,
            weight=train_weights, 
            baseline=baseline_train_predictions,
            cat_features=categorical_features,
            embedding_features=embedding_features)
        test_pool = Pool(
            test_x, 
            test_y, 
            weight=test_weights,
            baseline=baseline_test_predictions,
            cat_features=categorical_features,
            embedding_features=embedding_features)

        model = CatBoostRegressor(**cat_params, random_seed=RANDOM_SEED)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=early_stopping_round_count,
            verbose=False
        )

        models.append(model) 

        # GENERATE OOF PREDICTIONS.
        predictions = model.predict(test_x)
        if baseline_test_predictions is not None:
            predictions += baseline_test_predictions

        oof_predictions[test_index] = predictions

    base_rmse = root_mean_squared_error(y, oof_predictions)
    isotonic_models, isotonic_rmse, isotonic_oof_preds = TrainIsotonicModels(oof_predictions, y, ruleset_names, fold_count, RANDOM_SEED)

    print(f'Average RMSE (base, isotonic): {base_rmse:.5f}, {isotonic_rmse:.5f}')

    return lsa_preprocessors, models, isotonic_models, base_rmse, isotonic_rmse, oof_predictions, isotonic_oof_preds

# Additional features that can be tuned on CPU:
# 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
# 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
def Objective(trial, fold_count, extra_train_paths, starting_eval_json_paths):
    GAMES_FILEPATH = 'data/train.csv'
    ruleset_names, lud_rules, train_test_df, game_fitness_scores = GetPreprocessedData(
        games_csv_path = GAMES_FILEPATH, 
        split_agent_features = True,
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df, _ = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
            split_agent_features = True,
        )

    cat_params = {
        'iterations': trial.suggest_int('iterations', 7000, 15000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.04, log=True),
        'depth': trial.suggest_int('depth', 8, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 0.002, log=True),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
        'grow_policy': 'SymmetricTree',
        'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 3, 8),
        'loss_function': 'RMSE',
        "task_type": "GPU"
    }

    lsa_params = None

    early_stopping_round_count = 50
    oof_prediction_path_prefix = None
    lsa_preprocessors, cat_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds = TrainModels(
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
        oof_prediction_path_prefix,
        game_fitness_scores,
        cat_params, 
        lsa_params,
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
        reannotated_features_path = 'data/reannotation/lud_to_features_to_estimates_v1.json',
        down_weight_negative_fitness = False
    )
    
    return isotonic_rmse

def GetOptimalConfig(trial_count, extra_train_paths, starting_eval_json_paths):
    study = optuna.create_study(direction='minimize')

    # study.enqueue_trial({
    #     'iterations': 10219, 
    #     'learning_rate': 0.010964241393786744, 
    #     'depth': 10, 
    #     'l2_leaf_reg': 0.0012480029901784353, 
    #     'grow_policy': "SymmetricTree", 
    #     'max_ctr_complexity': 6,
    #     'loss_function': "RMSE",
    #     'task_type': "GPU"
    # })

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
    output_filepath = f'configs/catboost_{int(best_score * 100000)}_{trial_count}trials.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

def SaveModels(lsa_preprocessors, gbdt_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix = ''):
    output_directory_path = f'models/catboost_iso_{int(base_rmse*100000)}_{int(isotonic_rmse*100000)}_{len(gbdt_models)}{output_directory_suffix}'
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

def CreateEnsemble(
        cat_params, 
        lsa_params,
        early_stopping_round_count, 
        fold_count, 
        starting_eval_json_paths,
        reannotated_features_path = None,
        include_ruleset_name = False,
        oof_prediction_path_prefix = None,
        extra_train_paths = None,
        extra_data_weight = 1.0,
        output_directory_suffix = ''):
    GAMES_FILEPATH = 'data/train.csv'
    ruleset_names, lud_rules, train_test_df, game_fitness_scores = GetPreprocessedData(
        games_csv_path = GAMES_FILEPATH, 
        split_agent_features = True,
    )

    extra_train_df = None
    if extra_train_paths is not None:
        extra_ruleset_names, extra_lud_rules, extra_train_df, _ = GetPreprocessedData(
            games_csv_path = extra_train_paths['games_csv_path'],
            split_agent_features = True,
        )

    if include_ruleset_name:
        train_test_df['GameRulesetName'] = ruleset_names
        extra_train_df['GameRulesetName'] = extra_ruleset_names

        train_test_df['GameRulesetName'] = train_test_df['GameRulesetName'].astype('category')
        extra_train_df['GameRulesetName'] = extra_train_df['GameRulesetName'].astype('category')

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
        extra_data_weight,
        oof_prediction_path_prefix,
        game_fitness_scores,
        cat_params, 
        lsa_params,
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count,
        reannotated_features_path = reannotated_features_path
    )

    if output_directory_suffix is not None:
        SaveModels(lsa_preprocessors, lgbm_models, isotonic_models, base_rmse, isotonic_rmse, base_oof_preds, isotonic_oof_preds, output_directory_suffix)

    return base_rmse, isotonic_rmse

'''
"V4" extra data, no OOF predictions as features or baseline:
    Partial reannotation (base, isotonic): 0.37073, 0.36706
    Full reannotation (base, isotonic): 0.36885, 0.36506

V1:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = 
    Base RMSEs: [0.37327643921363174, 0.3714806737528657, 0.3733506485684452], (mean = 0.37270)
    Isotonic RMSEs: [0.3694906799465184, 0.36767838414750054, 0.369541584381967], (mean = 0.36890)

    MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 0, v2_suffix = 
    Base RMSEs: [0.3696144725755721, 0.3683631872056405, 0.3688965140442072], (mean = 0.36896)
    Isotonic RMSEs: [0.3656878723665475, 0.3644366987178743, 0.36503452786081947], (mean = 0.36505)
V2 R1:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = _v2
    Base RMSEs: [0.36999692212304, 0.3666399304023105, 0.3689011707947357], (mean = 0.36851)
    Isotonic RMSEs: [0.36588665676582577, 0.3625360827529921, 0.3648079870847258], (mean = 0.36441)

    MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 0, v2_suffix = _v2
    Base RMSEs: [0.3683922457008345, 0.3666690500652381, 0.3681706118535782], (mean = 0.36774)
    Isotonic RMSEs: [0.36451133164040234, 0.3628195127406353, 0.36452624862885413], (mean = 0.36395)
V2 R2:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = _v2_r2
    Base RMSEs: [0.37226387286037604, 0.3704097706510854, 0.37057543513894836], (mean = 0.37108)
    Isotonic RMSEs: [0.3684867088526859, 0.36669331215978007, 0.36689100604058583], (mean = 0.36736)

    MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 0, v2_suffix = _v2_r2
    Base RMSEs: [0.36844059092732595, 0.36637804732352175, 0.3676405609284163], (mean = 0.36749)
    Isotonic RMSEs: [0.36485575533741516, 0.36296663953708314, 0.36387819556770556], (mean = 0.36390)
V2 R1 + R2:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1 + r2
    Base RMSEs: [0.3694422535607253, 0.3684721035619669, 0.36847647618322654], (mean = 0.36880)
    Isotonic RMSEs: [0.3656157921181675, 0.3642883834666442, 0.3643242673237657], (mean = 0.36474)

    MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 0, v2_suffix = r1 + r2
    Base RMSEs: [0.3678237389678751, 0.36599816348407477, 0.366511297817491], (mean = 0.36678)
    Isotonic RMSEs: [0.36392649357690854, 0.3622273285132406, 0.3626360912575423], (mean = 0.36293)
V2 R1 - R4:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-4
    Base RMSEs: [0.3684702066475484, 0.36819056333898875, 0.36833237203051467], (mean = 0.36833)
    Isotonic RMSEs: [0.3641896139535153, 0.36433320995245894, 0.36441563651590414], (mean = 0.36431)

    MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 0, v2_suffix = r1-4
    Base RMSEs: [0.3680228368868103, 0.3646663765036787, 0.36671203837879157], (mean = 0.36647)
    Isotonic RMSEs: [0.36413966612261356, 0.36075043237645643, 0.36270279351364876], (mean = 0.36253)
V2 R1 - R5 (Leak fix, no augmentation):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-5
    Base RMSEs: [0.3709215788199145, 0.36921476565352784, 0.370263032317012], (mean = 0.37013)
    Isotonic RMSEs: [0.3669955019666404, 0.36537336608199744, 0.36658437093679386], (mean = 0.36632)
V2 R1 - R3 (Leak fix, augmented x2)
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-3
    Base RMSEs: [0.3707486406331564, 0.368626273528007, 0.36853847312100857], (mean = 0.36930)
    Isotonic RMSEs: [0.36679811082030184, 0.3648236427234251, 0.36458846229755887], (mean = 0.36540)
V2 R1 - R5 (Leak fix, augmented x4):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-5
    Base RMSEs: [0.36746957792866863, 0.3677887151671246, 0.3692300167716935], (mean = 0.36816)
    Isotonic RMSEs: [0.3631173829220106, 0.36387351268126317, 0.3654801224834776], (mean = 0.36416)

V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.0):
    TODO: Record rerun result - make sure you didn't f*ck this up.
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10
    Base RMSEs: [0.3690883606231034, 0.370196696192837, 0.3681214670712998], (mean = 0.36914)
    Isotonic RMSEs: [0.3648328069353167, 0.3662350049416467, 0.3638666836283148], (mean = 0.36498)
V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.125):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.125
    Base RMSEs: [0.36819604233868014, 0.3692718213922619, 0.36769023998317546], (mean = 0.36839)
    Isotonic RMSEs: [0.36391782446175486, 0.3652987891135068, 0.36340041685831137], (mean = 0.36421)
V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.25):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.25
    Base RMSEs: [0.3668214068402067, 0.36880750177832433, 0.3672351644615371], (mean = 0.36762)
    Isotonic RMSEs: [0.36256371706784973, 0.36502494896853255, 0.36299852410936606], (mean = 0.36353)
V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.33):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.36778325097270487, 0.3682548612284008, 0.3677453381211919], (mean = 0.36793)
    Isotonic RMSEs: [0.36371595463879325, 0.364612297642497, 0.36365431052021324], (mean = 0.36399)  <-- 0.36373 with the ratio metric, 0.36418 w/o ratio metric but using cfg 2, 0.36357 w/o ratio metric and using cfg 3
V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.5):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.5
    Base RMSEs: [0.36764694051632757, 0.36848893645739, 0.36719254331208034], (mean = 0.36778)
    Isotonic RMSEs: [0.3635606721081913, 0.3646633379513148, 0.36338213702394423], (mean = 0.36387)
V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.67):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.67
    Base RMSEs: [0.36838248190180317, 0.3689099903124891, 0.36892446411677754], (mean = 0.36874)
    Isotonic RMSEs: [0.36432070991399246, 0.36519367568235206, 0.3651073981770686], (mean = 0.36487)
V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.75):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.75
    Base RMSEs: [0.3675884775304768, 0.3689154962407502, 0.36924925095054206], (mean = 0.36858)
    Isotonic RMSEs: [0.363523748043206, 0.3651024395143728, 0.36568856191132526], (mean = 0.36477)
V2 R1 - R10 (Leak fix, augmented x9, global average weight 1.0):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=1.0
    Base RMSEs: [0.3680434377411894, 0.37143284057410225, 0.36777283263859306], (mean = 0.36908)
    Isotonic RMSEs: [0.3643446722742313, 0.3679433729097052, 0.36397705614934944], (mean = 0.36542)

V2 R1 - R10 (Leak fix, augmented x9, global average weight 0.5, border_count=254):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.5
    Base RMSEs: [0.368579833284986, 0.3690791240035932, 0.3673483521177039], (mean = 0.36834)
    Isotonic RMSEs: [0.3646736522605923, 0.36550989265759193, 0.363394358488412], (mean = 0.36453)

No added ratios:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.36778325097270487, 0.3682548612284008, 0.3677453381211919], (mean = 0.36793)
    Isotonic RMSEs: [0.36371595463879325, 0.364612297642497, 0.36365431052021324], (mean = 0.36399)

    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.367527060011438, 0.3676808593299746, 0.36772932129225144], (mean = 0.36765)
    Isotonic RMSEs: [0.36334940846713687, 0.3638979835958537, 0.36344831041035647], (mean = 0.36357)

    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 4, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.3691464814209377, 0.3693312062613787, 0.3676115955986807], (mean = 0.36870)
    Isotonic RMSEs: [0.36397860631839285, 0.3646578121174116, 0.3624784472218232], (mean = 0.36370)
Added DurationToComplexityRatio (ef9):
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.3677923866426754, 0.3683887031845124, 0.3679074312424344], (mean = 0.36803)
    Isotonic RMSEs: [0.3638613827577631, 0.3648114481356147, 0.36382634417709103], (mean = 0.36417)

    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.36736750458948275, 0.3678145017604958, 0.36723731572441154], (mean = 0.36747)
    Isotonic RMSEs: [0.3630848284588601, 0.3641262404674239, 0.3633052231342], (mean = 0.36351)
With LSA:
    10 LSA features:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.3697424578850705, 0.3722058017538827, 0.3697673772206686], (mean = 0.37057)
    Isotonic RMSEs: [0.36535119566232, 0.3682360411993173, 0.36557648371188606], (mean = 0.36639)

    20 LSA features:
    MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
    Base RMSEs: [0.37031534887770895, 0.3717496916870923, 0.37007011018342456], (mean = 0.37071)
    Isotonic RMSEs: [0.3653530059620077, 0.3677307368399865, 0.3652601565434509], (mean = 0.36611)

hparam comparison:
    1.41421356237-random-false:
        15s:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36885006180356456, 0.36911058453104895, 0.36730563726459187], (mean = 0.36842)
            Isotonic RMSEs: [0.3648311205503173, 0.3653152401917709, 0.3632346361077452], (mean = 0.36446)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36713520569201674, 0.36786421471351566, 0.3668471236449581], (mean = 0.36728)
            Isotonic RMSEs: [0.3627925063258664, 0.36407790478147495, 0.36264505424372184], (mean = 0.36317)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 4, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36901513529076446, 0.36921209143921346, 0.3674590942757495], (mean = 0.36856)
            Isotonic RMSEs: [0.3639823137010619, 0.364560657226986, 0.3622080274100025], (mean = 0.36358)
        30s:
            MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3667903294102891, 0.36453644452800804, 0.3656299715950625], (mean = 0.36565)
            Isotonic RMSEs: [0.3631028156794413, 0.3606510361715193, 0.3616448666266484], (mean = 0.36180)

            MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36729064633280534, 0.3630659373925017, 0.36463637362587614], (mean = 0.36500)
            Isotonic RMSEs: [0.36363200773637555, 0.35889444032854884, 0.3604389692007203], (mean = 0.36099)

            MCTS config = 1.41421356237-random-false, runtime = 30s, cat config = 4, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36828004628517413, 0.3647712699261342, 0.36612420486687614], (mean = 0.36639)
            Isotonic RMSEs: [0.36357631055999706, 0.35959532524892535, 0.36103087669656697], (mean = 0.36140)
     0.6-random-true:
        15s:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3679146152229138, 0.36691352201153443, 0.36668218810122444], (mean = 0.36717)
            Isotonic RMSEs: [0.36380817949660915, 0.36323251946990287, 0.362605019462364], (mean = 0.36322)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3667037337195271, 0.36561616415745096, 0.36535903791825414], (mean = 0.36589)
            Isotonic RMSEs: [0.3626381721903092, 0.3617173695079871, 0.36098196650902376], (mean = 0.36178)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 4, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3687502852648577, 0.3678409056186324, 0.3664962550618223], (mean = 0.36770)
            Isotonic RMSEs: [0.3639138142487281, 0.3632212635694714, 0.3612181271973436], (mean = 0.36278)
        30s:
            MCTS config = 0.6-random-true, runtime = 30s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36603261710448426, 0.36571620883563327, 0.36575575618683015], (mean = 0.36583)
            Isotonic RMSEs: [0.3625654909796285, 0.36214026464994664, 0.3620978830376169], (mean = 0.36227)

            MCTS config = 0.6-random-true, runtime = 30s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36566770157963124, 0.36373707283106926, 0.3652835141102975], (mean = 0.36490)
            Isotonic RMSEs: [0.36205828468869955, 0.35988338853904356, 0.3615971440617082], (mean = 0.36118)

            MCTS config = 0.6-random-true, runtime = 30s, cat config = 4, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3671981630122325, 0.36557879779547736, 0.3672230610194631], (mean = 0.36667)
            Isotonic RMSEs: [0.36254952295566945, 0.36101578921131655, 0.3624391022734853], (mean = 0.36200)

With rannotation aug (rand uniform):
    1.41421356237-random-false:
        15s:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36872434105197566, 0.36889180516979475, 0.3671999818035315], (mean = 0.36827)
            Isotonic RMSEs: [0.36422258831472604, 0.36471245493434407, 0.36260197742990075], (mean = 0.36385)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3677127490356511, 0.3679009781309041, 0.36458322157282697], (mean = 0.36673)
            Isotonic RMSEs: [0.3628433707597356, 0.36361186652603683, 0.35984473697844854], (mean = 0.36210)
     0.6-random-true:
        15s:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36911180704053637, 0.3680877034720578, 0.3668816812068461], (mean = 0.36803)
            Isotonic RMSEs: [0.3650747426010953, 0.36412701570169836, 0.36219651733662656], (mean = 0.36380)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3676430179166995, 0.3655793762472613, 0.3647740773347868], (mean = 0.36600)
            Isotonic RMSEs: [0.36325701722904974, 0.36126052807078, 0.35998389300770833], (mean = 0.36150)      <-- Current best

With rannotation aug (60/40 consistent):
    1.41421356237-random-false:
        15s:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3709061610250254, 0.3703096134834021, 0.36884248201565], (mean = 0.37002)
            Isotonic RMSEs: [0.366199653752428, 0.36616801201090143, 0.36433510051188084], (mean = 0.36557)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36972373591758395, 0.3695490721726136, 0.3676496846917567], (mean = 0.36897)
            Isotonic RMSEs: [0.36472151350034376, 0.365348903344075, 0.36287700749819823], (mean = 0.36432)
    0.6-random-true:
        15s:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.37119654888758025, 0.36953056420376107, 0.3684713125676225], (mean = 0.36973)
            Isotonic RMSEs: [0.3669704598391064, 0.36544337252060183, 0.36400619923305944], (mean = 0.36547)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36983153763398, 0.3680725188230965, 0.367220302178899], (mean = 0.36837)
            Isotonic RMSEs: [0.3655821827319131, 0.3636738438307009, 0.36243366857638476], (mean = 0.36390)
        
With reannotation aug, biased 10% towards original values:
    1.41421356237-random-false:
        15s:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3698206186493547, 0.36876998137959255, 0.36757515677027264], (mean = 0.36872)
            Isotonic RMSEs: [0.3654519426717462, 0.3645954900423522, 0.3630886882519947], (mean = 0.36438)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36816170249597646, 0.36781614974869786, 0.36526603635071114], (mean = 0.36708)
            Isotonic RMSEs: [0.3634254300231516, 0.3632783899758487, 0.36086578989312246], (mean = 0.36252)
    0.6-random-true:
        15s:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3704517171676583, 0.36738405821945136, 0.3671905123830095], (mean = 0.36834)
            Isotonic RMSEs: [0.36649995835210475, 0.3630820488710823, 0.3623772506284988], (mean = 0.36399)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36897566134985404, 0.3667554026794929, 0.36525959331570934], (mean = 0.36700)
            Isotonic RMSEs: [0.3647314847761107, 0.362458687925893, 0.36060363283307495], (mean = 0.36260)

Dropped cols with ludii player discrepancy (v4 data, no rean):
    1.41421356237-random-false:
        15s:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3669886264683284, 0.36875572364782816, 0.3672144384056297], (mean = 0.36765)
            Isotonic RMSEs: [0.36286927233207333, 0.36492384240010917, 0.36309454715473743], (mean = 0.36363)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36719136850221945, 0.36747758247622453, 0.36626777570592256], (mean = 0.36698)
            Isotonic RMSEs: [0.36288728503272305, 0.36372918170309143, 0.3620424424614047], (mean = 0.36289)
    0.6-random-true:
        15s:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3673943762225111, 0.3670070644502877, 0.36531879502572817], (mean = 0.36657)
            Isotonic RMSEs: [0.36348336956067906, 0.3632017773764793, 0.3610783427036597], (mean = 0.36259)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3661640387329997, 0.3657213214238731, 0.3653102858109494], (mean = 0.36573)
            Isotonic RMSEs: [0.3620280155280676, 0.36180545237648853, 0.36104245989053657], (mean = 0.36163)        <-- Second best (tie)
            
Cat + rean + drop (v4 data):
    1.41421356237-random-false:
        15s:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36859093543009736, 0.3688591315578262, 0.36685623600056455], (mean = 0.36810)
            Isotonic RMSEs: [0.36407274341616047, 0.36469573404598654, 0.3622659736924521], (mean = 0.36368)

            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36798122452555043, 0.36744972976753904, 0.36493264028690825], (mean = 0.36679)
            Isotonic RMSEs: [0.36335795745919147, 0.3632234844049032, 0.36019259170618806], (mean = 0.36226)
    0.6-random-true:
        15s:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3700843162848014, 0.36706813321796505, 0.3657308306414391], (mean = 0.36763)
            Isotonic RMSEs: [0.3657318253324252, 0.36279433760335383, 0.36107703422248333], (mean = 0.36320)

            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3686874077684907, 0.3657101422222174, 0.3644700723153365], (mean = 0.36629)
            Isotonic RMSEs: [0.3643953800172566, 0.3613852941760435, 0.35952429207279796], (mean = 0.36177)

Extra data v6 (rean aug (v1), drop):
    1.41421356237-random-false:
        Extra data weight 0.25:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36865341749853037, 0.36885063458235756, 0.3671818452150237], (mean = 0.36823)
            Isotonic RMSEs: [0.3641026316204479, 0.36464922026345464, 0.3625394203481939], (mean = 0.36376)
        Extra data weight 0.5:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3683077166484912, 0.36905363126682894, 0.3673779999369738], (mean = 0.36825)
            Isotonic RMSEs: [0.36381542203659045, 0.3648106986625942, 0.36269073857644835], (mean = 0.36377)
        Extra data weight 0.75:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36847531116960996, 0.3692445977017419, 0.3677079018244229], (mean = 0.36848)
            Isotonic RMSEs: [0.3639219483419355, 0.3650386061274562, 0.36300976275756425], (mean = 0.36399)
        Extra data weight 1.0:
            MCTS config = 1.41421356237-random-false, runtime = 15s, cat config = 0, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36842157288719574, 0.3691925651895927, 0.3673443367260406], (mean = 0.36832)
            Isotonic RMSEs: [0.3638832487689593, 0.3649765407546956, 0.36270443534447916], (mean = 0.36385)
    0.6-random-true:
        Extra data weight 0.25:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3686209556956796, 0.3653660998810349, 0.3643488892898091], (mean = 0.36611)
            Isotonic RMSEs: [0.36439315666237515, 0.36109976003222705, 0.3594036726030906], (mean = 0.36163)        <-- Second best (tie)
        Extra data weight 0.5:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3686205508461204, 0.365502222561894, 0.3643023377262288], (mean = 0.36614)
            Isotonic RMSEs: [0.3644175039986683, 0.36123230600943546, 0.3593567307991127], (mean = 0.36167)
        Extra data weight 0.75:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.3685993305426898, 0.3658809871979399, 0.364564564928791], (mean = 0.36635)
            Isotonic RMSEs: [0.3643392413980329, 0.36160250964322793, 0.3595283659689604], (mean = 0.36182)
        Extra data weight 1.0:
            MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33
            Base RMSEs: [0.36866020347973705, 0.36619106000826157, 0.36432513593941684], (mean = 0.36639)
            Isotonic RMSEs: [0.36438473233628815, 0.36188581913029505, 0.3592554901569161], (mean = 0.36184)

Extra data v6, rean aug v2, no drop:
    MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33, w25
    Base RMSEs: [0.36812189863205746, 0.3666574460342166, 0.36476193891187336], (mean = 0.36651)
    Isotonic RMSEs: [0.3639473742093733, 0.36232930462728763, 0.35973809370503995], (mean = 0.36200)

    MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33, w50
    Base RMSEs: [0.3679890748503363, 0.3664393103315174, 0.36474541176207875], (mean = 0.36639)
    Isotonic RMSEs: [0.3638418542236138, 0.3621309158586325, 0.3598071765900194], (mean = 0.36193)

Extra data v6, drop, extra data weight 0.25, CIR:
    Rean aug v2:
        MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33, w25
        Base RMSEs: [0.3689762713830082, 0.36634412509555436, 0.3645613159166088], (mean = 0.36663)
        Isotonic RMSEs: [0.3645560199236602, 0.3619595492495794, 0.3593990500443237], (mean = 0.36197)
    Rean aug v1:
        MCTS config = 0.6-random-true, runtime = 15s, cat config = 3, v2_suffix = r1-10, gaw=0.33, w25
        Base RMSEs: [0.36831644640189987, 0.3656170851508474, 0.36452408497232774], (mean = 0.36615)
        Isotonic RMSEs: [0.3640853689827418, 0.3612433566355856, 0.3594720638116485], (mean = 0.36160)

'''
if __name__ == '__main__':
    # extra_train_paths = {
    #     'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
    #     'starting_position_evals_json_paths': [
    #         # f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
    #         f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-0.6-random-true_15s_v2_r{i+1}.json'
    #         for i in range(10)
    #     ]
    # }
    # starting_eval_json_paths = [
    #     # f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
    #     f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-0.6-random-true_15s_v2_r{i+1}.json'
    #     for i in range(10)
    # ]

    # RANDOM_SEED = 574932
    # GetOptimalConfig(50, extra_train_paths, starting_eval_json_paths)

    #'''
    CAT_CONFIGS = [
        { # Tuned ~70 iterations without extra train or MCTS evals.
            "iterations": 10219, 
            "learning_rate": 0.010964241393786744, 
            "depth": 10, 
            "l2_leaf_reg": 0.0012480029901784353, 
            "grow_policy": "SymmetricTree", 
            "max_ctr_complexity": 6,
            "task_type": "GPU"
        },
        { # Tuned extra 64 iterations (using previous as starting guess) with V4 extra train & MCTS evals (30s 1.4-rand-false).
            "iterations": 10946,
            "learning_rate": 0.00832923539677879,
            "depth": 12,
            "l2_leaf_reg": 6.062503465034492e-07,
            "grow_policy": "SymmetricTree",
            "max_ctr_complexity": 5,
            "task_type": "GPU"
        },
        { # Retuned from scratch 25 iters, random strength & bagging temperature included.
            "iterations": 10575,
            "learning_rate": 0.023249727358874244,
            "depth": 10,
            "l2_leaf_reg": 0.0002045822216586399,
            "random_strength": 1.9992188951707355,
            "bagging_temperature": 0.5689516739875569,
            "grow_policy": "SymmetricTree",
            "max_ctr_complexity": 4,
            "task_type": "GPU"
        },
        { # Retuned from scratch 30 iters, random strength & bagging temperature included.
            "iterations": 12925,
            "learning_rate": 0.010048614733345623,
            "depth": 11,
            "l2_leaf_reg": 0.0005752849201335927,
            "random_strength": 1.2015370456710703,
            "bagging_temperature": 0.7113849924519451,
            "grow_policy": "SymmetricTree",
            "max_ctr_complexity": 7,
            "task_type": "GPU"
        },
        { # Retuned from scratch 50 iters, random strength & bagging temperature included.
            "iterations": 12202,
            "learning_rate": 0.005527276188601579,
            "depth": 12,
            "l2_leaf_reg": 0.00010022476523025375,
            "random_strength": 1.2965440615827322,
            "bagging_temperature": 0.613231100046889,
            "grow_policy": "SymmetricTree",
            "max_ctr_complexity": 6,
            "task_type": "GPU"
        }
    ]

    # LSA_CONFIG = {
    #     'analyzer': 'word', 
    #     'ngram_range': (4, 4),
    #     'n_components': 100, 
    #     'max_df': 0.8705023665904886, 
    #     'min_df': 0.05016684092193893, 
    #     'kept_feature_count': 10
    # }
    LSA_CONFIG = None

    DROPPED_COLUMNS += ['MancalaBoard', 'MancalaStores', 'MancalaTwoRows', 'MancalaThreeRows', 'MancalaFourRows', 'MancalaSixRows', 'MancalaCircular', 'AlquerqueBoard', 'AlquerqueBoardWithOneTriangle', 'AlquerqueBoardWithTwoTriangles', 'AlquerqueBoardWithFourTriangles', 'AlquerqueBoardWithEightTriangles', 'ThreeMensMorrisBoard', 'ThreeMensMorrisBoardWithTwoTriangles', 'NineMensMorrisBoard', 'StarBoard', 'CrossBoard', 'KintsBoard', 'PachisiBoard', 'FortyStonesWithFourGapsBoard', 'Sow', 'SowCW', 'SowCCW', 'GraphStyle', 'ChessStyle', 'GoStyle', 'MancalaStyle', 'PenAndPaperStyle', 'ShibumiStyle', 'BackgammonStyle', 'JanggiStyle', 'XiangqiStyle', 'ShogiStyle', 'TableStyle', 'SurakartaStyle', 'TaflStyle', 'NoBoard', 'MarkerComponent', 'StackType', 'Stack', 'Symbols', 'ShowPieceValue', 'ShowPieceState']

    REANNOTATION_PATHS = [
        'data/RecomputedFeatureEstimates.json',
    ]
    REANNOTATION_VERSION_NUMBERS = [2]

    MCTS_CONFIG_NAMES = [
        '0.6-random-true',
    ]
    MCTS_RUNTIMES_SEC = [15]
    for mcts_config_name in MCTS_CONFIG_NAMES:
        for mcts_runtime_sec in MCTS_RUNTIMES_SEC:
            for cat_config_index, cat_params in enumerate(CAT_CONFIGS):
                for reann_index, reannotated_features_path in enumerate(REANNOTATION_PATHS):
                    if cat_config_index not in [3]:
                        continue

                    extra_train_paths = {
                        'games_csv_path': 'data/ExtraAnnotatedGames_v4.csv',
                        'starting_position_evals_json_paths': [
                            f'data/StartingPositionEvals/StartingPositionEvals/merged_extra_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                            for i in range(10)
                        ]
                    }
                    starting_eval_json_paths = [
                        f'data/StartingPositionEvals/StartingPositionEvals/organizer_UCB1Tuned-{mcts_config_name}_{mcts_runtime_sec}s_v2_r{i+1}.json'
                        for i in range(10)
                    ]

                    base_rmses, isotonic_rmses = [], []
                    extra_data_weight = 0.25
                    reann_suffix = f'_reann_v{REANNOTATION_VERSION_NUMBERS[reann_index]}'
                    for RANDOM_SEED in [4444, 5555]:
                        base_rmse, isotonic_rmse = CreateEnsemble(
                            cat_params = cat_params,
                            lsa_params = LSA_CONFIG,
                            fold_count = 10,
                            early_stopping_round_count = 100,
                            starting_eval_json_paths = starting_eval_json_paths,
                            reannotated_features_path = reannotated_features_path,
                            include_ruleset_name = False,
                            extra_train_paths = extra_train_paths,
                            extra_data_weight = extra_data_weight,
                            oof_prediction_path_prefix = None,
                            output_directory_suffix = f"_et_v6_w{int(extra_data_weight*100)}_{mcts_config_name}_{mcts_runtime_sec}s_cfg{cat_config_index}_seed{RANDOM_SEED}_v2_r1-10_aug_gaw033_{reann_suffix}_drop"
                        )
                        base_rmses.append(base_rmse)
                        isotonic_rmses.append(isotonic_rmse)

                        print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, cat config = {cat_config_index}, v2_suffix = r1-10, gaw=0.33, w{int(extra_data_weight*100)}, seed = {RANDOM_SEED}')
                        print(f'Base RMSE: {base_rmse:.5f}, Isotonic RMSE: {isotonic_rmse:.5f}\n')

                    print(f'\nMCTS config = {mcts_config_name}, runtime = {mcts_runtime_sec}s, cat config = {cat_config_index}, v2_suffix = r1-10, gaw=0.33, w{int(extra_data_weight*100)}')
                    print(f'Base RMSEs: {base_rmses}, (mean = {np.mean(base_rmses):.5f})')
                    print(f'Isotonic RMSEs: {isotonic_rmses}, (mean = {np.mean(isotonic_rmses):.5f})\n')
    #'''