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

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(split_agent_features, include_english_lsa, include_lud_lsa):
    # df = pl.read_csv('DataGeneration/CompleteDatasets/OrganizerGamesAndFeatures_4Agents.csv')
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    ruleset_names = df['GameRulesetName'].to_pandas()
    english_rules = df['EnglishRules'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    if split_agent_features:
        for col in AGENT_COLS:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")

    df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in AGENT_COLS])            
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    
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

    # MAYBE ADD MCTS EVALUATION FEATURES.
    with open('StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json') as f:
        luds_to_mcts_evals = json.load(f)

    preexisting_column_count = df.shape[1]
    df.insert(
        preexisting_column_count, 
        "mcts_eval",
        [luds_to_mcts_evals[lud] for lud in lud_rules]
    )

    print(f'Data shape: {df.shape}')
    return ruleset_names, df

def SelectFeatures(train_x, train_y, lgb_params, kept_feature_count) -> list[str]:
    model = lgb.LGBMRegressor(**lgb_params, random_state=RANDOM_SEED)
    model.fit(train_x, train_y)

    feature_importances = model.feature_importances_
    feature_names = train_x.columns
    feature_importance_tuples = list(zip(feature_names, feature_importances))
    feature_importance_tuples.sort(key=lambda x: x[1], reverse=True)

    selected_feature_names = [tup[0] for tup in feature_importance_tuples[:kept_feature_count]]

    return selected_feature_names

def TrainModels(
        ruleset_names, 
        train_test_df, 
        lgbm_feature_selector_params, 
        cat_params, 
        kept_feature_count, 
        early_stopping_round_count, 
        fold_count):
    models = []
    rmse_scores = []

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

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
        print(f'Fold {fold_index+1}/{fold_count}...', end=' ', flush=True)

        # SPLIT DATA.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        # SELECT FEATURES.
        selected_feature_names = SelectFeatures(train_x, train_y, lgbm_feature_selector_params, kept_feature_count)
        train_x = train_x[selected_feature_names]
        test_x = test_x[selected_feature_names]

        # TRAIN REGRESSOR.
        train_pool = Pool(
            train_x, 
            train_y, 
            cat_features=categorical_features,
            embedding_features=embedding_features)
        test_pool = Pool(
            test_x, 
            test_y, 
            cat_features=categorical_features,
            embedding_features=embedding_features)

        model = CatBoostRegressor(**cat_params, random_seed=RANDOM_SEED)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=early_stopping_round_count,
            verbose=False
        )

        models.append({
            'model': model,
            'selected_feature_names': selected_feature_names
        })
        rmse_scores.append(model.best_score_['validation']['RMSE'])

        print(f'RMSE: {rmse_scores[-1]:.4f}')
    
    mean_score = np.mean(rmse_scores)
    print(f'Average RMSE: {mean_score:.4f}')

    return models, mean_score

def SaveModels(trained_models, test_rmse, output_directory_suffix = ''):
    output_directory_path = f'models/fs_catboost_{int(test_rmse*100000)}_{len(trained_models)}_{output_directory_suffix}'
    os.makedirs(output_directory_path)

    for fold_index, model in enumerate(trained_models):
        output_filepath = f'{output_directory_path}/{fold_index}.p'
        joblib.dump(model, output_filepath)

def CreateEnsemble(
        lgbm_feature_selector_params,
        cat_params, 
        kept_feature_count,
        early_stopping_round_count, 
        fold_count, 
        include_english_lsa,
        include_lud_lsa,
        output_directory_suffix = ''):
    ruleset_names, train_test_df = GetPreprocessedData(
        split_agent_features = True,
        include_english_lsa = include_english_lsa,
        include_lud_lsa = include_lud_lsa
    )
    trained_models, test_rmse = TrainModels(
        ruleset_names,
        train_test_df, 
        lgbm_feature_selector_params,
        cat_params, 
        kept_feature_count,
        early_stopping_round_count = early_stopping_round_count, 
        fold_count = fold_count
    )

    if output_directory_suffix is not None:
        SaveModels(trained_models, test_rmse, output_directory_suffix)

    return test_rmse

if __name__ == '__main__':
    scores = []
    # for seed in [1111, 2222]:
    for seed in [2222]:
        RANDOM_SEED = seed
        print(f'Seed: {RANDOM_SEED}')
        # for kept_feature_count in [600, 500, 400, 300, 200]:
        for kept_feature_count in [200, 300, 400, 500, 600]:
            rmse = CreateEnsemble(
                #region config_1 (200 iter tuned LGBM, 72 iter tuned catboost)
                lgbm_feature_selector_params = {
                    "n_estimators": 7_500,
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
                cat_params = {
                    "iterations": 10219, 
                    "learning_rate": 0.010964241393786744, 
                    "depth": 10, 
                    "l2_leaf_reg": 0.0012480029901784353, 
                    "grow_policy": "SymmetricTree", 
                    "max_ctr_complexity": 6,
                    "task_type": "GPU"
                },
                #endregion
                
                #region config_2 (lucky guess LGBM, 20 iter catboost)
                # lgbm_feature_selector_params = {
                #     "n_estimators": 10000,
                #     "learning_rate": 0.02556180399737767,
                #     "num_leaves": 247,
                #     "max_depth": 16,
                #     "min_child_samples": 72,
                #     "subsample": 0.6261097426808137,
                #     "colsample_bytree": 0.8480868413996958,
                #     "reg_alpha": 3.7437548643071,
                #     "reg_lambda": 0.0010293013793948255,
                #     "verbose": -1
                # },
                # cat_params = {
                #     "iterations": 4000,
                #     "learning_rate": 0.060251941838463344,
                #     "depth": 10,
                #     "l2_leaf_reg": 1.0977610434783105e-06,
                #     "task_type": "GPU"
                # },
                #endregion
                kept_feature_count = kept_feature_count,
                early_stopping_round_count = 100,
                fold_count = 10,
                include_english_lsa = False,
                include_lud_lsa = False,
                # output_directory_suffix = f"config2_{kept_feature_count}"
                output_directory_suffix = None
            )

            scores.append(rmse)

        print('Kept feature count:')
        print('Scores:', scores)
        print('Mean:', np.mean(scores))