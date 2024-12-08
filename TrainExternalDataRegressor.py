import polars as pl
import json
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from GroupKFoldShuffle import GroupKFoldShuffle
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
import joblib
import pandas as pd
from catboost import CatBoostRegressor, Pool

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(games_csv_path, starting_evals_json_path, split_agent_features):
    df = pl.read_csv(games_csv_path)

    ruleset_names = df['GameRulesetName'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    if split_agent_features:
        for col in AGENT_COLS:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")

    df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in AGENT_COLS])            
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    
    df = df.to_pandas()

    # ADD MCTS EVALUATION FEATURES.
    with open(starting_evals_json_path) as f:
        luds_to_mcts_evals = json.load(f)

    preexisting_column_count = df.shape[1]
    df.insert(
        preexisting_column_count, 
        "mcts_eval",
        [luds_to_mcts_evals[lud] for lud in lud_rules]
    )

    print(f'Data shape: {df.shape}')
    return ruleset_names, df

def PickLinearFeatures(features_df, targets, min_correlation_strength):
    selected_features = []
    for col in features_df.columns:
        feature_values = features_df[col].values

        try:
            if np.std(feature_values) == 0:
                continue

            r, p = pearsonr(feature_values, targets)
        except:
            continue

        if abs(r) > min_correlation_strength:
            selected_features.append(col)

    print(f'Selected features: {selected_features}')

    return selected_features

def TrainLinearModel(
        ruleset_names, 
        train_test_df, 
        fold_count, 
        feature_names = None,
        min_correlation_strength = None,
        random_seed = None):
    models = []
    rmse_scores = []
    all_oof_preds = np.empty(len(train_test_df))

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(tqdm(folds)):
        # SPLIT DATA.
        train_x = X.iloc[train_index]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        # SELECT FEATURES.
        if feature_names is None:
            feature_names = PickLinearFeatures(train_x, train_y, min_correlation_strength)
            train_x = train_x[feature_names]
            test_x = test_x[feature_names]
        else:
            train_x = train_x[feature_names]
            test_x = test_x[feature_names]

        # TRAIN MODEL.
        model = LinearRegression()
        model.fit(
            train_x,
            train_y
        )

        # RECORD RESULT.
        models.append({
            'model': model,
            'feature_names': feature_names
        })

        oof_predictions = model.predict(test_x)
        rmse_scores.append(root_mean_squared_error(test_y, oof_predictions))

        all_oof_preds[test_index] = oof_predictions

    return models, rmse_scores, all_oof_preds

def GetLgbmFeatures(train_test_df):
    feature_names = [
        'AdvantageP1', 
        'Balance',
        'PlayoutsPerSecond', 
        'MovesPerSecond',
        'DurationTurnsStdDev',
        'mcts_eval',
    ] + [col for col in train_test_df.columns if col[:6] in AGENT_COLS]

    return feature_names

def TrainLgbmModel(
        ruleset_names, 
        train_test_df, 
        lgb_params, 
        early_stopping_round_count, 
        fold_count, 
        feature_names,
        input_predictions,
        use_residuals,
        random_seed = None):
    models = []
    rmse_scores = []
    all_oof_preds = np.empty(len(train_test_df))

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    existing_column_count = X.shape[1]
    X.insert(existing_column_count, "input_predictions", input_predictions)
    if use_residuals:
        y = y - input_predictions

    if random_seed is None:
        group_kfold = GroupKFold(n_splits=fold_count)
    else:
        group_kfold = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)

    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(tqdm(folds)):
        # SPLIT DATA.
        train_x = X.iloc[train_index][feature_names]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index][feature_names]
        test_y = y.iloc[test_index]

        # TRAIN MODEL.
        model = lgb.LGBMRegressor(**lgb_params, random_state=random_seed)
        model.fit(
            train_x,
            train_y,
            eval_set=[(test_x, test_y)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(early_stopping_round_count)])

        # RECORD RESULT.
        models.append({
            'model': model,
            'feature_names': feature_names
        })
        rmse_scores.append(np.mean(model.evals_result_['valid_0']['rmse']))

        oof_predictions = model.predict(test_x)
        all_oof_preds[test_index] = oof_predictions

    return models, rmse_scores, all_oof_preds

def TrainCatboostModel(
        ruleset_names, 
        train_test_df, 
        cat_params, 
        early_stopping_round_count, 
        fold_count, 
        feature_names,
        input_predictions,
        use_residuals,
        random_seed = None):
    models = []
    rmse_scores = []
    all_oof_preds = np.empty(len(train_test_df))

    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    existing_column_count = X.shape[1]
    X.insert(existing_column_count, "input_predictions", input_predictions)
    if use_residuals:
        y = y - input_predictions

    categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']

    group_kfold = GroupKFold(n_splits=fold_count)

    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    for fold_index, (train_index, test_index) in enumerate(tqdm(folds)):
        # SPLIT DATA.
        train_x = X.iloc[train_index][feature_names]
        train_y = y.iloc[train_index]

        test_x = X.iloc[test_index][feature_names]
        test_y = y.iloc[test_index]

        # TRAIN MODEL.
        train_pool = Pool(train_x, train_y, cat_features=categorical_features)
        test_pool = Pool(test_x, test_y, cat_features=categorical_features)

        model = CatBoostRegressor(**cat_params, random_seed=random_seed)
        model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=early_stopping_round_count,
            verbose=False)

        # RECORD RESULT.
        models.append({
            'model': model,
            'feature_names': feature_names
        })
        rmse_scores.append(model.get_best_score()['validation']['RMSE'])

        oof_predictions = model.predict(test_x)
        all_oof_preds[test_index] = oof_predictions

    return models, rmse_scores, all_oof_preds

def GenerateOrganizerPredictions(stacked_ensemble, output_filepath):
    # LOAD DATA.
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    lud_rules = df['LudRules'].to_pandas()
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    for col in AGENT_COLS:
        df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")
    
    df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in AGENT_COLS])            
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])

    df = df.to_pandas()

    with open('StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json') as f:
        luds_to_mcts_evals = json.load(f)

    preexisting_column_count = df.shape[1]
    df.insert(
        preexisting_column_count, 
        "mcts_eval",
        [luds_to_mcts_evals[lud] for lud in lud_rules]
    )

    # GENERATE LINEAR PREDICTIONS.
    linear_predictions = []
    for linear_model_and_feature_names in stacked_ensemble['linear_models']:
        model = linear_model_and_feature_names['model']
        feature_names = linear_model_and_feature_names['feature_names']

        X = df[feature_names]
        linear_predictions.append(model.predict(X))

    linear_predictions = np.mean(linear_predictions, axis=0)

    if 'input_predictions' in stacked_ensemble['gbdt_models'][0]['feature_names']:
        df.insert(X.shape[1], "input_predictions", linear_predictions)
    
    # GENERATE GBDT PREDICTIONS.
    gbdt_predictions = []
    for gbdt_model_and_feature_names in stacked_ensemble['gbdt_models']:
        model = gbdt_model_and_feature_names['model']
        feature_names = gbdt_model_and_feature_names['feature_names']
    
        X = df[feature_names]

        gbdt_predictions.append(model.predict(X))

    gbdt_predictions = np.mean(gbdt_predictions, axis=0)

    # PRINT SCORE.
    rmse = root_mean_squared_error(df['utility_agent1'], gbdt_predictions)
    print(f'Organizer RMSE: {rmse:.4f}')

    # SAVE PREDICTIONS.
    if output_filepath is not None:
        output_df = pd.DataFrame({
            'prediction': gbdt_predictions,
            'utility_agent1': df['utility_agent1'],
        })

        output_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    # SEED = 0
    SEED = 3333

    # LOAD DATA.
    GAMES_CSV_PATH = 'GAVEL/generated_csvs/complete_datasets/2024-10-19_12-04-19.csv'
    STARTING_POSITION_EVALS_JSON_PATH = 'GAVEL/generated_csvs/complete_datasets/starting_position_evals_133_16s.json'
    ruleset_names, train_test_df = GetPreprocessedData(GAMES_CSV_PATH, STARTING_POSITION_EVALS_JSON_PATH, split_agent_features=True)

    # TRAIN LINEAR MODELS.
    linear_models, linear_rmse_scores, linear_oof_preds = TrainLinearModel(
        ruleset_names,
        train_test_df,
        fold_count=10,
        feature_names=['AdvantageP1', 'mcts_eval'],
        random_seed=SEED
    )

    print(f'Linear RMSE: {linear_rmse_scores}')
    print(f'Average Linear RMSE: {np.mean(linear_rmse_scores):.4f}')

    # TRAIN GBDT MODELS.
    ## 0.5438
    # gbdt_models, gbdt_rmse_scores, gbdt_oof_preds = TrainLgbmModel(
    #     ruleset_names,
    #     train_test_df,
    #     lgb_params={
    #         'n_estimators': 2000,
    #         'learning_rate': 0.11,
    #         'num_leaves': 7,
    #         'max_depth': 16,
    #         'min_child_samples': 40,
    #         'subsample': 0.5,
    #         'colsample_bytree': 0.75,
    #         'reg_alpha': 0.25,
    #         'reg_lambda': 0.25,
    #     },
    #     early_stopping_round_count=100,
    #     fold_count=10,
    #     feature_names=GetLgbmFeatures(train_test_df) + ['input_predictions'],
    #     random_seed=SEED,
    #     input_predictions = linear_oof_preds,
    #     use_residuals = False
    # )

    gbdt_models, gbdt_rmse_scores, gbdt_oof_preds = TrainCatboostModel(
        ruleset_names,
        train_test_df,
        cat_params={
            'iterations': 2000,
            'depth': 5,
            'l2_leaf_reg': 0.1,
            'subsample': 0.5,
            'colsample_bylevel': 0.9,
        },
        early_stopping_round_count=100,
        fold_count=10,
        feature_names=GetLgbmFeatures(train_test_df) + ['input_predictions'],
        random_seed=SEED,
        input_predictions = linear_oof_preds,
        use_residuals = False
    )

    print(f'GBDT RMSE: {gbdt_rmse_scores}')
    print(f'Average GBDT RMSE: {np.mean(gbdt_rmse_scores):.4f}')

    rmse = root_mean_squared_error(train_test_df['utility_agent1'], gbdt_oof_preds)
    print(f'Final RMSE: {rmse:.4f}')

    # SAVE STACKED ENSEMBLE.
    stacked_ensemble = {
        'linear_models': linear_models,
        'gbdt_models': gbdt_models,
    }
    ensemble_filepath = f'models/external_linear_catboost_{int(rmse * 100000)}_seed{SEED}.pkl'
    joblib.dump(stacked_ensemble, ensemble_filepath)

    # GENERATE ORGANIZER PREDICTIONS.
    predictions_filepath = ensemble_filepath.replace('models', 'predictions').replace('.pkl', '.csv')
    GenerateOrganizerPredictions(stacked_ensemble, predictions_filepath)