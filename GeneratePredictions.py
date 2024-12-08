import joblib
import polars as pl
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS
from TrainCatBoostUtilityStatsPredictor import GetPreprocessedData as GetUtilityStatsPreprocessedData

# GameRulesetName is not dropped, can split on it.
def GetPreprocessedData(split_agent_features, include_lud_stats, drop_agent_features = False):
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')
        
    ruleset_names = df['GameRulesetName'].to_pandas()
    english_rules = df['EnglishRules'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()

    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    if split_agent_features and not drop_agent_features:
        for col in AGENT_COLS:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")
        
        df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in AGENT_COLS])            
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    elif drop_agent_features:
        df = df.drop([col for col in df.columns if col in AGENT_COLS])
    
    df = df.to_pandas()

    if include_lud_stats:
        preexisting_column_count = df.shape[1]
        df.insert(preexisting_column_count, "EnglishRulesLength", [len(rule) for rule in english_rules])
        df.insert(preexisting_column_count + 1, "LudRulesLength", [len(rule) for rule in lud_rules])
        df.insert(preexisting_column_count + 2, "LudIfStatementCount", [rule.count('if') for rule in english_rules])
        df.insert(preexisting_column_count + 3, "LudArrayCount", [rule.count('array') for rule in english_rules])
        df.insert(preexisting_column_count + 4, "LudVariableCount", [rule.count('var') for rule in english_rules])
        df.insert(preexisting_column_count + 5, "LudToCount", [rule.count('to') for rule in english_rules])

    print(f'Data shape: {df.shape}')
    
    return ruleset_names, df

def GenerateCompetitionTargetRegressorPredictions():
    # SPECIFY MODEL DIRECTORY PATH.
    MODEL_DIRECTORY_PATH = 'models/catboost_41406_10_72iter_100round_gpu_LudiStats_drop-NumTopSites'

    # MAYBE UPDATE DROPPED COLUMNS.
    if 'drop-NumTopSites' in MODEL_DIRECTORY_PATH:
        DROPPED_COLUMNS.append('NumTopSites')

    # LOAD DATA.
    include_lud_stats = 'LudiStats' in MODEL_DIRECTORY_PATH
    ruleset_names, train_test_df = GetPreprocessedData(split_agent_features = True, include_lud_stats = include_lud_stats)
    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    # SETUP CV.
    group_kfold = GroupKFold(n_splits=len(os.listdir(MODEL_DIRECTORY_PATH)))
    folds = list(group_kfold.split(X, y, groups=ruleset_names))

    # GENERATE PREDICTIONS.
    all_predictions = []
    all_targets = []
    all_fold_ids = []
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{len(folds)}...')

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        model = joblib.load(f'{MODEL_DIRECTORY_PATH}/{fold_index}.p')
        predictions = model.predict(test_x)
        
        all_predictions.extend(predictions)
        all_targets.extend(test_y)
        all_fold_ids.extend([fold_index] * len(test_y))

    # SAVE PREDICTIONS.
    output_filepath = MODEL_DIRECTORY_PATH.replace('models', 'predictions') + '.csv'
    output_df = pd.DataFrame({
        'fold_id': all_fold_ids,
        'utility_agent1': all_targets,
        'prediction': all_predictions,
    })
    output_df.to_csv(output_filepath)

def GenerateUtilityStatsPredictions():
    # SPECIFY MODEL DIRECTORY PATH.
    # MODEL_DIRECTORY_PATH = 'models/utilitystats_catboost_8575_10absolute_test'
    # TARGET_COLUMN_NAME = 'mean_absolute_agent1_utilities'

    MODEL_DIRECTORY_PATH = 'models/utilitystats_catboost_16181_10raw_test'
    TARGET_COLUMN_NAME = 'mean_agent1_utilities'
    
    # GENERATE PREDICTIONS FOR EACH GAME.
    ruleset_names, test_df = GetUtilityStatsPreprocessedData(include_lud_stats = True)
    
    X = test_df.drop([TARGET_COLUMN_NAME], axis=1)
    y = test_df[TARGET_COLUMN_NAME]

    group_kfold = GroupKFold(n_splits=len(os.listdir(MODEL_DIRECTORY_PATH)))
    folds = list(group_kfold.split(X, y, groups=ruleset_names))

    ruleset_names_to_predictions = {}
    for fold_index, (train_index, test_index) in enumerate(folds):
        print(f'Fold {fold_index+1}/{len(folds)}...')

        test_x = X.iloc[test_index]
        test_y = y.iloc[test_index]

        model = joblib.load(f'{MODEL_DIRECTORY_PATH}/{fold_index}.p')
        predictions = model.predict(test_x)
        
        for ruleset_name, prediction in zip(ruleset_names.iloc[test_index], predictions):
            assert ruleset_name not in ruleset_names_to_predictions
            ruleset_names_to_predictions[ruleset_name] = prediction

    # CORRELATE PREDICTIONS WITH THE ORIGINAL COMPETITION DATA ENTRIES.
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    test_ruleset_names = df['GameRulesetName'].to_pandas()
    all_predictions = []
    default_prediction = np.mean(list(ruleset_names_to_predictions.values()))
    for ruleset_name in test_ruleset_names:
        all_predictions.append(ruleset_names_to_predictions.get(ruleset_name, default_prediction))

    # SAVE PREDICTIONS.
    output_filepath = MODEL_DIRECTORY_PATH.replace('models', 'predictions') + '.csv'
    output_df = pd.DataFrame({
        'prediction': all_predictions,
    })

    output_df.to_csv(output_filepath)

if __name__ == '__main__':
    # GenerateCompetitionTargetRegressorPredictions()
    GenerateUtilityStatsPredictions()