import numpy as np
import pandas as pd
import os

import sys
sys.path.append('./')
from TrainLGBM import GetPreprocessedData, TrainModels

# Adds up to 229 features.
# The dataframe will be modified in-place.
def AddFeatures(df: pd.DataFrame, feature_ids: list[int], verbose = False):
    # Feature combinations.
    if 0 in feature_ids:
        df['area'] = df['NumRows'] * df['NumColumns']
    if 1 in feature_ids:
        df['row_equal_col'] = (df['NumColumns'] == df['NumRows']).astype(np.int8)
    if 2 in feature_ids:
        df['Playouts/Moves'] = df['PlayoutsPerSecond'] / (df['MovesPerSecond'] + 1e-15)
    if 3 in feature_ids:
        df['EfficiencyPerPlayout'] = df['MovesPerSecond'] / (df['PlayoutsPerSecond'] + 1e-15)
    if 4 in feature_ids:
        df['TurnsDurationEfficiency'] = df['DurationActions'] / (df['DurationTurnsStdDev'] + 1e-15)
    if 5 in feature_ids:
        df['AdvantageBalanceRatio'] = df['AdvantageP1'] / (df['Balance'] + 1e-15)
    if 6 in feature_ids:
        df['ActionTimeEfficiency'] = df['DurationActions'] / (df['MovesPerSecond'] + 1e-15)
    if 7 in feature_ids:
        df['StandardizedTurnsEfficiency'] = df['DurationTurnsStdDev'] / (df['DurationActions'] + 1e-15)
    if 8 in feature_ids:
        df['AdvantageTimeImpact'] = df['AdvantageP1'] / (df['DurationActions'] + 1e-15)
    if 9 in feature_ids:
        df['DurationToComplexityRatio'] = df['DurationActions'] / (df['StateTreeComplexity'] + 1e-15)
    if 10 in feature_ids:
        df['NormalizedGameTreeComplexity'] = df['GameTreeComplexity'] / (df['StateTreeComplexity'] + 1e-15)
    if 11 in feature_ids:
        df['ComplexityBalanceInteraction'] = df['Balance'] * df['GameTreeComplexity']
    if 12 in feature_ids:
        df['OverallComplexity'] = df['StateTreeComplexity'] + df['GameTreeComplexity']
    if 13 in feature_ids:
        df['ComplexityPerPlayout'] = df['GameTreeComplexity'] / (df['PlayoutsPerSecond'] + 1e-15)
    if 14 in feature_ids:
        df['TurnsNotTimeouts/Moves'] = df['DurationTurnsNotTimeouts'] / (df['MovesPerSecond'] + 1e-15)
    if 15 in feature_ids:
        df['Timeouts/DurationActions'] = df['Timeouts'] / (df['DurationActions'] + 1e-15)
    if 16 in feature_ids:
        df['OutcomeUniformity/AdvantageP1'] = df['OutcomeUniformity'] / (df['AdvantageP1'] + 1e-15)
    if 17 in feature_ids:
        df['ComplexDecisionRatio'] = df['StepDecisionToEnemy'] + df['SlideDecisionToEnemy'] + df['HopDecisionMoreThanOne']
    if 18 in feature_ids:
        df['AggressiveActionsRatio'] = df['StepDecisionToEnemy'] + df['HopDecisionEnemyToEnemy'] + df['HopDecisionFriendToEnemy'] + df['SlideDecisionToEnemy']
    if 19 in feature_ids:
        df['PlayoutsPerSecond/MovesPerSecond'] = df['PlayoutsPerSecond'] / df['MovesPerSecond']

    # One-hot copies of low-cardinality features.
    CANDIDATE_ONEHOT_COLS = [['NumOffDiagonalDirections', [0.0, 4.82, 2.0, 5.18, 3.08, 0.06]], ['NumLayers', [1, 0, 4, 5]], ['NumPhasesBoard', [3, 2, 1, 5, 4]], ['NumContainers', [1, 4, 3, 2]], ['NumDice', [0, 2, 1, 4, 6, 3, 5, 7]], ['ProposeDecisionFrequency', [0.0, 0.05, 0.01]], ['PromotionDecisionFrequency', [0.0, 0.01, 0.03, 0.02, 0.11, 0.05, 0.04]], ['SlideDecisionToFriendFrequency', [0.0, 0.19, 0.06]], ['LeapDecisionToEnemyFrequency', [0.0, 0.04, 0.01, 0.02, 0.07, 0.03, 0.14, 0.08]], ['HopDecisionFriendToFriendFrequency', [0.0, 0.13, 0.09]], ['HopDecisionEnemyToEnemyFrequency', [0.0, 0.01, 0.2, 0.03]], ['HopDecisionFriendToEnemyFrequency', [0.0, 0.01, 0.09, 0.25, 0.02]], ['FromToDecisionFrequency', [0.0, 0.38, 1.0, 0.31, 0.94, 0.67]], ['ProposeEffectFrequency', [0.0, 0.01, 0.03]], ['PushEffectFrequency', [0.0, 0.5, 0.96, 0.25]], ['FlipFrequency', [0.0, 0.87, 1.0, 0.96]], ['SetCountFrequency', [0.0, 0.62, 0.54, 0.02]], ['DirectionCaptureFrequency', [0.0, 0.55, 0.54]], ['EncloseCaptureFrequency', [0.0, 0.08, 0.1, 0.07, 0.12, 0.02, 0.09]], ['InterveneCaptureFrequency', [0.0, 0.01, 0.14, 0.04]], ['SurroundCaptureFrequency', [0.0, 0.01, 0.03, 0.02]], ['NumPlayPhase', [1, 2, 3, 4, 5, 6, 7, 8]], ['LineLossFrequency', [0.0, 0.96, 0.87, 0.46, 0.26, 0.88, 0.94]], ['ConnectionEndFrequency', [0.0, 0.19, 1.0, 0.23, 0.94, 0.35, 0.97]], ['ConnectionLossFrequency', [0.0, 0.54, 0.78]], ['GroupEndFrequency', [0.0, 1.0, 0.11, 0.79]], ['GroupWinFrequency', [0.0, 0.11, 1.0]], ['LoopEndFrequency', [0.0, 0.14, 0.66]], ['LoopWinFrequency', [0.0, 0.14, 0.66]], ['PatternEndFrequency', [0.0, 0.63, 0.35]], ['PatternWinFrequency', [0.0, 0.63, 0.35]], ['NoTargetPieceWinFrequency', [0.0, 0.72, 0.77, 0.95, 0.32, 1.0]], ['EliminatePiecesLossFrequency', [0.0, 0.85, 0.96, 0.68]], ['EliminatePiecesDrawFrequency', [0.0, 0.03, 0.91, 1.0, 0.36, 0.86]], ['NoOwnPiecesLossFrequency', [0.0, 1.0, 0.68]], ['FillEndFrequency', [0.0, 1.0, 0.04, 0.01, 0.99, 0.72]], ['FillWinFrequency', [0.0, 1.0, 0.04, 0.01, 0.99]], ['ReachDrawFrequency', [0.0, 0.9, 0.98]], ['ScoringLossFrequency', [0.0, 0.6, 0.62]], ['NoMovesLossFrequency', [0.0, 1.0, 0.13, 0.06]], ['NoMovesDrawFrequency', [0.0, 0.01, 0.04, 0.03, 0.22]], ['BoardSitesOccupiedChangeNumTimes', [0.0, 0.06, 0.42, 0.12, 0.14, 0.94]], ['BranchingFactorChangeNumTimesn', [0.0, 0.3, 0.02, 0.07, 0.04, 0.13, 0.01, 0.21, 0.03]], ['PieceNumberChangeNumTimes', [0.0, 0.06, 0.42, 0.12, 0.14, 1.0]]]

    feature_id = 19
    for feature_id_offset, (col_name, categories) in enumerate(CANDIDATE_ONEHOT_COLS):
        for category in categories:
            feature_id += 1
            if feature_id in feature_ids:
                df[f'{col_name}_{category}'] = (df[col_name] == category).astype(np.int8)

    if verbose:
        print(f'Added {len(feature_ids)}/{feature_id+1} features.')

def GetFeatureSetScore(feature_ids, lgbm_config, random_seed):
    # LOAD DATA.
    MCTS_CONFIG_NAME = '1.41421356237-random-false'
    MCTS_RUNTIME_SEC = 15
    GAME_CACHE_FILEPATH = '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv'
    COMMON_GAMES_FILEPATH = 'data/from_organizers/train.csv'

    starting_eval_json_paths = [
        f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-1.41421356237-random-false_15s_v2_r{i+1}.json'
        for i in range(10)
    ]
    extra_train_paths = {
        'games_csv_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-23_15-10-16.csv',
        'starting_position_evals_json_paths': [
            f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra_v5_UCB1Tuned-{MCTS_CONFIG_NAME}_{MCTS_RUNTIME_SEC}s_v2_r{i+1}.json'
            for i in range(5)
        ]
    }

    ruleset_names, lud_rules, train_test_df = GetPreprocessedData(
        games_csv_path = GAME_CACHE_FILEPATH if os.path.exists(GAME_CACHE_FILEPATH) else COMMON_GAMES_FILEPATH, 
    )

    extra_ruleset_names, extra_lud_rules, extra_train_df = GetPreprocessedData(
        games_csv_path = extra_train_paths['games_csv_path'],
    )

    # ADD FEATURES.
    AddFeatures(train_test_df, feature_ids, verbose = True)
    AddFeatures(extra_train_df, feature_ids, verbose = True)

    # TRAIN MODEL.
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
        feature_importances_dir=None,
        dropped_feature_count=0,
        lgb_params = lgbm_config, 
        early_stopping_round_count = 50, 
        fold_count = 5,
        random_seed=random_seed
    )

    return isotonic_rmse

def SelectFeatures(lgbm_config, candidate_feature_ids, random_seed):
    # GET BASELINE SCORE.
    best_extra_feature_ids = []
    best_score = GetFeatureSetScore(best_extra_feature_ids, lgbm_config, random_seed)

    # ADD FEATURES ONE-BY-ONE.
    while len(candidate_feature_ids) > 0:
        candidate_addition_scores = []
        reasonable_candidates = []
        for candidate_feature_id in candidate_feature_ids:
            candidate_feature_ids = best_extra_feature_ids + [candidate_feature_id]
            candidate_score = GetFeatureSetScore(candidate_feature_ids, lgbm_config, random_seed)
            candidate_addition_scores.append((candidate_feature_id, candidate_score))

            if candidate_score < best_score:
                reasonable_candidates.append(candidate_feature_id)

        best_candidate_feature_id, best_candidate_score = min(candidate_addition_scores, key=lambda x: x[1])
        if best_candidate_score < best_score:
            best_extra_feature_ids.append(best_candidate_feature_id)
            best_score = best_candidate_score
        else:
            break

        candidate_feature_ids = [feature_id for feature_id in reasonable_candidates if (feature_id != best_candidate_feature_id)]
        
        print('Remaining unselected candidate:', len(candidate_feature_ids))
        print('Best extra feature ids:', best_extra_feature_ids)
        print('Best score:', best_score)

    return best_extra_feature_ids

# Selected 1 features: [9]
if __name__ == '__main__':
    lgbm_configs = [
        {
            'verbose': -1,
        },
        {
            "n_estimators": 10000,
            "learning_rate": 0.03184567466358953,
            "num_leaves": 196,
            "max_depth": 17,
            "min_child_samples": 94,
            "colsample_bytree": 0.9612980174610098,
            "colsample_bynode": 0.6867101420064379,
            "reg_alpha": 1.593152807295967e-05,
            "reg_lambda": 4.8636580199114866e-08,
            "extra_trees": False,
            "verbose": -1
        },
        {
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
        },
    ]

    RANDOM_SEEDS = [111111, 222222, 333333]
    candidate_feature_ids = list(range(229))
    for lgbm_config, random_seed in zip(lgbm_configs, RANDOM_SEEDS):
        selected_features = SelectFeatures(lgbm_config, candidate_feature_ids, random_seed)
        
        print('###############################################################')
        print(f'Selected {len(selected_features)} features:', selected_features)
        print('###############################################################')
        candidate_feature_ids = selected_features