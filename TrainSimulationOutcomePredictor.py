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

from catboost import CatBoostClassifier, Pool

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

def GetPreprocessedData():
    df = pl.read_csv('DataGeneration/CompleteDatasets/OrganizerGamesAndFeatures_4Agents_Dedup_NoOrigLabels.csv')

    ruleset_names = df['GameRulesetName'].to_pandas()
    english_rules = df['EnglishRules'].to_pandas()
    lud_rules = df['LudRules'].to_pandas()
    
    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in AGENT_COLS])            
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    
    df = df.to_pandas()

    return ruleset_names, lud_rules, df

def TrainModels(
        ruleset_names, 
        lud_rules,
        train_test_df, 
        cat_params, 
        early_stopping_round_count, 
        fold_count,
        target_count,
        random_seed):
    luds_to_predictions = {}
    target_names_to_models = {}
    log_losses = []

    for target_col_id in range(target_count):
        # EXTRACT TARGET COLUMN.
        all_target_col_names = [f'match_{i}_outcome' for i in range(target_count)]
        X = train_test_df.drop(columns=all_target_col_names)

        target_col_name = f'match_{target_col_id}_outcome'
        y = train_test_df[[target_col_name]]

        # TRANSLATE Y VALUES TO INTS.
        tartet_mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
        y[target_col_name] = y[target_col_name].apply(lambda x: tartet_mapping[x])

        # # LOG Y VALUES & COUNTS.
        # print(f"Unique y values: {y[target_col_name].unique()}")
        # print(f"y value counts: {y[target_col_name].value_counts()}")

        # TRAIN & TEST MODELS.
        group_kfold_shuffle = GroupKFoldShuffle(n_splits=fold_count, shuffle=True, random_state=random_seed)
        folds = list(group_kfold_shuffle.split(X, y, ruleset_names))
        for fold_index, (train_index, test_index) in enumerate(folds):
            print(f"Fold {fold_index + 1}/{fold_count}")

            # SPLIT DATA.
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lud_rules_train, lud_rules_test = lud_rules.iloc[train_index], lud_rules.iloc[test_index]

            train_pool = Pool(X_train, y_train)
            test_pool = Pool(X_test, y_test)

            # TRAIN MODEL.
            model = CatBoostClassifier(**cat_params, random_seed=random_seed)
            model.fit(
                train_pool,
                eval_set=test_pool,
                early_stopping_rounds=early_stopping_round_count,
                verbose=False
            )

            # STORE MODEL.
            if target_col_name not in target_names_to_models:
                target_names_to_models[target_col_name] = []

            target_names_to_models[target_col_name].append(model)

            # STORE LOG LOSS.
            log_losses.append(model.get_best_score()['validation']['MultiClass'])

            # STORE PREDICTIONS.
            predictions = model.predict_proba(X_test)

            for test_lud_index, lud_text in enumerate(lud_rules_test):
                if lud_text not in luds_to_predictions:
                    EXPECTED_CLASS_COUNT = 3
                    luds_to_predictions[lud_text] = np.empty((target_count, EXPECTED_CLASS_COUNT))

                luds_to_predictions[lud_text][target_col_id] = predictions[test_lud_index]

    print(f"Log losses: {log_losses}")

    mean_log_loss = np.mean(log_losses)
    print(f"Mean log loss: {mean_log_loss}")

    return target_names_to_models, luds_to_predictions, mean_log_loss

def CreateEnsemble(
        cat_params,
        early_stopping_round_count,
        fold_count,
        target_count,
        output_filename_prefix):
    # TRAIN & TEST MODELS.
    ruleset_names, lud_rules, df = GetPreprocessedData()
    target_names_to_models, luds_to_predictions, mean_log_loss = TrainModels(
        ruleset_names=ruleset_names,
        lud_rules=lud_rules,
        train_test_df=df,
        cat_params=cat_params,
        early_stopping_round_count=early_stopping_round_count,
        fold_count=fold_count,
        target_count=target_count,
        random_seed=12345
    )

    # MAYBE SAVE RESULTS.
    if output_filename_prefix is not None:
        # SAVE MODELS.
        loss_text = f'{int(mean_log_loss * 10000)}'
        with open(f'models/{output_filename_prefix}_{loss_text}.pkl', 'wb') as models_file:
            joblib.dump(target_names_to_models, models_file)

        # SAVE PREDICTIONS.
        with open(f'predictions/{output_filename_prefix}_pred_match_outcomes_{loss_text}.json', 'w') as predictions_file:
            luds_to_prediction_lists = {
                lud: predictions.tolist() 
                for lud, predictions 
                in luds_to_predictions.items()
            }
            json.dump(luds_to_prediction_lists, predictions_file, indent=4)

    return mean_log_loss

def Objective(trial, fold_count):
    ruleset_names, lud_rules, df = GetPreprocessedData()

    cat_params = {
        "iterations": trial.suggest_int('iterations', 200, 15000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "depth": trial.suggest_int("depth", 2, 15),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10, log=True),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ['Bayesian', 'Bernoulli', 'Poisson', 'No']),
        "grow_policy": 'SymmetricTree',
        "loss_function": 'MultiClass',
        "task_type": "GPU"
    }

    target_names_to_models, luds_to_predictions, mean_log_loss = TrainModels(
        ruleset_names=ruleset_names,
        lud_rules=lud_rules,
        train_test_df=df,
        cat_params=cat_params,
        early_stopping_round_count=50,
        fold_count=fold_count,
        target_count=4,
        random_seed=42
    )

    return mean_log_loss

def GetOptimalConfig(fold_count, trial_count):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: Objective(trial, fold_count), n_trials=trial_count)

    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=2))

    best_score = study.best_trial.value
    output_filepath = f'configs/catboost_sim_predictor_{int(best_score * 100000)}_{trial_count}trials.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)

if __name__ == '__main__':
    # GetOptimalConfig(fold_count=5, trial_count=25)

    ## 0.8025706970275595
    # CreateEnsemble(
    #     cat_params = {
    #         "iterations": 10219, 
    #         "learning_rate": 0.010964241393786744, 
    #         "depth": 10, 
    #         "l2_leaf_reg": 0.0012480029901784353, 
    #         "grow_policy": "SymmetricTree", 
    #         "max_ctr_complexity": 6,
    #         "loss_function": 'MultiClass',
    #         "task_type": "GPU"
    #     },
    #     early_stopping_round_count = 50,
    #     fold_count = 5,
    #     target_count = 4,
    #     output_filename_prefix = 'organizer_game_baseline'
    # )

    ## 0.7952134290668611
    CreateEnsemble(
        cat_params = {
            "iterations": 733,
            "learning_rate": 0.00869727888616565,
            "depth": 7,
            "l2_leaf_reg": 2.30645929367163e-05,
            "bootstrap_type": "Poisson",
            "grow_policy": 'SymmetricTree',
            "loss_function": 'MultiClass',
            "task_type": "GPU"
        },
        early_stopping_round_count = 50,
        fold_count = 5,
        target_count = 4,
        output_filename_prefix = 'organizer_game'
    )