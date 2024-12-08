import json
import numpy as np

from TrainLGBM import GetPreprocessedData, TrainModels
# from TrainCatBoost import GetPreprocessedData, TrainModels

# Dropped features:
#   - Run 1: ['NumTopSites']
#   - Run 2: ['MovesPerSecond']
#   - Run 3: ['Arithmetic', 'MoveDistanceChangeAverage']
if __name__ == '__main__':
    ruleset_names, all_data = GetPreprocessedData(split_agent_features = True)

    # # Running left.
    # gbt_params = {
    #     "num_iterations": 10000,
    #     "learning_rate": 0.03184567466358953,
    #     "num_leaves": 196,
    #     "max_depth": 17,
    #     "min_child_samples": 94,
    #     "subsample": 0.8854325308371437,
    #     "colsample_bytree": 0.9612980174610098,
    #     "colsample_bynode": 0.6867101420064379,
    #     "reg_alpha": 1.593152807295967e-05,
    #     "reg_lambda": 4.8636580199114866e-08,
    #     "extra_trees": False
    # }

    # Running center.
    gbt_params = {
        "num_iterations": 10000,
        "learning_rate": 0.02556180399737767,
        "num_leaves": 247,
        "max_depth": 16,
        "min_child_samples": 72,
        "subsample": 0.6261097426808137,
        "colsample_bytree": 0.8480868413996958,
        "reg_alpha": 3.7437548643071,
        "reg_lambda": 0.0010293013793948255,
    }

    # RANDOM_SEED = 42 # Running left.
    RANDOM_SEED = 42069 # Running center.
    _, best_rmse = TrainModels(ruleset_names, all_data, gbt_params, early_stopping_round_count = 50, fold_count = 5, random_seed = RANDOM_SEED)
    print(f'Baseline RMSE: {best_rmse}')

    candidate_dropped_columns = [col_name for col_name in all_data.columns if col_name != 'utility_agent1']
    best_dropped_columns = []

    while True:
        # MEASURE IMPACT OF DROPPING EACH COLUMN.
        candidate_drop_scores = []
        for col in candidate_dropped_columns:
            if col in best_dropped_columns:
                continue

            dropped_columns = best_dropped_columns + [col]
            data = all_data.drop(dropped_columns, axis=1)
            _, score = TrainModels(ruleset_names, data, gbt_params, early_stopping_round_count = 50, fold_count = 5, random_seed = RANDOM_SEED)

            candidate_drop_scores.append(score)

        # MAYBE STOP DROPPING COLUMNS.
        if len(candidate_drop_scores) == 0:
            break

        best_candidate_score = min(candidate_drop_scores)
        if best_candidate_score >= best_rmse:
            break

        # UPDATE COLUMN SETS.
        best_candidate_index = candidate_drop_scores.index(best_candidate_score)
        best_dropped_columns.append(candidate_dropped_columns[best_candidate_index])

        candidate_dropped_columns = list(np.array(candidate_dropped_columns)[np.array(candidate_drop_scores) < best_rmse])
        candidate_dropped_columns = list(set(candidate_dropped_columns) - set(best_dropped_columns))

        print(f'Best candidate RMSE: {best_candidate_score} with dropped columns: {candidate_dropped_columns}')
        print(f'Remaining candidates: {len(candidate_dropped_columns)}')

        # UPDATE BEST RMSE.
        best_rmse = best_candidate_score

    # RECORD BEST RESULT.
    print(f'Best RMSE: {best_rmse} with dropped columns: {best_dropped_columns}')

    output_filepath = f'configs/lgbm_{int(best_rmse * 100000)}_feature_selection.json'
    with open(output_filepath, 'w') as f:
        json.dump(best_dropped_columns, f, indent=4)