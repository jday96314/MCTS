import os
import polars as pl
import numpy as np
import pandas as pd
import joblib

from TrainRegressors import COMPONENT_NAMES_TO_OPTIONS

def GeneratePredictions(model_filepath, lud_features_path, data_to_annotate_path, output_path):
    # GENERATE PREDICTIONS.
    components_to_models = joblib.load(model_filepath)
    lud_features = pd.read_csv(lud_features_path)
    
    luds_to_components_to_predictions = {
        lud: {
            component_name: []
            for component_name, options in COMPONENT_NAMES_TO_OPTIONS.items()
        }
        for lud in lud_features['LudRules'].values
    }
    for agent_component_name, models_by_fold in components_to_models.items():
        for model_data in models_by_fold:
            model = model_data['model']
            feature_names = model_data['selected_feature_names']
            training_luds = model_data['training_luds']

            oof_rows = lud_features[~lud_features['LudRules'].isin(training_luds)]
            oof_features = oof_rows[feature_names]

            oof_predictions = model.predict(oof_features)

            for lud, prediction in zip(oof_rows['LudRules'].values, oof_predictions):
                luds_to_components_to_predictions[lud][agent_component_name].append(prediction)

    luds_to_components_to_predictions = {
        lud: {
            component_name: np.mean(predictions, axis=0)
            for component_name, predictions in components_to_predictions.items()
        }
        for lud, components_to_predictions in luds_to_components_to_predictions.items()
    }

    luds_to_components_to_options_to_predictions = {}
    for lud, components_to_predictions in luds_to_components_to_predictions.items():
        components_to_options_to_predictions = {}
        for component_name, options in COMPONENT_NAMES_TO_OPTIONS.items():
            options_to_predictions = {}
            for option_index, option in enumerate(options):
                option_predictions = components_to_predictions[component_name][option_index]
                options_to_predictions[option] = option_predictions

            components_to_options_to_predictions[component_name] = options_to_predictions

        luds_to_components_to_options_to_predictions[lud] = components_to_options_to_predictions

    # ADD ANNOTATIONS TO FULL DATASET.
    dataset_to_annotate = pl.read_csv(data_to_annotate_path).to_pandas()
    
    agent1_selection_elos = []
    agent1_exploration_const_elos = []
    agent1_playout_elos = []
    agent1_score_bounds_elos = []

    agent2_selection_elos = []
    agent2_exploration_const_elos = []
    agent2_playout_elos = []
    agent2_score_bounds_elos = []

    for _, row in dataset_to_annotate.iterrows():
        lud = row.LudRules
        agent1 = row.agent1
        agent2 = row.agent2

        agent1_option_values = agent1.split('-')[1:]
        agent1_selection_elos.append(luds_to_components_to_options_to_predictions[lud]['selection'][agent1_option_values[0]])
        agent1_exploration_const_elos.append(luds_to_components_to_options_to_predictions[lud]['exploration_const'][agent1_option_values[1]])
        agent1_playout_elos.append(luds_to_components_to_options_to_predictions[lud]['playout'][agent1_option_values[2].replace('random', 'Random200')])
        agent1_score_bounds_elos.append(luds_to_components_to_options_to_predictions[lud]['score_bounds'][agent1_option_values[3]])

        agent2_option_values = agent2.split('-')[1:]
        agent2_selection_elos.append(luds_to_components_to_options_to_predictions[lud]['selection'][agent2_option_values[0]])
        agent2_exploration_const_elos.append(luds_to_components_to_options_to_predictions[lud]['exploration_const'][agent2_option_values[1]])
        agent2_playout_elos.append(luds_to_components_to_options_to_predictions[lud]['playout'][agent2_option_values[2].replace('random', 'Random200')])
        agent2_score_bounds_elos.append(luds_to_components_to_options_to_predictions[lud]['score_bounds'][agent2_option_values[3]])

    # dataset_to_annotate['agent1_selection_elo'] = agent1_selection_elos
    # dataset_to_annotate['agent1_exploration_const_elo'] = agent1_exploration_const_elos
    # dataset_to_annotate['agent1_playout_elo'] = agent1_playout_elos
    # dataset_to_annotate['agent1_score_bounds_elo'] = agent1_score_bounds_elos

    # dataset_to_annotate['agent2_selection_elo'] = agent2_selection_elos
    # dataset_to_annotate['agent2_exploration_const_elo'] = agent2_exploration_const_elos
    # dataset_to_annotate['agent2_playout_elo'] = agent2_playout_elos
    # dataset_to_annotate['agent2_score_bounds_elo'] = agent2_score_bounds_elos

    mean_agent1_elos = np.mean([
        agent1_selection_elos,
        agent1_exploration_const_elos,
        agent1_playout_elos,
        agent1_score_bounds_elos
    ], axis=0)
    mean_agent2_elos = np.mean([
        agent2_selection_elos,
        agent2_exploration_const_elos,
        agent2_playout_elos,
        agent2_score_bounds_elos
    ], axis=0)

    elo_deltas = mean_agent1_elos - mean_agent2_elos

    dataset_to_annotate['mean_elo_delta'] = elo_deltas

    # SAVE PREDICTIONS.
    dataset_to_annotate.to_csv(output_path, index=False)

if __name__ == '__main__':
    MODEL_FILEPATHS = [
        'ELO/models/run_0_614088.pkl',
        'ELO/models/run_1_614965.pkl',
        'ELO/models/run_2_620859.pkl'
    ]
    DATASET_PATHS = [
        {
            'lud_features_path': 'ELO/CSV/labeled_organizer_data_0.csv',
            'data_to_annotate_path': '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv',
            # 'output_path': 'ELO/CSV/organizer_oof_predictions_{}.csv'
            'output_path': 'ELO/CSV/organizer_oof_pred_deltas_{}.csv'
        },
        {
            'lud_features_path': 'ELO/CSV/supplemental_data.csv',
            'data_to_annotate_path': 'GAVEL/generated_csvs/complete_datasets/2024-10-20_20-05-49.csv',
            # 'output_path': 'ELO/CSV/extra_v3_predictions_{}.csv'
            'output_path': 'ELO/CSV/extra_v3_pred_deltas_{}.csv'
        },
    ]

    for run_id, model_filepath in enumerate(MODEL_FILEPATHS):
        for dataset_paths in DATASET_PATHS:
            temp_dataset_paths = dataset_paths.copy()
            temp_dataset_paths['output_path'] = dataset_paths['output_path'].format(run_id)
            GeneratePredictions(model_filepath, **temp_dataset_paths)