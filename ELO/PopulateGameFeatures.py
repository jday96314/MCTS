import polars as pl
import json

import sys
sys.path.append('./')
from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

def GetPreprocessedData(games_csv_path, starting_evals_json_path):
    # LOAD & FILTER DATA.
    df = pl.read_csv(games_csv_path)

    df = df.to_pandas()
    df = df.drop_duplicates(subset=['LudRules'])
    df = pl.DataFrame(df)

    ruleset_names = df['GameRulesetName']
    lud_rules = df['LudRules']
    
    all_dropped_columns = DROPPED_COLUMNS + ['utility_agent1'] + AGENT_COLS
    df = df.drop(filter(lambda x: x in df.columns, all_dropped_columns))

    # ADD MCTS EVALUATION FEATURES.
    with open(starting_evals_json_path) as f:
        luds_to_mcts_evals = json.load(f)

    df = df.to_pandas()
    df['mcts_eval'] = [luds_to_mcts_evals[lud] for lud in lud_rules]

    # ENSURE LUD RULES WEREN'T DROPPED.
    df['LudRules'] = lud_rules

    return lud_rules, df

if __name__ == '__main__':
    # LOAD LUDS & FEATURES.
    organizer_rules, organizer_df = GetPreprocessedData(
        '/mnt/data01/data/TreeSearch/data/from_organizers/train.csv',
        'StartingPositionEvaluation/Evaluations/OrganizerGames/JSON/MCTS-UCB1Tuned-1.41421356237-random-false-16s.json'
    )

    SUPPLEMENTAL_DATASET_NAME = '2024-10-23_15-10-16'
    supplemental_rules, supplemental_df = GetPreprocessedData(
        f'GAVEL/generated_csvs/complete_datasets/{SUPPLEMENTAL_DATASET_NAME}.csv',
        'GAVEL/generated_csvs/complete_datasets/starting_position_evals_233_16s.json'
    )

    supplemental_df = supplemental_df[organizer_df.columns]

    print('Organizer data shape:', organizer_df.shape)
    print('Supplemental data shape:', supplemental_df.shape)

    # SAVE (UNLABLED) SUPPLEMENTAL DATA.
    supplemental_df.to_csv('ELO/CSV/supplemental_data.csv', index=False)

    # MERGE WITH ELO RATINGS.
    for elo_computation_run in range(3):
        labeled_organizer_df = organizer_df.copy()
        
        with open(f'ELO/ground_truth/run_{elo_computation_run}.json') as f:
            elo_data = json.load(f)

        luds_to_components_to_options_to_elos = elo_data['elos']
        luds_to_components_to_options_to_match_counts = elo_data['match_counts']
        luds_to_components_to_options_to_unique_agent_counts = elo_data['unique_agent_counts']

        for lud in organizer_rules:
            for component in luds_to_components_to_options_to_elos[lud]:
                for option in luds_to_components_to_options_to_elos[lud][component]:
                    elo = luds_to_components_to_options_to_elos[lud][component][option]
                    min_match_count = min(luds_to_components_to_options_to_match_counts[lud][component].values())

                    if min_match_count == 0:
                        elo = None

                    label_name = f'{component}_{option}_elo'
                    lud_row_index = (organizer_rules == lud).to_numpy().nonzero()[0][0]
                    labeled_organizer_df.at[lud_row_index, label_name] = elo

        output_filepath = f'ELO/CSV/labeled_organizer_data_{elo_computation_run}.csv'
        labeled_organizer_df.to_csv(output_filepath, index=False)