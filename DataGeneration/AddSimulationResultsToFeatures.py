import json
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import polars as pl

def GetLudsToOutcomes(config_directory_paths, expected_agent_pair_count):
    # FIND CONFIGS.
    config_filepaths = []
    for config_directory_path in config_directory_paths:
        config_filepaths += glob.glob(f'{config_directory_path.rstrip('/')}/*.json')

    # LOAD OUTCOMES.
    luds_to_outcomes = {}
    failure_count = 0
    for config_filepath in tqdm(config_filepaths):
        # LOAD CONFIG.
        with open(config_filepath, 'r') as config_file:
            config = json.load(config_file)

        # LOAD LUDII GAME RULES.
        lud_path = config['gameName']
        try:
            with open(lud_path, 'r') as lud_file:
                lud = lud_file.read()
        except:
            lud_path = f'DataGeneration/{lud_path}'
            with open(lud_path, 'r') as lud_file:
                lud = lud_file.read()

        # LOAD OUTCOME.
        outcome_path = f"{config['outDir']}/raw_results.csv"
        try:
            outcome_df = pd.read_csv(outcome_path)
        except:
            try:
                outcome_path = f"DataGeneration/{outcome_path.lstrip('./')}"
                outcome_df = pd.read_csv(outcome_path)
            except Exception as e:
                # print(f"Error loading outcome at {outcome_path}: {e}")
                failure_count += 1

                # Create placeholder outcome df.
                outcome_df = pd.DataFrame({
                    'utilities': ['0;0']
                })

        assert len(outcome_df) == 1

        outcome_text = outcome_df['utilities'].iloc[0]
        agent_1_utility = float(outcome_text.split(';')[0])

        # DETERMINE AGENT PAIR.
        agent_pair_id = int(config['outDir'].split('/')[-1].split('_')[-1])

        # STORE OUTCOME.
        if lud not in luds_to_outcomes:
            luds_to_outcomes[lud] = np.array([np.nan] * expected_agent_pair_count)

        luds_to_outcomes[lud][agent_pair_id] = agent_1_utility

    print(f"Failed to load {failure_count}/{len(config_filepaths)} outcomes.")

    return luds_to_outcomes

def GetGameDf():
    # LOAD GAME DATA.
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    df = df.to_pandas()

    return df

def FormDataset(
        config_directory_paths, 
        agent_pair_count,
        drop_duplicate_luds, 
        dropped_columns,
        output_filepath):
    # LOAD DATA.
    luds_to_outcomes = GetLudsToOutcomes(config_directory_paths, agent_pair_count)
    games_df = GetGameDf()

    # ADD OUTCOMES TO GAMES.
    for match_index in range(agent_pair_count):
        games_df[f'match_{match_index}_outcome'] = games_df['LudRules'].apply(
            lambda lud: luds_to_outcomes[lud][match_index]
        )

    # MAYBE DROP DUPLICATE LUDS.
    if drop_duplicate_luds:
        games_df = games_df.drop_duplicates(subset=['LudRules'])

    # MAYBE DROP COLUMNS.
    games_df = games_df.drop(columns=dropped_columns)

    # SAVE GAMES.
    games_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    FormDataset(
        config_directory_paths = glob.glob('DataGeneration/ReannotationConfigs_4Agents_Seed42/*'),
        agent_pair_count = 4,
        drop_duplicate_luds = True,
        dropped_columns = ['utility_agent1', 'agent1', 'agent2'],
        output_filepath = 'DataGeneration/CompleteDatasets/OrganizerGamesAndFeatures_4Agents_Dedup_NoOrigLabels.csv'
    )