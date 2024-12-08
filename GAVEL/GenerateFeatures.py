import json
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import sys
import numpy as np
import socket

sys.path.append('GAVEL/lib')
from Game import Game

sys.path.append('DataGeneration/NewGameGeneration')
from RuleMinification import MinifyRules

def GenerateFeatures(fold_id):
    concepts_futures = []
    ruleset_names = []
    minified_luds = []
    with ThreadPoolExecutor(max_workers=4) as worker_pool:
        if fold_id is not None:
            with open(f'GAVEL/elite_generated_luds/fold_{fold_id}.json', 'r') as elite_luds_file:
                luds = json.load(elite_luds_file)
        else:
            with open('GAVEL/organizer_lud_preprocessing/UniqueLuds.json', 'r') as luds_file:
                all_luds = json.load(luds_file)

            np.random.seed(0)
            luds = np.random.choice(all_luds, 16, replace=False).tolist()

        print(len(luds))
        for lud_text in tqdm(luds, desc='Beginning to generate features'):
            game = Game(lud_text)
            
            trial_count = 100
            regenerate = True
            concepts_future = worker_pool.submit(
                game.GetConceptsDf, 
                trial_count, 
                regenerate,
                f'_{trial_count}'
            )
            concepts_futures.append(concepts_future)

            minified_luds.append(MinifyRules(lud_text))

            ruleset_name_pattern = r'\(game \"(.*?)\"'
            ruleset_name_match = re.search(ruleset_name_pattern, lud_text, re.DOTALL)
            ruleset_name = ruleset_name_match.group(1)
            ruleset_names.append(ruleset_name)

    concepts = []
    for concepts_future in tqdm(concepts_futures, desc='Getting concepts'):
        concepts_df = concepts_future.result()
        if concepts_df is not None:
            concepts.append(concepts_df)

    all_concepts_df = pd.concat(concepts)

    column_count = len(all_concepts_df.columns)
    all_concepts_df.insert(column_count, 'LudRules', minified_luds)
    all_concepts_df.insert(column_count+1, 'GameRulesetName', ruleset_names)

    if fold_id is not None:
        all_concepts_df.to_csv(f'GAVEL/generated_csvs/fold_{fold_id}_games_and_features.csv', index=False)
    else:
        hostname = socket.gethostname()
        all_concepts_df.to_csv(f'GAVEL/generated_csvs/organizer_{hostname}_games_and_features.csv', index=False)

if __name__ == '__main__':
    # 0, 1, 3, 4 ran on mlserver3
    # 2, 5 ran on UbuntuDesktop
    for fold_id in [5]:
        GenerateFeatures(fold_id)

    # GenerateFeatures(None)