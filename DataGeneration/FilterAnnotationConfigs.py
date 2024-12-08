# python FilterAnnotationConfigs.py ReannotationConfigs_4Agents_Seed42/Machine1/ ReannotationConfigs_4Agents_Seed42/Machine1_Filtered

import sys
import json
import os
import glob

raw_configs_path = sys.argv[1].rstrip('/')
filtered_configs_path = sys.argv[2].rstrip('/')

os.makedirs(filtered_configs_path, exist_ok=True)

checked_game_count, saved_game_count = 0, 0
for raw_config_path in glob.glob(f'{raw_configs_path}/*.json'):
    checked_game_count += 1

    with open(raw_config_path, 'r') as raw_config_file:
        raw_config = json.load(raw_config_file)
    
    output_already_exists = os.path.exists(raw_config['outDir'])
    if output_already_exists:
        continue

    lud_path = raw_config['gameName']
    with open(lud_path, 'r') as lud_file:
        lud = lud_file.read()

    if '(players 2)' not in lud:
        continue

    with open(f'{filtered_configs_path}/{os.path.basename(raw_config_path)}', 'w') as filtered_config_file:
        json.dump(raw_config, filtered_config_file, indent=4)

    saved_game_count += 1

print(f'Checked {checked_game_count} games, saved {saved_game_count} games to {filtered_configs_path}.')