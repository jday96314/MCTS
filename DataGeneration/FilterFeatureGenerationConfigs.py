import os
import json
import glob

MACHINE_ID = 1

INPUT_CONFIGS_PATH = f'DataGeneration/FeatureGenerationConfigs/GamesFromOrganizers/Machine{MACHINE_ID}'
FEATURES_DIRECTORY_PATH = f'DataGeneration/GameFeatures/GamesFromOrganizers'
OUTPUT_CONFIGS_PATH = f'{INPUT_CONFIGS_PATH.rstrip('/')}_Filtered'

os.makedirs(OUTPUT_CONFIGS_PATH, exist_ok=True)

checked_game_count, saved_game_count = 0, 0
for raw_config_path in glob.glob(f'{INPUT_CONFIGS_PATH}/*.json'):
    checked_game_count += 1

    with open(raw_config_path, 'r') as raw_config_file:
        raw_config = json.load(raw_config_file)
    
    output_already_exists = os.path.exists(raw_config['conceptsDir'])
    if output_already_exists:
        continue

    with open(f'{OUTPUT_CONFIGS_PATH}/{os.path.basename(raw_config_path)}', 'w') as filtered_config_file:
        json.dump(raw_config, filtered_config_file, indent=4)

    saved_game_count += 1

print(f'Checked {checked_game_count} games, saved {saved_game_count} games to {OUTPUT_CONFIGS_PATH}.')