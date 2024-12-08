import os
import hashlib
import json
import subprocess
import polars as pl
import datetime
import glob
import pandas as pd
import re
import argparse

def CreateGenerationConfigs(luds, trials_per_lud = 15):
    os.makedirs('temp/luds', exist_ok = True)
    os.makedirs('temp/configs', exist_ok = True)

    config_filepaths = []
    for lud in luds:
        if '(pair 2123)' in lud:
            print('Skipping poisonous lud...')
            continue
    
        # SAVE LUD.
        lud_hash = hashlib.md5(lud.encode('utf-8')).hexdigest()
        
        if os.path.exists(f'temp/concepts/{lud_hash}/Concepts.csv'):
            print('Skipping completed lud...')
            continue
        
        lud_path = f'temp/luds/{lud_hash}.lud'
        with open(lud_path, 'w') as lud_file:
            lud_file.write(lud)

        # FORM CONFIG.
        config = {
            "gameName": lud_path,
            "ruleset": "",
            "numTrials": trials_per_lud,
            "trialsDir": f'temp/trials/{lud_hash}',
            "conceptsDir": f'temp/concepts/{lud_hash}',
            "treatGameNameAsFilepath": True
        }

        # SAVE CONFIG.
        config_filepath = f'temp/configs/{lud_hash}.json'
        with open(config_filepath, 'w') as config_file:
            json.dump(config, config_file, indent=4)

        config_filepaths.append(config_filepath)

    return config_filepaths

def GenerateFeatures(config_filepaths):
    # GENERATE "CONCEPTS" FILES.
    LUDII_FILEPATH = 'Ludii-1.3.13.jar'
    config_filepaths_text = ' '.join(config_filepaths)
    # Originally used "--num-cores-total 4" - bumped to 8 for SMT run.
    command = f'java -Xms7g -Xmx7g -da -dsa -XX:+UseStringDeduplication -jar {LUDII_FILEPATH} --parallel-compute-concepts-multiple-games --num-cores-total 8 --num-threads-per-job 4 --json-files {config_filepaths_text} --max-wall-time 800'
    subprocess.run(command, shell=True, check=True)

    # RETRY FOR MISSING RESULTS.
    for _ in range(3):
        filtered_config_filepaths = []
        for config_filepath in config_filepaths:
            with open(config_filepath, 'r') as config_file:
                config = json.load(config_file)
    
            concepts_filepath = f'{config["conceptsDir"]}/Concepts.csv'
            if not os.path.exists(concepts_filepath):
                filtered_config_filepaths.append(config_filepath)
    
        config_filepaths_text = ' '.join(filtered_config_filepaths)
        # Originally used "--num-cores-total 4" - bumped to 8 for SMT run.
        command = f'taskset 0xf java -Xms7g -Xmx7g -da -dsa -XX:+UseStringDeduplication -jar {LUDII_FILEPATH} --parallel-compute-concepts-multiple-games --num-cores-total 8 --num-threads-per-job 4 --json-files {config_filepaths_text} --max-wall-time 800'
        subprocess.run(command, shell=True, check=True)
    
def GenerateFeaturesCsv(original_csv_filepath, output_filepath):
    # GENERATE FEATURES.
    input_df = pl.read_csv(original_csv_filepath)
    unique_luds = input_df['LudRules'].unique().sort()
    annotation_config_filepaths = CreateGenerationConfigs(unique_luds)

    start_time = datetime.datetime.now()
    GenerateFeatures(annotation_config_filepaths)
    end_time = datetime.datetime.now()

    print('Runtime:', end_time - start_time)

    # LOAD RESULTS.
    processed_ruleset_names = []
    processed_luds = []
    concept_dfs = []
    for config_filepath in annotation_config_filepaths:
        # LOAD CONFIG.
        with open(config_filepath, 'r') as config_file:
            config = json.load(config_file)

        lud_filepath = config['gameName']
        concepts_filepath = f'{config["conceptsDir"]}/Concepts.csv'

        # VERIFY RESULT EXISTS.
        if not os.path.exists(concepts_filepath):
            print(f'WARNING: No result found at {concepts_filepath}')
            continue

        # LOAD LUD & RULESET NAME.
        with open(lud_filepath, 'r') as lud_file:
            lud = lud_file.read()

        processed_luds.append(lud)
            
        ruleset_name_pattern = r'\(game \"(.*?)\"'
        ruleset_name_match = re.search(ruleset_name_pattern, lud, re.DOTALL)
        ruleset_name = ruleset_name_match.group(1)
        processed_ruleset_names.append(ruleset_name)

        # LOAD CONCEPTS (I.E. FEATURES).
        concepts_df = pd.read_csv(concepts_filepath)
        concept_dfs.append(concepts_df)

    # SAVE RESULTS.
    all_concepts_df = pd.concat(concept_dfs)
    all_concepts_df['LudRules'] = processed_luds

    all_concepts_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_name', type=str, required=True)
    parser.add_argument('--run_id', type=int, required=True)
    args = parser.parse_args()

    GenerateFeaturesCsv(original_csv_filepath = 'train.csv', output_filepath = f'reannotated_organizer_games_{args.device_name}_r{args.run_id}.csv')
    GenerateFeaturesCsv(original_csv_filepath = '2024-11-08_16-32-19.csv', output_filepath = f'reannotated_extra_games_{args.device_name}_r{args.run_id}.csv')
    GenerateFeaturesCsv(original_csv_filepath = '2024-11-25_21-41-25_new.csv', output_filepath = f'reannotated_extra_new_games_{args.device_name}_r{args.run_id}.csv')
