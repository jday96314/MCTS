# java -Xms14g -Xmx14g -da -dsa -XX:+UseStringDeduplication -jar DataGeneration/Ludii-1.3.13.jar --parallel-compute-concepts-multiple-games --num-cores-total 8 --num-threads-per-job 4 --json-files DataGeneration/FeatureGenerationConfigs/SyntheticGames/Batch1/Machine0/* --max-wall-time 800
# java -Xms28g -Xmx28g -da -dsa -XX:+UseStringDeduplication -jar DataGeneration/Ludii-1.3.13.jar --parallel-compute-concepts-multiple-games --num-cores-total 16 --num-threads-per-job 4 --json-files DataGeneration/FeatureGenerationConfigs/SyntheticGames/Batch1/Machine1/* --max-wall-time 800
# java -Xms28g -Xmx28g -da -dsa -XX:+UseStringDeduplication -jar DataGeneration/Ludii-1.3.13.jar --parallel-compute-concepts-multiple-games --num-cores-total 16 --num-threads-per-job 4 --json-files DataGeneration/FeatureGenerationConfigs/SyntheticGames/Batch1/Machine2/* --max-wall-time 800

import glob
import json
import numpy as np
import os
import subprocess

# Returns True if the config is valid, False otherwise.
def ValidateConfig(config_filepath):
    # Try running Ludii with the config.
    command = f'java -Xms14g -Xmx14g -da -dsa -XX:+UseStringDeduplication -jar DataGeneration/Ludii-1.3.13.jar --parallel-compute-concepts-multiple-games --num-cores-total 8 --num-threads-per-job 4 --json-files {config_filepath} --max-wall-time 800'
    
    try:
        subprocess.run(command, shell=True, check=True, timeout=2)
        return True
    except subprocess.TimeoutExpired:
        subprocess.run('pkill -f java', shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

if __name__ == '__main__':
    # INPUT_DIR_PATH = 'DataGeneration/UniqueLuds'
    # OUTPUT_TRIALS_PATH = 'DataGeneration/GameTrials/GamesFromOrganizers'
    # OUTPUT_FEATURES_PATH = 'DataGeneration/GameFeatures/GamesFromOrganizers'
    # OUTPUT_CONFIGS_PATH = 'DataGeneration/FeatureGenerationConfigs/GamesFromOrganizers'

    INPUT_DIR_PATH = 'DataGeneration/NewGameGeneration/GeneratedGames_qwen_5shot_V2'
    OUTPUT_TRIALS_PATH = 'DataGeneration/GameTrials/SyntheticGames/Batch1'
    OUTPUT_FEATURES_PATH = 'DataGeneration/GameFeatures/SyntheticGames/Batch1'
    OUTPUT_CONFIGS_PATH = 'DataGeneration/FeatureGenerationConfigs/SyntheticGames/Batch1'

    # FIND LUDS TO ANNOTATE.
    lud_paths = glob.glob(f'{INPUT_DIR_PATH}/*.lud')

    # lud_paths = lud_paths[:20]

    # DETERMINE HOW LOAD SHOULD BE DISTRIBUTED.
    COMPUTERS = [
        # 5800x3d
        {
            "thread_count": 8,
            "performance_multiplier": 1.32,
        },
        # 9950x
        {
            "thread_count": 16,
            "performance_multiplier": 1.35,
        },
        # 5950x
        {
            "thread_count": 16,
            "performance_multiplier": 0.688,
        }
    ]
    computer_selection_weights = [
        computer['thread_count'] * computer['performance_multiplier']
        for computer in COMPUTERS
    ]
    computer_selection_weights = np.array(computer_selection_weights) / np.sum(computer_selection_weights)

    # GENERATE TASKS.
    json_configs_by_worker = [[] for _ in range(len(COMPUTERS))]
    for lud_path in lud_paths:
        # GENERATE CONFIG.
        config = {
            "gameName": lud_path,
            "ruleset": "",
            "numTrials": 100,
            "trialsDir": f"{OUTPUT_TRIALS_PATH}/{lud_path.split('/')[-1].replace('.lud', '')}",
            "conceptsDir": f"{OUTPUT_FEATURES_PATH}/{lud_path.split('/')[-1].replace('.lud', '')}",
            "treatGameNameAsFilepath": True
        }

        # SELECT COMPUTER.
        computer_index = np.random.choice(range(len(COMPUTERS)), p=computer_selection_weights)
        
        # SAVE CONFIG.
        json_configs_by_worker[computer_index].append(config)

    # SAVE TASKS.
    os.makedirs(OUTPUT_CONFIGS_PATH, exist_ok=True)

    success_count = 0
    total_count = 0
    for worker_index, json_configs in enumerate(json_configs_by_worker):
        # ENSURE OUTPUT DIRECTORY EXISTS.
        machine_output_directory = f'{OUTPUT_CONFIGS_PATH}/Machine{worker_index}'
        os.makedirs(machine_output_directory, exist_ok=True)

        # SAVE & FILTER CONFIGS.
        for config_index, config in enumerate(json_configs):
            # WRITE CONFIG.
            output_filepath = f'{machine_output_directory}/{config_index}.json'
            with open(output_filepath, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            # VALIDATE CONFIG.
            if not ValidateConfig(output_filepath):
                os.remove(output_filepath)
            else:
                success_count += 1

            total_count += 1
    
    print(f'Success rate: {success_count}/{total_count} ({success_count / total_count * 100:.2f}%)')