import numpy as np
import pandas as pd
import datetime
import glob
import hashlib
import json
import os

def GenerateAnalysisConfigs(lud_count, thinking_seconds_per_lud):
    unique_lud_paths = glob.glob('DataGeneration/UniqueLuds/*.lud')
    selected_lud_paths = np.random.choice(unique_lud_paths, lud_count, replace=False)

    config_paths = []
    for lud_path in selected_lud_paths:
        config = {
            "gameName": lud_path,
            "ruleset": "Standard",
            "agentString": "algorithm=MCTS;tree_reuse=false;selection=UCB1Tuned,explorationconstant=1.41421356237;qinit=PARENT;playout=random,playoutturnlimit=200;use_score_bounds=false;num_threads=2;friendly_name=MCTS-UCB1Tuned-1.41421356237-random-false",
            "outputFilepath": f"StartingPositionEvaluation/RuntimeAnalysis/MctsEvals/{hashlib.md5(lud_path.encode()).hexdigest()}.txt",
            "treatGameNameAsFilepath": True,
            "thinkingTime": thinking_seconds_per_lud
        }
        config_text = json.dumps(config, indent=4)

        config_hash = hashlib.md5(config_text.encode()).hexdigest()
        config_filepath = f'StartingPositionEvaluation/RuntimeAnalysis/Configs/{config_hash}.json'
        with open(config_filepath, 'w') as config_file:
            config_file.write(config_text)

        config_paths.append(config_filepath)

    return config_paths

def MeasureBatchRuntime(config_filepaths):
    start_time = datetime.datetime.now()

    ANALYZER_FILEPATH = 'StartingPositionEvaluation/AnalyzeGameStartingPosition.jar'
    config_args = ' '.join(config_filepaths)
    os.system(f'java -Xms14g -Xmx14g -da -dsa -XX:+UseStringDeduplication -jar {ANALYZER_FILEPATH} --json-files {config_args}')

    end_time = datetime.datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    return runtime_seconds

if __name__ == '__main__':
    lud_counts = []
    thinking_seconds_per_lud = []
    total_runtimes = []
    for _ in range(10):
        lud_count = np.random.randint(1, 100)
        thinking_seconds = np.random.uniform(1, 30)
        analysis_configs = GenerateAnalysisConfigs(lud_count, thinking_seconds)

        total_runtime_seconds = MeasureBatchRuntime(analysis_configs)
        
        lud_counts.append(lud_count)
        thinking_seconds_per_lud.append(thinking_seconds)
        total_runtimes.append(total_runtime_seconds)

    df = pd.DataFrame({
        'LudCount': lud_counts,
        'ThinkingSecondsPerLud': thinking_seconds_per_lud,
        'TotalRuntimeSeconds': total_runtimes
    })
    output_filepath = f'StartingPositionEvaluation/RuntimeAnalysis/Measurements/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df.to_csv(output_filepath, index=False)