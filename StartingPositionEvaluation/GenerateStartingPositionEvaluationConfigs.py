# java -Xms7g -Xmx7g -da -dsa -XX:+UseStringDeduplication -jar StartingPositionEvaluation/AnalyzeGameStartingPosition.jar --json-files StartingPositionEvaluation/EvalConfigs/OrganizerGames/*/*.json
# java -Xms7g -Xmx7g -da -dsa -XX:+UseStringDeduplication -jar StartingPositionEvaluation/AnalyzeGameStartingPosition.jar --json-files StartingPositionEvaluation/EvalConfigs/OrganizerGames/MCTS-UCB1Tuned-1.41421356237-MAST-false/*/*.json

import glob
import json
import os
from tqdm import tqdm

INPUT_LUDS_PATH = 'DataGeneration/UniqueLuds'
OUTPUT_CONFIGS_PATH = 'StartingPositionEvaluation/EvalConfigs/OrganizerGames'
OUTPUT_EVALS_PATH = 'StartingPositionEvaluation/Evaluations/OrganizerGames'

# SELECTION = 'UCB1Tuned' # 'UCB1', 'UCB1GRAVE', 'ProgressiveHistory', 'UCB1Tuned'
# EXPLORATION_CONST = '1.41421356237' # '0.1', '0.6', '1.41421356237'
# PLAYOUT = 'random' # 'random', 'MAST', 'NST'
# SCORE_BOUNDS = 'false' # 'true', 'false'

SELECTION = 'ProgressiveHistory'
EXPLORATION_CONST = '0.1'
PLAYOUT = 'NST'
SCORE_BOUNDS = 'true'

lud_paths = glob.glob('DataGeneration/UniqueLuds/*.lud')

# for thinking_seconds in tqdm([1,2,4,8,16]):
for thinking_seconds in tqdm([0.75]):
    runtime_text = f'{thinking_seconds}s'
    if thinking_seconds < 1:
        runtime_text = f'{int(thinking_seconds * 1000)}ms'

    for lud_path in lud_paths:
        lud_name = lud_path.split('/')[-1].replace('.lud', '')
        friendly_name = f'MCTS-{SELECTION}-{EXPLORATION_CONST}-{PLAYOUT}-{SCORE_BOUNDS}'
        config = {
            "gameName": lud_path,
            "ruleset": "Standard",
            "agentString": f"algorithm=MCTS;tree_reuse=true;selection={SELECTION},explorationconstant={EXPLORATION_CONST};qinit=PARENT;playout={PLAYOUT.lower()},playoutturnlimit=200;use_score_bounds={SCORE_BOUNDS};num_threads=2;friendly_name={friendly_name}",
            "outputFilepath": f"{OUTPUT_EVALS_PATH}/{friendly_name}/{runtime_text}/{lud_name}.txt",
            "treatGameNameAsFilepath": True,
            "thinkingTime": thinking_seconds
        }

        os.makedirs(f'{OUTPUT_CONFIGS_PATH}/{friendly_name}/{runtime_text}', exist_ok=True)
        with open(f'{OUTPUT_CONFIGS_PATH}/{friendly_name}/{runtime_text}/{lud_name}.json', 'w') as config_file:
            json.dump(config, config_file, indent=4)


        # config = {
        #     "gameName": lud_path,
        #     "ruleset": "Standard",
        #     "agentString": "algorithm=MCTS;tree_reuse=true;selection=UCB1Tuned,explorationconstant=1.41421356237;qinit=PARENT;playout=mast,playoutturnlimit=200;use_score_bounds=true;num_threads=2;friendly_name=MCTS-UCB1Tuned-1.41421356237-MAST-true",
        #     "outputFilepath": f"{OUTPUT_EVALS_PATH}/{thinking_seconds}s/{lud_name}.txt",
        #     "treatGameNameAsFilepath": True,
        #     "thinkingTime": thinking_seconds
        # }

        # os.makedirs(f'{OUTPUT_CONFIGS_PATH}/{thinking_seconds}s', exist_ok=True)
        # with open(f'{OUTPUT_CONFIGS_PATH}/{thinking_seconds}s/{lud_name}.json', 'w') as config_file:
        #     json.dump(config, config_file, indent=4)