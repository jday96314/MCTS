# java -Xms20g -Xmx20g -da -dsa -XX:+UseStringDeduplication -jar DataGeneration/Ludii-1.3.13.jar --parallel-eval-multi-games-multi-agents --num-cores-total 8 --num-threads-per-trial 4 --json-files DataGeneration/MatchConfigs/Batch1/*.json --max-wall-time 800

import os
import numpy as np
import json
import glob
import hashlib

# PICK RANDOM AGENTS.
SELECTIONS = ['UCB1', 'UCB1GRAVE', 'ProgressiveHistory', 'UCB1Tuned']
EXPLORATION_CONSTS = ['0.1', '0.6', '1.41421356237']
PLAYOUTS = ['random', 'MAST', 'NST']
SCORE_BOUNDS = ['true', 'false']

AGENT_PAIRS_PER_GAME = 15
TRIALS_PER_CONFIG = 8

# DETERMINE HOW LOAD SHOULD BE DISTRIBUTED.
COMPUTERS = [
    # 5800x3d
    {
        "thread_count": 8,
        "performance_multiplier": 1.66,
    },
    # # 9950x
    # {
    #     "thread_count": 16,
    #     "performance_multiplier": 1.35,
    # },
    # # 5950x
    # {
    #     "thread_count": 16,
    #     "performance_multiplier": 0.688,
    # }
]
computer_selection_weights = [
    computer['thread_count'] * computer['performance_multiplier']
    for computer in COMPUTERS
]
computer_selection_weights = np.array(computer_selection_weights) / np.sum(computer_selection_weights)


# FIND LUDS.
game_names = os.listdir('DataGeneration/GameFeatures/SyntheticGames/Batch1')
lud_paths = [
    f'DataGeneration/NewGameGeneration/GeneratedGames_qwen_5shot_V2/{game_name}.lud' 
    for game_name 
    in game_names
]
for lud_path in lud_paths:
    assert os.path.exists(lud_path)

# GENERATE TASKS.
json_configs_by_worker = [[] for _ in range(len(COMPUTERS))]
for lud_path in lud_paths:
    for _ in range(AGENT_PAIRS_PER_GAME):
        # PICK AGENT PAIR.
        selection_0 = np.random.choice(SELECTIONS)
        exploration_const_0 = np.random.choice(EXPLORATION_CONSTS)
        playout_0 = np.random.choice(PLAYOUTS)
        score_bounds_0 = np.random.choice(SCORE_BOUNDS)
        agent_0_str=f'algorithm=MCTS;tree_reuse=true;selection={selection_0},explorationconstant={exploration_const_0};qinit=PARENT;playout={playout_0.lower()},playoutturnlimit=200;use_score_bounds={score_bounds_0};num_threads=4;friendly_name=MCTS-{selection_0}-{exploration_const_0}-{playout_0}-{score_bounds_0}'

        selection_1 = np.random.choice(SELECTIONS)
        exploration_const_1 = np.random.choice(EXPLORATION_CONSTS)
        playout_1 = np.random.choice(PLAYOUTS)
        score_bounds_1 = np.random.choice(SCORE_BOUNDS)
        agent_1_str=f'algorithm=MCTS;tree_reuse=true;selection={selection_1},explorationconstant={exploration_const_1};qinit=PARENT;playout={playout_1.lower()},playoutturnlimit=200;use_score_bounds={score_bounds_1};num_threads=4;friendly_name=MCTS-{selection_1}-{exploration_const_1}-{playout_1}-{score_bounds_1}'
        
        agent_pair = [agent_0_str, agent_1_str]

        agent_pair_hash = hashlib.sha256(json.dumps(agent_pair).encode()).hexdigest()[:8]
        
        # SELECT COMPUTER.
        computer_index = np.random.choice(range(len(COMPUTERS)), p=computer_selection_weights)
        computer_performance_multiplier = COMPUTERS[computer_index]['performance_multiplier']
        
        # VALIDATE LUD.
        lud_name = lud_path.split('/')[-1].replace('.lud', '')
        root_features_dir_path = 'DataGeneration/GameFeatures/SyntheticGames/Batch1'
        features_dir_path = f'{root_features_dir_path}/{lud_name}'
        if not os.path.exists(features_dir_path):
            continue

        # GENERATE CONFIG.
        config = {
            "iterationLimit": 75000,
            "outputRawResults": True,
            "numTrials": TRIALS_PER_CONFIG,
            "treatGameNameAsFilepath": True,
            "ruleset": "",
            "outDir": f"./DataGeneration/Matches/Batch1/game_{lud_name}/agent_pair_{agent_pair_hash}",
            "outputAlphaRankData": False,
            "agentStrings": agent_pair,
            "gameName": lud_path,
            "outputSummary": True,
            "gameLengthCap": 650,
            "warmingUpSecs": 1,
            "thinkingTime": 1 / computer_performance_multiplier,
        }

        # UPDATE WORK ASSIGNED TO COMPUTER.
        json_configs_by_worker[computer_index].append(config)

# SAVE TASKS.
root_configs_dir = f'DataGeneration/MatchConfigs/Batch1'
os.makedirs(root_configs_dir, exist_ok=True)

for worker_index, json_configs in enumerate(json_configs_by_worker):
    machine_configs_dir = f'{root_configs_dir}/Machine{worker_index}'
    os.makedirs(machine_configs_dir, exist_ok=True)

    for config_index, config in enumerate(json_configs):
        with open(f'{machine_configs_dir}/config_{config_index}.json', 'w') as f:
            json.dump(config, f, indent=4)