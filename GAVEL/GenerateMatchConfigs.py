# java -Xms30g -Xmx30g -da -dsa -XX:+UseStringDeduplication -jar DataGeneration/Ludii-1.3.13.jar --parallel-eval-multi-games-multi-agents --num-cores-total 16 --num-threads-per-trial 4 --json-files GAVEL/configs/match_configs/fold_0\&1/* --max-wall-time 800

import os
import numpy as np
import json
import glob
import hashlib
import sys

sys.path.append('GAVEL/lib')
from Game import Game

# PICK RANDOM AGENTS.
SELECTIONS = ['UCB1', 'UCB1GRAVE', 'ProgressiveHistory', 'UCB1Tuned']
EXPLORATION_CONSTS = ['0.1', '0.6', '1.41421356237']
PLAYOUTS = ['random', 'MAST', 'NST']
SCORE_BOUNDS = ['true', 'false']
# MACHINE_SPEED_COEF = 1.82 # 9950x
MACHINE_SPEED_COEF = 0.96 # 5950x
AGENT_PAIRS_PER_GAME = 15
TRIALS_PER_CONFIG = 8

FOLD_ID = 5

configs_dir = f'GAVEL/configs/match_configs/fold_{FOLD_ID}'
os.makedirs(configs_dir, exist_ok=True)

if __name__ == '__main__':
    # FIND LUDS.
    with open(f'GAVEL/elite_generated_luds/fold_{FOLD_ID}.json', 'r') as elite_luds_file:
        luds = json.load(elite_luds_file)

    games = [Game(lud) for lud in luds]
    game_filepaths = [game.LudFilepath for game in games]

    # GENERATE MATCH CONFIGS.
    for lud_filepath, lud in zip(game_filepaths, luds):
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

            # PICK OUTPUT PATH.
            agent_pair_hash = hashlib.sha256(json.dumps(agent_pair).encode()).hexdigest()[:8]
            lud_hash = hashlib.sha256(lud.encode()).hexdigest()[:8]

            match_out_dir = f'GAVEL/ludii_player_outputs/matches/game_{lud_hash}/match_{agent_pair_hash}'

            # GENERATE CONFIG.
            config = {
                "iterationLimit": 75000,
                "outputRawResults": True,
                "numTrials": TRIALS_PER_CONFIG,
                "treatGameNameAsFilepath": True,
                "ruleset": "",
                "outDir": match_out_dir,
                "outputAlphaRankData": False,
                "agentStrings": agent_pair,
                "gameName": lud_filepath,
                "outputSummary": True,
                "gameLengthCap": 650,
                "warmingUpSecs": 1,
                "thinkingTime": 1 / MACHINE_SPEED_COEF,
            }

            with open(f'{configs_dir}/{lud_hash}_{agent_pair_hash}.json', 'w') as config_file:
                json.dump(config, config_file, indent=4)