import hashlib
import json
import subprocess
import re
from scipy.stats import hmean
import os
import pandas as pd
import numpy as np

class Game:
    def __init__(self, lud, path_prefix = ''):
        self.Lud = lud
        self.PathPrefix = path_prefix

        self.LudHash = hashlib.sha256(lud.encode()).hexdigest()
        self.LudFilepath = f'{self.PathPrefix}GAVEL/generated_luds/{self.LudHash}.lud'
        with open(self.LudFilepath, 'w') as lud_file:
            lud_file.write(lud)

    def IsPlayable(self):
        if self.Lud.count('(') != self.Lud.count(')'):
            return False

        ANALYSIS_RUNTIME_SECONDS = 0.25
        starting_position_evaluation = self.GetStartingPositionEvaluation(ANALYSIS_RUNTIME_SECONDS)

        return starting_position_evaluation is not None
    
    # Determined via concepts:
    # * Balance         - Balance  
    # * Agency          - DecisionMoves
    # * Coverage        - BoardCoverageDefault
    # * Completion      - Completion
    # * Decisiveness    - (1 - Drawishness)
    #
    # Determined via simulated matches (10 trials, 0.25s per move, 50 moves max)
    # * Strategic depth - "the proportion of games won by an MCTS agent against a random agent over n = 10 playouts"
    def ComputeOverallQuality(self):
        # SANITY CHECK GAME, 
        starting_position_eval = self.GetStartingPositionEvaluation(analysis_runtime_seconds=4)
        game_is_playable = starting_position_eval is not None
        if not game_is_playable:
            return -3
        
        game_is_reasonably_balanced = abs(starting_position_eval) < 0.5
        if not game_is_reasonably_balanced:
            return -2

        # ATTEMPT TO EVALUATE GAME.
        try:
            # COMPUTE CONCEPT BASED METRICS.
            balance, agency, coverage, completion, drawishness = self.GetConceptBasedMetrics()
            decisiveness = 1 - drawishness

            if agency < 0.5:
                return -1

            # COMPUTE STRATEGIC DEPTH.
            strategic_depth = self.GetStrategicDepth()

            # COMPUTE OVERALL QUALITY (HARMONIC MEAN OF METRICS).
            all_metrics = np.array([balance, agency, coverage, decisiveness, completion, strategic_depth])
            clipped_metrics = np.clip(all_metrics, 0.01, 1)
            overall_quality = hmean(clipped_metrics)
            
            return overall_quality
        except Exception as error:
            print(f'Error computing overall quality for {self.LudHash}: {error}')
            # Assumed to be runtime error caused by invalid game rules, so return bad quality.
            return -1
    
    def GetConceptBasedMetrics(self):
        concepts_df = self.GetConceptsDf()
        balance = concepts_df['Balance'].values[0]
        agency = concepts_df['DecisionMoves'].values[0]
        coverage = concepts_df['BoardCoverageDefault'].values[0]
        completion = concepts_df['Completion'].values[0]
        drawishness = concepts_df['Drawishness'].values[0]

        return balance, agency, coverage, completion, drawishness
    
    def GetStrategicDepth(self, match_count = 10, regenerate = False):
        config_filepath = f'{self.PathPrefix}GAVEL/configs/strategic_depth/{self.LudHash}.json'
        output_dir = f'{self.PathPrefix}GAVEL/ludii_player_outputs/strategic_depth/{self.LudHash}'

        # MAYBE RUN ANALYSIS.
        if (not os.path.exists(config_filepath)) or regenerate:
            config = {
                "iterationLimit": 75000,
                "outputRawResults": True,
                "numTrials": match_count,
                "treatGameNameAsFilepath": True,
                "ruleset": "",
                "outDir": output_dir,
                "outputAlphaRankData": False,
                "agentStrings": [
                    "Random",
                    "algorithm=MCTS;tree_reuse=true;selection=UCB1Tuned,explorationconstant=0.6;qinit=PARENT;playout=random,playoutturnlimit=200;use_score_bounds=false;num_threads=4;friendly_name=MCTS-UCB1Tuned-0.6-random-false"
                ],
                "gameName": self.LudFilepath,
                "outputSummary": True,
                "gameLengthCap": 50,
                "warmingUpSecs": 0.25,
                "thinkingTime": 0.25,
            }

            with open(config_filepath, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            command = f'java -Xms7g -Xmx7g -da -dsa -XX:+UseStringDeduplication -jar {self.PathPrefix}DataGeneration/Ludii-1.3.13.jar --parallel-eval-multi-games-multi-agents --num-cores-total 4 --num-threads-per-trial 4 --json-files {config_filepath} --max-wall-time 800'

            try:
                subprocess.run(command, shell=True, check=True)
            except Exception as error:
                return None
            
        # ATTEMPT TO LOAD RESULTS.
        try:
            results_filepath = f'{output_dir}/raw_results.csv'
            results_df = pd.read_csv(results_filepath)

            win_count = 0
            for agents, outcome in zip(results_df['agents'], results_df['utilities']):
                if agents.startswith("('Random'") and (outcome == "-1.0;1.0"):
                    win_count += 1
                if agents.startswith("('MCTS-UCB1Tuned-0.6-random-false'") and (outcome == "1.0;-1.0"):
                    win_count += 1

            return win_count / match_count
        except:
            return None


    def GetStartingPositionEvaluation(
            self, 
            analysis_runtime_seconds, 
            selection = 'UCB1Tuned',
            exploration_constant = '1.41421356237',
            playout = 'random',
            score_bounds = 'false',
            regenerate = False):
        # CHECK IF RESULT IS ALREADY CACHED.
        config_filepath = f'{self.PathPrefix}GAVEL/configs/starting_position_eval/{self.LudHash}.json'
        analysis_result_filepath = f'{self.PathPrefix}GAVEL/ludii_player_outputs/starting_position_evaluations/{self.LudHash}.txt'
        result_exists = False
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r') as config_file:
                old_config = json.load(config_file)

            runtime_matches = old_config['thinkingTime'] == analysis_runtime_seconds
            result_exists = os.path.exists(analysis_result_filepath) and runtime_matches

        # MAYBE RUN ANALYSIS.
        if (not result_exists) or regenerate:
            # CREATE ANALYSIS CONFIG.
            config = {
                "gameName": self.LudFilepath,
                "ruleset": "Standard",
                "agentString": f"algorithm=MCTS;tree_reuse=true;selection={selection},explorationconstant={exploration_constant};qinit=PARENT;playout={playout},playoutturnlimit=200;use_score_bounds={score_bounds};num_threads=2;friendly_name=MCTS-{selection}-{exploration_constant}-{playout}-{score_bounds}",
                "outputFilepath": analysis_result_filepath,
                "treatGameNameAsFilepath": True,
                "thinkingTime": analysis_runtime_seconds
            }

            with open(config_filepath, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            timeout_seconds = max(analysis_runtime_seconds * 4, 2)

            # RUN ANALYSIS.
            ANALYZER_FILEPATH = f'{self.PathPrefix}StartingPositionEvaluation/AnalyzeGameStartingPosition.jar'
            command = f'java -Xms14g -Xmx14g -da -dsa -XX:+UseStringDeduplication -jar {ANALYZER_FILEPATH} --json-files {config_filepath}'
            try:
                subprocess.run(command, shell=True, check=True, timeout=timeout_seconds)
            except Exception as error:
                return None
        
        # ATTEMPT TO PARSE ANALYSIS RESULTS.
        try:
            with open(analysis_result_filepath, 'r') as analysis_result_file:
                analysis_result = analysis_result_file.read()

            evaluation_pattern = r'Evaluation: (.*)'

            # Get last group, i.e. most recent result.
            match_results = re.findall(evaluation_pattern, analysis_result)
            evaluation = float(match_results[-1])

            return evaluation
        except:
            return None
        
    def GetConceptsDf(self, trial_count = 30, regenerate = False, path_suffix = ''):
        config_filepath = f'{self.PathPrefix}GAVEL/configs/concepts/{self.LudHash}{path_suffix}.json'
        trials_dir = f"{self.PathPrefix}GAVEL/ludii_player_outputs/trials/{self.LudHash}{path_suffix}"
        concepts_dir = f'{self.PathPrefix}GAVEL/ludii_player_outputs/concepts/{self.LudHash}{path_suffix}'
        concepts_filepath = f'{concepts_dir}/Concepts.csv'

        # MAYBE RUN ANALYSIS.
        if (not os.path.exists(config_filepath)) or regenerate:
            config = {
                "gameName": self.LudFilepath,
                "ruleset": "",
                "numTrials": trial_count,
                "trialsDir": trials_dir,
                "conceptsDir": concepts_dir,
                "treatGameNameAsFilepath": True
            }

            with open(config_filepath, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            command = f'java -Xms7g -Xmx7g -da -dsa -XX:+UseStringDeduplication -jar {self.PathPrefix}DataGeneration/Ludii-1.3.13.jar --parallel-compute-concepts-multiple-games --num-cores-total 4 --num-threads-per-job 4 --json-files {config_filepath} --max-wall-time 800'
            try:
                subprocess.run(command, shell=True, check=True)
            except Exception as error:
                return None
            
            # Maybe perform one retry, sometimes the Ludii player does not save results.
            if not os.path.exists(concepts_filepath):
                try:
                    subprocess.run(command, shell=True, check=True)
                except Exception as error:
                    return None
            
        # ATTEMPT TO LOAD RESULTS.
        try:
            concepts_df = pd.read_csv(concepts_filepath)
            return concepts_df
        except:
            return None