# vllm serve codellama/CodeLlama-7b-hf --host 0.0.0.0 --gpu-memory-util 0.92 --enable-lora --lora-modules sql-lora=Desktop/Shared/ExpansionDrive2/TreeSearch/GAVEL/models/4en4_16rank_2spl/0/final_model --max-model-len 15000
# vllm serve codellama/CodeLlama-7b-hf --host 0.0.0.0 --gpu-memory-util 0.92 --enable-lora --lora-modules sql-lora=Desktop/Shared/ExpansionDrive2/TreeSearch/GAVEL/models/4en4_16rank_2spl/1/final_model --max-model-len 15000 --port 8001

import glob
import json
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import pandas as pd
import re

import sys
sys.path.append('DataGeneration/NewGameGeneration')
from RuleMinification import MinifyRules, UnminifyRules

sys.path.append('GAVEL/lib')
from Game import Game

def FindParentheticalExpressions(lud_text):
    expression_stack = []
    expressions = []
    start_end_index_pairs = []

    for char_index, char in enumerate(lud_text):
        if char == '(':
            expression_stack.append(char_index)
        elif char == ')':
            if len(expression_stack) > 0:
                start_char_index = expression_stack.pop()
                expressions.append(lud_text[start_char_index:char_index+1])
                start_end_index_pairs.append((start_char_index, char_index+1))

    return expressions, start_end_index_pairs

def GetRandomPrefixMiddleSuffix(lud_text):
    expressions, start_end_index_pairs = FindParentheticalExpressions(lud_text)
    assert len(start_end_index_pairs) > 0

    middle_start_index, middle_end_index = start_end_index_pairs[np.random.randint(len(start_end_index_pairs))]
    
    prefix = lud_text[:middle_start_index]
    middle = lud_text[middle_start_index:middle_end_index]
    suffix = lud_text[middle_end_index:]

    return prefix, middle, suffix

def FillInMiddle(api_client, prefix, suffix):
    try:
        prompt = f'▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>'
        completion = api_client.completions.create(
            model = 'codellama/CodeLlama-7b-hf',
            prompt = prompt,
            max_tokens = 512,
            temperature = 1,
            frequency_penalty = 0,
            stop = ['▁<EOT>', suffix[:80], "</s>"]
        )

        return completion.choices[0].text
    except Exception as error:
        print('Failed to generate completion due to', error)
        return None

def GetAllLuds():
    with open('GAVEL/organizer_lud_preprocessing/UniqueLuds.json', 'r') as luds_file:
        all_luds = json.load(luds_file)

    return all_luds

def GetFoldLudsAndQualityMetrics(fold_count, fold_id, min_quality):
    all_luds = GetAllLuds()

    quality_metrics_path = f'GAVEL/organizer_lud_preprocessing/fold_{fold_id}_luds_to_quality_metrics.json'
    with open(quality_metrics_path, 'r') as quality_metrics_file:
        luds_to_quality_metrics = json.load(quality_metrics_file)

    filtered_luds_to_quality_metrics = {}
    all_fold_quality_metrics = []
    total_fold_lud_count = 0
    for lud_index, lud_text in enumerate(all_luds):
        if lud_index % fold_count != fold_id:
            continue
        else:
            total_fold_lud_count += 1

        if lud_text not in luds_to_quality_metrics.keys():
            continue

        quality_metric = luds_to_quality_metrics[lud_text]
        all_fold_quality_metrics.append(quality_metric)

        if quality_metric < min_quality:
            continue

        filtered_luds_to_quality_metrics[lud_text] = quality_metric

    print(f'Fold {fold_id}: {len(filtered_luds_to_quality_metrics)}/{total_fold_lud_count} luds had quality >= {min_quality}, average quality: {np.mean(all_fold_quality_metrics):.4f}')

    return filtered_luds_to_quality_metrics

def PrepareMapElites(fold_count, fold_id, min_quality):
    # LOAD LUDS AND QUALITY METRICS.
    filtered_luds_to_quality_metrics = GetFoldLudsAndQualityMetrics(fold_count, fold_id, min_quality)
    
    # SAMPLE LUDS.
    np.random.seed(0)
    luds = list(filtered_luds_to_quality_metrics.keys())
    luds = np.random.choice(luds, size=20, replace=False)

    # LOAD PREPROCESSING DATA..
    with open('GAVEL/organizer_lud_preprocessing/pca_pipeline.pkl', 'rb') as pca_pipeline_file:
        pca_pipeline = pickle.load(pca_pipeline_file)

    with open('GAVEL/organizer_lud_preprocessing/luds_to_reduced_features.json') as luds_to_reduced_features_file:
        luds_to_reduced_features = json.load(luds_to_reduced_features_file)

    # MAP ELITES PARAMETERS.
    dim_0_range = (-10, 32)
    dim_1_range = (-12, 14)
    dimension_bin_count = 40

    # PREPARE GRIDS.
    elite_fitness_scores = np.ones((dimension_bin_count, dimension_bin_count)) * -1
    elite_luds = np.empty((dimension_bin_count, dimension_bin_count), dtype=object)
    for lud_text in luds:
        if lud_text not in luds_to_reduced_features.keys():
            print(f'WARNING: Lud not in reduced features!')
            continue

        x, y = luds_to_reduced_features[lud_text]
        dim_0_index = int((x - dim_0_range[0]) / (dim_0_range[1] - dim_0_range[0]) * dimension_bin_count)
        dim_0_index = np.clip(dim_0_index, 0, dimension_bin_count-1)

        dim_1_index = int((y - dim_1_range[0]) / (dim_1_range[1] - dim_1_range[0]) * dimension_bin_count)
        dim_1_index = np.clip(dim_1_index, 0, dimension_bin_count-1)

        if elite_fitness_scores[dim_0_index, dim_1_index] < filtered_luds_to_quality_metrics[lud_text]:
            elite_fitness_scores[dim_0_index, dim_1_index] = filtered_luds_to_quality_metrics[lud_text]
            elite_luds[dim_0_index, dim_1_index] = lud_text

    return elite_fitness_scores, elite_luds, dim_0_range, dim_1_range, dimension_bin_count, pca_pipeline

def TryGetMutatedGame(lud_text, api_client):
    unminified_lud_text = UnminifyRules(lud_text)
    prefix, middle, suffix = GetRandomPrefixMiddleSuffix(unminified_lud_text)
    predicted_middle = FillInMiddle(api_client, prefix, suffix)

    if predicted_middle is None:
        return None
    
    middle_without_whitespace = middle.replace(' ', '').replace('\n', '').replace('\t', '')
    predicted_middle_without_whitespace = predicted_middle.replace(' ', '').replace('\n', '').replace('\t', '')
    if middle_without_whitespace == predicted_middle_without_whitespace:
        return None

    mutated_lud_text = f'{prefix}{predicted_middle}{suffix}'
    mutated_game = Game(mutated_lud_text)
    if not mutated_game.IsPlayable():
        return None

    return mutated_game

def TryGetMutatedGames(luds_text, api_client, thread_count, attempt_count):
    mutated_game_futures = []
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        for _ in range(attempt_count):
            lud_text = np.random.choice(luds_text)
            mutated_game_future = executor.submit(TryGetMutatedGame, lud_text, api_client)
            mutated_game_futures.append(mutated_game_future)

    mutated_games = []
    for mutated_game_future in tqdm(mutated_game_futures, desc='Getting mutated games'):
        mutated_game = mutated_game_future.result()
        if mutated_game is not None:
            mutated_games.append(mutated_game)

    return mutated_games

if __name__ == '__main__':
    # INITIALIZE API CLIENT.
    api_client = OpenAI(
        api_key='EMPTY',
        base_url='http://mlserver:8005/v1'
    )

    # PREPARE MAP ELITES.
    FOLD_ID = 5
    elite_fitness_scores, elite_luds, dim_0_range, dim_1_range, dimension_bin_count, pca_pipeline = PrepareMapElites(
        fold_count=6,
        fold_id=FOLD_ID,
        min_quality=0.5)
    
    # GET ELITE MUTATED GAMES
    GENERATION_COUNT = 240
    for generation in tqdm(range(GENERATION_COUNT), desc="Generation"):
        # GET MUTATED GAMES.
        elite_luds_text = elite_luds.flatten()
        elite_luds_text = [lud_text for lud_text in elite_luds_text if lud_text is not None]
        mutated_games = TryGetMutatedGames(elite_luds_text, api_client, thread_count=5, attempt_count=150)
        
        print(f'Got {len(mutated_games)} mutated games.')

        # EVALUATE MUTATED GAME FITNESS SCORES.
        fitness_futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for mutated_game in mutated_games:
                fitness_future = executor.submit(mutated_game.ComputeOverallQuality)
                fitness_futures.append(fitness_future)
                
        mutated_game_fitness_scores = []
        for fitness_future in tqdm(fitness_futures, desc='Getting fitness scores'):
            mutated_game_fitness_scores.append(fitness_future.result())

        # UPDATE ELITES.
        new_elite_count = 0
        for mutated_game, fitness_score in zip(mutated_games, mutated_game_fitness_scores):
            concepts_df = mutated_game.GetConceptsDf()
            if concepts_df is None:
                continue

            concepts_vector = concepts_df.to_numpy().flatten()
            concepts_vector = np.nan_to_num(concepts_vector)
            
            x, y = pca_pipeline.transform([concepts_vector])[0]
            dim_0_index = int((x - dim_0_range[0]) / (dim_0_range[1] - dim_0_range[0]) * dimension_bin_count)
            dim_0_index = np.clip(dim_0_index, 0, dimension_bin_count-1)

            dim_1_index = int((y - dim_1_range[0]) / (dim_1_range[1] - dim_1_range[0]) * dimension_bin_count)
            dim_1_index = np.clip(dim_1_index, 0, dimension_bin_count-1)

            if elite_fitness_scores[dim_0_index, dim_1_index] < fitness_score:
                elite_fitness_scores[dim_0_index, dim_1_index] = fitness_score
                elite_luds[dim_0_index, dim_1_index] = mutated_game.Lud

                new_elite_count += 1

        print(f'Updated {new_elite_count} elites.')

        if (generation + 1) % 10 == 0:
            with open(f'GAVEL/elite_generated_luds/fold_{FOLD_ID}.json', 'w') as elite_luds_file:
                json.dump(elite_luds.flatten().tolist(), elite_luds_file)

    # SAVE NEW ELITES.
    filtered_elite_luds = elite_luds.flatten()
    filtered_elite_luds = [lud_text for lud_text in filtered_elite_luds if lud_text is not None]
    
    all_original_luds = GetAllLuds()
    filtered_elite_luds = [lud_text for lud_text in filtered_elite_luds if lud_text not in all_original_luds]

    with open(f'GAVEL/elite_generated_luds/fold_{FOLD_ID}.json', 'w') as elite_luds_file:
        json.dump(filtered_elite_luds, elite_luds_file)


    # Checking random variance in starting position evaluations.
    # std +/- 0.02 for games with non-zero evaluations (which seem constant).
    '''
    lud_filepaths = glob.glob('DataGeneration/UniqueLuds/*.lud')
    games = []
    for lud_filepath in lud_filepaths:
        with open(lud_filepath, 'r') as lud_file:
            game = Game(lud_file.read())
            games.append(game)

    games = np.random.choice(games, size=10, replace=False)

    balances_by_game = [[] for game in games]
    for _ in range(10):
        for game_index, game in enumerate(games):
            balance = game.GetStartingPositionEvaluation(analysis_runtime_seconds=16)
            balances_by_game[game_index].append(balance)

    stds = [np.std(balances) for balances in balances_by_game]
    print(stds)
    print(np.mean(stds))
    '''