import polars as pl
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import numpy as np
import os
from tqdm import tqdm
import gc
import re

import sys
sys.path.append('DataGeneration/NewGameGeneration')

from RuleMinification import MinifyRules, UnminifyRules

# vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --tensor-parallel-size 2 --max-model-len 8192 --kv-cache-dtype fp8 --host 0.0.0.0 --gpu-memory-util 0.95
# vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --host 0.0.0.0 --gpu-memory-util 0.95 --tensor-parallel-size 2

def GetUniqueLudsToEnglish():
    # LOAD GAME DATA.
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    df = df.to_pandas()

    # EXTRACT RULES.
    luds_to_english: dict[str, list[str]] = {}
    luds_to_ruleset_names: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        # STORE RULES.
        lud = row['LudRules']
        english = row['EnglishRules']
        if lud not in luds_to_english:
            luds_to_english[lud] = []

        luds_to_english[lud].append(english)

        # STORE RULESET NAME.
        ruleset_name = row['GameRulesetName']
        if lud not in luds_to_ruleset_names:
            luds_to_ruleset_names[lud] = []

        luds_to_ruleset_names[lud].append(ruleset_name)

    # CLEANUP.
    del df
    gc.collect()

    return luds_to_english, luds_to_ruleset_names

def GenerateNewGame_V1(
        api_client: OpenAI, 
        sampling_temperature: float,
        example_game_names: list[str], 
        example_english_rules: list[str], 
        example_lud_rules: list[str], 
        output_directory_path: str):
    # FORM PROMPT.
    conversation_history = [{
        "role": "system",
        "content": "You are an expert at inventing creative games and describing them with Ludii game rules. The rules you create in each response will be different from the ones you created previously in the conversation. All Ludii game code will be syntactially correct."
    }]
    USER_INSTRUCTIONS = "Generate rules for a new game. Express the rules both in English and in Ludii game code. Your response should be formatted as JSON with a game name, an English description of the rules, and a Ludii description of the rules.\n\nBe creative and feel free to create new games no one has ever heard of."
    for game_name, english, lud in zip(example_game_names, example_english_rules, example_lud_rules):
        conversation_history.append({
            "role": "user",
            "content": USER_INSTRUCTIONS
        })

        conversation_history.append({
            "role": "assistant",
            "content": json.dumps({
                "game_name": game_name,
                "english_rules": english,
                "lud_rules": lud
            }, indent=4)
        })

    conversation_history.append({
        "role": "user",
        "content": USER_INSTRUCTIONS
    })

    # SPECIFY RESPONSE SCHEMA.
    JSON_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "game_name": {
                "type": "string"
            },
            "english_rules": {
                "type": "string"
            },
            "lud_rules": {
                "type": "string"
            }
        },
        "required": ["game_name", "english_rules", "lud_rules"]
    }

    # GENERATE GAME.
    try:
        response = api_client.chat.completions.create(
            # model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            model="solidrust/Qwen2.5-32B-Instruct-AWQ",
            messages=conversation_history,
            max_tokens=2048,
            frequency_penalty=0.02,
            temperature=sampling_temperature,
            extra_body={"guided_json": JSON_RESPONSE_SCHEMA}
        )

        response_json_text = response.choices[0].message.content
        parsed_response = json.loads(response_json_text)
    except:
        # Try again with more aggressive truncation.
        try:
            response = api_client.chat.completions.create(
                # model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                model="solidrust/Qwen2.5-32B-Instruct-AWQ",
                messages=conversation_history[-3:],
                max_tokens=2048,
                frequency_penalty=0.02,
                temperature=sampling_temperature,
                extra_body={"guided_json": JSON_RESPONSE_SCHEMA}
            )

            response_json_text = response.choices[0].message.content
            parsed_response = json.loads(response_json_text)
        except Exception as error:
            print(f'Error running game generation request: {error}')
            return None
        
    # VALIDATE GAME.
    game_name = parsed_response['game_name']
    if (game_name in example_game_names) or (len(game_name.strip()) == 0):
        return None
    
    english_rules = parsed_response['english_rules']
    if (english_rules in example_english_rules) or (len(english_rules.strip()) == 0):
        return None
    
    lud_rules = parsed_response['lud_rules']
    if (lud_rules in example_lud_rules) or (len(lud_rules.strip()) == 0):
        return None

    if lud_rules.count('(') != lud_rules.count(')'):
        return None
    
    if lud_rules.count('{') != lud_rules.count('}'):
        return None
    
    # FIX WEIRD QUOTES.
    english_rules = english_rules.replace('„', '"').replace('“', '"')
    lud_rules = lud_rules.replace('„', '"').replace('“', '"')

    # RECORD MINIFIED RULES.
    parsed_response = {
        "game_name": game_name,
        "english_rules": english_rules,
        "lud_rules": lud_rules
    }
    
    # SAVE GAME.
    game_hash_hex = hashlib.sha256(game_name.encode()).hexdigest()[:8]
    game_filepath = f'{output_directory_path}/{game_hash_hex}.json'

    os.makedirs(output_directory_path, exist_ok=True)
    with open(game_filepath, 'w') as file:
        json.dump(parsed_response, file, indent=4)
        print('Saved game to:', game_filepath)

def ParseV2Response(response_text: str):
    response_pattern = r'.*?Game Name: "(.*)"\n\nEnglish Rules: "(.*)"\n\nLudii Rules: `(.*)`'

    match = re.match(response_pattern, response_text.strip(), re.DOTALL)
    if match is None:
        return None
    
    game_name = match.group(1)
    english_rules = match.group(2)
    lud_rules = match.group(3)

    return {
        "game_name": game_name,
        "english_rules": english_rules,
        "lud_rules": lud_rules
    }

def GenerateNewGame_V2(
        api_client: OpenAI, 
        sampling_temperature: float,
        example_game_names: list[str], 
        example_english_rules: list[str], 
        example_lud_rules: list[str], 
        output_directory_path: str):
    # FORM PROMPT.
    conversation_history = [{
        "role": "system",
        "content": "You are an expert at inventing creative games and describing them with Ludii game rules. The rules you create in each response will be different from the ones you created previously in the conversation. All Ludii game code will be syntactially correct."
    }]
    USER_INSTRUCTIONS = "Generate rules for a new game. Format your response like ```Game Name: \"{game_name}\"\n\nEnglish Rules: \"{english_rules}\"\n\nLudii Rules: `{ludii_game_code}`\n```\n\nBe creative and feel free to create new games no one has ever heard of."
    for game_name, english, lud in zip(example_game_names, example_english_rules, example_lud_rules):
        conversation_history.append({
            "role": "user",
            "content": USER_INSTRUCTIONS
        })

        conversation_history.append({
            "role": "assistant",
            "content": f'Game Name: "{game_name}"\n\nEnglish Rules: "{english}"\n\nLudii Rules: `{lud}`'
        })

    conversation_history.append({
        "role": "user",
        "content": USER_INSTRUCTIONS
    })

    # GENERATE GAME.
    try:
        response = api_client.chat.completions.create(
            model="solidrust/Qwen2.5-32B-Instruct-AWQ",
            messages=conversation_history,
            max_tokens=2048,
            frequency_penalty=0.02,
            temperature=sampling_temperature,
        )

        response_text = response.choices[0].message.content
        parsed_response = ParseV2Response(response_text)
    except:
        # Try again with more aggressive truncation.
        try:
            response = api_client.chat.completions.create(
                model="Qwen/Qwen2.5-32B-Instruct-AWQ",
                messages=conversation_history[-5:],
                max_tokens=2048,
                frequency_penalty=0.02,
                temperature=sampling_temperature,
            )

            response_text = response.choices[0].message.content
            parsed_response = ParseV2Response(response_text)
        except Exception as error:
            print(f'Error running game generation request: {error}')
            return None
        

    # VALIDATE GAME.
    if parsed_response is None:
        return None
    
    game_name = parsed_response['game_name']
    if (game_name in example_game_names) or (len(game_name.strip()) == 0):
        return None
    
    english_rules = parsed_response['english_rules']
    if (english_rules in example_english_rules) or (len(english_rules.strip()) == 0):
        return None
    
    lud_rules = parsed_response['lud_rules']
    if (lud_rules in example_lud_rules) or (len(lud_rules.strip()) == 0):
        return None

    if lud_rules.count('(') != lud_rules.count(')'):
        return None
    
    if lud_rules.count('{') != lud_rules.count('}'):
        return None
    
    # FIX WEIRD QUOTES.
    english_rules = english_rules.replace('„', '"').replace('“', '"')
    lud_rules = lud_rules.replace('„', '"').replace('“', '"')

    # RECORD MINIFIED RULES.
    parsed_response = {
        "game_name": game_name,
        "english_rules": english_rules,
        "lud_rules": lud_rules
    }
    
    # SAVE GAME.
    game_hash_hex = hashlib.sha256(game_name.encode()).hexdigest()[:8]
    game_filepath = f'{output_directory_path}/{game_hash_hex}.json'

    os.makedirs(output_directory_path, exist_ok=True)
    with open(game_filepath, 'w') as file:
        json.dump(parsed_response, file, indent=4)
        print('Saved game to:', game_filepath)

if __name__ == '__main__':
    # LOAD DATA.
    luds_to_english, luds_to_ruleset_names = GetUniqueLudsToEnglish()
    unique_luds = np.array(list(luds_to_english.keys()))

    # INITIALIZE API CLIENT.
    api_client = OpenAI(
        api_key='EMPTY',
        base_url='http://mlserver:8000/v1'
    )

    # GENERATE NEW GAMES.
    # NEW_GAME_COUNT = 10_000
    NEW_GAME_COUNT = 200
    futures = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for _ in range(NEW_GAME_COUNT):
            SHOT_COUNT = 3
            example_lud_rules = np.random.choice(unique_luds, size=SHOT_COUNT, replace=False)
            example_game_names = [
                np.random.choice(luds_to_ruleset_names[lud])
                for lud in example_lud_rules
            ]
            example_english_rules = [
                np.random.choice(luds_to_english[lud])
                for lud in example_lud_rules
            ]

            TEMPERATURE = 0.7
            # OUTPUT_DIRECTORY_PATH = 'DataGeneration/NewGameGeneration/GeneratedGames_qwen_5shot'
            OUTPUT_DIRECTORY_PATH = 'DataGeneration/NewGameGeneration/GeneratedGames_qwen_3shot'
            future = executor.submit(
                GenerateNewGame_V2, 
                api_client, 
                TEMPERATURE, 
                example_game_names, 
                example_english_rules, 
                example_lud_rules, 
                OUTPUT_DIRECTORY_PATH
            )
            futures.append(future)

    for future in tqdm(futures):
        future.result()