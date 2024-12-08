import json
import glob

game_filepaths = glob.glob('DataGeneration/NewGameGeneration/GeneratedGames_qwen_5shot_V2/*.json')
for game_filepath in game_filepaths:
    with open(game_filepath, 'r') as game_file:
        game = json.load(game_file)
    
    lud_rules = game['lud_rules']

    output_filepath = game_filepath.replace('.json', '.lud')
    with open(output_filepath, 'w') as output_file:
        output_file.write(lud_rules)