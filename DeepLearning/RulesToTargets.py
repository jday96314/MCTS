import polars as pl
import numpy as np
import json

# LOAD RULES.
try:
    df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
except:
    df = pl.read_csv('data/from_organizers/train.csv')

all_lud_rules = df['LudRules'].to_list()
all_english_rules = df['EnglishRules'].to_list()

lud_rules_to_english_rules = {}
for lud_rules, english_rules in zip(all_lud_rules, all_english_rules):
    if lud_rules not in lud_rules_to_english_rules:
        lud_rules_to_english_rules[lud_rules] = set()

    lud_rules_to_english_rules[lud_rules] = english_rules

# LOAD TARGETS.
target_filepaths = [
    'data/rules_to_pca_utilities.json',
    'data/rules_to_mean_agent1_utilities.json',
    'data/rules_to_mean_absolute_agent1_utilities.json',
    'data/lud_rules_to_clusters/10_both_players.json',
    'data/lud_rules_to_clusters/10_player1.json',
    'data/lud_rules_to_clusters/10_player2.json',
]
target_names = [
    'pca_utilities',
    'mean_agent1_utilities',
    'mean_absolute_agent1_utilities',
    'both_players_clusters',
    'player1_clusters',
    'player2_clusters',
]
target_dicts = [json.load(open(filepath)) for filepath in target_filepaths]

# MERGE RULES & TARGETS.
records = []
for lud_rules, english_rules in lud_rules_to_english_rules.items():
    record = {'lud_rules': lud_rules, 'english_rules': english_rules}

    for target_name, target_dict in zip(target_names, target_dicts):
        record[target_name] = target_dict[lud_rules]

    records.append(record)

df = pl.DataFrame(records).to_pandas()
df.to_csv('data/rules_to_derived_targets.csv', index=False)