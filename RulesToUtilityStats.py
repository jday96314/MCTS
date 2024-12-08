import polars as pl
import numpy as np
import json

from ColumnNames import IRRELEVANT_COLS, OUTCOME_COUNT_COLS, AGENT_COLS, POISONOUS_COLS

# LOAD RULES & FEATURES.
try:
    df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
except:
    df = pl.read_csv('data/from_organizers/train.csv')

# LOAD TARGETS.
target_filepaths = [
    'data/rules_to_mean_agent1_utilities.json',
    'data/rules_to_mean_absolute_agent1_utilities.json',
]
target_names = [
    'mean_agent1_utilities',
    'mean_absolute_agent1_utilities'
]
target_dicts = [json.load(open(filepath)) for filepath in target_filepaths]

# MERGE RULES, FEATURES, & TARGETS.
records = []
already_seen_lud_rules = set()
for row in df.to_pandas().iterrows():
    # Skip already processed games.
    lud_rules = row[1]['LudRules']
    if lud_rules in already_seen_lud_rules:
        continue

    already_seen_lud_rules.add(lud_rules)

    # Record rules & features.
    record = {}
    for col_name in row[1].index:
        dropped_cols = IRRELEVANT_COLS + OUTCOME_COUNT_COLS + AGENT_COLS + POISONOUS_COLS
        if col_name in dropped_cols:
            continue

        record[col_name] = row[1][col_name]

    # Record targets.
    for target_name, target_dict in zip(target_names, target_dicts):
        record[target_name] = target_dict[lud_rules]

    records.append(record)

# SAVE TO CSV.
df = pl.DataFrame(records).to_pandas()
df.to_csv('data/games_to_utility_stats.csv', index=False)