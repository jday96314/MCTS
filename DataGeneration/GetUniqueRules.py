import json
import numpy as np

with open('data/lsa/lud_rules_to_selected_features.json', 'r') as f:
    lud_rules_to_selected_features = json.load(f)

unique_rules = set(lud_rules_to_selected_features.keys())

for rule_index, rule_text in enumerate(unique_rules):
    with open(f'DataGeneration/UniqueLuds/{rule_index}.lud', 'w') as f:
        f.write(rule_text)

# unique_rules_subset = np.random.choice(list(unique_rules), 16, replace=False)
# for rule_index, rule_text in enumerate(unique_rules_subset):
#     with open(f'DataGeneration/UniqueLudsSubset/{rule_index}.lud', 'w') as f:
#         f.write(rule_text)
