from torch.utils.data import Dataset
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import numpy as np

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS

# TODO: Add multimodal variant which provides tokenized text fields!
class TabularCompetitionDataset(Dataset):
    def __init__(
            self, 
            features: pd.DataFrame, 
            targets: pd.Series,
            agent_1_feature_mask: list[bool] = None,
            agent_2_feature_mask: list[bool] = None,
            game_feature_mask: list[bool] = None):
        self.Features = features
        self.Targets = targets

        self.Agent1FeatureMask = agent_1_feature_mask
        self.Agent2FeatureMask = agent_2_feature_mask
        self.GameFeatureMask = game_feature_mask

    def __len__(self):
        return len(self.Features)
    
    def __getitem__(self, index):
        all_features = self.Features[index]

        agent_1_features = all_features[self.Agent1FeatureMask].astype('int64')
        agent_2_features = all_features[self.Agent2FeatureMask].astype('int64')
        game_features = all_features[self.GameFeatureMask].astype('float16')

        target = self.Targets.iloc[index]

        return agent_1_features, agent_2_features, game_features, target

class TextCompetitionDataset(Dataset):
    def __init__(
            self, 
            rules_to_targets: pd.DataFrame,
            tokenizer: object,
            max_sequence_length: int):
        self.RulesToTargets = rules_to_targets
        self.Tokenizer = tokenizer
        self.MaxSequenceLength = max_sequence_length

    def __len__(self):
        return len(self.RulesToTargets)
    
    def __getitem__(self, index):
        record = self.RulesToTargets.iloc[index]

        # LOAD RULES.
        lud_rules = record['lud_rules']
        lud_rules_tokenized = self.Tokenizer(lud_rules, max_length=self.MaxSequenceLength, padding='max_length', truncation=True, return_tensors='pt')

        english_rules = record['english_rules']
        english_rules_tokenized = self.Tokenizer(english_rules, max_length=self.MaxSequenceLength, padding='max_length', truncation=True, return_tensors='pt')

        # LOAD TARGETS.
        pca_utilities = eval(record['pca_utilities'].replace('\n', '').replace(' ', ',').replace(',,', ',').replace(',,', ',').replace(',,', ',').replace('[,', '['))
        mean_agent1_utilities = record['mean_agent1_utilities']
        mean_absolute_agent1_utilities = record['mean_absolute_agent1_utilities']
        both_players_clusters = record['both_players_clusters']
        player1_clusters = record['player1_clusters']
        player2_clusters = record['player2_clusters']

        return {
            'lud_rules_text': lud_rules,
            'lud_rule_input_ids': lud_rules_tokenized['input_ids'][0],
            'lud_rule_attention_mask': lud_rules_tokenized['attention_mask'][0],
            'english_rules_text': english_rules,
            'english_rule_input_ids': english_rules_tokenized['input_ids'][0],
            'english_rule_attention_mask': english_rules_tokenized['attention_mask'][0],
            'pca_utilities': np.array(pca_utilities, dtype=np.float16),
            'mean_agent1_utilities': mean_agent1_utilities,
            'mean_absolute_agent1_utilities': mean_absolute_agent1_utilities,
            'both_players_clusters': both_players_clusters,
            'player1_clusters': player1_clusters,
            'player2_clusters': player2_clusters,
        }
        
class MultimodalCompetitionDataset(Dataset):
    def __init__(
            self, 
            features: pd.DataFrame, 
            targets: pd.Series,
            agent_1_feature_mask: list[bool],
            agent_2_feature_mask: list[bool],
            game_feature_mask: list[bool],
            lud_rules: pd.Series,
            english_rules: pd.Series,
            tokenizer: object,
            max_sequence_length: int):
        self.Features = features
        self.Targets = targets

        self.Agent1FeatureMask = agent_1_feature_mask
        self.Agent2FeatureMask = agent_2_feature_mask
        self.GameFeatureMask = game_feature_mask

        self.LudRules = lud_rules
        self.EnglishRules = english_rules
        self.Tokenizer = tokenizer
        self.MaxSequenceLength = max_sequence_length

    def __len__(self):
        return len(self.Features)
    
    def __getitem__(self, index):
        all_features = self.Features[index]

        agent_1_features = all_features[self.Agent1FeatureMask].astype('int64')
        agent_2_features = all_features[self.Agent2FeatureMask].astype('int64')
        game_features = all_features[self.GameFeatureMask].astype('float16')

        target = self.Targets.iloc[index]

        lud_rules = self.LudRules.iloc[index]
        lud_rules_tokenized = self.Tokenizer(lud_rules, max_length=self.MaxSequenceLength, padding='max_length', truncation=True, return_tensors='pt')

        english_rules = self.EnglishRules.iloc[index]
        english_rules_tokenized = self.Tokenizer(english_rules, max_length=self.MaxSequenceLength, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'agent_1_features': agent_1_features,
            'agent_2_features': agent_2_features,
            'game_features': game_features,
            'target': target,
            'lud_rules_text': lud_rules,
            'lud_rule_input_ids': lud_rules_tokenized['input_ids'][0],
            'lud_rule_attention_mask': lud_rules_tokenized['attention_mask'][0],
            'english_rules_text': english_rules,
            'english_rule_input_ids': english_rules_tokenized['input_ids'][0],
            'english_rule_attention_mask': english_rules_tokenized['attention_mask'][0],
        }


def GetUnnormalizedData(split_agent_features):
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    # df = df.sample(n = 200)
        
    lud_rules = df['LudRules'].to_pandas()
    english_rules = df['EnglishRules'].to_pandas()
    ruleset_names = df['GameRulesetName'].to_pandas()

    df = df.drop(filter(lambda x: x in df.columns, DROPPED_COLUMNS))

    if split_agent_features:
        for col in AGENT_COLS:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")
        
        df = df.with_columns([pl.col(col).cast(pl.String) for col in df.columns if col[:6] in AGENT_COLS])            
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in AGENT_COLS])
    
    print(f'Data shape: {df.shape}')
    
    return lud_rules, english_rules, ruleset_names, df.to_pandas()

def PrepareDatasets(split_agent_features, fold_index, fold_count, multimodal=False, tokenizer=None, max_sequence_length=None):
    # LOAD DATA.
    lud_rules, english_rules, ruleset_names, train_test_df = GetUnnormalizedData(split_agent_features)

    # SEPARATE FEATURES & TARGETS.
    X = train_test_df.drop(['utility_agent1'], axis=1)
    y = train_test_df['utility_agent1']

    # SPLIT INTO TRAIN & TEST PARTITIONS.
    group_kfold = GroupKFold(n_splits=fold_count)
    folds = list(group_kfold.split(X, y, groups=ruleset_names))
    train_index, test_index = folds[fold_index]

    train_x = X.iloc[train_index]
    train_y = y.iloc[train_index]

    test_x = X.iloc[test_index]
    test_y = y.iloc[test_index]

    # PREPROCESS FEATURES.
    numerical_cols = [col for col in train_x.columns if train_x[col].dtype.name != 'object']
    categorical_cols = [col for col in train_x.columns if train_x[col].dtype.name == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('agent1', OrdinalEncoder(), ['agent1_1', 'agent1_2', 'agent1_3', 'agent1_4']),
            ('agent2', OrdinalEncoder(), ['agent2_1', 'agent2_2', 'agent2_3', 'agent2_4']),
            ('num', QuantileTransformer(output_distribution='uniform', random_state=0), numerical_cols),
        ],
        remainder='passthrough'
    )

    train_x = preprocessor.fit_transform(train_x)
    test_x = preprocessor.transform(test_x)

    # CREATE FEATURE MASKS.
    feature_names = train_test_df.columns.drop('utility_agent1')
    agent_1_feature_mask = [col[:6] == 'agent1' for col in feature_names]
    agent_2_feature_mask = [col[:6] == 'agent2' for col in feature_names]
    game_feature_mask = [col[:6] not in ['agent1', 'agent2'] for col in feature_names]


    # MAYBE OUTPUT TABULAR DATASET.
    if not multimodal:
        train_dataset = TabularCompetitionDataset(train_x, train_y, agent_1_feature_mask, agent_2_feature_mask, game_feature_mask)
        test_dataset = TabularCompetitionDataset(test_x, test_y, agent_1_feature_mask, agent_2_feature_mask, game_feature_mask)
        
        return preprocessor, train_dataset, test_dataset
    
    # OUTPUT MULTIMODAL DATASET.
    train_dataset = MultimodalCompetitionDataset(
        train_x, train_y, agent_1_feature_mask, agent_2_feature_mask, game_feature_mask,
        lud_rules.iloc[train_index], english_rules.iloc[train_index],
        tokenizer, max_sequence_length)
    test_dataset = MultimodalCompetitionDataset(
        test_x, test_y, agent_1_feature_mask, agent_2_feature_mask, game_feature_mask,
        lud_rules.iloc[test_index], english_rules.iloc[test_index],
        tokenizer, max_sequence_length)
    
    return preprocessor, train_dataset, test_dataset
