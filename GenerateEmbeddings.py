import polars as pl
import pickle
import os
from sentence_transformers import SentenceTransformer

def GetRules():
    try:
        df = pl.read_csv('/mnt/data01/data/TreeSearch/data/from_organizers/train.csv')
    except:
        df = pl.read_csv('data/from_organizers/train.csv')

    english_rules = df['EnglishRules'].to_numpy()
    lud_rules = df['LudRules'].to_numpy()

    return english_rules, lud_rules

if __name__ == '__main__':
    # LOAD MODEL
    MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    model = SentenceTransformer(
        MODEL_NAME, 
        trust_remote_code=True,
        model_kwargs={'torch_dtype': 'float16'})
    model.max_seq_length = 2048
    
    # LOAD RULES.
    english_rules, lud_rules = GetRules()
    english_rules = list(set(english_rules))
    lud_rules = list(set(lud_rules))

    # GENERATE EMBEDDINGS.
    english_rule_embeddings = model.encode(english_rules, show_progress_bar=True)
    lud_rule_embeddings = model.encode(lud_rules, show_progress_bar=True)

    # SAVE EMBEDDINGS.
    output_directory_path = f'cached_embeddings/{MODEL_NAME.split("/")[-1]}'
    os.makedirs(output_directory_path, exist_ok=True)

    english_output_path = f'{output_directory_path}/english_rule_embeddings.p'
    with open(english_output_path, 'wb') as english_output_file:
        english_rules_to_embeddings = dict(zip(english_rules, english_rule_embeddings))
        pickle.dump(english_rules_to_embeddings, english_output_file)

    lud_output_path = f'{output_directory_path}/lud_rule_embeddings.p'
    with open(lud_output_path, 'wb') as lud_output_file:
        lud_rules_to_embeddings = dict(zip(lud_rules, lud_rule_embeddings))
        pickle.dump(lud_rules_to_embeddings, lud_output_file)