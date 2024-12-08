import json
import numpy as np
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset

import sys
sys.path.append('DataGeneration/NewGameGeneration')
from RuleMinification import MinifyRules, UnminifyRules

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

def GetDatasets(
        fold_count,
        holdout_fold,
        samples_per_lud):
    with open('GAVEL/UniqueLuds.json', 'r') as luds_file:
        luds = json.load(luds_file)

    formatted_training_sequences = []
    formatted_validation_sequences = []
    for lud_index, lud in enumerate(luds):
        for _ in range(samples_per_lud):
            unminified_lud = UnminifyRules(lud)
            prefix, middle, suffix = GetRandomPrefixMiddleSuffix(unminified_lud)
            formatted_sequence = f'▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>{middle}▁<EOT>'

            if lud_index % fold_count == holdout_fold:
                formatted_validation_sequences.append(formatted_sequence)
            else:
                formatted_training_sequences.append(formatted_sequence)

    train_dataset = Dataset.from_dict({'text': formatted_training_sequences})
    test_dataset = Dataset.from_dict({'text': formatted_validation_sequences})

    return train_dataset, test_dataset

def TrainModel(fold_count, fold_id, samples_per_lud):
    MAX_SEQ_LENGTH = 4096
    output_dir = f"GAVEL/models/4en4_16rank_{samples_per_lud}spl/{fold_id}"
    RANK = 16
    LEARNING_RATE = 4e-4

    # Failed to use 13B due to https://github.com/unslothai/unsloth/issues/638
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/codellama-7b-bnb-4bit",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    tokenizer.truncation_side = 'left'

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r = RANK,
        target_modules = target_modules,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = False, # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    train_dataset, test_dataset = GetDatasets(
        fold_count=fold_count, 
        holdout_fold=fold_id, 
        samples_per_lud=samples_per_lud
    )
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    data_collator = DataCollatorForCompletionOnlyLM("▁<MID>", tokenizer=tokenizer)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        eval_packing = False,
        data_collator = data_collator,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_ratio=0.1,
            num_train_epochs = 1,
            # max_steps = 10,
            learning_rate = LEARNING_RATE,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
        )
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    model.save_pretrained(f'{output_dir}/final_model')

if __name__ == '__main__':
    fold_count = 6
    for fold_id in range(1, fold_count):
        TrainModel(fold_count, fold_id, samples_per_lud=4)