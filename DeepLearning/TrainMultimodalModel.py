import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from transformers import AutoTokenizer

from CompetitionDataset import PrepareDatasets
from TabularModel import MultimodalUtilityPredictor

def PrepareModel(
        agent_embedding_size = 12,
        agent_mlp_layer_size = 64,
        agent_dropout_rate = 0.4,
        game_mlp_layer_size = 832,
        game_dropout_rate = 0.3,
        game_rules_backbone_name = 'microsoft/deberta-v3-base',
        game_rules_dropout_rate = 0.3,
        game_rules_output_feature_count = 32,
        classifier_mlp_layer_1_size = 320,
        classifier_mlp_layer_2_size = 160,
        classifier_dropout_rate_1 = 0.4,
        classifier_dropout_rate_2 = 0.0,
        classifier_hidden_layer_count = 1,
        activation_str = 'ReLU',
        multi_step_dropout_count = 2):
    GAME_FEATURE_COUNT = 588
    model = MultimodalUtilityPredictor(
        agent_encoder_kwargs = {
            'embedding_sizes': [agent_embedding_size] * 4,
            'mlp_layer_sizes': [agent_mlp_layer_size],
            'dropout_rates': [agent_dropout_rate],
            'activation_str': activation_str
        },
        game_encoder_kwargs = {
            'game_feature_count': GAME_FEATURE_COUNT,
            'mlp_layer_sizes': [game_mlp_layer_size],
            'dropout_rates': [game_dropout_rate],
            'activation_str': activation_str
        },
        game_rules_encoder_kwargs = {
            'backbone_name': game_rules_backbone_name,
            'dropout_rate': game_rules_dropout_rate,
            'output_feature_count': game_rules_output_feature_count
        },
        mlp_layer_sizes = [classifier_mlp_layer_1_size, classifier_mlp_layer_2_size][:classifier_hidden_layer_count] + [1],
        dropout_rates = [classifier_dropout_rate_1, classifier_dropout_rate_2][:classifier_hidden_layer_count] + [0],
        hidden_activations_str = activation_str,
        multi_sample_dropout_count = multi_step_dropout_count
    )

    return model.cuda()

def PrepareDataloaders(fold_index, fold_count, train_batch_size, tokenizer, max_sequence_length):
    preprocessor, train_dataset, test_dataset = PrepareDatasets(
        split_agent_features=True, 
        fold_index=fold_index, 
        fold_count=fold_count,
        multimodal=True,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        num_workers=torch.get_num_threads(),
        pin_memory=True,
        shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=train_batch_size * 2, 
        num_workers=torch.get_num_threads(),
        pin_memory=True,
        shuffle=False)

    return preprocessor, train_dataloader, test_dataloader

def TrainModel(
        fold_index,
        fold_count,
        hyperparameters,
        log_dir: str,
        run_description: str,
        model_dir: str):
    # PREPARE FOR TRAINING.
    model: MultimodalUtilityPredictor = PrepareModel(**hyperparameters['model_kwargs'])
    preprocessor, train_dataloader, test_dataloader = PrepareDataloaders(
        fold_index, 
        fold_count, 
        hyperparameters['batch_size'],
        tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model_kwargs']['game_rules_backbone_name']),
        max_sequence_length = 512
    )

    optimizer = optim.AdamW(model.parameters(), weight_decay=hyperparameters['weight_decay'])
    criterion = nn.MSELoss()

    scaler = GradScaler()
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hyperparameters['max_lr'],
        steps_per_epoch=len(train_dataloader),
        epochs=hyperparameters['epoch_count'],
        anneal_strategy='linear',
        pct_start=0.1,
    )

    writer = SummaryWriter(os.path.join(log_dir, run_description))

    # TRAIN MODEL.
    best_test_rmse = float('inf')
    for epoch in range(hyperparameters['epoch_count']):
        model.train()
        train_losses = []
        for batch_index, batch in enumerate(tqdm(train_dataloader)):
            # ZERO GRADIENTS.
            optimizer.zero_grad()

            # EXTRACT BATCH.
            agent_1_features = batch['agent_1_features'].cuda()
            agent_2_features = batch['agent_2_features'].cuda()
            game_features = batch['game_features'].cuda()
            lud_rule_input_ids = batch['lud_rule_input_ids'].cuda()
            lud_rule_attn_mask = batch['lud_rule_attention_mask'].cuda()
            english_rule_input_ids = batch['english_rule_input_ids'].cuda()
            english_rule_attn_mask = batch['english_rule_attention_mask'].cuda()

            target = batch['target'].cuda()

            # FORWARD PASS.
            with autocast(dtype=torch.bfloat16):
                predictions = model(
                    agent_1_features,
                    agent_2_features,
                    game_features,
                    lud_rule_input_ids,
                    lud_rule_attn_mask,
                    english_rule_input_ids,
                    english_rule_attn_mask
                )

                loss = criterion(predictions.reshape(-1), target)

            # BACKWARD PASS.
            normalized_loss = loss / hyperparameters['grad_accumulation_step_count']
            scaler.scale(normalized_loss).backward()

            if (batch_index + 1) % hyperparameters['grad_accumulation_step_count'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                lr_schedule.step()

            # LOGGING.
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_dataloader) + batch_index)
            train_losses.append(loss.item())

        # EVALUATE MODEL.
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                agent_1_features = batch['agent_1_features'].cuda()
                agent_2_features = batch['agent_2_features'].cuda()
                game_features = batch['game_features'].cuda()
                lud_rule_input_ids = batch['lud_rule_input_ids'].cuda()
                lud_rule_attn_mask = batch['lud_rule_attention_mask'].cuda()
                english_rule_input_ids = batch['english_rule_input_ids'].cuda()
                english_rule_attn_mask = batch['english_rule_attention_mask'].cuda()

                target = batch['target'].cuda()

                with autocast(dtype=torch.bfloat16):
                    predictions = model(
                        agent_1_features,
                        agent_2_features,
                        game_features,
                        lud_rule_input_ids,
                        lud_rule_attn_mask,
                        english_rule_input_ids,
                        english_rule_attn_mask
                    )

                    loss = criterion(predictions.reshape(-1), target)
                
                test_losses.append(loss.item())
        
        # LOGGING.
        train_rmse = np.mean(train_losses) ** 0.5
        test_rmse = np.mean(test_losses) ** 0.5
        print(f'Epoch {epoch} Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')
        writer.add_scalar('Loss/Test', test_rmse, epoch)

    # SAVE MODEL.
    if model_dir is not None:
        backbone_name = hyperparameters['model_kwargs']['game_rules_backbone_name']
        model_filename = f'{backbone_name.split('/')[-1]}_mm_{int(test_rmse * 10000)}_{run_description}.pth'
        torch.save(model.state_dict(), os.path.join(model_dir, model_filename))

    return test_rmse

if __name__ == '__main__':
    hyperparameters = {
        'model_kwargs': {
            'agent_embedding_size': 12,
            'agent_mlp_layer_size': 64,
            'agent_dropout_rate': 0.4,
            'game_mlp_layer_size': 832,
            'game_dropout_rate': 0.3,
            'game_rules_backbone_name': 'microsoft/deberta-v3-base',
            'game_rules_dropout_rate': 0.3,
            'game_rules_output_feature_count': 32,
            'classifier_mlp_layer_1_size': 320,
            'classifier_mlp_layer_2_size': 160,
            'classifier_dropout_rate_1': 0.4,
            'classifier_dropout_rate_2': 0.0,
            'classifier_hidden_layer_count': 1,
            'activation_str': 'ReLU',
            'multi_step_dropout_count': 2
        },
        'batch_size': 8,
        'max_lr': 1e-5,
        'weight_decay': 0.001,
        'grad_accumulation_step_count': 1,
        'epoch_count': 8
    }

    score = TrainModel(
        fold_index=0,
        fold_count=5,
        hyperparameters=hyperparameters,
        log_dir='logs',
        run_description='1en5_8e',
        model_dir='models'
    )

    # fold_count = 5
    # for fold_index in range(fold_count):
    #     TrainModel(
    #         fold_index,
    #         fold_count,
    #         hyperparameters,
    #         log_dir='logs',
    #         run_description=f'fold_{fold_index}',
    #         model_dir='models'
    #     )