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

from CompetitionDataset import TextCompetitionDataset
from InterpretableTextEncoder import InterpretableTextEncoder

def CreateDataLoaders(
        rules_to_targets_filepath: str,
        tokenizer: object,
        max_sequence_length: int,
        batch_size: int,
        fold_count: int,
        test_fold_index: int):
    # LOAD RULES TO TARGETS.
    rules_to_targets = pd.read_csv(rules_to_targets_filepath)

    # CREATE FOLDS.
    kf = KFold(n_splits=fold_count, shuffle=True, random_state=42)
    fold_indices = list(kf.split(rules_to_targets))

    # CREATE DATALOADERS.
    train_indices, test_indices = fold_indices[test_fold_index]
    train_dataset = TextCompetitionDataset(rules_to_targets.iloc[train_indices], tokenizer, max_sequence_length)
    test_dataset = TextCompetitionDataset(rules_to_targets.iloc[test_indices], tokenizer, max_sequence_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def TrainDebertaModel(
        rules_to_targets_filepath: str,
        backbone_name: str,
        pca_component_count: int,
        regression_activations: list[str],
        classification_target_counts: list[int],
        dropout_rate: float,
        max_sequence_length: int,
        classification_loss_weight: float,
        pca_loss_weight: float,
        regression_loss_weight: float,
        batch_size: int,
        grad_accumulation_step_count: int,
        fold_count: int,
        test_fold_index: int,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        log_dir: str,
        run_description: str,
        model_dir: str):
    # CREATE MODEL.
    model = InterpretableTextEncoder(backbone_name, pca_component_count, regression_activations, classification_target_counts, dropout_rate)
    model = model.cuda()

    # CREATE DATA LOADERS.
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    train_dataloader, test_dataloader = CreateDataLoaders(
        rules_to_targets_filepath, 
        tokenizer, 
        max_sequence_length, 
        batch_size, 
        fold_count, 
        test_fold_index)
    
    # SETUP TRAINING ALGORITHM.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=max_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,
        anneal_strategy='linear',
        # div_factor=2,
    )

    pca_loss_function = nn.MSELoss()
    regression_loss_function = nn.MSELoss()
    classification_loss_function = nn.CrossEntropyLoss()

    # CREATE TENSORBOARD WRITER.
    writer = SummaryWriter(os.path.join(log_dir, run_description))

    # TRAIN MODEL.
    train_batch_index = 0
    for epoch in range(max_epochs):
        # TRAIN.
        model.train()
        total_training_losses = []
        for batch_id, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{max_epochs} - Train')):
            lud_rules_input_ids = batch['lud_rule_input_ids'].cuda()
            lud_rules_attn_mask = batch['lud_rule_attention_mask'].cuda()
            english_rules_input_ids = batch['english_rule_input_ids'].cuda()
            english_rules_attn_mask = batch['english_rule_attention_mask'].cuda()

            pca_utilities = batch['pca_utilities'].cuda()
            mean_agent1_utilities = batch['mean_agent1_utilities'].float().cuda()
            mean_absolute_agent1_utilities = batch['mean_absolute_agent1_utilities'].float().cuda()
            both_players_clusters = batch['both_players_clusters'].cuda()
            player1_clusters = batch['player1_clusters'].cuda()
            player2_clusters = batch['player2_clusters'].cuda()

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                predicted_pca_utilities, regression_predictions, classification_predictions = model(lud_rules_input_ids, lud_rules_attn_mask, english_rules_input_ids, english_rules_attn_mask)

                pca_loss = pca_loss_function(predicted_pca_utilities, pca_utilities)
                regression_losses = [regression_loss_function(prediction, target) for prediction, target in zip(regression_predictions, [mean_agent1_utilities, mean_absolute_agent1_utilities])]
                classification_losses = [classification_loss_function(prediction, target) for prediction, target in zip(classification_predictions, [both_players_clusters, player1_clusters, player2_clusters])]

                total_regression_loss = sum(regression_losses)
                total_classification_loss = sum(classification_losses)
                loss = (pca_loss_weight * pca_loss) + (regression_loss_weight * total_regression_loss) + (classification_loss_weight * total_classification_loss)

            normalized_loss = loss / grad_accumulation_step_count
            scaler.scale(normalized_loss).backward()

            if (batch_id + 1) % grad_accumulation_step_count == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

            writer.add_scalar('TrainLoss/Pca', pca_loss.item(), train_batch_index)
            writer.add_scalar('TrainLoss/Regression', total_regression_loss.item(), train_batch_index)
            writer.add_scalar('TrainLoss/Classification', total_classification_loss.item(), train_batch_index)
            writer.add_scalar('TrainLoss/Total', loss.item(), train_batch_index)
            train_batch_index += 1

            total_training_losses.append(loss.item())

        # EVALUATE.
        model.eval()
        test_pca_losses = []
        test_regression_losses = []
        test_classification_losses = []
        test_total_losses = []

        lud_rules_to_predictions = {}

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{max_epochs} - Test'):
                lud_rules_input_ids = batch['lud_rule_input_ids'].cuda()
                lud_rules_attn_mask = batch['lud_rule_attention_mask'].cuda()
                english_rules_input_ids = batch['english_rule_input_ids'].cuda()
                english_rules_attn_mask = batch['english_rule_attention_mask'].cuda()

                pca_utilities = batch['pca_utilities'].cuda()
                mean_agent1_utilities = batch['mean_agent1_utilities'].float().cuda()
                mean_absolute_agent1_utilities = batch['mean_absolute_agent1_utilities'].float().cuda()
                both_players_clusters = batch['both_players_clusters'].cuda()
                player1_clusters = batch['player1_clusters'].cuda()
                player2_clusters = batch['player2_clusters'].cuda()

                with autocast(dtype=torch.bfloat16):
                    predicted_pca_utilities, regression_predictions, classification_predictions = model(lud_rules_input_ids, lud_rules_attn_mask, english_rules_input_ids, english_rules_attn_mask)

                    pca_loss = pca_loss_function(predicted_pca_utilities, pca_utilities)
                    regression_losses = [regression_loss_function(prediction, target) for prediction, target in zip(regression_predictions, [mean_agent1_utilities, mean_absolute_agent1_utilities])]
                    classification_losses = [classification_loss_function(prediction, target) for prediction, target in zip(classification_predictions, [both_players_clusters, player1_clusters, player2_clusters])]

                    total_regression_loss = sum(regression_losses)
                    total_classification_loss = sum(classification_losses)
                    loss = (pca_loss_weight * pca_loss) + (regression_loss_weight * total_regression_loss) + (classification_loss_weight * total_classification_loss)

                test_pca_losses.append(pca_loss.item())
                test_regression_losses.append(total_regression_loss.item())
                test_classification_losses.append(total_classification_loss.item())
                test_total_losses.append(loss.item())

                for lud_rules_index, lud_rules in enumerate(batch['lud_rules_text']):
                    lud_rules_to_predictions[lud_rules] = {
                        'pca_utilities': predicted_pca_utilities[lud_rules_index].float().cpu().numpy().tolist(),
                        'mean_agent1_utilities': regression_predictions[0][lud_rules_index].float().cpu().numpy().tolist(),
                        'mean_absolute_agent1_utilities': regression_predictions[1][lud_rules_index].float().cpu().numpy().tolist(),
                        'both_players_clusters': classification_predictions[0][lud_rules_index].float().softmax(dim=0).cpu().numpy().tolist(),
                        'player1_clusters': classification_predictions[1][lud_rules_index].float().softmax(dim=0).cpu().numpy().tolist(),
                        'player2_clusters': classification_predictions[2][lud_rules_index].float().softmax(dim=0).cpu().numpy().tolist(),
                    }

        writer.add_scalar('TestLoss/Pca', np.mean(test_pca_losses), epoch)
        writer.add_scalar('TestLoss/Regression', np.mean(test_regression_losses), epoch)
        writer.add_scalar('TestLoss/Classification', np.mean(test_classification_losses), epoch)
        writer.add_scalar('TestLoss/Total', np.mean(test_total_losses), epoch)

        print(f'Epoch {epoch+1}/{max_epochs} - Train Loss: {np.mean(total_training_losses):.4f} - Test Loss: {np.mean(test_total_losses):.4f}')

    # SAVE MODEL.
    total_loss = np.mean(test_total_losses)
    if model_dir is not None:
        model_filename = f'{backbone_name.split('/')[-1]}_{int(total_loss * 10000)}_{run_description}.pth'
        torch.save(model.state_dict(), f'{model_dir}/{model_filename}.pth')

        predictions_filename = f'{backbone_name.split('/')[-1]}_{int(total_loss * 10000)}_{run_description}.json'
        with open(f'predictions/{predictions_filename}', 'w') as file:
            json.dump(lud_rules_to_predictions, file, indent=4)

    writer.close()

    return total_loss

# 1.34 - 5 epochs, default div factor
# 1.381 - 6 epochs, default div factor
# 1.354 - 6 epochs, div factor=2
if __name__ == '__main__':
    fold_test_losses = []
    for fold_index in range(5):
        test_loss = TrainDebertaModel(
            rules_to_targets_filepath = 'data/rules_to_derived_targets.csv',
            backbone_name = 'microsoft/deberta-v3-base',
            pca_component_count = 32,
            regression_activations = ['Tanh', 'Sigmoid'],
            classification_target_counts = [10, 10, 10],
            dropout_rate = 0.1,
            max_sequence_length = 640,
            ## BASELINE.
            classification_loss_weight = 0.04,
            regression_loss_weight = 0.17,
            pca_loss_weight = 1,
            ## REG 5.
            # classification_loss_weight = 0.04,
            # regression_loss_weight = 5,
            # pca_loss_weight = 0.17,
            ## REG 6.
            # classification_loss_weight = 0.004,
            # regression_loss_weight = 6,
            # pca_loss_weight = 0.017,
            batch_size = 8,
            grad_accumulation_step_count = 1,
            fold_count = 5,
            test_fold_index = fold_index,
            learning_rate = 2e-5,
            weight_decay = 0.01,
            max_epochs = 5,
            log_dir = 'logs',
            run_description = f'CTX640_fold{fold_index}_e5_rerun',
            model_dir = 'models'
            # log_dir = 'logs',
            # run_description = 'debug',
            # model_dir = None
        )

        fold_test_losses.append(test_loss)

    print(f'Mean test loss: {np.mean(fold_test_losses)}')

    # best_config = {}
    # best_score = 1e9
    # for lr in [5e-6, 1e-5, 2e-5, 4e-5, 8e-5]:
    #     for dropout_rate in [0, 0.1, 0.2]:
    #         for max_epochs in [3, 5, 10]:
    #             score = TrainDebertaModel(
    #                 rules_to_targets_filepath = 'data/rules_to_derived_targets.csv',
    #                 backbone_name = 'microsoft/deberta-v3-base',
    #                 pca_component_count = 32,
    #                 regression_activations = ['Tanh', 'Sigmoid'],
    #                 classification_target_counts = [10, 10, 10],
    #                 dropout_rate = dropout_rate,
    #                 max_sequence_length = 640,
    #                 classification_loss_weight = 0.04,
    #                 regression_loss_weight = 0.17,
    #                 batch_size = 8,
    #                 grad_accumulation_step_count = 1,
    #                 fold_count = 5,
    #                 # test_fold_index = 0,
    #                 # test_fold_index = 1,
    #                 test_fold_index = 2,
    #                 learning_rate = lr,
    #                 weight_decay = 0.01,
    #                 max_epochs = max_epochs,
    #                 log_dir = 'logs',
    #                 run_description = f'640_{lr}_{dropout_rate}_{max_epochs}',
    #                 model_dir = None
    #             )

    #             if score < best_score:
    #                 best_score = score
    #                 best_config = {
    #                     'learning_rate': lr,
    #                     'dropout_rate': dropout_rate,
    #                     'max_epochs': max_epochs
    #                 }

    #                 print(f'Best config: {best_config} - Best score: {best_score}')

    # print(f'Best config: {best_config} - Best score: {best_score}')