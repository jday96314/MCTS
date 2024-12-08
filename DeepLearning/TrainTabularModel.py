from TabularModel import UtilityPredictor
import torch
import polars as pl
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error
import optuna
import json

from ColumnNames import DROPPED_COLUMNS, AGENT_COLS
from TabularModel import UtilityPredictor
from CompetitionDataset import TabularCompetitionDataset, PrepareDatasets

def PrepareModel(
        embedding_size = 8,
        agent_mlp_layer_size = 32,
        agent_dropout_rate = 0.25,
        game_mlp_layer_size = 256,
        game_dropout_rate = 0.25,
        classifier_mlp_layer_1_size = 256,
        classifier_mlp_layer_2_size = 128,
        classifier_dropout_rate_1 = 0.25,
        classifier_dropout_rate_2 = 0.25,
        classifier_hidden_layer_count = 2,
        activation_str = 'ReLU',
        multi_step_dropout_count = 1):
    GAME_FEATURE_COUNT = 588
    model = UtilityPredictor(
        agent_encoder_kwargs = {
            'embedding_sizes': [embedding_size] * 4,
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
        mlp_layer_sizes = [classifier_mlp_layer_1_size, classifier_mlp_layer_2_size][:classifier_hidden_layer_count] + [1],
        dropout_rates = [classifier_dropout_rate_1, classifier_dropout_rate_2][:classifier_hidden_layer_count] + [0],
        hidden_activations_str = activation_str,
        multi_sample_dropout_count = multi_step_dropout_count
    )

    return model.cuda()

def PrepareDataloaders(fold_index, fold_count, train_batch_size):
    preprocessor, train_dataset, test_dataset = PrepareDatasets(
        split_agent_features=True, 
        fold_index=fold_index, 
        fold_count=fold_count)
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

    return train_dataloader, test_dataloader

def TrainModel(
        fold_index,
        fold_count,
        hyperparameters,
        output_directory_path,
        output_suffix,
        max_save_rmse):
    # PREPARE FOR TRAINING.
    model = PrepareModel(**hyperparameters['model_kwargs'])
    train_dataloader, test_dataloader = PrepareDataloaders(
        fold_index, 
        fold_count, 
        hyperparameters['batch_size'])

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=hyperparameters['weight_decay'])
    criterion = torch.nn.MSELoss()

    scaler = GradScaler()
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hyperparameters['max_lr'],
        steps_per_epoch=len(train_dataloader),
        epochs=hyperparameters['epoch_count'],
        anneal_strategy='linear',
        pct_start=0.1,
    )

    # TRAIN & TEST MODEL.
    best_test_rmse = float('inf')
    for epoch in range(hyperparameters['epoch_count']):
        model.train()
        train_losses = []
        progress_description = f'Train Epoch {epoch}'
        for agent_1_features, agent_2_features, game_features, targets in tqdm(train_dataloader, desc=progress_description):
            optimizer.zero_grad()

            agent_1_features = agent_1_features.cuda()
            agent_2_features = agent_2_features.cuda()
            game_features = game_features.cuda()
            targets = targets.cuda()

            with autocast(dtype=torch.bfloat16):
                predictions = model(agent_1_features, agent_2_features, game_features)
                loss = criterion(predictions.reshape(-1), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_schedule.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        for agent_1_features, agent_2_features, game_features, targets in tqdm(test_dataloader, desc='Test Epoch'):
            agent_1_features = agent_1_features.cuda()
            agent_2_features = agent_2_features.cuda()
            game_features = game_features.cuda()
            targets = targets.cuda()

            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    predictions = model(agent_1_features, agent_2_features, game_features)
                    loss = criterion(predictions.reshape(-1), targets)

            test_losses.append(loss.item())

        train_rmse = (sum(train_losses) / len(train_losses)) ** 0.5
        test_rmse = (sum(test_losses) / len(test_losses)) ** 0.5
        print(f'Epoch {epoch} - Train RMSE: {train_rmse} - Test RMSE: {test_rmse}')

        if (test_rmse < best_test_rmse):
            best_test_rmse = test_rmse
            if (test_rmse < max_save_rmse):
                torch.save(model, f'{output_directory_path}/dl_{int(best_test_rmse*100000)}{output_suffix}.pth')

    return best_test_rmse

def Objective(trial):
    hyperparameters = {
        # 'batch_size': trial.suggest_int('batch_size', 64, 1024, log=True),
        # 'max_lr': trial.suggest_float('max_lr', 1e-5, 1e-2, log=True),
        'batch_size': 8,
        'max_lr': 2e-5,
        'epoch_count': trial.suggest_int('epoch_count', 3, 20),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'model_kwargs': {
            'embedding_size': trial.suggest_int('embedding_size', 4, 16),
            'agent_mlp_layer_size': trial.suggest_int('agent_mlp_layer_size', 16, 64),
            'agent_dropout_rate': trial.suggest_float('agent_dropout_rate', 0.0, 0.6),
            'game_mlp_layer_size': trial.suggest_int('game_mlp_layer_size', 64, 1024, log=True),
            'game_dropout_rate': trial.suggest_float('game_dropout_rate', 0.0, 0.6),
            'classifier_mlp_layer_1_size': trial.suggest_int('classifier_mlp_layer_1_size', 128, 512),
            'classifier_mlp_layer_2_size': trial.suggest_int('classifier_mlp_layer_2_size', 64, 256),
            'classifier_dropout_rate_1': trial.suggest_float('classifier_dropout_rate_1', 0.0, 0.6),
            'classifier_dropout_rate_2': trial.suggest_float('classifier_dropout_rate_2', 0.0, 0.6),
            'classifier_hidden_layer_count': trial.suggest_int('classifier_hidden_layer_count', 0, 2),
            'activation_str': trial.suggest_categorical('activation_str', ['ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU', 'Tanh', 'PReLU']),
            'multi_step_dropout_count': trial.suggest_int('multi_step_dropout_count', 1, 4)
        }
    }

    rmse_scores = []
    FOLD_COUNT = 5
    for fold_index in range(FOLD_COUNT):
        rmse = TrainModel(
            fold_index=0,
            fold_count=5,
            hyperparameters=hyperparameters,
            output_directory_path='models',
            output_suffix=f'_{trial.number}',
            max_save_rmse=0.42
        )
        rmse_scores.append(rmse)

    mean_score = sum(rmse_scores) / len(rmse_scores)

    return mean_score

'''
Best is trial 22 with value: 0.4768811678731346.
Best hyperparameters:
{
    "epoch_count": 19,
    "weight_decay": 6.137866378944886e-05,
    "embedding_size": 8,
    "agent_mlp_layer_size": 29,
    "agent_dropout_rate": 0.2853988776708428,
    "game_mlp_layer_size": 802,
    "game_dropout_rate": 0.45814244959984657,
    "classifier_mlp_layer_1_size": 199,
    "classifier_mlp_layer_2_size": 81,
    "classifier_dropout_rate_1": 0.3323531451474906,
    "classifier_dropout_rate_2": 0.07458501581256304,
    "classifier_hidden_layer_count": 2,
    "activation_str": "PReLU",
    "multi_step_dropout_count": 2
}
'''
if __name__ == '__main__':
    # TrainModel(
    #     fold_index=0,
    #     fold_count=5,
    #     hyperparameters={
    #         'batch_size': 256,
    #         'max_lr': 1e-3,
    #         'epoch_count': 10,
    #         'model_kwargs': {}
    #     },
    #     output_directory_path='models',
    #     output_suffix='temp',
    #     max_save_rmse=0.42
    # )

    study = optuna.create_study(direction='minimize')
    study.optimize(Objective, n_trials=30)

    print('Best hyperparameters:')
    print(json.dumps(study.best_params, indent=4))

    best_score = study.best_trial.value
    output_filepath = f'configs/dl_{int(best_score * 100000)}.json'
    with open(output_filepath, 'w') as f:
        json.dump(study.best_params, f, indent=4)