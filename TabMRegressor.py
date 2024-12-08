import math
import random
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm
import sklearn

from tabm.tabm_reference import Model, make_parameter_groups
import rtdl_num_embeddings


# Based on https://github.com/yandex-research/tabm/blob/main/example.ipynb
class TabMRegressor:

    def __init__(
        self,
        arch_type: str = 'tabm',
        backbone: dict = {'type': 'MLP', 'n_blocks': 3, 'd_block': 512, 'dropout': 0.1},
        use_embeddings: bool = True,
        d_embedding: int = 16,
        bin_count: int = 48,
        k: int = 32,
        learning_rate: float = 2e-3,
        weight_decay: float = 3e-4,
        clip_grad_norm: bool = True,
        max_epochs: int = 1000,
        patience: int = 16,
        batch_size: int = 256,
        compile_model: bool = False,
        device: Optional[str] = 'cuda:0',
        random_state: int = 0,
        verbose: bool = True
    ):
        self.arch_type = arch_type
        self.use_embeddings = use_embeddings
        self.backbone = backbone
        self.k = k
        self.d_embedding = d_embedding
        self.bin_count = bin_count
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        eval_set: Tuple[pd.DataFrame, np.array],
        sample_weight: Optional[np.array] = None
    ):
        # PREPROCESS DATA.
        X_cat_train, X_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)
        X_cat_val, X_cont_val, _, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

        # CREATE MODEL & TRAINING ALGO.
        bins = rtdl_num_embeddings.compute_bins(X_cont_train, n_bins=self.bin_count) if self.use_embeddings else None
        self.model = Model(
            n_num_features=X_cont_train.shape[1],
            cat_cardinalities=cat_cardinalities,
            n_classes=None,
            backbone=self.backbone,
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': self.d_embedding,
                    'activation': False,
                    'version': 'B',
                }
            ),
            arch_type=self.arch_type,
            k=self.k,
        ).to(self.device)
        optimizer = torch.optim.AdamW(make_parameter_groups(self.model), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.compile_model:
            self.model = torch.compile(self.model)

        if sample_weight is None:
            loss_fn = torch.nn.MSELoss().to(self.device)
        else:
            loss_fn = torch.nn.MSELoss(reduction='none').to(self.device)
            eval_loss_fn = torch.nn.MSELoss().to(self.device)
            sample_weights_tensor = torch.tensor(sample_weight, device=self.device)

        # TRAIN & TEST MODEL.
        best = {
            'epoch': -1,
            'eval_loss': math.inf,
            'model_state_dict': None,
        }
        remaining_patience = self.patience
        epoch_size = math.ceil(len(X) / self.batch_size)
        for epoch in range(self.max_epochs):
            # TRAIN.
            optimizer.zero_grad()
            train_losses = []
            progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
            progress_bar = tqdm(progress_bar, desc=f'Epoch {epoch}', total=epoch_size) if self.verbose else progress_bar
            for batch_idx in progress_bar:
                self.model.train()

                with torch.amp.autocast(device_type='cuda', dtype = torch.bfloat16):
                    y_pred = self.model(
                        X_cont_train[batch_idx],
                        X_cat_train[batch_idx],
                    ).squeeze(-1).float()
                    
                if sample_weight is None:
                    loss = loss_fn(y_pred.flatten(0, 1), y_train[batch_idx].repeat_interleave(self.k))
                else:
                    sample_losses = loss_fn(y_pred.flatten(0, 1), y_train[batch_idx].repeat_interleave(self.k))
                    batch_sample_weights = sample_weights_tensor[batch_idx].repeat_interleave(self.k)
                    loss = (sample_losses * batch_sample_weights).mean()

                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # EVALUATE.
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_idx in torch.arange(0, len(y_val), self.batch_size, device=self.device):
                    y_pred = self.model(
                        X_cont_val[batch_idx:batch_idx+self.batch_size],
                        X_cat_val[batch_idx:batch_idx+self.batch_size],
                    ).squeeze(-1).float()

                    if sample_weight is None:
                        loss = loss_fn(y_pred.flatten(0, 1), y_val[batch_idx:batch_idx+self.batch_size].repeat_interleave(self.k))
                    else:
                        loss = eval_loss_fn(y_pred.flatten(0, 1), y_val[batch_idx:batch_idx+self.batch_size].repeat_interleave(self.k))

                    val_losses.append(loss.item())

            # PRINT INFO.
            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)
            if self.verbose:
                print(f'Epoch {epoch} | Train Loss: {mean_train_loss} | Val Loss: {mean_val_loss}')

            # COMPARE TO BEST.
            if mean_val_loss < best['eval_loss']:
                best['epoch'] = epoch
                best['eval_loss'] = mean_val_loss
                best['model_state_dict'] = self.model.state_dict()
                remaining_patience = self.patience
                
                if self.verbose:
                    print('🌸 New best epoch! 🌸')
            else:
                remaining_patience -= 1

            # EARLY STOPPING.
            if remaining_patience == 0:
                break

        # RESTORE BEST MODEL.
        self.model.load_state_dict(best['model_state_dict'])

    def predict(
        self,
        X: pd.DataFrame,
        batch_size: Optional[int] = 8096
    ) -> np.ndarray:
        # PREPROCESS DATA.
        X_cat, X_cont, _, _ = self._preprocess_data(X, y=None, training=False)

        # PREDICT.
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_idx in torch.arange(0, len(X), batch_size, device=self.device):
                y_pred.append(
                    self.model(
                        X_cont[batch_idx:batch_idx+batch_size],
                        X_cat[batch_idx:batch_idx+batch_size],
                    ).squeeze(-1).float().cpu().numpy()
                )
        y_pred = np.concatenate(y_pred)

        # DENORMALIZE TARGETS.
        y_pred = y_pred * self._target_std + self._target_mean

        # COMPUTE ENSEMBLE MEAN.
        y_pred = np.mean(y_pred, axis=1)

        return y_pred

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, training: bool):
        # PICK NON-CONSTANT COLUMNS.
        if training:
            self._non_constant_columns = X.columns[X.nunique() > 1]

        X = X[self._non_constant_columns]

        # SEPARATE CATEGORICAL & CONTINUOUS FEATURES.
        categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']
        X_cat = X[categorical_features].to_numpy()
        X_cont = X.drop(columns=categorical_features).to_numpy()

        # ENCODE CATEGORICAL FEATURES.
        cat_cardinalities = [X[col].nunique() for col in categorical_features]

        if training:
            self._categorical_encoders = [
                sklearn.preprocessing.OrdinalEncoder()
                for _ in range(X_cat.shape[1])
            ]
            X_cat = np.concatenate([
                encoder.fit_transform(X_cat[:, i:i+1])
                for i, encoder in enumerate(self._categorical_encoders)
            ], axis=1)
        else:
            X_cat = np.concatenate([
                encoder.transform(X_cat[:, i:i+1])
                for i, encoder in enumerate(self._categorical_encoders)
            ], axis=1)

        # NORMALIZE TARGETS.
        if training:
            self._target_mean = y.mean()
            self._target_std = y.std()

            y = (y - self._target_mean) / self._target_std

        # SCALE CONTINUOUS FEATURES.
        if training:
            noise = (
                np.random.default_rng(0)
                .normal(0.0, 1e-5, X_cont.shape)
                .astype(X_cont.dtype)
            )
            self._cont_feature_preprocessor = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(len(X) // 30, 1000), 10),
                output_distribution='normal',
                subsample=10**9,
            ).fit(X_cont + noise)

        X_cont = self._cont_feature_preprocessor.transform(X_cont)

        # CONVERT TO TENSORS.
        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)

        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        return X_cat, X_cont, cat_cardinalities, y
