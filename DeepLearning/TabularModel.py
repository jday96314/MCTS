from torch import nn
import torch
from transformers import AutoModelForSequenceClassification

class AgentEncoder(nn.Module):
    def __init__(
            self, 
            embedding_sizes: list[int], 
            mlp_layer_sizes: list[int], 
            dropout_rates: list[float], 
            activation_str: str = 'ReLU'
        ):
        super(AgentEncoder, self).__init__()
        
        self.SelectionEmbedding = nn.Embedding(4, embedding_sizes[0])
        self.ExplorationConstEmbedding = nn.Embedding(3, embedding_sizes[1])
        self.PlayoutEmbedding = nn.Embedding(3, embedding_sizes[2])
        self.ScoreBoundsEmbedding = nn.Embedding(2, embedding_sizes[3])

        activation = getattr(nn, activation_str)()

        self.MLP = nn.ModuleList()
        input_dim_count = sum(embedding_sizes)
        for layer_dim_count, dropout_rate in zip(mlp_layer_sizes, dropout_rates):
            self.MLP.append(nn.Linear(input_dim_count, layer_dim_count))
            self.MLP.append(nn.Dropout(dropout_rate))
            self.MLP.append(activation)

            input_dim_count = layer_dim_count

    def forward(self, x):
        selection_embedding = self.SelectionEmbedding(x[:, 0])
        exploration_const_embedding = self.ExplorationConstEmbedding(x[:, 1])
        playout_embedding = self.PlayoutEmbedding(x[:, 2])
        score_bounds_embedding = self.ScoreBoundsEmbedding(x[:, 3])
        
        x = torch.cat(
            [
                selection_embedding,
                exploration_const_embedding,
                playout_embedding,
                score_bounds_embedding
            ],
            dim=1
        )
        
        for layer in self.MLP:
            x = layer(x)
        
        return x
    
class GameEncoder(nn.Module):
    def __init__(self, game_feature_count, mlp_layer_sizes: list[int], dropout_rates: list[float], activation_str: str = 'ReLU'):
        super(GameEncoder, self).__init__()
        
        activation = getattr(nn, activation_str)()

        self.MLP = nn.ModuleList()
        input_dim_count = game_feature_count
        for layer_dim_count, dropout_rate in zip(mlp_layer_sizes, dropout_rates):
            self.MLP.append(nn.Linear(input_dim_count, layer_dim_count))
            self.MLP.append(nn.Dropout(dropout_rate))
            self.MLP.append(activation)

            input_dim_count = layer_dim_count

    def forward(self, x):
        for layer in self.MLP:
            x = layer(x)
        
        return x
    
class GameRulesEncoder(nn.Module):
    def __init__(self, backbone_name: str, dropout_rate: float, output_feature_count: int):
        super(GameRulesEncoder, self).__init__()
        self.LudBackbone = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        self.EnglishBackbone = AutoModelForSequenceClassification.from_pretrained(backbone_name)

        self.Downsampler = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.LudBackbone.config.hidden_size + self.EnglishBackbone.config.hidden_size, output_feature_count),
            nn.Tanh()
        )

    def forward(self, lud_rules_input_ids, lud_rules_attn_mask, english_rules_input_ids, english_rules_attn_mask):
        lud_features = self.LudBackbone.deberta(input_ids=lud_rules_input_ids, attention_mask=lud_rules_attn_mask)
        english_features = self.EnglishBackbone.deberta(input_ids=english_rules_input_ids, attention_mask=english_rules_attn_mask)

        pooled_lud_features = self.LudBackbone.pooler(lud_features[0])
        pooled_english_features = self.EnglishBackbone.pooler(english_features[0])

        concatenated_features = torch.cat([pooled_lud_features, pooled_english_features], dim=1)
        downsampled_features = self.Downsampler(concatenated_features)

        return downsampled_features

class UtilityPredictor(nn.Module):
    def __init__(
            self, 
            agent_encoder_kwargs: dict, 
            game_encoder_kwargs: dict,
            mlp_layer_sizes: list[int], 
            dropout_rates: list[float], 
            hidden_activations_str: str = 'ReLU',
            multi_sample_dropout_count: int = 1):
        super(UtilityPredictor, self).__init__()
        
        self.FirstPlayerAgentEncoder = AgentEncoder(**agent_encoder_kwargs)
        self.SecondPlayerAgentEncoder = AgentEncoder(**agent_encoder_kwargs)
        self.GameEncoder = GameEncoder(**game_encoder_kwargs)

        self.MLP = nn.ModuleList()
        input_dim_count = (agent_encoder_kwargs['mlp_layer_sizes'][-1] * 2) + game_encoder_kwargs['mlp_layer_sizes'][-1]
        for layer_index, (layer_dim_count, dropout_rate) in enumerate(zip(mlp_layer_sizes, dropout_rates)):
            self.MLP.append(nn.Linear(input_dim_count, layer_dim_count))
            self.MLP.append(nn.Dropout(dropout_rate))
            
            if layer_index < len(mlp_layer_sizes) - 1:
                self.MLP.append(getattr(nn, hidden_activations_str)())
            else:
                self.MLP.append(nn.Tanh())

            input_dim_count = layer_dim_count

        self.MultiSampleDropoutCount = multi_sample_dropout_count

    # player_1 has shape (batch_size, agent_feature_count)
    # player_2 has shape (batch_size, agent_feature_count)
    # game has shape (batch_size, game_feature_count)
    # Where agent feature count is ~4 and game feature count is ~597.
    def forward(self, player_1, player_2, game):
        raw_predictions = []

        for _ in range(self.MultiSampleDropoutCount):
            first_player_agent_features = self.FirstPlayerAgentEncoder(player_1)
            second_player_agent_features = self.SecondPlayerAgentEncoder(player_2)
            game_features = self.GameEncoder(game)

            x = torch.cat(
                [
                    first_player_agent_features,
                    second_player_agent_features,
                    game_features
                ],
                dim=1
            )

            for layer in self.MLP:
                x = layer(x)

            raw_predictions.append(x)

        averaged_predictions = torch.stack(raw_predictions, dim=0).mean(dim=0)

        return averaged_predictions
    
class UtilityPredictorEnsemble(nn.Module):
    def __init__(self, **kwargs):
        super(UtilityPredictorEnsemble, self).__init__()
        
        self.WeakPredictors = nn.ModuleList()
        for i in range(3):
            self.WeakPredictors.append(UtilityPredictor(**kwargs))

    def forward(self, player_1, player_2, game):
        predictions = []
        for weak_predictor in self.WeakPredictors:
            predictions.append(weak_predictor(player_1, player_2, game))

        return torch.stack(predictions, dim=1).mean(dim=1)
    
class MultimodalUtilityPredictor(nn.Module):
    def __init__(
            self, 
            agent_encoder_kwargs: dict, 
            game_encoder_kwargs: dict,
            game_rules_encoder_kwargs: dict,
            mlp_layer_sizes: list[int], 
            dropout_rates: list[float], 
            hidden_activations_str: str = 'ReLU',
            multi_sample_dropout_count: int = 1):
        super(MultimodalUtilityPredictor, self).__init__()
        
        self.FirstPlayerAgentEncoder = AgentEncoder(**agent_encoder_kwargs)
        self.SecondPlayerAgentEncoder = AgentEncoder(**agent_encoder_kwargs)
        self.GameEncoder = GameEncoder(**game_encoder_kwargs)
        self.GameRulesEncoder = GameRulesEncoder(**game_rules_encoder_kwargs)

        self.MLP = nn.ModuleList()
        input_dim_count = (agent_encoder_kwargs['mlp_layer_sizes'][-1] * 2) + game_encoder_kwargs['mlp_layer_sizes'][-1] + game_rules_encoder_kwargs['output_feature_count']
        for layer_index, (layer_dim_count, dropout_rate) in enumerate(zip(mlp_layer_sizes, dropout_rates)):
            self.MLP.append(nn.Linear(input_dim_count, layer_dim_count))
            self.MLP.append(nn.Dropout(dropout_rate))
            
            if layer_index < len(mlp_layer_sizes) - 1:
                self.MLP.append(getattr(nn, hidden_activations_str)())
            else:
                self.MLP.append(nn.Tanh())

            input_dim_count = layer_dim_count

        self.MultiSampleDropoutCount = multi_sample_dropout_count

    # player_1 has shape (batch_size, agent_feature_count)
    # player_2 has shape (batch_size, agent_feature_count)
    # game has shape (batch_size, game_feature_count)
    # Where agent feature count is ~4 and game feature count is ~597.
    def forward(self, player_1, player_2, game, lud_rules_input_ids, lud_rules_attn_mask, english_rules_input_ids, english_rules_attn_mask):
        raw_predictions = []

        game_rules_features = self.GameRulesEncoder(lud_rules_input_ids, lud_rules_attn_mask, english_rules_input_ids, english_rules_attn_mask)
        
        for _ in range(self.MultiSampleDropoutCount):
            first_player_agent_features = self.FirstPlayerAgentEncoder(player_1)
            second_player_agent_features = self.SecondPlayerAgentEncoder(player_2)
            game_features = self.GameEncoder(game)

            x = torch.cat(
                [
                    first_player_agent_features,
                    second_player_agent_features,
                    game_features,
                    game_rules_features
                ],
                dim=1
            )

            for layer in self.MLP:
                x = layer(x)

            raw_predictions.append(x)

        averaged_predictions = torch.stack(raw_predictions, dim=0).mean(dim=0)

        return averaged_predictions