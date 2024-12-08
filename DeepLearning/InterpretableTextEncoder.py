from torch import nn
import torch
from transformers import AutoModelForSequenceClassification

class InterpretableTextEncoder(nn.Module):
    def __init__(
            self, 
            backbone_name: str,
            pca_component_count: int,
            regression_activations: list[str],
            classification_target_counts: list[int],
            dropout_rate: float = 0.1):
        super(InterpretableTextEncoder, self).__init__()
        self.LudBackbone = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        self.EnglishBackbone = AutoModelForSequenceClassification.from_pretrained(backbone_name)

        total_hidden_size = self.LudBackbone.config.hidden_size + self.EnglishBackbone.config.hidden_size

        self.PcaRegressionHead = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_hidden_size, total_hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(total_hidden_size // 2, pca_component_count),
        )

        self.RegressionHeads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(total_hidden_size, total_hidden_size // 2),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(total_hidden_size // 2, 1),
                getattr(torch.nn, activation)()
            ) for activation in regression_activations
        ])

        self.ClassificationHeads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(total_hidden_size, total_hidden_size // 2),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(total_hidden_size // 2, target_count),
            ) for target_count in classification_target_counts
        ])

    def forward(self, lud_rules_input_ids, lud_rules_attn_mask, english_rules_input_ids, english_rules_attn_mask):
        # ENCODE RULES.
        lud_features = self.LudBackbone.deberta(input_ids=lud_rules_input_ids, attention_mask=lud_rules_attn_mask)
        english_features = self.EnglishBackbone.deberta(input_ids=english_rules_input_ids, attention_mask=english_rules_attn_mask)

        pooled_lud_features = self.LudBackbone.pooler(lud_features[0])
        pooled_english_features = self.EnglishBackbone.pooler(english_features[0])

        # CONCATENATE FEATURES.
        concatenated_features = torch.cat([pooled_lud_features, pooled_english_features], dim=1)

        # PREDICTION HEADS.
        pca_components = self.PcaRegressionHead(concatenated_features)
        regression_predictions = [head(concatenated_features) for head in self.RegressionHeads]
        classification_predictions = [head(concatenated_features) for head in self.ClassificationHeads]

        return pca_components, regression_predictions, classification_predictions