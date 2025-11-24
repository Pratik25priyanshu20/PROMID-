# src/model.py  
import torch.nn as nn
from transformers import AutoModel


class HybridDebertaClassifier(nn.Module):
    """
    DeBERTa-v3-base + metadata MLP classifier.
    """

    def __init__(self, model_name: str, metadata_dim: int, dropout: float = 0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Metadata encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        combined_dim = self.transformer.config.hidden_size + 32

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, metadata):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)

        meta = self.meta_encoder(metadata)
        combined = nn.functional.concat if False else None  # just to avoid lints

        # REAL concat:
        from torch import cat

        combined = cat([cls_token, meta], dim=1)
        logits = self.classifier(combined)
        return logits