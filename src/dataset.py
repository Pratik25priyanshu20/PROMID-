# src/dataset.py
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class MisinfoDataset(Dataset):
    """
    PyTorch Dataset for (text, metadata, label) samples.
    """

    def __init__(
        self,
        texts: List[str],
        metadata: np.ndarray,
        labels: Optional[np.ndarray],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
    ):
        self.texts = texts
        self.metadata = metadata.astype("float32")
        self.labels = labels.astype("float32") if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item["label"] = torch.tensor(0.0, dtype=torch.float)

        return item