# src/inference.py
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import Config
from .dataset import MisinfoDataset
from .model import HybridDebertaClassifier
from .utils import get_device


def run_inference(
    cfg: Config,
    test_df: pd.DataFrame,
    meta_features: List[str],
    model_paths: List[str],
    fold_thresholds: List[float],
    fold_f1_scores: List[float],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load fold models, run ensemble prediction, build submission DataFrame.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    test_dataset = MisinfoDataset(
        test_df["clean_text"].tolist(),
        test_df[meta_features].values,
        labels=None,
        tokenizer=tokenizer,
        max_len=cfg.MAX_LEN,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_fold_probs = []

    for fold_idx, model_path in enumerate(model_paths, start=1):
        print(f"Predicting with fold {fold_idx} model...")
        model = HybridDebertaClassifier(
            cfg.MODEL_NAME, metadata_dim=len(meta_features)
        ).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        probs = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                metadata = batch["metadata"].to(device)

                with autocast(enabled=(cfg.USE_AMP and device.type == "cuda")):
                    logits = model(input_ids, attention_mask, metadata)
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                probs.extend(p)

        all_fold_probs.append(np.array(probs))
        del model
        torch.cuda.empty_cache()

    all_fold_probs = np.stack(all_fold_probs, axis=0)

    # Weighted ensemble by fold F1
    weights = np.array(fold_f1_scores)
    weights = weights / weights.sum()
    ensemble_probs = (weights[:, None] * all_fold_probs).sum(axis=0)
    avg_thr = float(np.mean(fold_thresholds))
    print(f"Using ensemble threshold={avg_thr:.3f}")

    ensemble_preds = (ensemble_probs >= avg_thr).astype(int)

    # Build submission
    if cfg.LABEL_MODE == "string":
        label_map = {0: "nonmisinfo", 1: "misinfo"}
        labels = [label_map[int(p)] for p in ensemble_preds]
    else:
        labels = ensemble_preds

    submission = pd.DataFrame(
        {"id": test_df["id"].values, "label": labels}
    )

    return submission, ensemble_probs