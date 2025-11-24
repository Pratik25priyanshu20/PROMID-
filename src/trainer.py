# src/trainer.py
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import Config
from .dataset import MisinfoDataset
from .loss import FocalLoss
from .model import HybridDebertaClassifier
from .utils import get_device


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    low: float,
    high: float,
    step: float,
) -> Tuple[float, float]:
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.arange(low, high, step):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)


def train_kfold(
    cfg: Config, train_df, meta_features: List[str]
) -> Tuple[List[str], List[float], List[float]]:
    """
    3-fold Stratified CV training.
    Returns:
        model_paths, fold_thresholds, fold_f1_scores
    """
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    X_text = train_df["clean_text"].tolist()
    X_meta = train_df[meta_features].values
    y = train_df["label"].values.astype("float32")

    skf = StratifiedKFold(
        n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED
    )

    model_paths = []
    fold_thresholds = []
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y), start=1):
        print(f"\n==== FOLD {fold}/{cfg.N_FOLDS} ====")

        X_train_text = [X_text[i] for i in train_idx]
        X_val_text = [X_text[i] for i in val_idx]
        X_train_meta = X_meta[train_idx]
        X_val_meta = X_meta[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        train_dataset = MisinfoDataset(
            X_train_text, X_train_meta, y_train, tokenizer, cfg.MAX_LEN
        )
        val_dataset = MisinfoDataset(
            X_val_text, X_val_meta, y_val, tokenizer, cfg.MAX_LEN
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        model = HybridDebertaClassifier(
            cfg.MODEL_NAME, metadata_dim=len(meta_features), dropout=0.3
        ).to(device)

        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        pos_weight_val = neg / max(pos, 1)
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float).to(device)

        criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)

        optimizer = AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
        )
        total_steps = len(train_loader) * cfg.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        scaler = GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

        best_fold_f1 = 0.0
        best_state_dict = None
        best_thr = 0.5

        for epoch in range(1, cfg.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
            model.train()
            running_loss = 0.0

            for step_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                if scaler is not None:
                    with autocast(enabled=True):
                        logits = model(input_ids, attention_mask, metadata)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(input_ids, attention_mask, metadata)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / max(len(train_loader), 1)
            print(f"Train loss: {avg_train_loss:.4f}")

            # ---- validation ----
            model.eval()
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    metadata = batch["metadata"].to(device)
                    labels = batch["label"].to(device).unsqueeze(1)

                    if scaler is not None:
                        with autocast(enabled=True):
                            logits = model(
                                input_ids, attention_mask, metadata
                            )
                    else:
                        logits = model(input_ids, attention_mask, metadata)

                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy().flatten())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels).astype(int)

            thr, f1 = find_best_threshold(
                all_probs,
                all_labels,
                cfg.THRESH_LOW,
                cfg.THRESH_HIGH,
                cfg.THRESH_STEP,
            )
            print(f"Val F1={f1:.4f} @ thr={thr:.2f}")

            if f1 > best_fold_f1:
                best_fold_f1 = f1
                best_state_dict = model.state_dict()
                best_thr = thr

        print(f"Fold {fold} best F1={best_fold_f1:.4f} thr={best_thr:.2f}")

        # Save best model for this fold
        model_path = cfg.MODELS_DIR / f"deberta_fold{fold}.pt"
        torch.save(best_state_dict, model_path)
        print(f"Saved fold {fold} model to {model_path}")

        model_paths.append(str(model_path))
        fold_thresholds.append(best_thr)
        fold_f1_scores.append(best_fold_f1)

        # Free memory
        del model
        torch.cuda.empty_cache()

    print("\n==== CROSS-VALIDATION SUMMARY ====")
    print("Fold F1:", [round(x, 4) for x in fold_f1_scores])
    print(
        f"Mean F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}"
    )
    print(f"Mean threshold: {np.mean(fold_thresholds):.3f}")

    return model_paths, fold_thresholds, fold_f1_scores