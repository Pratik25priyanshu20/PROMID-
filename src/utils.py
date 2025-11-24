# src/utils.py
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .config import Config


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(cfg: Config):
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load misinfo/nonmisinfo train sets and test set.
    We ignore any existing 'label' column and assign:
        misinfo -> 1
        nonmisinfo -> 0
    """
    print("Loading raw data...")
    mis = pd.read_csv(cfg.MISINFO_PATH, low_memory=False)
    non = pd.read_csv(cfg.NONMISINFO_PATH, low_memory=False)
    test = pd.read_csv(cfg.TEST_PATH, low_memory=False)

    # Minimal columns
    def minimal(df: pd.DataFrame, label_val: int) -> pd.DataFrame:
        out = pd.DataFrame()
        out["id"] = df["id"]
        out["text"] = df["text"]
        out["lang"] = df["lang"] if "lang" in df.columns else "und"
        out["label"] = label_val
        return out

    train_df = pd.concat(
        [minimal(mis, 1), minimal(non, 0)], ignore_index=True
    )
    print(
        f"Train: {train_df.shape}, positives: {train_df['label'].sum()} "
        f"({train_df['label'].mean():.2%})"
    )
    print(f"Test:  {test.shape}, columns: {list(test.columns)}")

    # Standardize test columns
    if "text" not in test.columns:
        raise ValueError("Test CSV must contain a 'text' column.")
    if "id" not in test.columns:
        raise ValueError("Test CSV must contain an 'id' column.")
    if "lang" not in test.columns:
        test["lang"] = "und"

    test_df = test[["id", "text", "lang"]].copy()
    return train_df, test_df


def save_metrics(
    cfg: Config, fold_f1s, fold_thresholds, out_path: Path = None
):
    if out_path is None:
        out_path = cfg.MODELS_DIR / "cv_metrics.json"
    payload = {
        "fold_f1": [float(x) for x in fold_f1s],
        "fold_thresholds": [float(x) for x in fold_thresholds],
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved CV metrics to {out_path}")


def zip_submission(cfg: Config):
    import zipfile

    zip_path = cfg.SUBMISSION_ZIP
    csv_path = cfg.SUBMISSION_CSV
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
    print(f"Zipped submission to: {zip_path}")