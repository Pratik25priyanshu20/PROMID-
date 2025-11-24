# src/config.py
from pathlib import Path


class Config:
    """
    Central configuration for paths, model and training hyperparameters.
    Adjust file names under DATA_DIR to match your local CSVs.
    """

    # --------- Paths ---------
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    SUBMISSIONS_DIR = BASE_DIR / "submissions"

    # Train files – make sure these filenames exist in data/
    MISINFO_PATH = DATA_DIR / "misinfo_train.csv"
    NONMISINFO_PATH = DATA_DIR / "nonmisinfo_train.csv"

    # Test file – rename if yours is different
    TEST_PATH = DATA_DIR / "test_data_without_label.csv"
    # e.g. if your file is `test_final_merge_withoutlabel (1).csv`:
    # TEST_PATH = DATA_DIR / "test_final_merge_withoutlabel (1).csv"

    # --------- Model / Training hyperparams ---------
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 96          # tweets rarely need >100 tokens
    N_FOLDS = 3           # good balance of robustness / time
    EPOCHS = 3
    BATCH_SIZE = 16       # per-GPU batch size
    LR = 3e-5
    WEIGHT_DECAY = 0.01
    SEED = 42

    GRAD_ACCUM_STEPS = 1  # set >1 if you want larger effective batch
    USE_AMP = True        # mixed precision if CUDA is available

    # Threshold search range for F1
    THRESH_LOW = 0.05
    THRESH_HIGH = 0.7
    THRESH_STEP = 0.01

    # Output label format in submission: "string" → "misinfo"/"nonmisinfo"
    LABEL_MODE = "string"  # or "int" for 0/1

    # Submission file names
    SUBMISSION_CSV = SUBMISSIONS_DIR / "deberta_3fold_submission.csv"
    SUBMISSION_ZIP = SUBMISSIONS_DIR / "submission.zip"