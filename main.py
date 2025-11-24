# main.py
from src.config import Config
from src.feature_engineering import build_features
from src.trainer import train_kfold
from src.inference import run_inference
from src.utils import (
    ensure_dirs,
    load_raw_data,
    save_metrics,
    seed_everything,
    zip_submission,
)


def main():
    cfg = Config()
    ensure_dirs(cfg)
    seed_everything(cfg.SEED)

    # 1. Load data
    train_df, test_df = load_raw_data(cfg)

    # 2. Feature engineering
    print("Engineering features...")
    train_fe, test_fe, meta_features = build_features(train_df, test_df)
    print(f"Meta feature count: {len(meta_features)}")
    print("Example meta features:", meta_features[:10])

    # 3. Train with K-fold
    model_paths, fold_thresholds, fold_f1_scores = train_kfold(
        cfg, train_fe, meta_features
    )

    # 4. Save CV metrics
    save_metrics(cfg, fold_f1_scores, fold_thresholds)

    # 5. Inference on test set
    submission, _ = run_inference(
        cfg, test_fe, meta_features, model_paths, fold_thresholds, fold_f1_scores
    )

    # 6. Save CSV
    cfg.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(cfg.SUBMISSION_CSV, index=False)
    print(f"Saved submission CSV to: {cfg.SUBMISSION_CSV}")

    # 7. Zip for Codabench
    zip_submission(cfg)


if __name__ == "__main__":
    main()