This project presents a production-grade system for multilingual misinformation detection developed for the PROMID Codabench Challenge. The goal is to classify social media posts into misinformation or nonmisinformation using a hybrid deep learning pipeline. The solution combines a transformer-based language model (DeBERTa-v3-Base) with a comprehensive metadata feature layer capturing linguistic signals, user-behavior patterns, and propaganda-oriented cues. A rigorous 3-fold cross-validation strategy is used to ensure robust generalization, supported by Focal Loss and class-balancing techniques to address extreme label imbalance in the dataset. The pipeline incorporates efficient preprocessing, feature engineering, and an optimized training loop featuring mixed-precision acceleration for performance. During inference, predictions are aggregated through a weighted ensemble of fold models and calibrated using threshold optimization to maximize F1-score. The system automatically generates Codabench-ready submission files and is organized with modular architecture suitable for extension, deployment, and reproducibility. This repository reflects a full end-to-end machine learning solution—from data ingestion to final competition submission—built with an emphasis on model performance, engineering quality, and production readiness.






PROMID/
│
├── data/                      # training + test data
├── models/                    # saved model weights per fold
├── submissions/               # generated CSV & ZIP for Codabench
│
├── src/
│   ├── config.py              # global configs
│   ├── dataset.py
│   ├── feature_engineering.py
│   ├── model.py               # DeBERTa hybrid classifier
│   ├── loss.py                # Focal Loss
│   ├── trainer.py             # cross-validation training
│   ├── inference.py           # ensemble prediction
│   └── utils.py
│
├── main.py                    # entry point for full pipeline
├── requirements.txt
└── README.md




This project implements a state-of-the-art multilingual misinformation detection system trained on the Russia–Ukraine social media dataset from the PROMID challenge (Codabench). The model combines:
	•	Transformer embeddings (DeBERTa-v3 / DistilBERT-multilingual)
	•	Metadata features (engagement, user behavior, bot indicators)
	•	Propaganda signals (emotion words, conspiratorial patterns, sensational terms)
	•	Language features
	•	3-Fold Ensemble Training for robust generalization
	•	Focal Loss + Calibrated Thresholding for extreme class imbalance
	•	Fast inference + reproducible submission pipeline

The system achieves strong F1 performance even under heavy class imbalance and small positive sample count.




git clone https://github.com/<yourusername>/PROMID.git
cd PROMID
pip install -r requirements.txt


python src/main.py 

or


python src/main.py --mode train \
    --train_misinfo data/train/misinfo_train.csv \
    --train_nonmis data/train/nonmisinfo_train.csv \
    --folds 3 \
    --model deberta \
    --epochs 3