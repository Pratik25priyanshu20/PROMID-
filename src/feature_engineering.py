# src/feature_engineering.py
import re
from typing import List, Tuple

import emoji
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_tweet(text: str) -> str:
    """Basic tweet cleaning: URLs, mentions, emojis, repeated chars, lowercase."""
    if pd.isna(text):
        return ""
    text = str(text)

    # URLs & mentions
    text = re.sub(r"http\S+|www\.\S+", "<url>", text)
    text = re.sub(r"@\w+", "<user>", text)

    # Space before hashtags to keep them as separate tokens
    text = re.sub(r"#", " #", text)

    # Emojis → descriptive tokens
    try:
        text = emoji.demojize(text, delimiters=(" ", " "))
    except Exception:
        pass

    # Collapse character repetitions ("cooool" -> "cool")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text.lower().strip()


def engineer_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create text and propaganda-style features."""
    df = df.copy()
    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["text"].apply(clean_tweet)

    # Basic text stats
    df["text_len"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().apply(len)
    df["avg_word_len"] = df["text_len"] / (df["word_count"] + 1)

    df["exclamation_cnt"] = df["clean_text"].str.count("!")
    df["question_cnt"] = df["clean_text"].str.count(r"\?")
    df["uppercase_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    df["url_count"] = df["text"].str.count(r"http\S+|www\.\S+")
    df["mention_count"] = df["text"].str.count("@")
    df["hashtag_count"] = df["text"].str.count("#")

    # Propaganda cues / sensationalism
    df["has_breaking"] = df["text"].str.contains(
        r"BREAKING|JUST IN|URGENT|ALERT", case=False, regex=True
    ).astype(int)

    df["has_shocking"] = df["text"].str.contains(
        r"SHOCKING|UNBELIEVABLE|MUST SEE|INSANE|WOW", case=False, regex=True
    ).astype(int)

    df["has_conspiracy"] = df["text"].str.contains(
        r"deep state|false flag|wake up|sheeple|bio lab|biolab|bioweapon",
        case=False,
        regex=True,
    ).astype(int)

    df["has_fear_words"] = df["text"].str.contains(
        r"dangerous|threat|attack|destroy|kill|war|death|terror",
        case=False,
        regex=True,
    ).astype(int)

    df["multiple_exclamation"] = df["text"].str.contains(r"!{2,}").astype(int)
    df["multiple_question"] = df["text"].str.contains(r"\?{2,}").astype(int)

    df["caps_word_ratio"] = df["text"].apply(
        lambda x: len(
            [w for w in str(x).split() if w.isupper() and len(w) > 2]
        )
        / (len(str(x).split()) + 1)
    )

    return df


def add_language_features(
    df: pd.DataFrame, top_langs: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode top N languages."""
    df = df.copy()
    if "lang" not in df.columns:
        df["lang"] = "und"
    df["lang"] = df["lang"].fillna("und")

    if top_langs is None:
        top_langs = df["lang"].value_counts().head(5).index.tolist()

    for lg in top_langs:
        df[f"lang_{lg}"] = (df["lang"] == lg).astype(int)

    return df, top_langs


def build_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Apply feature engineering + scaling to train and test.
    Returns:
      train_df_fe, test_df_fe, meta_feature_names
    """
    # --- Text features ---
    train_df = engineer_text_features(train_df)
    test_df = engineer_text_features(test_df)

    # --- Language one-hots (fit on train, apply to test) ---
    train_df, top_langs = add_language_features(train_df, None)
    test_df, _ = add_language_features(test_df, top_langs)

    # --- Numeric features to scale ---
    numeric_cols = [
        "text_len",
        "word_count",
        "avg_word_len",
        "exclamation_cnt",
        "question_cnt",
        "uppercase_ratio",
        "url_count",
        "mention_count",
        "hashtag_count",
        "caps_word_ratio",
    ]

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    # Boolean/binary features that we keep as-is
    bool_cols = [
        "has_breaking",
        "has_shocking",
        "has_conspiracy",
        "has_fear_words",
        "multiple_exclamation",
        "multiple_question",
    ]

    # Language features
    lang_cols = [c for c in train_df.columns if c.startswith("lang_")]

    meta_features = numeric_cols + bool_cols + lang_cols

    return train_df, test_df, meta_features