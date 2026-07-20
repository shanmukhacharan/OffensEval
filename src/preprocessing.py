"""
preprocessing.py

Shared utilities for loading and cleaning the OffensEval dataset.
Used by both the classical ML pipeline (train_classical.py) and the
deep learning pipeline (train_deep.py).
"""

import re
import pandas as pd


def load_dataset(path: str = "data/TBO_4k_train.xlsx") -> pd.DataFrame:
    """Load the raw dataset and keep only the columns needed for the
    binary harmful/not-harmful classification task."""
    df = pd.read_excel(path)
    df = df[["id", "text", "T1 Harmful"]]
    df["label"] = df["T1 Harmful"].map({"YES": 1, "NO": 0})
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset="text")
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip mentions/URLs/hashtags/punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"@[\w]+", "", text)            # remove @mentions
    text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
    text = re.sub(r"#\w+", "", text)               # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)            # keep only letters/whitespace
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def add_bigram_text(text: str) -> str:
    """Produce a bigram-augmented version of a cleaned string, used for the
    CNN+LSTM model which is trained on bigram-enhanced sequences."""
    tokens = text.split()
    bigrams = ["_".join(pair) for pair in zip(tokens, tokens[1:])]
    return " ".join(tokens + bigrams)
