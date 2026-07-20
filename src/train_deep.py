"""
train_deep.py

Trains and evaluates the two deep learning models used in this project:
  - BiLSTM on unigram token sequences
  - CNN+LSTM hybrid on bigram-enhanced token sequences

This is the script version of notebooks/02_deep_learning.ipynb.
Requires TensorFlow (see requirements.txt).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_dataset, add_clean_text_column, add_bigram_text
from evaluate import evaluate_model, plot_roc

MAX_VOCAB = 10000
MAX_LEN = 50
EMBED_DIM = 128


def make_sequences(train_texts, test_texts, max_vocab=MAX_VOCAB, max_len=MAX_LEN):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len)
    return X_train_seq, X_test_seq, tokenizer


def build_bilstm(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_lstm(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = load_dataset("data/TBO_4k_train.xlsx")
    df = add_clean_text_column(df)
    df["bigram_text"] = df["clean_text"].apply(add_bigram_text)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(df["label"]), y=df["label"]
    )
    class_weight_dict = dict(enumerate(class_weights))
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # --- BiLSTM on unigram sequences ---
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_train_seq, X_test_seq, tok_uni = make_sequences(X_train_txt, X_test_txt)

    bilstm = build_bilstm(vocab_size=min(MAX_VOCAB, len(tok_uni.word_index) + 1))
    bilstm.fit(
        X_train_seq, y_train, validation_split=0.1, epochs=15, batch_size=32,
        class_weight=class_weight_dict, callbacks=[early_stop], verbose=1,
    )
    y_prob_bilstm = bilstm.predict(X_test_seq).ravel()
    y_pred_bilstm = (y_prob_bilstm > 0.5).astype(int)
    evaluate_model(y_test, y_pred_bilstm, "BiLSTM")
    plot_roc(y_test, y_prob_bilstm, "BiLSTM")

    # --- CNN+LSTM on bigram-enhanced sequences ---
    Xb_train_txt, Xb_test_txt, yb_train, yb_test = train_test_split(
        df["bigram_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    Xb_train_seq, Xb_test_seq, tok_bi = make_sequences(Xb_train_txt, Xb_test_txt)

    cnn_lstm = build_cnn_lstm(vocab_size=min(MAX_VOCAB, len(tok_bi.word_index) + 1))
    cnn_lstm.fit(
        Xb_train_seq, yb_train, validation_split=0.1, epochs=15, batch_size=32,
        class_weight=class_weight_dict, callbacks=[early_stop], verbose=1,
    )
    y_prob_cnn_lstm = cnn_lstm.predict(Xb_test_seq).ravel()
    y_pred_cnn_lstm = (y_prob_cnn_lstm > 0.5).astype(int)
    evaluate_model(yb_test, y_pred_cnn_lstm, "CNN LSTM")
    plot_roc(yb_test, y_prob_cnn_lstm, "CNN LSTM")

    # --- Save models ---
    import os
    os.makedirs("saved_models", exist_ok=True)
    bilstm.save("saved_models/bilstm_model.h5")
    cnn_lstm.save("saved_models/cnn_lstm_model.h5")
    print("\nSaved deep learning models to saved_models/")


if __name__ == "__main__":
    main()
