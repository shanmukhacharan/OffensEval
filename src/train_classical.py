"""
train_classical.py

Trains and evaluates the classical ML baselines for OffensEval:
  - TF-IDF + Logistic Regression
  - TF-IDF + Random Forest (class-weighted)
  - TF-IDF + Random Forest tuned with GridSearchCV

This is the script version of notebooks/01_classical_ml.ipynb — run this
directly if you just want the trained metrics without opening Jupyter.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from preprocessing import load_dataset, add_clean_text_column
from evaluate import evaluate_model


def main():
    # --- Load and clean data ---
    df = load_dataset("data/TBO_4k_train.xlsx")
    df = add_clean_text_column(df)
    print("Dataset shape after cleaning:", df.shape)
    print(df["label"].value_counts())

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # --- TF-IDF vectorization ---
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # --- Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_tfidf, y_train)
    y_pred_lr = log_reg.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred_lr, "Logistic Regression")

    # --- Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_tfidf, y_train)
    y_pred_rf = rf_model.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred_rf, "Random Forest")

    # --- Random Forest tuned with GridSearchCV ---
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid, cv=5, n_jobs=-1, verbose=1,
    )
    grid_rf.fit(X_train_tfidf, y_train)
    best_rf = grid_rf.best_estimator_
    print("\nBest RF params:", grid_rf.best_params_)
    y_pred_best_rf = best_rf.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred_best_rf, "Random Forest Tuned")

    # --- Save the best model + vectorizer for the demo app ---
    joblib.dump(log_reg, "saved_models/logreg_model.joblib")
    joblib.dump(tfidf, "saved_models/tfidf_vectorizer.joblib")
    print("\nSaved model + vectorizer to saved_models/")


if __name__ == "__main__":
    import os
    os.makedirs("saved_models", exist_ok=True)
    main()
