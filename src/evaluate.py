"""
evaluate.py

Shared evaluation and plotting helpers used across the classical and deep
learning pipelines, so every model is scored and visualized the same way.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)


def evaluate_model(y_true, y_pred, model_name: str, save_dir: str = "results"):
    """Print accuracy + classification report, save a confusion matrix figure."""
    os.makedirs(save_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} — Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Not Harmful", "Harmful"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Harmful", "Harmful"],
        yticklabels=["Not Harmful", "Harmful"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

    return acc


def plot_roc(y_true, y_score, model_name: str, save_dir: str = "results"):
    """Plot and save an ROC curve for a model that outputs probabilities."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {out_path}  (AUC={roc_auc:.3f})")

    return roc_auc
