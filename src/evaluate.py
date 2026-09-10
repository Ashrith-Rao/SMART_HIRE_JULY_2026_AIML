"""Evaluation metrics and visualization generators for all SmartHire models."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from src.config import FIGURES_DIR


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_cm_plot: bool = True,
    plot_filename: str = "confusion_matrix.png",
) -> Dict[str, object]:
    """Compute comprehensive classification metrics and optionally generate a confusion matrix plot.
    
    Returns
    -------
    dict
        {
            "accuracy": float,
            "precision_weighted": float,
            "precision_macro": float,
            "recall_weighted": float,
            "recall_macro": float,
            "f1_weighted": float,
            "f1_macro": float,
            "report": str,
            "confusion_matrix": np.ndarray
        }
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_wt = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    p_mac = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_wt = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    r_mac = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if save_cm_plot:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto",
        )
        plt.title("Resume Category Classification — Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Category", fontsize=12)
        plt.ylabel("Actual Category", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plot_path = FIGURES_DIR / plot_filename
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Confusion matrix plot saved to {plot_path}")

    return {
        "accuracy": round(acc, 4),
        "precision_weighted": round(p_wt, 4),
        "precision_macro": round(p_mac, 4),
        "recall_weighted": round(r_wt, 4),
        "recall_macro": round(r_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "f1_macro": round(f1_mac, 4),
        "report": report,
        "confusion_matrix": cm,
    }


def evaluate_clustering(
    X_matrix,
    labels: np.ndarray,
    sample_size: int = 2000,
) -> Dict[str, float]:
    """Compute clustering quality metrics including silhouette score."""
    if len(set(labels)) < 2:
        return {"silhouette_score": 0.0}

    # Silhouette score on TF-IDF matrix (sample if matrix is large)
    n_samples = X_matrix.shape[0]
    if n_samples > sample_size:
        sil = float(silhouette_score(X_matrix, labels, sample_size=sample_size, random_state=42))
    else:
        sil = float(silhouette_score(X_matrix, labels))

    return {
        "silhouette_score": round(sil, 4),
        "n_clusters": len(set(labels)),
    }
