"""
Evaluation module for Bank Term Deposit Subscription Prediction.
Computes comprehensive metrics, finds optimal business thresholds,
and generates publication-quality visualization figures.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes key classification metrics at a given decision threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except Exception:
        pr_auc = 0.0

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, num_steps: int = 100
) -> tuple:
    """
    Search for the decision threshold that maximizes the F1-Score.
    """
    thresholds = np.linspace(0.05, 0.95, num_steps)
    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = {}

    for t in thresholds:
        m = compute_metrics(y_true, y_proba, threshold=t)
        if m["F1-Score"] > best_f1:
            best_f1 = m["F1-Score"]
            best_thresh = t
            best_metrics = m

    return best_thresh, best_metrics


def plot_confusion_matrices(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    save_path: Path,
):
    """
    Plots a multi-panel comparison of confusion matrices.
    """
    n_models = len(model_predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (name, y_pred) in enumerate(model_predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array(
            [[f"{count}\n({pct:.1%})" for count, pct in zip(r_c, r_p)]
             for r_c, r_p in zip(cm, cm_norm)]
        )
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
        )
        axes[i].set_title(f"{name}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    model_probas: Dict[str, np.ndarray],
    save_path: Path,
):
    """
    Plots comparative ROC and Precision-Recall curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for name, y_proba in model_probas.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = average_precision_score(y_true, y_proba)
        ax2.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", lw=2)

    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    ax1.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    baseline_rate = y_true.sum() / len(y_true)
    ax2.plot([0, 1], [baseline_rate, baseline_rate], "k--", lw=1.5, label=f"No Skill ({baseline_rate:.1%})")
    ax2.set_title("Precision-Recall Curves Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_leaderboard_bars(leaderboard_df: pd.DataFrame, save_path: Path):
    """
    Plots a horizontal bar chart comparing models across ROC-AUC, F1, and Recall.
    """
    df = leaderboard_df.sort_values("ROC-AUC", ascending=True)
    metrics_to_plot = ["ROC-AUC", "F1-Score", "Recall", "Precision"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    palette = sns.color_palette("viridis", len(df))

    for i, metric in enumerate(metrics_to_plot):
        axes[i].barh(df["Model"], df[metric], color=palette)
        axes[i].set_title(f"Model Comparison: {metric}", fontsize=12, fontweight="bold")
        axes[i].set_xlim(0, 1.05)
        axes[i].grid(axis="x", alpha=0.3)
        for j, val in enumerate(df[metric]):
            axes[i].text(val + 0.02, j, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
