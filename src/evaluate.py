"""
Model evaluation module.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
and generates performance comparison charts.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier pipeline on a held-out test set.
    Computes all standard classification metrics and curves.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = None

    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
    elif hasattr(pipeline, "decision_function"):
        try:
            scores = pipeline.decision_function(X_test)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception:
            y_prob = None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    roc_auc_val = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ROC curve points
    roc_data = None
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": roc_auc_val,
        }

    # PR curve points
    pr_data = None
    if y_prob is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_data = {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "avg_precision": float(auc(recall_curve, precision_curve)),
        }

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc_val,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_data": roc_data,
        "pr_data": pr_data,
    }


def plot_metrics_comparison(results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Renders a grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    data = []
    for r in results:
        for m, label in zip(metrics, metric_labels):
            val = r.get(m)
            if val is not None:
                data.append({"Model": r["model_name"], "Metric": label, "Score": val})

    comp_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(
        data=comp_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette="crest",
        ax=ax,
        edgecolor="none"
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("Model Performance Comparison (Held-out Test Set)", fontsize=13, pad=12)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                xytext=(0, 2),
                textcoords="offset points",
            )

    fig.tight_layout()
    return fig


def plot_confusion_matrices(results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Renders confusion matrix heatmaps side-by-side for all evaluated models.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 3.8))
    if n_models == 1:
        axes = [axes]

    for i, res in enumerate(results):
        cm = np.array(res["confusion_matrix"])
        cm_pct = cm / cm.sum() * 100

        # Annotations with counts and percentages
        labels = [
            [f"{cm[0, 0]}\n({cm_pct[0, 0]:.1f}%)", f"{cm[0, 1]}\n({cm_pct[0, 1]:.1f}%)"],
            [f"{cm[1, 0]}\n({cm_pct[1, 0]:.1f}%)", f"{cm[1, 1]}\n({cm_pct[1, 1]:.1f}%)"],
        ]

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["Pred Retained", "Pred Churned"],
            yticklabels=["Actual Retained", "Actual Churned"],
        )
        axes[i].set_title(f"{res['model_name']}", fontsize=11, pad=8)

    fig.tight_layout()
    return fig


def plot_roc_curves(results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Plots multi-model ROC curves on a single plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2b5c8f", "#2ca02c", "#d9534f", "#9467bd"]

    for i, res in enumerate(results):
        roc_data = res.get("roc_data")
        if roc_data:
            auc_val = roc_data.get("auc", 0.0)
            ax.plot(
                roc_data["fpr"],
                roc_data["tpr"],
                label=f"{res['model_name']} (AUC = {auc_val:.3f})",
                color=colors[i % len(colors)],
                linewidth=2,
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_precision_recall_curves(results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Plots Precision-Recall curves on a single plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2b5c8f", "#2ca02c", "#d9534f", "#9467bd"]

    for i, res in enumerate(results):
        pr_data = res.get("pr_data")
        if pr_data:
            ap_val = pr_data.get("avg_precision", 0.0)
            ax.plot(
                pr_data["recall"],
                pr_data["precision"],
                label=f"{res['model_name']} (AP = {ap_val:.3f})",
                color=colors[i % len(colors)],
                linewidth=2,
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves", fontsize=12, pad=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
