"""
src/evaluation/visualize.py — All diagnostic charts for training and evaluation
================================================================================
Charts generated:
  Training (read from log CSV):
    plot_training_curves()    Train/Val loss + F1 + Recall per epoch
    plot_grad_norm_history()  Gradient norm per epoch (LSTM/GRU stability check)

  Evaluation (called from evaluate.py):
    plot_roc_curve()          ROC-AUC curve
    plot_pr_curve()           Precision-Recall (more reliable under imbalance)
    plot_confusion_matrix()   TP/FP/TN/FN heatmap
    plot_threshold_sweep()    P/R/F1 across all decision thresholds
    plot_model_comparison()   All 4 models side by side (ablation chart)

HOW TO ADD A NEW PLOT:
  1. Write a function below that ends with: plt.savefig(path); plt.close(); print(...)
  2. Call it from evaluate.py where relevant
  No registry, no boilerplate — just write the function and call it.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

MODEL_COLORS = {
    "cnn_baseline": "#e74c3c",
    "cnn_lstm":     "#2980b9",
    "cnn_gru":      "#27ae60",
    "tcn":          "#8e44ad",
}
_STYLE = {
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
}


# ── Training diagnostics ──────────────────────────────────────────────────────

def plot_training_curves(log_csv_path: str, model_name: str, save_dir: str):
    """
    Reads the training log CSV and plots Loss, F1, and Recall over epochs.

    How to interpret:
      Train << Val loss      → Overfitting. Increase dropout or weight_decay.
      Both losses plateau    → Underfitting. Increase model capacity or epochs.
      Val F1 still rising at last epoch → Train more epochs.
      Best epoch dashed line very early → Model peaked fast, check LR.
    """
    df = pd.read_csv(log_csv_path)
    if df.empty:
        print(f"  Skipping training curves — empty log for {model_name}")
        return

    best_epoch = df["val_f1"].idxmax() + 1
    color      = MODEL_COLORS.get(model_name, "#2c3e50")

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{model_name.upper()} — Training Curves", fontsize=14, fontweight="bold")

        panels = [
            ("Loss",   "train_loss", "val_loss"),
            ("F1",     None,         "val_f1"),
            ("Recall", None,         "val_recall"),
        ]

        for (metric, train_key, val_key), ax in zip(panels, axes):
            if train_key and train_key in df.columns:
                ax.plot(df["epoch"], df[train_key], color=color,
                        linestyle="--", alpha=0.7, label="Train")
            ax.plot(df["epoch"], df[val_key], color=color, linewidth=2, label="Val")
            ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7,
                       label=f"Best ({best_epoch})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=9)

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved: {path}")


def plot_grad_norm_history(log_csv_path: str, model_name: str, save_dir: str):
    """
    Plots gradient L2 norm per epoch on a log scale.
    Useful for debugging LSTM/GRU training instability.

    Healthy: norms gradually decrease as training converges.
    Spikes to 10× baseline → exploding gradients → lower LR or check grad_clip.
    Near zero from epoch 1  → vanishing gradients → check BatchNorm, LSTM init.
    """
    df = pd.read_csv(log_csv_path)
    if "grad_norm" not in df.columns or df.empty:
        return

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["epoch"], df["grad_norm"], color="#e67e22", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm (log scale)")
        ax.set_title(
            f"{model_name.upper()} — Gradient Norm History\n"
            "Healthy: decreasing  ·  Spikes: exploding  ·  ~0: vanishing"
        )
        ax.set_yscale("log")
        path = os.path.join(save_dir, f"{model_name}_grad_norms.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grad norm history saved: {path}")


# ── Evaluation plots ──────────────────────────────────────────────────────────

def plot_roc_curve(
    all_labels: np.ndarray, all_probs: np.ndarray,
    model_name: str, save_dir: str, auc: float = None
):
    """
    ROC curve. AUC = 1.0 is perfect; AUC = 0.5 = random.
    Can be optimistic under class imbalance — check PR curve alongside this.
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    color        = MODEL_COLORS.get(model_name, "#2c3e50")
    auc_str      = f" (AUC = {auc:.4f})" if auc is not None else ""

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{model_name}{auc_str}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
        ax.fill_between(fpr, tpr, alpha=0.1, color=color)
        ax.set_xlabel("False Positive Rate  (1 − Specificity)")
        ax.set_ylabel("True Positive Rate  (Recall)")
        ax.set_title(f"{model_name.upper()} — ROC Curve")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"ROC curve saved: {path}")


def plot_pr_curve(
    all_labels: np.ndarray, all_probs: np.ndarray,
    model_name: str, save_dir: str, pr_auc: float = None
):
    """
    Precision-Recall curve. More informative than ROC when seizures are < 5% of windows.
    The no-skill baseline is a horizontal line at the seizure rate.
    """
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    baseline             = float(all_labels.mean())
    color                = MODEL_COLORS.get(model_name, "#2c3e50")
    pr_str               = f" (PR-AUC = {pr_auc:.4f})" if pr_auc is not None else ""

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{model_name}{pr_str}")
        ax.axhline(baseline, color="gray", linestyle="--", alpha=0.7,
                   label=f"No-skill baseline ({baseline:.3f})")
        ax.fill_between(recall, precision, baseline, alpha=0.1, color=color)
        ax.set_xlabel("Recall  (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title(f"{model_name.upper()} — Precision-Recall Curve")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        path = os.path.join(save_dir, f"{model_name}_pr_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PR curve saved: {path}")


def plot_confusion_matrix(
    cm: np.ndarray, model_name: str, save_dir: str, threshold: float = 0.5
):
    """
    Heatmap of [[TN, FP], [FN, TP]].

    For seizure detection:
      FN (bottom-left) = missed seizures = most dangerous → lower threshold
      FP (top-right)   = false alarms   = annoying       → raise threshold
    """
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: Normal", "Pred: Seizure"],
            yticklabels=["True: Normal", "True: Seizure"],
            linewidths=0.5, ax=ax,
        )
        ax.set_title(
            f"{model_name.upper()} — Confusion Matrix  (threshold = {threshold:.2f})\n"
            f"TN={tn}  FP={fp}  FN={fn}  TP={tp}   ← FN = missed seizures"
        )
        plt.tight_layout()
        path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved: {path}")


def plot_threshold_sweep(sweep: dict, model_name: str, save_dir: str):
    """
    P/R/F1 across all thresholds. Use this to pick your operating point.

    How to use:
      1. Decide recall target (e.g. ≥ 0.90)
      2. Find where the red recall line hits 0.90
      3. Read the threshold on the x-axis at that point
      4. Set evaluation.threshold in configs/config.yaml
    """
    t      = sweep["thresholds"]
    best_t = sweep["best_f1_threshold"]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, sweep["precisions"], label="Precision", color="#3498db", linewidth=2)
        ax.plot(t, sweep["recalls"],    label="Recall",    color="#e74c3c", linewidth=2)
        ax.plot(t, sweep["f1s"],        label="F1",        color="#2ecc71", linewidth=2)
        ax.axvline(best_t, color="gray",  linestyle=":",  alpha=0.8,
                   label=f"Best F1 ({best_t:.2f})")
        ax.axvline(0.5,    color="black", linestyle="--", alpha=0.4,
                   label="Default (0.5)")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(
            f"{model_name.upper()} — Threshold Sweep\n"
            "Lower threshold → higher recall (catches more seizures)"
        )
        ax.legend()
        ax.set_xlim([0.05, 0.95])
        ax.set_ylim([0, 1.02])
        path = os.path.join(save_dir, f"{model_name}_threshold_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Threshold sweep saved: {path}")


def plot_model_comparison(results_dict: dict, save_dir: str):
    """
    Side-by-side bar chart comparing all 4 models on key metrics.
    Equivalent to the chest X-ray project's ablation_comparison.png.

    results_dict format:
        {"cnn_baseline": {recall: ..., f1: ..., roc_auc: ..., pr_auc: ...}, ...}

    How to interpret:
      CNN+LSTM/GRU > CNN baseline on recall + F1 → temporal modeling proved valuable
      TCN competitive → parallelism is a real option for deployment
      If baseline wins → local morphology alone is sufficient at 5s window size
    """
    model_names     = list(results_dict.keys())
    metrics_to_plot = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    n_models        = len(model_names)
    x               = np.arange(len(metrics_to_plot))
    width           = min(0.18, 0.8 / n_models)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(13, n_models * 3), 6))

        for i, name in enumerate(model_names):
            vals = [results_dict[name].get(m, 0.0) for m in metrics_to_plot]
            bars = ax.bar(
                x + i * width, vals, width,
                label=name,
                color=MODEL_COLORS.get(name, "#7f8c8d"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot], fontsize=10)
        ax.set_ylim([0, 1.18])
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Test Set Performance (Patient-Level Split)",
                     fontsize=13)
        ax.legend(loc="upper right", fontsize=9)

        path = os.path.join(save_dir, "ablation_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Ablation chart saved: {path}")
