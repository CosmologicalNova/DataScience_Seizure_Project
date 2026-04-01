"""
src/evaluation/metrics.py — Evaluation metrics for binary EEG seizure classification
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)


def evaluate_model(
    model,
    dataloader,
    device,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on a dataloader and compute all classification metrics.

    Args:
        threshold: Probability cutoff for binary prediction.
                   Lower → higher recall (catches more seizures, more false alarms)
                   Higher → higher precision (fewer false alarms, misses more seizures)
                   Use the threshold sweep plot to pick the best value.

    Returns:
        results    : dict with accuracy, precision, recall, f1, specificity,
                     roc_auc, pr_auc, threshold, tp, fp, tn, fn
        all_probs  : (N,) sigmoid probabilities
        all_labels : (N,) ground truth
        all_preds  : (N,) binary predictions at threshold
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_probs  = 1 / (1 + np.exp(-all_logits))    # numerically stable sigmoid
    all_preds  = (all_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    specificity    = tn / (tn + fp + 1e-8)

    results = {
        "accuracy":    float(accuracy_score(all_labels, all_preds)),
        "precision":   float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":      float(recall_score(all_labels,   all_preds, zero_division=0)),
        "f1":          float(f1_score(all_labels,       all_preds, zero_division=0)),
        "specificity": float(specificity),
        "roc_auc":     float(roc_auc_score(all_labels,  all_probs)),
        "pr_auc":      float(average_precision_score(all_labels, all_probs)),
        "threshold":   float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }

    return results, all_probs, all_labels, all_preds


def compute_threshold_sweep(
    all_labels: np.ndarray,
    all_probs:  np.ndarray,
    thresholds: np.ndarray = None,
) -> dict:
    """
    Computes precision, recall, F1 across a range of thresholds.

    Use the threshold_sweep plot from visualize.py to:
        1. Decide your recall target (e.g. "must catch ≥90% of seizures")
        2. Find the threshold where recall = 0.90
        3. Read off the corresponding precision and F1
        4. Set that threshold in configs/config.yaml → evaluation.threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 50)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        precisions.append(precision_score(all_labels, preds, zero_division=0))
        recalls.append(recall_score(all_labels, preds, zero_division=0))
        f1s.append(f1_score(all_labels, preds, zero_division=0))

    f1s_arr = np.array(f1s)
    return {
        "thresholds":         thresholds,
        "precisions":         np.array(precisions),
        "recalls":            np.array(recalls),
        "f1s":                f1s_arr,
        "best_f1_threshold":  float(thresholds[np.argmax(f1s_arr)]),
    }


def print_results_table(results: dict, model_name: str = ""):
    """Prints a clean aligned metrics table matching the chest X-ray project style."""
    header = f"── {model_name} " if model_name else "── "
    print(f"\n{header}{'─' * (55 - len(header))}")
    print(f"  {'Metric':<16} {'Value':>8}")
    print(f"  {'─'*26}")
    print(f"  {'Recall':<16} {results['recall']:>8.4f}   ← primary (missed seizures = FN)")
    print(f"  {'Precision':<16} {results['precision']:>8.4f}")
    print(f"  {'F1':<16} {results['f1']:>8.4f}")
    print(f"  {'Specificity':<16} {results['specificity']:>8.4f}")
    print(f"  {'ROC-AUC':<16} {results['roc_auc']:>8.4f}")
    print(f"  {'PR-AUC':<16} {results['pr_auc']:>8.4f}   ← use under class imbalance")
    print(f"  {'Accuracy':<16} {results['accuracy']:>8.4f}")
    print(f"  {'Threshold':<16} {results['threshold']:>8.2f}")
    print(f"  TP={results['tp']}  FP={results['fp']}  "
          f"TN={results['tn']}  FN={results['fn']}")
