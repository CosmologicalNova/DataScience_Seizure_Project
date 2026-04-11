# src/evaluation/roc_pr_curves.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Support direct execution like:
#   python src/evaluation/plot_roc_pr_curves.py ...
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    required_keys = ["probs", "targets", "model"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")

    probs = np.asarray(data["probs"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.int32)
    model_name = str(data["model"])

    if probs.ndim != 1:
        probs = probs.reshape(-1)

    if targets.ndim != 1:
        targets = targets.reshape(-1)

    if len(probs) != len(targets):
        raise ValueError(
            f"Mismatched lengths in {path}: len(probs)={len(probs)}, len(targets)={len(targets)}"
        )

    if len(np.unique(targets)) < 2:
        raise ValueError(
            f"{path} has only one class in targets. ROC/PR curves require both classes."
        )

    return {
        "probs": probs,
        "targets": targets,
        "model": model_name,
        "threshold": data.get("threshold", None),
        "source_path": str(path),
    }


def compute_curve_metrics(targets: np.ndarray, probs: np.ndarray):
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = roc_auc_score(targets, probs)

    precision, recall, _ = precision_recall_curve(targets, probs)
    ap = average_precision_score(targets, probs)

    positive_rate = float(np.mean(targets))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "positive_rate": positive_rate,
    }


def pretty_model_name(name: str) -> str:
    name = name.lower().strip()
    if name == "cnn":
        return "CNN"
    if name == "cnn_lstm":
        return "CNN+LSTM"
    if name == "cnn_gru":
        return "CNN+GRU"
    if name == "tcn":
        return "TCN"
    return name.upper()


def save_roc_plot(curves: list[dict], out_path: Path):
    plt.figure(figsize=(7, 6))

    for item in curves:
        label = f"{item['display_name']} (AUC = {item['roc_auc']:.4f})"
        plt.plot(item["fpr"], item["tpr"], linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pr_plot(curves: list[dict], out_path: Path):
    plt.figure(figsize=(7, 6))

    for item in curves:
        label = f"{item['display_name']} (AP = {item['ap']:.4f})"
        plt.plot(item["recall"], item["precision"], linewidth=2, label=label)

    if curves:
        baseline = curves[0]["positive_rate"]
        plt.axhline(
            y=baseline,
            linestyle="--",
            linewidth=1,
            label=f"Positive rate baseline = {baseline:.4f}",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-preds",
        type=str,
        required=True,
        help="Path to CNN predictions.pt",
    )
    parser.add_argument(
        "--cnn-lstm-preds",
        type=str,
        required=True,
        help="Path to CNN+LSTM predictions.pt",
    )
    parser.add_argument(
        "--cnn-gru-preds",
        type=str,
        default=None,
        help="Optional path to CNN+GRU predictions.pt",
    )
    parser.add_argument(
        "--tcn-preds",
        type=str,
        default=None,
        help="Optional path to TCN predictions.pt",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/evaluation/comparison",
        help="Directory to save ROC/PR plots",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cnn_data = load_predictions(Path(args.cnn_preds))
    cnn_lstm_data = load_predictions(Path(args.cnn_lstm_preds))
    cnn_gru_data = (
        load_predictions(Path(args.cnn_gru_preds))
        if args.cnn_gru_preds is not None
        else None
    )
    tcn_data = (
        load_predictions(Path(args.tcn_preds))
        if args.tcn_preds is not None
        else None
    )

    results = []
    inputs = [cnn_data, cnn_lstm_data]
    if cnn_gru_data is not None:
        inputs.append(cnn_gru_data)
    if tcn_data is not None:
        inputs.append(tcn_data)

    for item in inputs:
        metrics = compute_curve_metrics(item["targets"], item["probs"])
        results.append(
            {
                **metrics,
                "model": item["model"],
                "display_name": pretty_model_name(item["model"]),
                "source_path": item["source_path"],
            }
        )

    save_roc_plot(results, outdir / "roc_comparison.png")
    save_pr_plot(results, outdir / "pr_comparison.png")

    print("\n================ ROC / PR SUMMARY ================")
    for item in results:
        print(f"\nModel: {item['display_name']}")
        print(f"  Source file   : {item['source_path']}")
        print(f"  ROC AUC       : {item['roc_auc']:.4f}")
        print(f"  Average Prec. : {item['ap']:.4f}")
        print(f"  Positive rate : {item['positive_rate']:.4f}")

    print("\nSaved:")
    print(f"  {outdir / 'roc_comparison.png'}")
    print(f"  {outdir / 'pr_comparison.png'}")


if __name__ == "__main__":
    main()
