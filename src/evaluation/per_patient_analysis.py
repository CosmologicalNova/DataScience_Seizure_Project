# src/evaluation/per_patient_analysis.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

# Support direct execution like:
#   python src/evaluation/per_patient_analysis.py ...
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    required = ["probs", "preds", "targets", "patient_id", "model"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    probs = np.asarray(data["probs"]).reshape(-1)
    preds = np.asarray(data["preds"]).reshape(-1).astype(int)
    targets = np.asarray(data["targets"]).reshape(-1).astype(int)
    patient_ids = np.asarray(data["patient_id"]).reshape(-1)

    if not (len(probs) == len(preds) == len(targets) == len(patient_ids)):
        raise ValueError(
            f"Length mismatch in {path}: "
            f"probs={len(probs)}, preds={len(preds)}, targets={len(targets)}, patient_id={len(patient_ids)}"
        )

    patient_ids = np.array([str(pid) for pid in patient_ids], dtype=object)

    return {
        "probs": probs,
        "preds": preds,
        "targets": targets,
        "patient_id": patient_ids,
        "model": str(data["model"]),
        "threshold": data.get("threshold", None),
        "source_path": str(path),
    }


def safe_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = safe_confusion_counts(y_true, y_pred)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def build_per_patient_table(data: dict) -> pd.DataFrame:
    probs = data["probs"]
    preds = data["preds"]
    targets = data["targets"]
    patient_ids = data["patient_id"]

    rows = []
    unique_patients = sorted(np.unique(patient_ids).tolist())

    for pid in unique_patients:
        mask = patient_ids == pid

        y_true = targets[mask]
        y_pred = preds[mask]
        y_prob = probs[mask]

        metrics = compute_binary_metrics(y_true, y_pred)

        positives = int((y_true == 1).sum())
        negatives = int((y_true == 0).sum())

        rows.append(
            {
                "patient_id": pid,
                "n_samples": int(mask.sum()),
                "positives": positives,
                "negatives": negatives,
                "positive_rate": safe_div(positives, len(y_true)),
                "avg_prob": float(np.mean(y_prob)) if len(y_prob) > 0 else 0.0,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["f1", "recall", "precision", "positives", "patient_id"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)

    return df


def add_overall_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    total = {
        "patient_id": "OVERALL",
        "n_samples": int(df["n_samples"].sum()),
        "positives": int(df["positives"].sum()),
        "negatives": int(df["negatives"].sum()),
        "tn": int(df["tn"].sum()),
        "fp": int(df["fp"].sum()),
        "fn": int(df["fn"].sum()),
        "tp": int(df["tp"].sum()),
    }

    total["positive_rate"] = safe_div(total["positives"], total["n_samples"])
    total["avg_prob"] = np.nan
    total["precision"] = safe_div(total["tp"], total["tp"] + total["fp"])
    total["recall"] = safe_div(total["tp"], total["tp"] + total["fn"])
    total["f1"] = safe_div(
        2 * total["precision"] * total["recall"],
        total["precision"] + total["recall"],
    )
    total["specificity"] = safe_div(total["tn"], total["tn"] + total["fp"])
    total["accuracy"] = safe_div(
        total["tp"] + total["tn"],
        total["tp"] + total["tn"] + total["fp"] + total["fn"],
    )

    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def save_bar_plot(df: pd.DataFrame, metric: str, title: str, out_path: Path):
    plot_df = df[df["patient_id"] != "OVERALL"].copy()

    if plot_df.empty:
        print(f"Skipping plot for {metric}: no patient rows available.")
        return

    plot_df = plot_df.sort_values(by=metric, ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["patient_id"], plot_df[metric])
    plt.ylim(0, 1.0)
    plt.xlabel("Patient ID")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_support_plot(df: pd.DataFrame, out_path: Path):
    plot_df = df[df["patient_id"] != "OVERALL"].copy()

    if plot_df.empty:
        print("Skipping support plot: no patient rows available.")
        return

    plot_df = plot_df.sort_values(by="positives", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, plot_df["positives"], width=width, label="Positives")
    plt.bar(x + width / 2, plot_df["negatives"], width=width, label="Negatives")
    plt.xticks(x, plot_df["patient_id"], rotation=45)
    plt.xlabel("Patient ID")
    plt.ylabel("Count")
    plt.title("Per-Patient Sample Support")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_summary(df: pd.DataFrame, model_name: str, source_path: str):
    print("\n================ PER-PATIENT SUMMARY ================")
    print(f"Model       : {model_name}")
    print(f"Source file : {source_path}")

    overall = df[df["patient_id"] == "OVERALL"]
    if not overall.empty:
        row = overall.iloc[0]
        print("\nOverall:")
        print(f"  Samples    : {int(row['n_samples'])}")
        print(f"  Positives  : {int(row['positives'])}")
        print(f"  Negatives  : {int(row['negatives'])}")
        print(f"  TN / FP    : {int(row['tn'])} / {int(row['fp'])}")
        print(f"  FN / TP    : {int(row['fn'])} / {int(row['tp'])}")
        print(f"  Precision  : {row['precision']:.4f}")
        print(f"  Recall     : {row['recall']:.4f}")
        print(f"  F1         : {row['f1']:.4f}")
        print(f"  Accuracy   : {row['accuracy']:.4f}")
        print(f"  Specificity: {row['specificity']:.4f}")

    patient_only = df[df["patient_id"] != "OVERALL"].copy()
    if not patient_only.empty:
        best_f1 = patient_only.sort_values(by="f1", ascending=False).iloc[0]
        worst_f1 = patient_only.sort_values(by="f1", ascending=True).iloc[0]

        print("\nBest patient by F1:")
        print(
            f"  {best_f1['patient_id']} | "
            f"F1={best_f1['f1']:.4f} | "
            f"Precision={best_f1['precision']:.4f} | "
            f"Recall={best_f1['recall']:.4f} | "
            f"Positives={int(best_f1['positives'])}"
        )

        print("\nWorst patient by F1:")
        print(
            f"  {worst_f1['patient_id']} | "
            f"F1={worst_f1['f1']:.4f} | "
            f"Precision={worst_f1['precision']:.4f} | "
            f"Recall={worst_f1['recall']:.4f} | "
            f"Positives={int(worst_f1['positives'])}"
        )

        print("\nPer-patient table:")
        display_cols = [
            "patient_id",
            "n_samples",
            "positives",
            "negatives",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]
        print(patient_only[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.pt from evaluate_confusion_matrix.py",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/evaluation/per_patient",
        help="Directory to save CSV and plots",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    data = load_predictions(pred_path)
    model_name = str(data["model"]).lower()

    outdir = Path(args.outdir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_per_patient_table(data)
    df = add_overall_row(df)

    csv_path = outdir / "per_patient_metrics.csv"

    df.to_csv(csv_path, index=False)

    save_bar_plot(
        df,
        metric="f1",
        title=f"{model_name.upper()} Per-Patient F1",
        out_path=outdir / "per_patient_f1.png",
    )

    save_bar_plot(
        df,
        metric="recall",
        title=f"{model_name.upper()} Per-Patient Recall",
        out_path=outdir / "per_patient_recall.png",
    )

    save_bar_plot(
        df,
        metric="precision",
        title=f"{model_name.upper()} Per-Patient Precision",
        out_path=outdir / "per_patient_precision.png",
    )

    save_support_plot(
        df,
        out_path=outdir / "per_patient_support.png",
    )

    print_summary(df, model_name=data["model"], source_path=data["source_path"])

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {outdir / 'per_patient_f1.png'}")
    print(f"  {outdir / 'per_patient_recall.png'}")
    print(f"  {outdir / 'per_patient_precision.png'}")
    print(f"  {outdir / 'per_patient_support.png'}")


if __name__ == "__main__":
    main()
