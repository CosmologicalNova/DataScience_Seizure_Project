"""
scripts/learning_curve.py — Plot F1 and Recall vs number of training patients
==============================================================================
Run after you have trained at multiple patient counts and renamed your logs.

Workflow (same pattern as the chest X-ray project):
  1. Set n_train_patients: 4  in configs/config.yaml
     python src/data/preprocess.py --config configs/config.yaml
     python train.py
     Rename logs: e.g. logs/cnn_lstm_log.csv → logs/cnn_lstm_4pat.csv

  2. Set n_train_patients: 8, repeat
  3. Set n_train_patients: 12, repeat
  4. Set n_train_patients: 16 (full), repeat

  5. python scripts/learning_curve.py

What it tells you:
  Curves still rising at 16 patients → more training patients would help (get more data)
  Curves flatten at 8 patients       → architecture bottleneck, not data size
  Large gap between models           → temporal modeling helps regardless of data size
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR    = "logs"
RESULTS_DIR = "results"
MODEL_NAMES = ["cnn_baseline", "cnn_lstm", "cnn_gru", "tcn"]
COLORS      = {
    "cnn_baseline": "#e74c3c",
    "cnn_lstm":     "#2980b9",
    "cnn_gru":      "#27ae60",
    "tcn":          "#8e44ad",
}


def get_best_metrics(csv_path: str) -> dict:
    """Returns the best val_f1 and corresponding val_recall from a training log CSV."""
    df = pd.read_csv(csv_path)
    best_idx = df["val_f1"].idxmax()
    return {
        "f1":     df.loc[best_idx, "val_f1"],
        "recall": df.loc[best_idx, "val_recall"],
    }


def parse_patient_count(filename: str) -> int:
    """
    Extracts patient count from renamed log filename.
    e.g. cnn_lstm_4pat.csv → 4
    """
    base = os.path.basename(filename).replace(".csv", "")
    part = base.split("_")[-1]          # e.g. "4pat"
    return int(part.replace("pat", ""))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, (ax_f1, ax_recall) = plt.subplots(1, 2, figsize=(14, 6))
    found_any = False

    for model_name in MODEL_NAMES:
        pattern  = os.path.join(LOGS_DIR, f"{model_name}_*pat.csv")
        log_files = sorted(glob.glob(pattern))

        if not log_files:
            print(f"No patient-count logs found for {model_name} — skipping.")
            print(f"  Expected files like: logs/{model_name}_4pat.csv")
            continue

        counts, f1s, recalls = [], [], []

        for log_file in log_files:
            try:
                n = parse_patient_count(log_file)
                metrics = get_best_metrics(log_file)
                counts.append(n)
                f1s.append(metrics["f1"])
                recalls.append(metrics["recall"])
                print(f"  {model_name} @ {n} patients → F1: {metrics['f1']:.4f}  "
                      f"Recall: {metrics['recall']:.4f}")
            except Exception as e:
                print(f"  Skipping {log_file}: {e}")

        if not counts:
            continue

        # Sort by patient count
        pairs = sorted(zip(counts, f1s, recalls))
        counts, f1s, recalls = zip(*pairs)
        color = COLORS.get(model_name, "gray")

        for ax, values, label in [(ax_f1, f1s, "F1"), (ax_recall, recalls, "Recall")]:
            ax.plot(counts, values, marker="o", label=model_name,
                    color=color, linewidth=2, markersize=8)
            for n, v in zip(counts, values):
                ax.annotate(f"{v:.3f}", xy=(n, v),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)

        found_any = True

    if not found_any:
        print("\nNo learning curve data found.")
        print("Train at multiple patient counts and rename logs before running this script.")
        print("Example after 4-patient run:")
        print("  Rename logs/cnn_lstm_log.csv → logs/cnn_lstm_4pat.csv")
        return

    for ax, title in [(ax_f1, "F1-Score"), (ax_recall, "Recall (Sensitivity)")]:
        ax.set_xlabel("Training Patients Used", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"Learning Curve — {title} vs Training Patients", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "learning_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nLearning curve saved: {path}")


if __name__ == "__main__":
    main()
