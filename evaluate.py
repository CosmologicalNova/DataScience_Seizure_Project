"""
evaluate.py — Full evaluation on test set + all diagnostic charts
==================================================================
Run AFTER train.py:
    python evaluate.py

What happens:
  1. Loads test_loader.pt (same test patients as training used)
  2. For each model: loads checkpoint → evaluates → prints metrics → saves plots
  3. Plots training curves from logs/ CSVs
  4. Plots gradient norm history (useful for LSTM/GRU debugging)
  5. Runs ablation comparison chart across all 4 models
  6. Saves metrics_summary.csv to results/

Lower threshold if recall is too low:
    python evaluate.py
    Then set evaluation.threshold: 0.3 in configs/config.yaml and re-run.
    The threshold_sweep plot shows you the optimal value.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd

from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_lstm     import CNNLSTM
from src.models.cnn_gru      import CNNGRU
from src.models.tcn          import TCNModel
from src.evaluation.metrics  import (evaluate_model, compute_threshold_sweep,
                                      print_results_table)
from src.evaluation.visualize import (plot_training_curves, plot_grad_norm_history,
                                       plot_roc_curve, plot_pr_curve,
                                       plot_confusion_matrix, plot_threshold_sweep,
                                       plot_model_comparison)


def load_model(model_class, checkpoint_path: str, device, **kwargs):
    """Loads a trained model from checkpoint. Raises clear error if checkpoint missing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"\n[evaluate] Checkpoint not found: {checkpoint_path}\n"
            f"  Train the model first:  python train.py"
        )
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                      weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    return model


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = config["paths"]["results"]
    logs_dir    = config["paths"]["logs"]
    ckpt_dir    = config["paths"]["checkpoints"]
    threshold   = config["evaluation"]["threshold"]
    os.makedirs(results_dir, exist_ok=True)

    num_channels   = config["data"]["num_channels"]
    window_samples = int(config["data"]["window_size_sec"] * config["data"]["sampling_rate"])

    # ── Load test dataloader ───────────────────────────────────────────────────
    test_loader_path = os.path.join(ckpt_dir, "test_loader.pt")
    if not os.path.exists(test_loader_path):
        raise FileNotFoundError(
            f"\n[evaluate] test_loader.pt not found at {test_loader_path}\n"
            f"  Run training first:  python train.py"
        )
    test_loader = torch.load(test_loader_path, weights_only=False)

    # ── Load all models ────────────────────────────────────────────────────────
    print("\nLoading trained models...")
    cnn_cfg  = config["cnn_baseline"]
    lstm_cfg = config["cnn_lstm"]
    gru_cfg  = config["cnn_gru"]
    tcn_cfg  = config["tcn"]

    models = {
        "cnn_baseline": load_model(
            CNNBaseline, os.path.join(ckpt_dir, "cnn_baseline_best.pt"), device,
            num_channels=num_channels, window_samples=window_samples,
            out_channels=cnn_cfg["out_channels"], kernel_size=cnn_cfg["kernel_size"],
            dropout=cnn_cfg["dropout"],
        ),
        "cnn_lstm": load_model(
            CNNLSTM, os.path.join(ckpt_dir, "cnn_lstm_best.pt"), device,
            num_channels=num_channels, window_samples=window_samples,
            out_channels=lstm_cfg["out_channels"], kernel_size=lstm_cfg["kernel_size"],
            lstm_hidden=lstm_cfg["lstm_hidden"], lstm_layers=lstm_cfg["lstm_layers"],
            bidirectional=lstm_cfg["bidirectional"], dropout=lstm_cfg["dropout"],
        ),
        "cnn_gru": load_model(
            CNNGRU, os.path.join(ckpt_dir, "cnn_gru_best.pt"), device,
            num_channels=num_channels, window_samples=window_samples,
            out_channels=gru_cfg["out_channels"], kernel_size=gru_cfg["kernel_size"],
            gru_hidden=gru_cfg["gru_hidden"], gru_layers=gru_cfg["gru_layers"],
            bidirectional=gru_cfg["bidirectional"], dropout=gru_cfg["dropout"],
        ),
        "tcn": load_model(
            TCNModel, os.path.join(ckpt_dir, "tcn_best.pt"), device,
            num_channels=num_channels, window_samples=window_samples,
            num_block_channels=tcn_cfg["num_channels"], kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
        ),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Training curves + grad norm history (from log CSVs)
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPlotting training curves...")
    for model_name in models:
        log_path = os.path.join(logs_dir, f"{model_name}_log.csv")
        if os.path.exists(log_path):
            plot_training_curves(log_path, model_name, results_dir)
            plot_grad_norm_history(log_path, model_name, results_dir)
        else:
            print(f"  No log found for {model_name} — skipping curves")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Per-model evaluation
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'─'*60}")
        print(f"Evaluating: {model_name.upper()}  |  threshold: {threshold}")

        results, all_probs, all_labels, all_preds = evaluate_model(
            model, test_loader, device, threshold=threshold
        )
        all_results[model_name] = results
        print_results_table(results, model_name)

        sweep = compute_threshold_sweep(all_labels, all_probs)
        if abs(sweep["best_f1_threshold"] - threshold) > 0.1:
            print(f"\n  ⚠  Consider setting evaluation.threshold: "
                  f"{sweep['best_f1_threshold']:.2f} in configs/config.yaml "
                  f"(current: {threshold:.2f})")

        plot_roc_curve(all_labels, all_probs, model_name, results_dir,
                       auc=results["roc_auc"])
        plot_pr_curve(all_labels, all_probs, model_name, results_dir,
                      pr_auc=results["pr_auc"])
        plot_confusion_matrix(results["confusion_matrix"], model_name,
                               results_dir, threshold=threshold)
        plot_threshold_sweep(sweep, model_name, results_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Ablation comparison chart
    # ══════════════════════════════════════════════════════════════════════════
    print("\nGenerating ablation comparison chart...")
    plot_model_comparison(all_results, results_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Summary table printed to console
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  {'Model':<18} {'Recall':>8} {'Precision':>10} {'F1':>8} "
          f"{'ROC-AUC':>9} {'PR-AUC':>8}")
    print(f"  {'─'*65}")
    for name, r in all_results.items():
        print(f"  {name:<18} {r['recall']:>8.4f} {r['precision']:>10.4f} "
              f"{r['f1']:>8.4f} {r['roc_auc']:>9.4f} {r['pr_auc']:>8.4f}")
    print(f"{'='*70}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Save metrics_summary.csv
    # ══════════════════════════════════════════════════════════════════════════
    rows = []
    for model_name, metrics in all_results.items():
        row = {"model": model_name}
        row.update({k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix",)})
        rows.append(row)

    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nAll results saved to {results_dir}/")
    print(f"Metrics CSV: {csv_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
