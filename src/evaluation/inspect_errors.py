# src/evaluation/inspect_errors.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

# Support direct execution like:
#   python src/evaluation/inspect_errors.py ...
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
            f"probs={len(probs)}, preds={len(preds)}, "
            f"targets={len(targets)}, patient_id={len(patient_ids)}"
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


def build_error_dataframe(data: dict) -> pd.DataFrame:
    probs = data["probs"]
    preds = data["preds"]
    targets = data["targets"]
    patient_ids = data["patient_id"]

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(probs)),
            "patient_id": patient_ids,
            "target": targets,
            "pred": preds,
            "prob": probs,
        }
    )

    df["error_type"] = "correct"
    df.loc[(df["target"] == 0) & (df["pred"] == 1), "error_type"] = "false_positive"
    df.loc[(df["target"] == 1) & (df["pred"] == 0), "error_type"] = "false_negative"

    # Confidence-style ranking:
    # - false positives: high seizure probability is worse
    # - false negatives: low seizure probability is worse
    df["error_score"] = np.nan
    fp_mask = df["error_type"] == "false_positive"
    fn_mask = df["error_type"] == "false_negative"

    df.loc[fp_mask, "error_score"] = df.loc[fp_mask, "prob"]
    df.loc[fn_mask, "error_score"] = 1.0 - df.loc[fn_mask, "prob"]

    return df


def summarize_errors_by_patient(error_df: pd.DataFrame) -> pd.DataFrame:
    err = error_df[error_df["error_type"] != "correct"].copy()

    if err.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "n_errors",
                "false_positives",
                "false_negatives",
                "avg_error_score",
                "avg_prob",
            ]
        )

    summary = (
        err.groupby("patient_id", dropna=False)
        .agg(
            n_errors=("error_type", "size"),
            false_positives=("error_type", lambda s: int((s == "false_positive").sum())),
            false_negatives=("error_type", lambda s: int((s == "false_negative").sum())),
            avg_error_score=("error_score", "mean"),
            avg_prob=("prob", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        by=["n_errors", "false_negatives", "false_positives", "avg_error_score", "patient_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return summary


def save_top_errors(error_df: pd.DataFrame, error_type: str, top_k: int, out_path: Path):
    subset = error_df[error_df["error_type"] == error_type].copy()

    if subset.empty:
        subset.to_csv(out_path, index=False)
        return subset

    subset = subset.sort_values(by="error_score", ascending=False).reset_index(drop=True)
    top_df = subset.head(top_k).copy()
    top_df.to_csv(out_path, index=False)
    return top_df


def save_all_errors(error_df: pd.DataFrame, out_path: Path):
    errors_only = error_df[error_df["error_type"] != "correct"].copy()
    errors_only = errors_only.sort_values(
        by=["error_type", "error_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    errors_only.to_csv(out_path, index=False)
    return errors_only


def print_summary(model_name: str, source_path: str, threshold, error_df: pd.DataFrame, patient_summary: pd.DataFrame):
    print("\n================ ERROR INSPECTION SUMMARY ================")
    print(f"Model       : {model_name}")
    print(f"Source file : {source_path}")
    print(f"Threshold   : {threshold}")

    total = len(error_df)
    fp = int((error_df["error_type"] == "false_positive").sum())
    fn = int((error_df["error_type"] == "false_negative").sum())
    correct = total - fp - fn

    print("\nCounts:")
    print(f"  Total samples : {total}")
    print(f"  Correct       : {correct}")
    print(f"  False positives: {fp}")
    print(f"  False negatives: {fn}")

    if not patient_summary.empty:
        print("\nPatients with most errors:")
        print(patient_summary.head(10).to_string(index=False))

    top_fp = (
        error_df[error_df["error_type"] == "false_positive"]
        .sort_values(by="error_score", ascending=False)
        .head(10)
    )
    top_fn = (
        error_df[error_df["error_type"] == "false_negative"]
        .sort_values(by="error_score", ascending=False)
        .head(10)
    )

    if not top_fp.empty:
        print("\nTop false positives:")
        print(
            top_fp[["sample_idx", "patient_id", "target", "pred", "prob", "error_score"]]
            .to_string(index=False)
        )

    if not top_fn.empty:
        print("\nTop false negatives:")
        print(
            top_fn[["sample_idx", "patient_id", "target", "pred", "prob", "error_score"]]
            .to_string(index=False)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.pt from evaluate_confusion_matrix.py",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top false positives / false negatives to save",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/evaluation/error_inspection",
        help="Directory to save CSV outputs",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    data = load_predictions(pred_path)
    model_name = str(data["model"]).lower()

    outdir = Path(args.outdir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    error_df = build_error_dataframe(data)
    patient_summary = summarize_errors_by_patient(error_df)

    all_errors_path = outdir / "all_errors.csv"
    fp_path = outdir / "top_false_positives.csv"
    fn_path = outdir / "top_false_negatives.csv"
    patient_path = outdir / "errors_by_patient.csv"

    save_all_errors(error_df, all_errors_path)
    save_top_errors(error_df, "false_positive", args.top_k, fp_path)
    save_top_errors(error_df, "false_negative", args.top_k, fn_path)
    patient_summary.to_csv(patient_path, index=False)

    print_summary(
        model_name=data["model"],
        source_path=data["source_path"],
        threshold=data["threshold"],
        error_df=error_df,
        patient_summary=patient_summary,
    )

    print("\nSaved:")
    print(f"  {all_errors_path}")
    print(f"  {fp_path}")
    print(f"  {fn_path}")
    print(f"  {patient_path}")


if __name__ == "__main__":
    main()
