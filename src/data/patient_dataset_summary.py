from __future__ import annotations
from pathlib import Path
import torch


DATA_DIR = Path("data/processed/windowed_splits")


def summarize_split(split_path: Path):
    data = torch.load(split_path, map_location="cpu")

    X = data["X"]
    y = data["y"].float().view(-1)

    total = len(y)
    positives = int(y.sum().item())
    negatives = total - positives

    pos_ratio = positives / total if total > 0 else 0.0
    pos_weight = negatives / positives if positives > 0 else float("inf")

    return {
        "windows": total,
        "positives": positives,
        "negatives": negatives,
        "pos_ratio": pos_ratio,
        "pos_weight": pos_weight,
        "shape": tuple(X.shape),
    }


def main():
    patient_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])

    print("=" * 90)
    print("PATIENT DATASET SUMMARY")
    print("=" * 90)

    for patient_dir in patient_dirs:
        train_file = patient_dir / "train.pt"

        if not train_file.exists():
            continue

        stats = summarize_split(train_file)

        print(f"\n{patient_dir.name}")
        print("-" * 50)
        print(f"X shape     : {stats['shape']}")
        print(f"Windows     : {stats['windows']}")
        print(f"Positives   : {stats['positives']}")
        print(f"Negatives   : {stats['negatives']}")
        print(f"Pos ratio   : {stats['pos_ratio']:.6f}")
        print(f"Pos weight  : {stats['pos_weight']:.4f}")


if __name__ == "__main__":
    main()