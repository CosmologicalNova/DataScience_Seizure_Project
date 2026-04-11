import argparse
from pathlib import Path

import torch


def load_split_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    data = torch.load(path, map_location="cpu")

    if "X" not in data or "y" not in data:
        raise ValueError(f"Invalid split file format: {path}")

    X = data["X"].float()
    y = data["y"].float().view(-1)

    if len(X) != len(y):
        raise ValueError(
            f"Mismatched X/y lengths in {path}: len(X)={len(X)}, len(y)={len(y)}"
        )

    return data, X, y


def summarize_split(split_name: str, X: torch.Tensor, y: torch.Tensor):
    total = len(y)
    pos = int(y.sum().item())
    neg = total - pos

    print(f"\n{split_name.upper()}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"Total windows: {total}")
    print(f"Positive: {pos}")
    print(f"Negative: {neg}")

    return total, pos, neg


def merge_one_split(
    split_name: str,
    patients: list[str],
    input_root: Path,
    output_root: Path,
):
    all_X = []
    all_y = []
    source_files = []
    patient_ids = []
    all_recording_ids = []
    all_window_indices = []

    print("\n" + "=" * 70)
    print(f"MERGING {split_name.upper()}")
    print("=" * 70)

    for patient in patients:
        split_path = input_root / patient / f"{split_name}.pt"
        data, X, y = load_split_file(split_path)

        recording_ids = data["recording_id"]
        window_indices = data["window_idx"]

        pos = int(y.sum().item())
        neg = len(y) - pos

        print(f"{patient}:")
        print(f"  source: {split_path}")
        print(f"  X shape: {tuple(X.shape)}")
        print(f"  y shape: {tuple(y.shape)}")
        print(f"  positive: {pos}")
        print(f"  negative: {neg}")

        all_X.append(X)
        all_y.append(y)
        source_files.append(str(split_path))
        patient_ids.extend([patient] * len(y))

        all_recording_ids.extend(recording_ids)
        all_window_indices.extend(window_indices)

    merged_X = torch.cat(all_X, dim=0)
    merged_y = torch.cat(all_y, dim=0)

    merged_obj = {
        "X": merged_X,
        "y": merged_y,
        "patient_id": patient_ids,
        "recording_id": all_recording_ids,
        "window_idx": all_window_indices,
        "patients": patients,
        "split": split_name,
        "source_files": source_files,
    }

    out_path = output_root / f"{split_name}.pt"
    torch.save(merged_obj, out_path)
    print(f"\nSaved merged {split_name} -> {out_path}")

    summarize_split(split_name, merged_X, merged_y)

    return merged_X, merged_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="data/processed/windowed_splits",
        help="Root folder containing per-patient split folders.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        required=True,
        help="Patients to merge, e.g. --patients chb01 chb02 chb03 chb04 chb05",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where merged train.pt / val.pt / test.pt will be saved. Defaults to data/processed/windowed_splits/multi_patient_<N>.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_root / f"multi_patient_{len(args.patients)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_X, train_y = merge_one_split(
        split_name="train",
        patients=args.patients,
        input_root=input_root,
        output_root=output_dir,
    )

    val_X, val_y = merge_one_split(
        split_name="val",
        patients=args.patients,
        input_root=input_root,
        output_root=output_dir,
    )

    test_X, test_y = merge_one_split(
        split_name="test",
        patients=args.patients,
        input_root=input_root,
        output_root=output_dir,
    )

    train_total = len(train_y)
    train_pos = int(train_y.sum().item())
    train_neg = train_total - train_pos
    pos_weight = train_neg / train_pos if train_pos > 0 else float("inf")

    print("\n" + "=" * 70)
    print("FINAL MERGED SUMMARY")
    print("=" * 70)
    print(f"TRAIN: X={tuple(train_X.shape)}, y={tuple(train_y.shape)}")
    print(f"VAL:   X={tuple(val_X.shape)}, y={tuple(val_y.shape)}")
    print(f"TEST:  X={tuple(test_X.shape)}, y={tuple(test_y.shape)}")
    print(f"Training positives: {train_pos}")
    print(f"Training negatives: {train_neg}")
    print(f"Training pos_weight: {pos_weight:.4f}")


if __name__ == "__main__":
    main()
