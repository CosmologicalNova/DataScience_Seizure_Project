import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chbmit_index import build_subject_index
from src.data.datasets import EEGWindowDatasetByFiles


def split_files_seizure_aware(records):
    """
    Create seizure-aware train/val/test file splits.

    Strategy:
    - Separate seizure EDF files from non-seizure EDF files
    - Distribute seizure files across train/val/test when possible
    - Fill remaining files with non-seizure EDFs
    - Keep chronological order inside each split
    """
    seizure_files = []
    non_seizure_files = []

    for r in sorted(records, key=lambda x: x["file_name"]):
        fname = r["file_name"]
        if len(r["seizure_ranges_sec"]) > 0:
            seizure_files.append(fname)
        else:
            non_seizure_files.append(fname)

    n_total = len(seizure_files) + len(non_seizure_files)
    if n_total < 3:
        raise ValueError(f"Need at least 3 EDF files, got {n_total}")

    train_files = []
    val_files = []
    test_files = []

    # --- distribute seizure files first ---
    n_seiz = len(seizure_files)

    if n_seiz >= 3:
        # at least one seizure EDF in each split
        train_files.append(seizure_files[0])
        val_files.append(seizure_files[1])
        test_files.append(seizure_files[2])

        # extra seizure files go mostly to train, then val, then test
        extras = seizure_files[3:]
        for i, f in enumerate(extras):
            if i % 3 == 0:
                train_files.append(f)
            elif i % 3 == 1:
                val_files.append(f)
            else:
                test_files.append(f)

    elif n_seiz == 2:
        train_files.append(seizure_files[0])
        val_files.append(seizure_files[1])

    elif n_seiz == 1:
        train_files.append(seizure_files[0])

    # --- fill with non-seizure files ---
    # target rough sizes ~60/20/20
    target_train = max(1, int(0.6 * n_total))
    target_val = max(1, int(0.2 * n_total))
    # test gets the rest

    for f in non_seizure_files:
        if len(train_files) < target_train:
            train_files.append(f)
        elif len(val_files) < target_val:
            val_files.append(f)
        else:
            test_files.append(f)

    # safeguard: if any split ended empty, rebalance from the largest split
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name in ["train", "val", "test"]:
        if len(splits[split_name]) == 0:
            donor_name = max(splits, key=lambda k: len(splits[k]))
            if len(splits[donor_name]) <= 1:
                raise ValueError("Could not create non-empty train/val/test splits.")
            moved = splits[donor_name].pop()
            splits[split_name].append(moved)

    train_files = sorted(splits["train"])
    val_files = sorted(splits["val"])
    test_files = sorted(splits["test"])

    return train_files, val_files, test_files, seizure_files, non_seizure_files


def dataset_to_tensors(dataset, split_name: str):
    xs = []
    ys = []
    recording_ids = []
    window_indices = []

    total = len(dataset)
    print(f"\nBuilding {split_name} tensors...")
    print(f"Total windows: {total}")

    for i in range(total):
        x, y = dataset[i]
        sample_meta = dataset.samples[i]

        xs.append(x)
        ys.append(y)
        recording_ids.append(sample_meta["file_name"])
        window_indices.append(sample_meta["window_idx"])

        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  processed {i + 1}/{total}")

    X = torch.stack(xs, dim=0).float()
    y = torch.stack(ys, dim=0).float().view(-1)

    pos = int(y.sum().item())
    neg = len(y) - pos

    print(f"{split_name}:")
    print(f"  X shape = {tuple(X.shape)}")
    print(f"  y shape = {tuple(y.shape)}")
    print(f"  positive = {pos}")
    print(f"  negative = {neg}")

    return X, y, recording_ids, window_indices


def save_split(
    records,
    subject,
    allowed_files,
    split_name,
    out_dir,
    overwrite=False,
    window_size_sec=10,
    stride_sec=5,
    overlap_threshold=0.0,
):
    out_path = out_dir / f"{split_name}.pt"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing split -> {out_path}")
        saved = torch.load(out_path, map_location="cpu")
        return saved["X"], saved["y"]

    dataset = EEGWindowDatasetByFiles(
        records=records,
        allowed_files=allowed_files,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        overlap_threshold=overlap_threshold,
    )

    X, y, recording_ids, window_indices = dataset_to_tensors(dataset, split_name)

    save_obj = {
        "X": X,
        "y": y,
        "patient_id": subject,
        "recording_id": recording_ids,
        "window_idx": window_indices,
        "split": split_name,
        "allowed_files": allowed_files,
        "window_size_sec": window_size_sec,
        "stride_sec": stride_sec,
        "overlap_threshold": overlap_threshold,
    }   

    torch.save(save_obj, out_path)
    print(f"Saved -> {out_path}")

    return X, y


def process_subject(
    data_root,
    subject,
    out_root,
    overwrite=False,
    window_size_sec=10,
    stride_sec=5,
    overlap_threshold=0.0,
):
    print("\n" + "=" * 70)
    print(f"PROCESSING {subject}")
    print("=" * 70)

    out_dir = out_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_subject_index(data_root, subject)

    if len(records) == 0:
        print(f"[WARNING] No records found for {subject}, skipping.")
        return

    train_files, val_files, test_files, seizure_files, non_seizure_files = split_files_seizure_aware(records)

    print(f"Total EDF files: {len(records)}")
    print(f"Seizure EDF files ({len(seizure_files)}): {seizure_files}")
    print(f"Non-seizure EDF files ({len(non_seizure_files)}): {non_seizure_files}")
    print(f"Train files ({len(train_files)}): {train_files}")
    print(f"Val files   ({len(val_files)}): {val_files}")
    print(f"Test files  ({len(test_files)}): {test_files}")

    train_X, train_y = save_split(
        records=records,
        subject=subject,
        allowed_files=train_files,
        split_name="train",
        out_dir=out_dir,
        overwrite=overwrite,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        overlap_threshold=overlap_threshold,
    )

    val_X, val_y = save_split(
        records=records,
        subject=subject,
        allowed_files=val_files,
        split_name="val",
        out_dir=out_dir,
        overwrite=overwrite,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        overlap_threshold=overlap_threshold,
    )

    test_X, test_y = save_split(
        records=records,
        subject=subject,
        allowed_files=test_files,
        split_name="test",
        out_dir=out_dir,
        overwrite=overwrite,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        overlap_threshold=overlap_threshold,
    )

    train_pos = int(train_y.sum().item())
    train_neg = len(train_y) - train_pos
    pos_weight = train_neg / train_pos if train_pos > 0 else float("inf")

    print("\n" + "-" * 70)
    print(f"SUMMARY FOR {subject}")
    print("-" * 70)
    print(f"TRAIN: X={tuple(train_X.shape)}, y={tuple(train_y.shape)}")
    print(f"VAL:   X={tuple(val_X.shape)}, y={tuple(val_y.shape)}")
    print(f"TEST:  X={tuple(test_X.shape)}, y={tuple(test_y.shape)}")
    print(f"Training pos_weight = {pos_weight:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="Path to the directory containing subject EDF folders."
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["chb01"],
        help="List of patient IDs, e.g. --patients chb01 chb02 chb03 chb04 chb05 chb06"
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/windowed_splits",
        help="Root directory where each patient's split folder will be saved."
    )
    parser.add_argument("--window-size-sec", type=int, default=10)
    parser.add_argument("--stride-sec", type=int, default=5)
    parser.add_argument("--overlap-threshold", type=float, default=0.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild split files even if matching .pt outputs already exist.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for subject in args.patients:
        process_subject(
            data_root=data_root,
            subject=subject,
            out_root=out_root,
            overwrite=args.overwrite,
            window_size_sec=args.window_size_sec,
            stride_sec=args.stride_sec,
            overlap_threshold=args.overlap_threshold,
        )


if __name__ == "__main__":
    main()
