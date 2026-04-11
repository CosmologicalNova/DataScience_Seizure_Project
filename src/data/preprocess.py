"""
src/data/preprocess.py — Converts raw CHB-MIT .edf files to windowed numpy arrays
===================================================================================
Run once before training (setup_data.py calls this automatically):
    python src/data/preprocess.py --config configs/config.yaml

Memory strategy:
    Two-pass approach — no temp files, no large in-memory arrays:
      Pass 1: count total windows across all patients (fast, no data stored)
      Pass 2: preallocate final .npy on disk as memmap, fill one EDF at a time

    Peak RAM = one EDF file at a time (~40-300 MB). Disk usage = final output only.
"""

import os
import re
import json
import argparse
import numpy as np
import yaml
from pathlib import Path

import pyedflib


# ── Summary file parsing ──────────────────────────────────────────────────────

def parse_summary_file(summary_path: Path) -> dict:
    annotations  = {}
    current_file = None
    num_seizures = 0

    with open(summary_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[1].strip()
            annotations[current_file] = []
            num_seizures = 0
        elif line.startswith("Number of Seizures in File:"):
            num_seizures = int(line.split(":", 1)[1].strip())
        elif re.match(r"Seizure\s*(\d+\s*)?Start Time:", line):
            if current_file and num_seizures > 0:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    annotations[current_file].append([int(m.group(1)), None])
        elif re.match(r"Seizure\s*(\d+\s*)?End Time:", line):
            if current_file and num_seizures > 0 and annotations[current_file]:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    annotations[current_file][-1][1] = int(m.group(1))

    return {
        fname: [(s, e) for s, e in seiz_list if e is not None]
        for fname, seiz_list in annotations.items()
    }


# ── EDF loading ───────────────────────────────────────────────────────────────

def load_edf(edf_path: Path, target_channels: int = 23) -> np.ndarray | None:
    try:
        f         = pyedflib.EdfReader(str(edf_path))
        n_ch      = f.signals_in_file
        n_samples = f.getNSamples()[0]
        data      = np.zeros((n_ch, n_samples), dtype=np.float32)
        for i in range(n_ch):
            data[i, :] = f.readSignal(i).astype(np.float32)
        f.close()

        if data.shape[0] > target_channels:
            data = data[:target_channels, :]
        elif data.shape[0] < target_channels:
            pad  = np.zeros((target_channels - data.shape[0], data.shape[1]),
                            dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        return data
    except Exception as e:
        print(f"  [WARNING] Could not load {edf_path.name}: {e}")
        return None


# ── Window count (pass 1 — no data stored) ───────────────────────────────────

def count_windows_for_edf(n_samples: int, window_samples: int,
                           stride_samples: int) -> int:
    count = 0
    start = 0
    while start + window_samples <= n_samples:
        count += 1
        start += stride_samples
    return count


def count_windows_for_patient(patient_dir: Path, cfg: dict) -> int:
    """Fast pass: count windows without loading signal data."""
    sr             = cfg["data"]["sampling_rate"]
    win_sec        = cfg["data"]["window_size_sec"]
    stride_sec     = cfg["data"]["stride_sec"]
    window_samples = int(win_sec * sr)
    stride_samples = int(stride_sec * sr)

    total = 0
    for edf_path in sorted(patient_dir.glob("*.edf")):
        try:
            f         = pyedflib.EdfReader(str(edf_path))
            n_samples = f.getNSamples()[0]
            f.close()
            total += count_windows_for_edf(n_samples, window_samples,
                                           stride_samples)
        except Exception:
            pass
    return total


# ── Window extraction ─────────────────────────────────────────────────────────

def extract_windows(data: np.ndarray, seizure_intervals: list,
                    sampling_rate: int, window_size_sec: float,
                    stride_sec: float, overlap_threshold: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(window_size_sec * sampling_rate)
    stride_samples = int(stride_sec * sampling_rate)
    windows, labels = [], []
    start = 0

    while start + window_samples <= data.shape[1]:
        end              = start + window_samples
        w_start          = start / sampling_rate
        w_end            = end   / sampling_rate
        label            = 0

        for s, e in seizure_intervals:
            overlap = max(0.0, min(w_end, e) - max(w_start, s))
            if overlap / window_size_sec >= overlap_threshold:
                label = 1
                break

        windows.append(data[:, start:end])
        labels.append(label)
        start += stride_samples

    return (np.array(windows, dtype=np.float32),
            np.array(labels,  dtype=np.int64))


def normalize(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=-1, keepdims=True)
    std  = windows.std(axis=-1,  keepdims=True) + 1e-8
    return (windows - mean) / std


# ── Two-pass processing ───────────────────────────────────────────────────────

def process_split(patient_ids: list, raw_dir: Path, cfg: dict,
                  out_win: Path, out_lab: Path) -> np.ndarray:
    """
    Two-pass processing for one split (train/val/test).

    Pass 1: count total windows → preallocate memmap output files on disk
    Pass 2: fill memmap one EDF at a time → never more than one EDF in RAM

    Returns the labels array (small, needed for pos_weight).
    """
    sr             = cfg["data"]["sampling_rate"]
    num_ch         = cfg["data"]["num_channels"]
    win_sec        = cfg["data"]["window_size_sec"]
    stride_sec     = cfg["data"]["stride_sec"]
    overlap_thresh = cfg["data"]["seizure_overlap_threshold"]
    window_samples = int(win_sec * sr)

    # ── Pass 1: count windows ─────────────────────────────────────────────────
    print("  Counting windows...")
    valid_patients = []
    total_windows  = 0

    for pid in patient_ids:
        patient_dir = raw_dir / pid
        if not patient_dir.exists():
            print(f"  [{pid}] Directory not found — skipping")
            continue
        summary_files = list(patient_dir.glob("*-summary.txt"))
        if not summary_files:
            print(f"  [{pid}] No summary file — skipping")
            continue
        n = count_windows_for_patient(patient_dir, cfg)
        if n > 0:
            valid_patients.append(pid)
            total_windows += n

    if total_windows == 0:
        np.save(out_win, np.zeros((0, num_ch, window_samples), dtype=np.float32))
        np.save(out_lab, np.zeros(0, dtype=np.int64))
        return np.zeros(0, dtype=np.int64)

    print(f"  Allocating output files for {total_windows:,} windows...")

    # Preallocate on disk as proper .npy memmaps (compatible with np.load)
    win_mm = np.lib.format.open_memmap(str(out_win), dtype="float32", mode="w+",
                                        shape=(total_windows, num_ch, window_samples))
    lab_mm = np.lib.format.open_memmap(str(out_lab), dtype="int64", mode="w+",
                                        shape=(total_windows,))

    # ── Pass 2: fill one EDF at a time ───────────────────────────────────────
    idx = 0
    for pid in valid_patients:
        patient_dir   = raw_dir / pid
        annotations   = parse_summary_file(
            list(patient_dir.glob("*-summary.txt"))[0]
        )
        edf_files     = sorted(patient_dir.glob("*.edf"))
        pat_n         = 0
        pat_seizure   = 0

        for edf_path in edf_files:
            data = load_edf(edf_path, target_channels=num_ch)
            if data is None:
                continue

            seizure_intervals = annotations.get(edf_path.name, [])
            windows, labels   = extract_windows(
                data, seizure_intervals, sr, win_sec, stride_sec, overlap_thresh
            )
            del data  # free immediately

            n = len(labels)
            if n == 0:
                continue

            # Normalize this EDF's windows in-place (small, ~1-3k windows)
            windows = normalize(windows)

            # Write directly to disk memmap
            win_mm[idx:idx + n] = windows
            lab_mm[idx:idx + n] = labels
            idx        += n
            pat_n      += n
            pat_seizure += int(labels.sum())
            del windows, labels

        if pat_n > 0:
            print(f"  [{pid}] {pat_n:,} windows  |  "
                  f"{pat_seizure} seizure "
                  f"({100*pat_seizure/pat_n:.1f}%)")

    win_mm.flush()
    lab_mm.flush()
    labels_out = np.array(lab_mm)
    del win_mm, lab_mm

    return labels_out


# ── Main ──────────────────────────────────────────────────────────────────────

def preprocess(cfg: dict):
    raw_dir       = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    sr             = cfg["data"]["sampling_rate"]
    win_sec        = cfg["data"]["window_size_sec"]
    stride_sec     = cfg["data"]["stride_sec"]
    n_channels     = cfg["data"]["num_channels"]
    window_samples = int(win_sec * sr)

    train_patients = cfg["data"]["train_patients"]
    val_patients   = cfg["data"]["val_patients"]
    test_patients  = cfg["data"]["test_patients"]
    n_train        = cfg["data"].get("n_train_patients", len(train_patients))
    n_val          = cfg["data"].get("n_val_patients",   len(val_patients))
    n_test         = cfg["data"].get("n_test_patients",  len(test_patients))
    train_patients = train_patients[:n_train]
    val_patients   = val_patients[:n_val]
    test_patients  = test_patients[:n_test]

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"\n[preprocess] Raw data directory not found: {raw_dir}\n"
            f"  Run:  python scripts/setup_data.py"
        )

    print(f"Window: {win_sec}s  |  Stride: {stride_sec}s  |  "
          f"Channels: {n_channels}")
    print(f"Train: {n_train} patients  |  "
          f"Val: {n_val} patients  |  "
          f"Test: {n_test} patients\n")

    split_labels = {}
    splits = {
        "train": train_patients,
        "val":   val_patients,
        "test":  test_patients,
    }

    for split, patient_ids in splits.items():
        print(f"── {split.upper()} patients ─────────────────────────")
        labels = process_split(
            patient_ids, raw_dir, cfg,
            processed_dir / f"windows_{split}.npy",
            processed_dir / f"labels_{split}.npy",
        )
        split_labels[split] = labels
        n   = len(labels)
        pos = int(labels.sum())
        print(f"  → {n:,} total windows  |  "
              f"{pos} seizure ({100*pos/max(n,1):.2f}%)\n")

    # pos_weight
    y_train    = split_labels["train"]
    n_pos      = max(int(y_train.sum()), 1)
    n_neg      = len(y_train) - n_pos
    pos_weight = n_neg / n_pos
    print(f"pos_weight (BCEWithLogitsLoss): {pos_weight:.2f}  "
          f"(neg={n_neg:,} / pos={n_pos:,})")

    np.save(processed_dir / "pos_weight.npy",
            np.array([pos_weight], dtype=np.float32))

    meta = {
        "window_size_sec":   win_sec,
        "stride_sec":        stride_sec,
        "sampling_rate":     sr,
        "num_channels":      n_channels,
        "n_train_patients":  n_train,
        "n_val_patients":    n_val,
        "n_test_patients":   n_test,
        "train_patients":    train_patients,
        "val_patients":      val_patients,
        "test_patients":     test_patients,
        "n_train":           int(len(split_labels["train"])),
        "n_val":             int(len(split_labels["val"])),
        "n_test":            int(len(split_labels["test"])),
        "seizure_pct_train": float(y_train.mean()) if len(y_train) > 0 else 0.0,
        "pos_weight":        float(pos_weight),
    }
    with open(processed_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPreprocessing complete. Files saved to: {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    preprocess(cfg)