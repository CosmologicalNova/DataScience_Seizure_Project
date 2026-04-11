# src/data/datasets.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import tempfile
import warnings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_DIR = _PROJECT_ROOT / ".runtime"
_TMP_DIR = _RUNTIME_DIR / "tmp"
_MPL_DIR = _RUNTIME_DIR / "matplotlib"

for runtime_dir in (_TMP_DIR, _MPL_DIR):
    runtime_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", str(_TMP_DIR))
os.environ.setdefault("TEMP", str(_TMP_DIR))
os.environ.setdefault("TMP", str(_TMP_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
tempfile.tempdir = str(_TMP_DIR)

import mne
import numpy as np
import torch
from torch.utils.data import Dataset


def _is_valid_channel_name(name: str) -> bool:
    # CHB14/15 contain placeholder channels like "--0" with incomplete metadata.
    return not name.startswith("--")


def _read_raw_edf(file_path: str | Path, preload: bool):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Channel names are not unique")
        warnings.filterwarnings(
            "ignore",
            message="Scaling factor is not defined in following channels:.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(str(file_path), preload=preload, verbose=False)

    # Rename duplicate channels so downstream code sees unique names
    seen = {}
    rename_counts = {}
    new_names = {}
    for name in raw.ch_names:
        rename_counts[name] = rename_counts.get(name, 0) + 1
        count = rename_counts[name]
        if count > 1:
            new_names[name] = f"{name}_{count}"
        seen[name] = count
    if new_names:
        raw.rename_channels(new_names)

    return raw


def load_edf_channel_names(file_path: str | Path) -> list[str]:
    raw = _read_raw_edf(file_path, preload=False)
    return list(raw.ch_names)


def load_edf_signal(
    file_path: str | Path,
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load one EDF file.

    Returns:
        signal: np.ndarray of shape [C, T]
        fs: sampling frequency
    """
    raw = _read_raw_edf(file_path, preload=True)

    if channel_names is not None:
        raw.pick(channel_names)

    signal = raw.get_data().astype(np.float32)
    fs = int(raw.info["sfreq"])
    return signal, fs


class EEGWindowDatasetByFiles(Dataset):
    """
    Dynamic sliding-window dataset built directly from raw EDF files.

    Each record is expected to come from build_subject_index(), like:
        {
            "subject": "chb01",
            "file_id": "chb01_03",
            "file_name": "chb01_03.edf",
            "file_path": "data/raw/chb01/chb01_03.edf",
            "seizure_ranges_sec": [(2996, 3036)],
        }

    This dataset preserves the same logic as your old file-based patient dataset:
    - split by EDF files
    - sliding windows
    - overlap-based seizure labels
    - optional per-window normalization
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        allowed_files: list[str],
        window_size_sec: float = 10.0,
        stride_sec: float = 5.0,
        overlap_threshold: float = 0.0,
        normalize_per_window: bool = True,
    ) -> None:
        super().__init__()

        self.allowed_files = set(allowed_files)
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.overlap_threshold = overlap_threshold
        self.normalize_per_window = normalize_per_window

        self.all_records = records
        self.records = [r for r in records if r["file_name"] in self.allowed_files]
        if not self.records:
            raise ValueError("No records matched allowed_files.")

        self.subject = self.records[0]["subject"]

        # IMPORTANT: resolve channels across ALL subject files
        self.channel_names = self._resolve_shared_channel_names(self.all_records)

        # Will hold lightweight sample metadata only
        self.samples: list[dict[str, Any]] = []

        # 1-file cache to avoid re-reading same EDF for nearby windows
        self._cache_path: str | None = None
        self._cache_signal: np.ndarray | None = None
        self._cache_fs: int | None = None

        self._build_index()

        print(
            f"[{self.subject}] files={len(self.records)} | "
            f"channels={len(self.channel_names)} | "
            f"windows={len(self.samples):,}"
        )

    def _resolve_shared_channel_names(self, records_for_channel_resolution: list[dict[str, Any]]) -> list[str]:
        reference_all_names = load_edf_channel_names(
            records_for_channel_resolution[0]["file_path"]
        )
        reference_names = [
            name for name in reference_all_names if _is_valid_channel_name(name)
        ]
        shared_names = set(reference_names)
        invalid_names = {
            name for name in reference_all_names if not _is_valid_channel_name(name)
        }

        for record in records_for_channel_resolution[1:]:
            all_channel_names = load_edf_channel_names(record["file_path"])
            invalid_names.update(
                name for name in all_channel_names if not _is_valid_channel_name(name)
            )
            channel_names = {
                name for name in all_channel_names if _is_valid_channel_name(name)
            }
            shared_names &= channel_names

        ordered_shared = [name for name in reference_names if name in shared_names]

        if not ordered_shared:
            raise ValueError(
                f"No shared channels found across files for subject {self.subject}."
            )

        if invalid_names:
            print(
                f"[{self.subject}] excluding placeholder channels: "
                f"{sorted(invalid_names)}"
            )

        if len(ordered_shared) != len(reference_names):
            removed = [name for name in reference_names if name not in shared_names]
            print(f"[{self.subject}] dropping non-shared channels: {removed}")

        return ordered_shared

    def _build_index(self) -> None:
        """
        Build window metadata from raw EDF files.

        Stores only:
        - file_path
        - file_name
        - start_sample
        - end_sample
        - label
        """
        for record in self.records:
            file_path = record["file_path"]
            file_name = record["file_name"]
            seizure_ranges_sec = record["seizure_ranges_sec"]

            signal, fs = load_edf_signal(file_path, channel_names=self.channel_names)
            total_samples = signal.shape[1]

            window_size = int(self.window_size_sec * fs)
            stride = int(self.stride_sec * fs)

            window_idx = 0
            start = 0

            while start + window_size <= total_samples:
                end = start + window_size
                label = self._window_label(
                    start=start,
                    end=end,
                    fs=fs,
                    seizure_ranges_sec=seizure_ranges_sec,
                )

                self.samples.append(
                    {
                        "file_path": file_path,
                        "file_name": file_name,
                        "start_sample": start,
                        "end_sample": end,
                        "label": label,
                        "window_idx": window_idx,
                    }
                )

                start += stride
                window_idx += 1
                
    def _window_label(
        self,
        start: int,
        end: int,
        fs: int,
        seizure_ranges_sec: list[tuple[int, int]],
    ) -> int:
        """
        Label = 1 if seizure overlap fraction >= overlap_threshold.

        overlap_threshold = 0.0 means any overlap gives positive label.
        """
        window_len = end - start
        start_sec = start / fs
        end_sec = end / fs

        for seiz_start_sec, seiz_end_sec in seizure_ranges_sec:
            overlap_sec = max(
                0.0,
                min(end_sec, float(seiz_end_sec)) - max(start_sec, float(seiz_start_sec))
            )

            if overlap_sec > 0:
                frac = overlap_sec / (self.window_size_sec + 1e-8)
                if frac >= self.overlap_threshold:
                    return 1

        return 0

    def _get_signal(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        1-file cache:
        if next sample comes from same EDF, reuse it.
        """
        if self._cache_path != file_path:
            signal, fs = load_edf_signal(file_path, channel_names=self.channel_names)
            self._cache_path = file_path
            self._cache_signal = signal
            self._cache_fs = fs

        return self._cache_signal, self._cache_fs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        signal, fs = self._get_signal(sample["file_path"])
        x = signal[:, sample["start_sample"]:sample["end_sample"]]  # [C, W]
        y = sample["label"]

        x = torch.tensor(x, dtype=torch.float32)

        if self.normalize_per_window:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-8
            x = (x - mean) / std

        y = torch.tensor(y, dtype=torch.float32)
        return x, y


if __name__ == "__main__":
    from src.data.chbmit_index import build_subject_index

    root = "data/raw"
    subject = "chb01"

    records = build_subject_index(root, subject)

    # Example split just to test
    train_files = [f"chb01_{i:02d}.edf" for i in range(1, 17)]

    ds = EEGWindowDatasetByFiles(
        records=records,
        allowed_files=train_files,
        window_size_sec=10.0,
        stride_sec=5.0,
        overlap_threshold=0.0,
        normalize_per_window=True,
    )

    positives = sum(sample["label"] for sample in ds.samples)
    negatives = len(ds) - positives

    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")

    x, y = ds[0]
    print(f"Sample x shape: {x.shape}")
    print(f"Sample y: {y}")
