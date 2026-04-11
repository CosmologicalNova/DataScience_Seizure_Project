# src/data/chbmit_index.py

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def parse_summary_file(summary_path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """
    Parse a CHB-MIT summary file and return seizure ranges per EDF file.

    Returns:
        dict like:
        {
            "chb01_03.edf": [(2996, 3036)],
            "chb01_04.edf": [],
            ...
        }
    """
    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    seizure_map: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    # Handles lines like:
    # "File Name: chb01_03.edf"
    file_pattern = re.compile(r"File Name:\s*(.+\.edf)", re.IGNORECASE)

    # Handles lines like:
    # "Seizure Start Time: 2996 seconds"
    start_pattern = re.compile(r"Seizure Start Time:\s*(\d+)\s*seconds", re.IGNORECASE)

    # Handles lines like:
    # "Seizure End Time: 3036 seconds"
    end_pattern = re.compile(r"Seizure End Time:\s*(\d+)\s*seconds", re.IGNORECASE)

    pending_start: int | None = None

    with summary_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            file_match = file_pattern.search(line)
            if file_match:
                current_file = file_match.group(1).strip()
                seizure_map.setdefault(current_file, [])
                pending_start = None
                continue

            start_match = start_pattern.search(line)
            if start_match:
                pending_start = int(start_match.group(1))
                continue

            end_match = end_pattern.search(line)
            if end_match and current_file is not None and pending_start is not None:
                seizure_end = int(end_match.group(1))
                seizure_map[current_file].append((pending_start, seizure_end))
                pending_start = None

    return seizure_map


def build_subject_index(data_root: str | Path, subject: str) -> list[dict[str, Any]]:
    """
    Build metadata records for one CHB-MIT subject.

    Expected folder structure:
        data_root/
            chb01/
                chb01_01.edf
                chb01_02.edf
                ...
                chb01-summary.txt

    Args:
        data_root: path to the folder containing subject folders
                   e.g. data/raw/
        subject: subject folder name, e.g. "chb01"

    Returns:
        List of records like:
        [
            {
                "subject": "chb01",
                "file_id": "chb01_03",
                "file_name": "chb01_03.edf",
                "file_path": "data/raw/chb01/chb01_03.edf",
                "seizure_ranges_sec": [(2996, 3036)],
            },
            ...
        ]
    """
    data_root = Path(data_root)
    subject_dir = data_root / subject

    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject folder not found: {subject_dir}")

    summary_path = subject_dir / f"{subject}-summary.txt"
    seizure_map = parse_summary_file(summary_path)

    edf_files = sorted(subject_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in: {subject_dir}")

    records: list[dict[str, Any]] = []

    for edf_path in edf_files:
        file_name = edf_path.name
        file_id = edf_path.stem

        records.append(
            {
                "subject": subject,
                "file_id": file_id,
                "file_name": file_name,
                "file_path": str(edf_path),
                "seizure_ranges_sec": seizure_map.get(file_name, []),
            }
        )

    return records


if __name__ == "__main__":
    # Example quick test
    root = "data/raw/"
    subject = "chb01"

    records = build_subject_index(root, subject)

    print(f"Total EDF files found for {subject}: {len(records)}")
    print("First 5 records:\n")

    for record in records[:5]:
        print(record)