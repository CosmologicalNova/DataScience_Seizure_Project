"""
scripts/setup_data.py — One-time setup: downloads CHB-MIT and runs preprocessing
=================================================================================
Run once from project root:
    python scripts/setup_data.py

CHB-MIT is Open Access — no account or credentials needed.

Download method priority:
  1. AWS S3  (fastest — same speed as Kaggle, recommended)
  2. requests over HTTP (fallback if AWS CLI not installed, slower)

After this script completes:
    python train.py
"""

import os
import sys
import re
import glob
import subprocess
from pathlib import Path

RAW_DIR           = "data/raw"
PROCESSED_DIR     = "data/processed"
S3_PATH           = "s3://physionet-open/chbmit/1.0.0/"
HTTP_BASE_URL     = "https://physionet.org/files/chbmit/1.0.0"
EXPECTED_PATIENTS = 24


# ── Check how many patients already downloaded ────────────────────────────────

def count_existing_patients() -> int:
    if not os.path.exists(RAW_DIR):
        return 0
    return len([d for d in os.listdir(RAW_DIR)
                if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("chb")])


# ── Method 1: AWS S3 (fast) ───────────────────────────────────────────────────

def has_aws_cli() -> bool:
    try:
        result = subprocess.run(["aws", "--version"],
                                capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_via_s3() -> bool:
    """
    Uses AWS CLI S3 sync — same speed as Kaggle downloads.
    --no-sign-request: no AWS account needed, dataset is public.
    --only-show-errors: suppresses per-file spam, shows progress cleanly.
    Safe to interrupt and re-run — sync skips already-downloaded files.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading via AWS S3 (fast)...")
    print(f"  {S3_PATH} → {RAW_DIR}/")
    print("Safe to interrupt and re-run — sync skips already-downloaded files.\n")

    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        "--only-show-errors",
        S3_PATH,
        RAW_DIR,
    ]

    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("\nS3 download complete.")
            return True
        else:
            print(f"\nS3 sync exited with code {result.returncode}.")
            return False
    except KeyboardInterrupt:
        print("\n\nInterrupted. Re-run to resume — S3 sync skips existing files.")
        sys.exit(0)


# ── Method 2: HTTP requests (fallback) ───────────────────────────────────────

def get_requests():
    try:
        import requests
        return requests
    except ImportError:
        os.system(f"{sys.executable} -m pip install requests --quiet")
        import requests
        return requests


def list_patient_files(requests, patient_id: str) -> list:
    url = f"{HTTP_BASE_URL}/{patient_id}/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', r.text)
    return [h for h in hrefs
            if h.endswith(".edf") or h.endswith(".txt")]


def download_file_http(requests, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100 * downloaded / total
                    mb  = downloaded / 1e6
                    print(f"\r    {dest.name}  {mb:.1f} MB  ({pct:.0f}%)",
                          end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n    ERROR: {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_via_http() -> bool:
    requests = get_requests()
    os.makedirs(RAW_DIR, exist_ok=True)
    patients = [f"chb{i:02d}" for i in range(1, EXPECTED_PATIENTS + 1)]

    print(f"Downloading via HTTP (slower — install AWS CLI for faster speeds).")
    print(f"Safe to interrupt and re-run.\n")

    failed = []
    for patient_id in patients:
        patient_dir = Path(RAW_DIR) / patient_id
        if list(patient_dir.glob("*.edf")):
            print(f"  [{patient_id}] already present — skipping")
            continue

        print(f"  [{patient_id}] fetching file list...")
        try:
            files = list_patient_files(requests, patient_id)
        except Exception as e:
            print(f"  [{patient_id}] ERROR: {e}")
            failed.append(patient_id)
            continue

        patient_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{patient_id}] downloading {len(files)} files...")
        any_failed = False
        for filename in files:
            url  = f"{HTTP_BASE_URL}/{patient_id}/{filename}"
            dest = patient_dir / filename
            if not download_file_http(requests, url, dest):
                any_failed = True
        if any_failed:
            failed.append(patient_id)

    if failed:
        print(f"\nWarning: {len(failed)} patients had errors: {failed}")
        print("Re-run to retry.")
        return False

    return True


# ── Main download orchestrator ────────────────────────────────────────────────

def download_dataset() -> bool:
    existing = count_existing_patients()
    if existing >= EXPECTED_PATIENTS:
        print(f"Dataset already downloaded ({existing} patients in {RAW_DIR}/)")
        return True

    if has_aws_cli():
        print("AWS CLI detected — using S3 (fast).")
        return download_via_s3()
    else:
        print("AWS CLI not found — falling back to HTTP (slower).")
        print()
        print("  To get fast download speeds, install AWS CLI:")
        print("    Windows: https://aws.amazon.com/cli/  (download the MSI installer)")
        print("    Then re-run this script.")
        print()
        return download_via_http()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def run_preprocessing():
    print("\nRunning preprocessing...")
    print("(Windowing EDF files, labeling seizure windows, patient-level split)\n")
    ret = os.system(
        f"{sys.executable} src/data/preprocess.py --config configs/config.yaml"
    )
    if ret != 0:
        print("\nPreprocessing failed. Check the error above.")
        sys.exit(1)


# ── Verification ──────────────────────────────────────────────────────────────

def verify():
    patients = sorted(glob.glob(os.path.join(RAW_DIR, "chb*")))
    edfs     = glob.glob(os.path.join(RAW_DIR, "chb*", "*.edf"))
    processed_files = [
        os.path.join(PROCESSED_DIR, f)
        for f in ["windows_train.npy", "labels_train.npy",
                  "windows_val.npy",   "labels_val.npy",
                  "windows_test.npy",  "labels_test.npy",
                  "pos_weight.npy",    "meta.json"]
    ]
    processed_ok = all(os.path.exists(f) for f in processed_files)

    print("\n" + "=" * 50)
    print(f"Patients downloaded : {len(patients)} / {EXPECTED_PATIENTS}")
    print(f"EDF files           : {len(edfs)}")
    print(f"Processed files     : {'Ready' if processed_ok else 'MISSING'}")

    if len(patients) >= EXPECTED_PATIENTS and processed_ok:
        print("\nSetup complete. Quick test run:")
        print("  1. Set epochs: 2 and n_train_patients: 4 in configs/config.yaml")
        print("  2. python train.py")
    else:
        if len(patients) < EXPECTED_PATIENTS:
            print(f"\nOnly {len(patients)} patient folders found.")
            print("Re-run this script to resume.")
        if not processed_ok:
            print("\nPreprocessed files missing — check errors above.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("EEG Seizure Detection — Data Setup")
    print("=" * 50)
    print()

    success = download_dataset()
    if success:
        run_preprocessing()
    verify()


if __name__ == "__main__":
    main()