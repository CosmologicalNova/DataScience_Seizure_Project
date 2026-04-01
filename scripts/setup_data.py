"""
scripts/setup_data.py — One-time setup: downloads CHB-MIT and runs preprocessing
=================================================================================
Run once from project root:
    python scripts/setup_data.py

Requires PHYSIONET_USERNAME and PHYSIONET_PASSWORD in your .env file.
Get a free PhysioNet account at: https://physionet.org/register/

What it does:
  1. Reads credentials from .env
  2. Downloads CHB-MIT Scalp EEG Database via wfdb (~45GB total)
     Returns instantly if already downloaded.
  3. Runs preprocessing automatically (windowing, labeling, patient-level split)

After this script completes:
    python train.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
DATASET_NAME  = "chbmit"           # PhysioNet database identifier
EXPECTED_PATIENTS = 24


def check_credentials():
    username = os.getenv("PHYSIONET_USERNAME", "").strip()
    password = os.getenv("PHYSIONET_PASSWORD", "").strip()

    if not username or not password:
        print("ERROR: PhysioNet credentials not found in .env file.")
        print()
        print("Steps to fix:")
        print("  1. Create a free account at https://physionet.org/register/")
        print("  2. Open the .env file in the project root")
        print("  3. Fill in:")
        print("       PHYSIONET_USERNAME=your_username")
        print("       PHYSIONET_PASSWORD=your_password")
        print("  4. Re-run: python scripts/setup_data.py")
        sys.exit(1)

    return username, password


def download_dataset(username, password):
    try:
        import wfdb
    except ImportError:
        print("Installing wfdb...")
        os.system(f"{sys.executable} -m pip install wfdb --quiet")
        import wfdb

    os.makedirs(RAW_DIR, exist_ok=True)

    # Check how many patients are already downloaded
    existing = [d for d in os.listdir(RAW_DIR)
                if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("chb")]

    if len(existing) >= EXPECTED_PATIENTS:
        print(f"Dataset already downloaded ({len(existing)} patients in {RAW_DIR})")
        return True

    print(f"Downloading CHB-MIT Scalp EEG Database to {RAW_DIR}/")
    print("~45GB on first run — returns cached path instantly if already downloaded.")
    print("This will take a while. Go get a coffee.\n")

    try:
        wfdb.dl_database(
            DATASET_NAME,
            dl_dir=RAW_DIR,
            username=username,
            password=password,
        )
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print()
        print("Common causes:")
        print("  - Wrong username or password in .env")
        print("  - You haven't accepted the PhysioNet data use agreement")
        print("    → Go to https://physionet.org/content/chbmit/1.0.0/ and click 'Access'")
        print("  - Network timeout on large download — re-run to resume")
        return False


def run_preprocessing():
    print("\nRunning preprocessing...")
    print("(Windowing EDF files, labeling seizure windows, patient-level split)")
    ret = os.system(f"{sys.executable} src/data/preprocess.py --config configs/config.yaml")
    if ret != 0:
        print("\nPreprocessing failed. Check the error above.")
        sys.exit(1)


def verify():
    import glob
    patients = sorted(glob.glob(os.path.join(RAW_DIR, "chb*")))
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
    print(f"Processed files     : {'Ready' if processed_ok else 'MISSING'}")

    if len(patients) >= EXPECTED_PATIENTS and processed_ok:
        print("\nSetup complete.")
        print("Set epochs: 2 in configs/config.yaml for a quick test, then run:")
        print("    python train.py")
    else:
        if len(patients) < EXPECTED_PATIENTS:
            print(f"\nOnly {len(patients)} patient folders found — download may be incomplete.")
            print("Re-run this script to resume.")
        if not processed_ok:
            print("\nPreprocessed files missing — check preprocess.py errors above.")


def main():
    print("=" * 50)
    print("EEG Seizure Detection — Data Setup")
    print("=" * 50)

    username, password = check_credentials()
    success = download_dataset(username, password)

    if success:
        run_preprocessing()

    verify()


if __name__ == "__main__":
    main()
