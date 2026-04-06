"""
src/data/dataset.py — PyTorch Dataset and DataLoader factory
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    """
    PyTorch Dataset for windowed EEG seizure detection data.

    Loaded with mmap_mode="r" so numpy doesn't load the entire array into RAM
    on init — it reads windows on-demand as the DataLoader requests them.
    This matters for the full CHB-MIT dataset which can be several GB.

    Augmentation (training only):
      Gaussian noise: simulates electrode noise
      Channel dropout: simulates bad electrode contact (common in real EEG)
    """

    def __init__(self, windows_path: str, labels_path: str, augment: bool = False):
        self.windows = np.load(windows_path, mmap_mode="r")
        self.labels  = np.load(labels_path)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.windows[idx].copy().astype(np.float32)
        y = float(self.labels[idx])

        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Apply to training data only — never to val/test.

        Gaussian noise: std = 1% of signal range.
        Channel dropout: zeros one random channel with 20% probability.
          (Simulates a bad electrode — very common in clinical EEG recordings.)

        To add more augmentation: add transforms here.
        To remove one: comment it out.
        """
        noise_std = 0.01 * (x.max() - x.min() + 1e-8)
        x = x + np.random.normal(0, noise_std, x.shape).astype(np.float32)

        if np.random.rand() < 0.2:
            drop_ch = np.random.randint(0, x.shape[0])
            x[drop_ch, :] = 0.0

        return x


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train, val, and test DataLoaders from preprocessed .npy files.

    Raises a clear FileNotFoundError if preprocessing hasn't been run yet,
    rather than a confusing numpy error deep in the stack.
    """
    processed_dir = Path(config["data"]["processed_dir"])

    required = ["windows_train.npy", "labels_train.npy",
                "windows_val.npy",   "labels_val.npy",
                "windows_test.npy",  "labels_test.npy"]

    missing = [f for f in required if not (processed_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\n[dataset] Processed data not found in '{processed_dir}'.\n"
            f"  Missing: {missing}\n\n"
            f"  Run setup first:\n"
            f"    python scripts/setup_data.py\n"
            f"  Or preprocess only:\n"
            f"    python src/data/preprocess.py --config configs/config.yaml"
        )

    train_ds = EEGDataset(processed_dir / "windows_train.npy",
                          processed_dir / "labels_train.npy",  augment=True)
    val_ds   = EEGDataset(processed_dir / "windows_val.npy",
                          processed_dir / "labels_val.npy",    augment=False)
    test_ds  = EEGDataset(processed_dir / "windows_test.npy",
                          processed_dir / "labels_test.npy",   augment=False)

    bs = config["training"]["batch_size"]
    nw = config["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    print(f"Train: {len(train_ds):,} windows  |  "
          f"Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def load_pos_weight(config: dict, device: torch.device) -> torch.Tensor:
    """
    Loads the pre-computed pos_weight for BCEWithLogitsLoss.
    pos_weight = (# non-seizure windows) / (# seizure windows) in training set.
    Computed from training patients only during preprocessing.
    """
    path   = Path(config["data"]["processed_dir"]) / "pos_weight.npy"
    weight = float(np.load(path)[0])
    print(f"pos_weight: {weight:.2f}  (penalizes missing a seizure {weight:.1f}x more)")
    return torch.tensor([weight], dtype=torch.float32).to(device)
