"""
src/training/trainer.py — Training loop with full diagnostics
==============================================================
Matches the chest X-ray project's Trainer class pattern.

Handles:
  - Train and validation loop per epoch
  - Gradient clipping (critical for LSTM/GRU)
  - AMP (automatic mixed precision) for faster GPU training
  - Loss, F1, Recall, and AUC logging to CSV
  - Early stopping on val F1 (or val recall if configured)
  - Checkpointing best weights

How to read training output:
    Train loss >> Val loss    → Underfitting (model too simple or LR too high)
    Train loss << Val loss    → Overfitting  (increase dropout or weight_decay)
    Both high and equal       → Underfitting (increase model capacity or epochs)
    Both low and equal        → Good fit
    Grad norm spikes          → Exploding gradients (lower LR or check grad_clip)
    Grad norm near zero       → Vanishing gradients (check BatchNorm + LSTM init)
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """
    General-purpose trainer — works with CNNBaseline, CNNLSTM, CNNGRU, TCNModel.

    Args:
        model       : PyTorch model (any of the 4 architectures)
        train_loader: Training DataLoader
        val_loader  : Validation DataLoader
        config      : Full config dict from configs/config.yaml
        model_name  : "cnn_baseline" | "cnn_lstm" | "cnn_gru" | "tcn"
        device      : torch.device
    """

    def __init__(self, model, train_loader, val_loader,
                 config: dict, model_name: str, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.model_name   = model_name
        self.device       = device

        t_cfg = config["training"]

        # ── Loss ──────────────────────────────────────────────────────────────
        # BCEWithLogitsLoss = sigmoid + binary cross entropy fused.
        # pos_weight is set in train.py after computing class frequencies.
        self.criterion = nn.BCEWithLogitsLoss()

        # ── Optimizer ─────────────────────────────────────────────────────────
        # AdamW: proper weight decay decoupled from gradient scaling.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
        )

        # ── LR Scheduler ──────────────────────────────────────────────────────
        # Cosine annealing: smoothly decays LR to eta_min over all epochs.
        # Prevents loss from bouncing near convergence.
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=t_cfg["epochs"],
            eta_min=1e-6,
        )

        # ── AMP ───────────────────────────────────────────────────────────────
        # Automatic mixed precision: float16 where safe, float32 elsewhere.
        # Speeds up training ~1.5-2x on NVIDIA GPU. Set mixed_precision: false if NaN loss.
        self.use_amp = t_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── Tracking ──────────────────────────────────────────────────────────
        self.grad_clip      = t_cfg.get("grad_clip", 1.0)
        self.monitor        = t_cfg.get("monitor_metric", "f1")
        self.patience       = t_cfg["early_stopping_patience"]
        self.epochs         = t_cfg["epochs"]
        self.threshold      = config["evaluation"]["threshold"]
        self.best_score     = -1.0
        self.patience_counter = 0

        # ── Paths ─────────────────────────────────────────────────────────────
        self.ckpt_path = os.path.join(config["paths"]["checkpoints"],
                                      f"{model_name}_best.pt")
        self.log_path  = os.path.join(config["paths"]["logs"],
                                      f"{model_name}_log.csv")

        os.makedirs(config["paths"]["checkpoints"], exist_ok=True)
        os.makedirs(config["paths"]["logs"],        exist_ok=True)

        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss",
                 "val_f1", "val_recall", "val_roc_auc", "lr", "grad_norm"]
            )

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_norm = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images).squeeze(1)
                loss   = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # Gradient clipping — critical for LSTM/GRU to prevent exploding gradients.
            # If loss suddenly goes NaN, this is likely the cause.
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            ).item()
            total_norm += norm

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss += loss.item()

        avg_norm = total_norm / len(self.train_loader)
        return total_loss / len(self.train_loader), avg_norm

    # ── Validation epoch ──────────────────────────────────────────────────────

    def _val_epoch(self) -> tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(images).squeeze(1)
                    loss   = self.criterion(logits, labels)

                total_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_probs  = 1 / (1 + np.exp(-all_logits))   # stable sigmoid
        all_preds  = (all_probs >= self.threshold).astype(int)

        f1     = f1_score(all_labels,  all_preds,  zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0   # Only one class present in val batch (rare edge case)

        return total_loss / len(self.val_loader), f1, recall, auc

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> float:
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name.upper()}")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}  |  AMP: {self.use_amp}  |  "
              f"Monitor: {self.monitor}")
        print(f"{'='*60}")
        print(f"{'Epoch':<8} {'Train Loss':<13} {'Val Loss':<11} "
              f"{'Val F1':<10} {'Val Recall':<12} {'Val AUC':<10} {'LR'}")
        print("─" * 72)

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_loss, grad_norm      = self._train_epoch()
            val_loss, f1, recall, auc  = self._val_epoch()
            self.scheduler.step()

            elapsed    = time.time() - start
            current_lr = self.optimizer.param_groups[0]["lr"]
            score      = f1 if self.monitor == "f1" else recall

            print(f"{epoch:<8} {train_loss:<13.4f} {val_loss:<11.4f} "
                  f"{f1:<10.4f} {recall:<12.4f} {auc:<10.4f} {current_lr:.2e}  "
                  f"({elapsed:.0f}s)")

            # ── CSV log ───────────────────────────────────────────────────────
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                     f"{f1:.4f}", f"{recall:.4f}", f"{auc:.4f}",
                     f"{current_lr:.6f}", f"{grad_norm:.4f}"]
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if score > self.best_score:
                self.best_score = score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f"  ✓ New best {self.monitor}: {score:.4f} — checkpoint saved")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")

            # ── Early stopping ────────────────────────────────────────────────
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best {self.monitor}: {self.best_score:.4f})")
                break

        print(f"\nTraining complete. Best val {self.monitor}: {self.best_score:.4f}")
        print(f"Checkpoint: {self.ckpt_path}")
        print(f"Log:        {self.log_path}")
        return self.best_score
