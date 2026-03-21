"""
Train Stage 1: Normal vs Fracture (binary classification)
─────────────────────────────────────────────────────────
Run from project root:
    python src/train_stage1.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from dataset     import get_dataloaders
from model_stage1 import build_stage1_model, unfreeze_all

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 64        # increase to 128 if GPU VRAM > 12 GB
NUM_WORKERS = 4
WARMUP_EPOCHS = 3       # freeze backbone, train head only
TOTAL_EPOCHS  = 30
LR_HEAD     = 1e-3      # learning rate during warmup
LR_FULL     = 3e-4      # learning rate after unfreezing
WEIGHT_DECAY = 1e-4
PATIENCE    = 7         # early stopping patience


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc"   : val_acc,
    }, path)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*60}")
    print(f"  Stage 1 Training — Normal vs Fracture")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Dataloaders
    loaders = get_dataloaders(DATA_ROOT, stage=1, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model — freeze backbone for warmup
    model = build_stage1_model(pretrained=True, freeze_backbone=True).to(DEVICE)
    print(f"\nModel loaded. Warming up head for {WARMUP_EPOCHS} epochs...\n")

    criterion = nn.CrossEntropyLoss()

    # Warmup optimizer (head only)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS)

    best_val_acc    = 0.0
    patience_counter = 0
    best_ckpt_path  = os.path.join(CKPT_DIR, "model1_best.pth")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        t0 = time.time()

        # Unfreeze full model after warmup
        if epoch == WARMUP_EPOCHS + 1:
            print(f"\n→ Unfreezing full backbone at epoch {epoch}\n")
            unfreeze_all(model)
            optimizer = optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS)

        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, DEVICE, train=True)
        val_loss,   val_acc   = run_epoch(model, loaders["val"],   criterion, optimizer, DEVICE, train=False)

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint saved to: {best_ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
