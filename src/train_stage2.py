"""
Train Stage 2: Hairline vs Major Crack (imbalanced multi-class)
───────────────────────────────────────────────────────────────
Run from project root:
    python src/train_stage2.py
"""

import os
import sys
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, os.path.dirname(__file__))
from dataset      import get_dataloaders
from model_stage2 import build_stage2_model, get_focal_loss, unfreeze_all

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 16        # small batch for small dataset
NUM_WORKERS = 4
WARMUP_EPOCHS = 5       # longer warmup — smaller dataset needs more head training first
TOTAL_EPOCHS  = 60      # more epochs — dataset is tiny
LR_HEAD     = 1e-3
LR_FULL     = 1e-4      # lower LR after unfreezing (tiny data = risk of overfitting)
WEIGHT_DECAY = 1e-3     # higher L2 regularisation for tiny dataset
PATIENCE    = 15        # more patience — F1 fluctuates on small sets

# Stage 2 class counts from your folder (sorted alphabetically = ImageFolder order)
# hairline=0, major_crack=1
CLASS_COUNTS = {"hairline": 77, "major_crack": 709}
NUM_CLASSES  = len(CLASS_COUNTS)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_f1, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"          : epoch,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_f1"         : val_f1,
    }, path)


def compute_f1_per_class(preds, labels, num_classes):
    """Macro F1 without sklearn dependency in the hot loop."""
    f1_scores = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / num_classes
    return macro_f1, f1_scores


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # Move focal loss alpha to device
    if hasattr(criterion, "alpha") and criterion.alpha is not None:
        criterion.alpha = criterion.alpha.to(device)

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    avg_loss = total_loss / len(labels)
    acc      = (preds == labels).float().mean().item()
    macro_f1, per_class_f1 = compute_f1_per_class(preds, labels, NUM_CLASSES)

    return avg_loss, acc, macro_f1, per_class_f1


# ─── Main ─────────────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*60}")
    print(f"  Stage 2 Training — Hairline vs Major Crack")
    print(f"  Device: {DEVICE}")
    print(f"  Class counts: {CLASS_COUNTS}")
    print(f"{'='*60}\n")

    loaders  = get_dataloaders(DATA_ROOT, stage=2, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    criterion = get_focal_loss(CLASS_COUNTS)

    model = build_stage2_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True).to(DEVICE)
    print(f"\nModel loaded. Warming up head for {WARMUP_EPOCHS} epochs...\n")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    # CosineAnnealingWarmRestarts: restarts LR every T_0 epochs — helps escape local minima on small data
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_f1      = 0.0
    patience_counter = 0
    best_ckpt_path   = os.path.join(CKPT_DIR, "model2_best.pth")
    class_names      = sorted(CLASS_COUNTS.keys())   # ["hairline", "major_crack"]

    for epoch in range(1, TOTAL_EPOCHS + 1):
        t0 = time.time()

        if epoch == WARMUP_EPOCHS + 1:
            print(f"\n→ Unfreezing full backbone at epoch {epoch}\n")
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        tr_loss, tr_acc, tr_f1, _ = run_epoch(model, loaders["train"], criterion, optimizer, DEVICE, train=True)
        vl_loss, vl_acc, vl_f1, vl_pcf1 = run_epoch(model, loaders["val"], criterion, optimizer, DEVICE, train=False)

        scheduler.step(epoch)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"Train Loss: {tr_loss:.4f} F1: {tr_f1:.4f} | "
            f"Val Loss: {vl_loss:.4f} F1: {vl_f1:.4f} Acc: {vl_acc:.4f} | "
            f"{elapsed:.1f}s"
        )
        # Per-class F1 detail
        for cls_name, f1 in zip(class_names, vl_pcf1):
            print(f"    {cls_name:20s} F1 = {f1:.4f}")

        # Best model saved by macro F1 (not accuracy — imbalanced dataset)
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            save_checkpoint(model, optimizer, epoch, vl_f1, best_ckpt_path)
            print(f"  ✓ Best model saved (macro_f1={vl_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val macro F1: {best_val_f1:.4f}")
    print(f"  Checkpoint saved to: {best_ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
