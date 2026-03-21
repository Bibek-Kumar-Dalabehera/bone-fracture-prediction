"""
Evaluate both stages on their test sets.
─────────────────────────────────────────
Run from project root:
    python src/evaluate.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from dataset      import get_dataloaders
from model_stage1 import build_stage1_model
from model_stage2 import build_stage2_model

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RESULTS   = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


# ─── Collect predictions ──────────────────────────────────────────────────────

def collect_preds(model, loader, device, return_probs=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()

    return (preds, labels, probs) if return_probs else (preds, labels)


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw counts", "Normalised (recall per class)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"{title} — {subtitle}", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_roc(labels, probs, class_names, title, save_path):
    """Binary ROC (Stage 1) or One-vs-Rest ROC (Stage 2)."""
    plt.figure(figsize=(8, 6))

    if probs.shape[1] == 2:
        # Binary
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc = roc_auc_score(labels, probs[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.4f}")
    else:
        # Multi-class OvR
        for i, cls in enumerate(class_names):
            bin_labels = (labels == i).astype(int)
            fpr, tpr, _ = roc_curve(bin_labels, probs[:, i])
            auc = roc_auc_score(bin_labels, probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{cls} AUROC = {auc:.4f}")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved → {save_path}")


# ─── Stage evaluations ────────────────────────────────────────────────────────

def evaluate_stage1():
    print(f"\n{'='*55}")
    print("  Evaluating Stage 1: Normal vs Fracture")
    print(f"{'='*55}")

    loaders = get_dataloaders(DATA_ROOT, stage=1, batch_size=64, num_workers=4)
    model   = build_stage1_model(pretrained=False).to(DEVICE)
    ckpt    = torch.load(os.path.join(CKPT_DIR, "model1_best.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    preds, labels, probs = collect_preds(model, loaders["test"], DEVICE, return_probs=True)
    class_names = ["normal", "fracture"]

    print("\n  Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    auroc = roc_auc_score(labels, probs[:, 1])
    print(f"  AUROC: {auroc:.4f}")

    plot_confusion_matrix(labels, preds, class_names,
                          "Stage 1 — Normal vs Fracture",
                          os.path.join(RESULTS, "confusion_matrix_stage1.png"))
    plot_roc(labels, probs, class_names,
             "Stage 1 — ROC Curve",
             os.path.join(RESULTS, "roc_stage1.png"))


def evaluate_stage2():
    print(f"\n{'='*55}")
    print("  Evaluating Stage 2: Hairline vs Major Crack")
    print(f"{'='*55}")

    loaders = get_dataloaders(DATA_ROOT, stage=2, batch_size=16, num_workers=4)
    model   = build_stage2_model(num_classes=2, pretrained=False).to(DEVICE)
    ckpt    = torch.load(os.path.join(CKPT_DIR, "model2_best.pth"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (val_f1={ckpt['val_f1']:.4f})")

    preds, labels, probs = collect_preds(model, loaders["test"], DEVICE, return_probs=True)
    class_names = ["hairline", "major_crack"]

    print("\n  Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    plot_confusion_matrix(labels, preds, class_names,
                          "Stage 2 — Fracture Type",
                          os.path.join(RESULTS, "confusion_matrix_stage2.png"))
    plot_roc(labels, probs, class_names,
             "Stage 2 — ROC Curve (One-vs-Rest)",
             os.path.join(RESULTS, "roc_stage2.png"))


if __name__ == "__main__":
    evaluate_stage1()
    evaluate_stage2()
    print("\nAll evaluation results saved to results/\n")
