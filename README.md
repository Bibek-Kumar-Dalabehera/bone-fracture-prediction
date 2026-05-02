# 🦴 Bone Fracture Detection — 2-Stage Deep Learning Pipeline

---

## 📌 Project Overview

This project detects and classifies bone fractures in X-ray images using a **two-stage deep learning approach** that mirrors real radiology workflow.

---

## 🧠 2-Stage Model Logic
```
X-ray Image
     │
     ▼
┌─────────────────────────────┐
│  STAGE 1 — EfficientNet-B4  │   Binary Classification
│  Normal  vs  Fracture       │
└─────────────────────────────┘
     │
     ├──── Normal ──────────────► ✅ Output: NORMAL (stop here)
     │
     └──── Fracture ────────────►
                                  ┌─────────────────────────────┐
                                  │  STAGE 2 — ResNet50          │   Multi-Class
                                  │  Hairline  vs  Major Crack   │
                                  └─────────────────────────────┘
                                       │              │
                                       ▼              ▼
                                  🦴 Hairline    💥 Major Crack
```

---

## 📁 Project Structure
```
bone_fracture_project/
├── data/
│   ├── stage1/
│   │   ├── train/   normal/  fracture/      (2492 + 5710 images)
│   │   ├── val/     normal/  fracture/      (532  + 1232 images)
│   │   └── test/    normal/  fracture/      (539  + 1238 images)
│   └── stage2/
│       ├── train/   hairline/  major_crack/ (77 + 222 → 693 + 1998 after aug)
│       ├── val/     hairline/  major_crack/ (17 + 48 images)
│       └── test/    hairline/  major_crack/ (17 + 48 images)
├── src/
│   ├── dataset.py          # Transforms + WeightedRandomSampler
│   ├── model_stage1.py     # EfficientNet-B4 — binary classifier
│   ├── model_stage2.py     # ResNet50 + FocalLoss — fracture type classifier
│   ├── train_stage1.py     # Stage 1 training loop
│   ├── train_stage2.py     # Stage 2 training loop
│   ├── augment_stage2.py   # Offline data augmentation (8x multiplier)
│   ├── evaluate.py         # Confusion matrix + ROC + Grad-CAM
│   └── inference.py        # 2-stage cascade prediction on new images
├── checkpoints/
│   ├── model1_best.pth     # Best Stage 1 weights (saved by fracture recall)
│   └── model2_best.pth     # Best Stage 2 weights (saved by macro F1)
└── results/
    ├── confusion_matrix_stage1.png
    ├── confusion_matrix_stage2.png
    ├── roc_curve_stage1.png
    └── gradcam_samples/
```
---


## 📦 Datasets Used

| Dataset | Used For | Direct Download |
|---|---|---|
| **Mendeley Fracture Dataset** | Stage 1 — Fracture class | [kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays](https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays) |
| **pkdarabi Bone Break** | Stage 2 — Hairline & Major Crack | [kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset) |

### Dataset Split Summary

| Stage | Class | Source | Train | Val | Test |
|---|---|---|---|---|---|
| Stage 1 | Normal | Mendeley | 2,492 | 532 | 539 |
| Stage 1 | Fracture | Mendeley | 5,710 | 1,232 | 1,238 |
| Stage 2 | Hairline | pkdarabi | 77 → **693** (8x aug) | 17 | 17 |
| Stage 2 | Major Crack | pkdarabi | 222 → **1,998** (8x aug) | 48 | 48 |


> ℹ️ **Note**  
> Stage 2 training counts reflect effective sample size due to oversampling and aggressive augmentation using `WeightedRandomSampler`.

---

## 🏗️ Model Architecture

### Stage 1 — EfficientNet-B4
| Property | Value |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Output classes | 2 — Normal / Fracture |
| Loss | FocalLoss (α=0.75, γ=2.0) |
| Key metric | **Fracture Recall ≥ 0.95** (missing a fracture is dangerous) |
| Decision threshold | 0.40 (not 0.50 — biased toward detecting fractures) |

### Stage 2 — ResNet50
| Property | Value |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained) |
| Output classes | 2 — Hairline / Major Crack |
| Loss | FocalLoss with inverse-frequency alpha weights |
| Key metric | **Macro F1 ≥ 0.80** |
| Imbalance fix | WeightedRandomSampler + 8x offline augmentation |

---

## ⚙️ Training Strategy

### Handling Class Imbalance (Stage 2)
- Hairline images: **77** originals → **693** after 8x augmentation
- Major crack images: **222** originals → **1,998** after 8x augmentation
- `WeightedRandomSampler` ensures balanced batches during training
- `FocalLoss` penalises the model more for missing minority class

### Transfer Learning Approach
```
Phase 1 (Warmup):   Freeze backbone → train head only   (fast convergence)
Phase 2 (Finetune): Unfreeze all   → train entire model (higher accuracy)
```

### Augmentation Pipeline (Stage 2)
- Random rotation ±25°
- Horizontal + vertical flip
- Affine transforms (translate, scale, shear)
- Color jitter (brightness, contrast)
- Gaussian blur
- Random erasing (simulates occlusion)

---

## 📊 Target Metrics

| Stage | Metric | Target |
|---|---|---|
| Stage 1 | AUROC | ≥ 0.95 |
| Stage 1 | Fracture Recall | ≥ 0.95 |
| Stage 2 | Macro F1 | ≥ 0.80 |
| Stage 2 | Hairline F1 | ≥ 0.75 |
