# Bone Fracture Detection Project

This project focuses on identifying bone fractures in X-ray images using deep learning. It employs a two-stage approach to efficiently detect and classify fractures.

## 📁 Directory Structure

```text
bone_fracture_project/
├── data/
│   ├── raw/                     # Original downloaded datasets (Mendeley, FracAtlas, pkdarabi)
│   ├── stage1/                  # Data organized for Stage 1 (Normal vs Fracture)
│   └── stage2/                  # Data organized for Stage 2 (Fracture Types)
├── src/
│   ├── model_stage1.py          # Model architecture for Stage 1
│   ├── model_stage2.py          # Model architecture for Stage 2
│   ├── train_stage1.py          # Training script for Stage 1
│   ├── train_stage2.py          # Training script for Stage 2
│   ├── prepare_data.py          # Data preprocessing and splitting script
│   ├── dataset.py               # PyTorch Dataset class definition
│   ├── evaluate.py              # Model evaluation script
│   ├── inference.py             # Script for running predictions on new images
│   └── bone_fracture.ipynb      # Jupyter Notebook for experimentation
├── checkpoints/                 # Saved model weights
├── results/                     # Training logs and performance metrics
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🧠 Project Logic & Workflow

The project follows a two-stage classification logic:

### 1. Data Preparation (`prepare_data.py`)
- Consolidates images from various sources (`Mendeley`, `FracAtlas`, `pkdarabi`).
- Preprocesses images (resizing, normalization).
- Splits the data into training, validation, and test sets.
- Organizes data into `stage1` (binary: Normal/Fracture) and `stage2` (multi-class: Fracture types).

### 2. Stage 1: Fracture Detection (`model_stage1.py`, `train_stage1.py`)
- **Objective**: Determine if an image contains a fracture or is normal.
- **Model**: Binary classifier (e.g., ResNet or EfficientNet adapted for two classes).
- **Output**: `Normal` or `Fracture`.

### 3. Stage 2: Fracture Classification (`model_stage2.py`, `train_stage2.py`)
- **Objective**: If a fracture is detected in Stage 1, classify its type.
- **Model**: Multi-class classifier trained only on fracture images.
- **Output**: Specific fracture type (e.g., transverse, oblique, comminuted, etc.).

### 4. Evaluation & Inference (`evaluate.py`, `inference.py`)
- **Evaluation**: Calculates metrics like accuracy, precision, recall, and F1-score for both stages.
- **Inference**: A unified script that takes an image as input, runs Stage 1 to detect a fracture, and if present, runs Stage 2 to classify the type.

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   ```bash
   python src/prepare_data.py
   ```

3. **Train Stage 1**:
   ```bash
   python src/train_stage1.py
   ```

4. **Train Stage 2**:
   ```bash
   python src/train_stage2.py
   ```

5. **Run Inference**:
   ```bash
   python src/inference.py --image_path path/to/your/image.jpg
   ```
