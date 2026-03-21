import torch
import torch.nn as nn
from torchvision import models

def build_stage1_model(pretrained=True, freeze_backbone=True):
    """
    Stage 1: Binary classification (Normal vs Fracture).
    Uses ResNet50 as the backbone.
    """
    # Load pretrained ResNet50
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Freeze backbone parameters if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Modify the final fully connected layer for binary classification
    # ResNet50 fc layer input features is 2048
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)  # 2 classes: Normal, Fracture
    )
    
    return model

def unfreeze_all(model):
    """Unfreeze all parameters in the model for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
