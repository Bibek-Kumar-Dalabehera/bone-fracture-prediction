import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def build_stage2_model(num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Stage 2: Multi-class classification (Hairline vs Major Crack).
    Uses ResNet50 as the backbone.
    """
    # Load pretrained ResNet50
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Freeze backbone parameters if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),  # higher dropout for smaller dataset
        nn.Linear(512, num_classes)
    )
    
    return model

def get_focal_loss(class_counts, gamma=2.0):
    """
    Compute Focal Loss weights based on class inverse frequencies.
    class_counts: dict {class_name: count}
    """
    # Sort by class name to match ImageFolder order
    counts = [class_counts[k] for k in sorted(class_counts.keys())]
    total  = sum(counts)
    
    # Inverse frequency weights
    # alpha = [total / count for count in counts]
    # More stable weights:
    alpha = [1.0 - (count / total) for count in counts]
    alpha = torch.tensor(alpha, dtype=torch.float32)
    
    return FocalLoss(alpha=alpha, gamma=gamma)

def unfreeze_all(model):
    """Unfreeze all parameters in the model for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
