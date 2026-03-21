import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(stage: int, split: str) -> transforms.Compose:
    """
    stage=1 → moderate augmentation (large balanced dataset)
    stage=2 → aggressive augmentation (tiny imbalanced dataset)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        if stage == 1:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:  # stage 2 — aggressive aug for tiny dataset
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
    else:  # val / test — no augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─── Weighted Sampler (fixes class imbalance) ─────────────────────────────────

def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Gives each sample a weight inversely proportional to its class frequency.
    Rare classes (hairline) get sampled more often → balanced batches.
    """
    class_counts = Counter(dataset.targets)
    total = sum(class_counts.values())
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: str,
    stage: int,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Returns dict with keys: 'train', 'val', 'test'
    Each value is a DataLoader.

    Args:
        data_root : path to bone_fracture_project/data/
        stage     : 1 or 2
        batch_size: images per batch
        num_workers: parallel data loading workers
    """
    stage_dir = os.path.join(data_root, f"stage{stage}")
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(stage_dir, split)
        tfm = get_transforms(stage, split)
        ds  = datasets.ImageFolder(root=split_dir, transform=tfm)

        if split == "train":
            sampler = make_weighted_sampler(ds)
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,          # WeightedSampler replaces shuffle=True
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        print(f"  Stage {stage} | {split:5s} | {len(ds):6,} images | classes: {ds.class_to_idx}")

    return loaders
