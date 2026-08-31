import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

def get_transforms(split="train"):
    """Return PyTorch data augmentation and preprocessing transforms."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])


class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images organized by split and class directories."""
    
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform if transform is not None else get_transforms(split)
        
        self.samples = []
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.dcm"):
                for filepath in class_dir.glob(ext):
                    self.samples.append((filepath, self.class_to_idx[class_name]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, str(filepath)


def get_dataloaders(task="chest_xray", batch_size=config.BATCH_SIZE, num_workers=0):
    """Factory function returning train, val, test dataloaders for the given task."""
    base_dir = config.CHEST_XRAY_DIR if task == "chest_xray" else config.SKIN_LESION_DIR
    
    train_dataset = MedicalImageDataset(base_dir, split="train", transform=get_transforms("train"))
    val_dataset = MedicalImageDataset(base_dir, split="val", transform=get_transforms("val"))
    test_dataset = MedicalImageDataset(base_dir, split="test", transform=get_transforms("test"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset.class_names
