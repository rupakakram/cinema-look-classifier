import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_loaders(data_dir="data", batch_size=32, val_split=0.2, seed=42):

    # Training data transformations
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation data transformations
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes # ['cinematic', 'normal']

    # Split into training and validation sets
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_ds.dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds,  batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Classes : {class_names}")
    print(f"Total : {total} images")
    print(f"Train : {train_size} images")
    print(f"Validation : {val_size} images")
    return train_loader, val_loader, class_names

if __name__ == "__main__":
    train_loader, val_loader, class_names = get_loaders()
    image, labels = next(iter(train_loader))
    print(f"Batch shape: {image.shape}")
    print(f"Label shape: {labels.shape}")
