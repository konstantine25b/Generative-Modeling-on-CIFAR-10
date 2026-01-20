import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

def get_cifar10_loaders(data_dir='./data', batch_size=64, num_workers=2, val_split=0.1):
    """
    Download and prepare CIFAR-10 data loaders.
    
    Args:
        data_dir: Directory to store the dataset
        batch_size: Batch size for training and validation
        num_workers: Number of subprocesses for data loading
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # NVAE typically works with mild augmentation or just standard normalization
    # We'll use standard normalization for CIFAR-10
    # Mean and Std for CIFAR-10
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # NVAE often uses raw [0,1] or [-1,1]. We'll stick to [0,1] for BCE loss stability
        # If using standard Normalization, output would be roughly [-2, 2], requiring MSE and different activation
        # For this implementation (BCE loss), we keep it [0, 1]
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download datasets
    # This will automatically download if not present
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Split training into train/val
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    
    train_dataset, val_dataset = random_split(
        train_set, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Data Loaders ready: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_set)}")
    
    return train_loader, val_loader, test_loader
