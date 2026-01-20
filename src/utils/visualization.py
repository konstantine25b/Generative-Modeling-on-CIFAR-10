import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_training_curves(train_losses, val_losses, train_bpd, val_bpd, save_path=None):
    """
    Plots training and validation loss/BPD curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Curve
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (ELBO)')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # BPD Curve
    ax2.plot(epochs, train_bpd, label='Train BPD')
    ax2.plot(epochs, val_bpd, label='Val BPD')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Bits Per Dimension')
    ax2.set_title('Training vs Validation BPD')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
        
    plt.show()

def show_samples(samples, nrow=8, title="Generated Samples", save_path=None):
    """
    Displays a grid of samples.
    Args:
        samples: Tensor [B, 3, H, W] in [0, 1]
    """
    samples = samples.detach().cpu()
    
    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=nrow, padding=2)
    
    # Convert to numpy for matplotlib [C, H, W] -> [H, W, C]
    grid_np = grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Sample grid saved to {save_path}")
        
    plt.show()
