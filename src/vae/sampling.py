import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from .model import NVAE

def generate_samples(model, num_samples=64, temperature=1.0, device='cpu'):
    """
    Generate samples from the trained NVAE model.
    
    Args:
        model: Trained NVAE model
        num_samples: Number of images to generate
        temperature: Sampling temperature (lower = more stability, higher = diversity)
        device: 'cuda' or 'cpu'
        
    Returns:
        samples: Tensor of shape [num_samples, 3, 32, 32] in range [0, 1]
    """
    model.eval()
    samples = model.sample(num_samples, device=device, temp=temperature)
    return samples

def save_sample_grid(samples, save_path, nrow=8):
    """
    Save a grid of generated samples to a file.
    """
    # Denormalize if necessary (if samples are not [0,1])
    # Assuming samples are already [0,1] from sigmoid output
    
    vutils.save_image(samples, save_path, nrow=nrow, normalize=True)
    print(f"Samples saved to {save_path}")

def load_model_for_sampling(checkpoint_path, config, device):
    """
    Load a trained model for sampling.
    """
    model = NVAE(
        hidden_dim=config.get('hidden_dim', 64),
        latent_dim=config.get('latent_dim', 20),
        num_scales=config.get('num_scales', 3)
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {checkpoint_path}")
    
    return model
