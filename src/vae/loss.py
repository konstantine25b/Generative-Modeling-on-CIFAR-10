import torch
import torch.nn.functional as F
import numpy as np

def vae_loss(recon_x, x, kl_losses, beta=1.0):
    """
    Calculates the VAE Loss (ELBO).
    
    Args:
        recon_x: Reconstructed images (logits) from the decoder
        x: Original input images [0, 1]
        kl_losses: List of KL divergences from each scale
        beta: Weight for KL term (used for annealing)
        
    Returns:
        total_loss: Scalar
        recon_loss: Scalar
        kl_loss: Scalar
        bpd: Bits per dimension scalar
    """
    # 1. Reconstruction Loss
    # We use MSE for simplicity (Gaussian likelihood with fixed sigma)
    # Alternatively, BCE could be used if data is strictly [0,1]
    # For logits input, we use BCEWithLogits or convert to probabilities first
    
    # Using BCE with logits (standard for [0,1] image data in VAEs)
    # This corresponds to a Bernoulli likelihood
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum') / x.size(0)
    
    # 2. KL Divergence
    # Sum over all scales (kl_losses is a list of tensors [batch_size])
    total_kl = 0
    for kl in kl_losses:
        total_kl += torch.mean(kl)
        
    # 3. Total Loss (ELBO)
    loss = recon_loss + beta * total_kl
    
    # 4. Bits Per Dimension (BPD)
    # BPD = Loss / (ln(2) * T) where T is total dimensions (32*32*3)
    # We add a constant for the discretization of pixel values if using continuous density,
    # but for BCE/MSE based "reconstruction error", this is an approximation.
    dims = 32 * 32 * 3
    bpd = loss / (np.log(2) * dims)
    
    return loss, recon_loss, total_kl, bpd
