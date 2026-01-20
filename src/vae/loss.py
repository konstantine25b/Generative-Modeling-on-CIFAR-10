import torch
import torch.nn.functional as F
import numpy as np

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def discretized_mix_logistic_loss(l, x):
    """
    Log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    Input:
        l: Output from the network [Batch, num_mix * 10, H, W]
        x: Target image [Batch, 3, H, W] in [-1, 1]
    """
    # N: Batch size, C: Channels (3), H, W
    # nr_mix: Number of mixtures
    
    # Unpack output
    # NVAE output shape: [B, 3*10 + 10, H, W] -> roughly 10 groups of params
    # Actually PixelCNN++ uses: 10 * (1 pi + 3 means + 3 scales + 3 coeffs) = 100 channels?
    # Standard implementation expects l to be [B, H, W, num_mix * 10]
    # So we might need to permute.
    
    # Assume l comes in as [B, M, H, W] where M = num_mix * 10
    batch_size, num_channels, H, W = l.shape
    nr_mix = num_channels // 10
    
    # Permute to [B, H, W, M] for easier slicing
    l = l.permute(0, 2, 3, 1)
    # x needs to be [B, H, W, 3]
    x = x.permute(0, 2, 3, 1)
    
    # Unpack parameters
    logit_probs = l[:, :, :, :nr_mix]                    # [B, H, W, nr_mix]
    l = l[:, :, :, nr_mix:].view(batch_size, H, W, 3, nr_mix * 3) # Remaining params
    means = l[:, :, :, :, :nr_mix]                       # [B, H, W, 3, nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2*nr_mix], min=-7.) # [B, H, W, 3, nr_mix]
    coeffs = torch.tanh(l[:, :, :, :, 2*nr_mix:3*nr_mix]) # [B, H, W, 3, nr_mix]
    
    # Adjustment for x (assumed [-1, 1])
    x = x.unsqueeze(-1) + torch.zeros_like(means) # Broadcast x to [B, H, W, 3, nr_mix]

    # Model means: 
    # m1 = mu1
    # m2 = mu2 + c1 * x1
    # m3 = mu3 + c2 * x1 + c3 * x2
    m1 = means[:, :, :, 0, :]
    m2 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    m3 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + \
                                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]

    means = torch.cat((m1.unsqueeze(3), m2.unsqueeze(3), m3.unsqueeze(3)), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    
    plus_in  = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    
    min_in   = inv_stdv * (centered_x - 1. / 255.)
    cdf_min  = torch.sigmoid(min_in)
    
    # Log probability for edge cases of 0 (x=-1) and 255 (x=1)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    
    cdf_delta = cdf_plus - cdf_min
    
    # Robust log(cdf_delta)
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    # if x < -0.999: log_cdf_plus
    # elif x > 0.999: log_one_minus_cdf_min
    # else: log(cdf_delta) -> approximate with log_pdf_mid - log(127.5) if delta is small?
    # Actually just use torch.where for numerical stability
    
    # Standard safe implementation:
    log_probs = torch.where(x < -0.999, log_cdf_plus, 
                            torch.where(x > 0.999, log_one_minus_cdf_min, 
                                        torch.where(cdf_delta > 1e-5, 
                                                    torch.log(torch.clamp(cdf_delta, min=1e-12)), 
                                                    log_pdf_mid - np.log(127.5))))
    
    log_probs = torch.sum(log_probs, dim=3) + log_sum_exp(logit_probs)
    
    # Loss is negative log likelihood
    # Return shape [B, H, W] -> Sum over H, W
    return -torch.sum(log_probs, dim=[1, 2])


def vae_loss(recon_x, x, kl_losses, beta=1.0, reduction='mean'):
    """
    Calculates the VAE Loss (ELBO).
    """
    
    # 1. Reconstruction Loss
    # Check if using DMOL (output channels > 3)
    if recon_x.size(1) > 3:
        # DMOL expects x in [-1, 1]. Data loader provides [0, 1].
        x_scaled = 2. * x - 1.
        recon_loss_batch = discretized_mix_logistic_loss(recon_x, x_scaled)
        
        if reduction == 'none':
            recon_loss = recon_loss_batch
        else:
            recon_loss = torch.mean(recon_loss_batch) # Mean over batch for training
            
    else:
        # Fallback to BCE (Old method)
        if reduction == 'none':
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none')
            recon_loss = torch.sum(recon_loss, dim=[1, 2, 3])
        else:
            recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum') / x.size(0)
    
    # 2. KL Divergence
    total_kl = 0
    for kl in kl_losses:
        if reduction == 'none':
            total_kl += kl
        else:
            total_kl += torch.mean(kl)
        
    # 3. Total Loss
    loss = recon_loss + beta * total_kl
    
    # 4. BPD
    dims = 32 * 32 * 3
    bpd = loss / (np.log(2) * dims)
    
    return loss, recon_loss, total_kl, bpd
