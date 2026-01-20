import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 1. Basic Layers & Activations
# -----------------------------------------------------------------------------

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Used extensively in NVAE instead of ReLU.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Reweights channels based on global context.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            Swish(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# -----------------------------------------------------------------------------
# 2. Advanced Convolutional Blocks (NVAE Components)
# -----------------------------------------------------------------------------

class ResidualCell(nn.Module):
    """
    The core residual cell of NVAE.
    Uses Depthwise Separable Convolutions to increase receptive field efficiently.
    BN -> Swish -> Conv -> BN -> Swish -> Conv ...
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=2, spectral_norm=True):
        super().__init__()
        
        hidden_dim = in_channels * expansion
        self.use_res_connect = stride == 1 and in_channels == out_channels

        # Helper for spectral norm
        def wrap_conv(module):
            return nn.utils.spectral_norm(module) if spectral_norm else module

        # Depthwise Separable Convolution sequence
        self.conv = nn.Sequential(
            # 1. 1x1 Conv (Expansion)
            wrap_conv(nn.Conv2d(in_channels, hidden_dim, 1, bias=False)),
            nn.BatchNorm2d(hidden_dim, momentum=0.05),
            Swish(),
            
            # 2. 5x5 Depthwise Conv (Spatial mixing)
            # Note: Spectral norm is usually applied to standard convs. 
            # For depthwise, it can be applied but is less common/critical than 1x1.
            # We apply it here for consistency with "stabilizing all layers".
            wrap_conv(nn.Conv2d(hidden_dim, hidden_dim, 5, stride=stride, padding=2, groups=hidden_dim, bias=False)),
            nn.BatchNorm2d(hidden_dim, momentum=0.05),
            Swish(),
            
            # 3. 1x1 Conv (Projection)
            wrap_conv(nn.Conv2d(hidden_dim, out_channels, 1, bias=False)),
            nn.BatchNorm2d(out_channels, momentum=0.05),
            
            # 4. Squeeze-and-Excitation
            SEBlock(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EncoderBlock(nn.Module):
    """
    Bottom-up Encoder Block.
    Downsamples features and prepares them for the latent hierarchy.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cell = ResidualCell(in_channels, out_channels, stride=2)
    
    def forward(self, x):
        return self.cell(x)

class DecoderBlock(nn.Module):
    """
    Top-down Decoder Block.
    Upsamples features and merges with latent samples.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsampling via interpolation + 1x1 conv is often cleaner than TransposedConv for VAEs
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.cell = ResidualCell(out_channels, out_channels, stride=1)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.cell(x)
        return x

def sample_from_discretized_mix_logistic(l, nr_mix):
    """
    Sample from the discretized mixture of logistics distribution.
    Input:
        l: Output from the network [Batch, num_mix * 10, H, W]
        nr_mix: Number of mixtures (10)
    Output:
        x: Sampled image [Batch, 3, H, W] in [0, 1] range
    """
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1) # [B, H, W, C]
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3] # [B, H, W, 3]
    
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].view(ls[0], ls[1], ls[2], 3, nr_mix * 3)
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2*nr_mix], -7.) # min_log_scale = -7
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2*nr_mix], min=-7.)
    coeffs = torch.tanh(l[:, :, :, :, 2*nr_mix:3*nr_mix])
    
    # sample mixture indicator from softmax
    logit_probs = F.softmax(logit_probs, dim=3)
    
    # Needs to be done on CPU or efficiently?
    # We can use Gumbel-Max or Categorical
    # Flatten: [B*H*W, nr_mix]
    flat_probs = logit_probs.view(-1, nr_mix)
    indices = torch.multinomial(flat_probs, 1).view(ls[0], ls[1], ls[2])
    
    # One-hot encoding of indices
    one_hot = F.one_hot(indices, num_classes=nr_mix).float().to(l.device) # [B, H, W, nr_mix]
    
    # Select parameters for the chosen mixture
    # means: [B, H, W, 3, nr_mix]
    # one_hot: [B, H, W, nr_mix] -> unsqueeze to [B, H, W, 1, nr_mix]
    one_hot_exp = one_hot.unsqueeze(3)
    
    # Reduce over mixture dim
    means = torch.sum(means * one_hot_exp, dim=4) # [B, H, W, 3]
    log_scales = torch.sum(log_scales * one_hot_exp, dim=4)
    coeffs = torch.sum(coeffs * one_hot_exp, dim=4)
    
    # Sample from logistic
    u = torch.rand(xs, device=l.device)
    x = torch.log(u) - torch.log(1. - u) + means
    x = x + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u)) # Wait, standard logistic sample is mu + s * log(u/(1-u))
    
    # Actually the paper uses: x = mu + s * logit(u)
    # Correct formula: sample = mean + scale * (log(u) - log(1-u))
    x = means + torch.exp(log_scales) * (torch.log(u + 1e-5) - torch.log(1. - u + 1e-5))
    
    # Autoregressive coupling for RGB
    # x0 = m0
    # x1 = m1 + c0 * x0
    # x2 = m2 + c1 * x0 + c2 * x1
    x0 = x[:, :, :, 0]
    x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0
    x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1
    
    x_out = torch.stack([x0, x1, x2], dim=3)
    
    # Rescale to [0, 1]
    # In loss we assumed x in [-1, 1]
    # So x_out is in [-1, 1] approx
    x_out = torch.clamp(x_out, -1., 1.)
    x_out = (x_out + 1.) / 2.
    
    return x_out.permute(0, 3, 1, 2) # [B, 3, H, W]

# -----------------------------------------------------------------------------
# 3. Main NVAE Model
# -----------------------------------------------------------------------------

class NVAE(nn.Module):
    """
    Hierarchical NVAE Model for CIFAR-10 (32x32).
    
    Architecture:
    - Pre-process: Conv2d mapping input to hidden dim
    - Encoder: Bottom-up extraction of feature maps at different scales
    - Latent Space: Hierarchical latents (z1, z2, z3...)
    - Decoder: Top-down generation, combining priors and posteriors
    """
    def __init__(self, hidden_dim=64, latent_dim=20, num_scales=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_scales = num_scales
        
        # -----------------------
        # Bottom-up Encoder
        # -----------------------
        # Input: 3x32x32
        self.stem = nn.Conv2d(3, hidden_dim, 3, padding=1)
        
        # We will extract features at 16x16, 8x8, 4x4
        self.enc_tower = nn.ModuleList([
            ResidualCell(hidden_dim, hidden_dim, stride=1),  # 32x32
            EncoderBlock(hidden_dim, hidden_dim*2),          # -> 16x16
            EncoderBlock(hidden_dim*2, hidden_dim*4),        # -> 8x8
            EncoderBlock(hidden_dim*4, hidden_dim*8),        # -> 4x4
        ])
        
        # -----------------------
        # Top-down Decoder / Generative Flow
        # -----------------------
        # We process from smallest scale (4x4) to largest (32x32)
        
        # Parameters for the smallest latent variable (z_bottom)
        self.decoder_input = nn.Parameter(torch.randn(1, hidden_dim*8, 4, 4))
        
        # Blocks for processing deterministic paths
        self.dec_tower_4x4 = ResidualCell(hidden_dim*8, hidden_dim*8)
        self.dec_upsample_8x8 = DecoderBlock(hidden_dim*8, hidden_dim*4)
        self.dec_upsample_16x16 = DecoderBlock(hidden_dim*4, hidden_dim*2)
        self.dec_upsample_32x32 = DecoderBlock(hidden_dim*2, hidden_dim)
        
        # Final output projection
        # Discretized Mixture of Logistics requires 10 output parameters per mixture:
        # 1 weight, 3 means, 3 scales, 3 coefficients
        # For 10 mixtures: 10 * 10 = 100 channels
        self.num_mix = 10
        out_channels = self.num_mix * 10
        
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1) # Output for DMOL
        )
        
        # -----------------------
        # Latent Parameters (q(z|x) and p(z))
        # -----------------------
        # Projections to get mean/log_var from features for each scale
        # 4x4 scale
        self.z4_proj = nn.Conv2d(hidden_dim*8, 2 * latent_dim, 1) # Inference q(z|x)
        self.z4_prior = nn.Conv2d(hidden_dim*8, 2 * latent_dim, 1) # Generative p(z)
        self.z4_to_feat = nn.Conv2d(latent_dim, hidden_dim*8, 1)   # Merge z back to features
        
        # 8x8 scale
        self.z8_proj = nn.Conv2d(hidden_dim*4, 2 * latent_dim, 1)
        self.z8_prior = nn.Conv2d(hidden_dim*4, 2 * latent_dim, 1)
        self.z8_to_feat = nn.Conv2d(latent_dim, hidden_dim*4, 1)
        
        # 16x16 scale
        self.z16_proj = nn.Conv2d(hidden_dim*2, 2 * latent_dim, 1)
        self.z16_prior = nn.Conv2d(hidden_dim*2, 2 * latent_dim, 1)
        self.z16_to_feat = nn.Conv2d(latent_dim, hidden_dim*2, 1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass (Training):
        1. Bottom-up encoder pass to get skip connections
        2. Top-down decoder pass to sample z's and reconstruct
        """
        
        # --- 1. Encoder Pass ---
        # Store features for skip connections (like U-Net / Ladder VAE)
        x_stem = self.stem(x)
        
        feat_32 = self.enc_tower[0](x_stem)
        feat_16 = self.enc_tower[1](feat_32)
        feat_8  = self.enc_tower[2](feat_16)
        feat_4  = self.enc_tower[3](feat_8)
        
        # --- 2. Decoder / Generative Pass ---
        # We start from a learned constant parameter
        batch_size = x.size(0)
        h = self.decoder_input.expand(batch_size, -1, -1, -1)
        h = self.dec_tower_4x4(h)
        
        kl_losses = []
        
        # --- Scale 1: 4x4 ---
        # Prior p(z) comes from top-down h
        params_prior = self.z4_prior(h)
        mu_prior, log_var_prior = torch.chunk(params_prior, 2, dim=1)
        
        # Posterior q(z|x) combines top-down h AND bottom-up feat_4
        # NVAE combines them usually by residual or concatenation. We'll use residual.
        params_post = self.z4_proj(feat_4) + params_prior # Residual parameterization
        mu_post, log_var_post = torch.chunk(params_post, 2, dim=1)
        
        # Sample z
        z4 = self.reparameterize(mu_post, log_var_post)
        
        # Calculate KL(q||p) for this scale
        kl_4 = self.calculate_kl(mu_post, log_var_post, mu_prior, log_var_prior)
        kl_losses.append(kl_4)
        
        # Add z back to top-down stream
        h = h + self.z4_to_feat(z4)
        
        # --- Scale 2: 8x8 ---
        # h = self.dec_upsample_8x8(h) # Upsample to 8x8
        # Using separate block to prevent variable overwriting issues if any
        h_8 = self.dec_upsample_8x8(h)
        
        params_prior = self.z8_prior(h_8)
        mu_prior, log_var_prior = torch.chunk(params_prior, 2, dim=1)
        
        # Combine with skip connection from encoder
        params_post = self.z8_proj(feat_8) + params_prior
        mu_post, log_var_post = torch.chunk(params_post, 2, dim=1)
        
        z8 = self.reparameterize(mu_post, log_var_post)
        kl_losses.append(self.calculate_kl(mu_post, log_var_post, mu_prior, log_var_prior))
        
        h = h_8 + self.z8_to_feat(z8)
        
        # --- Scale 3: 16x16 ---
        h_16 = self.dec_upsample_16x16(h) # Upsample to 16x16
        
        params_prior = self.z16_prior(h_16)
        mu_prior, log_var_prior = torch.chunk(params_prior, 2, dim=1)
        
        params_post = self.z16_proj(feat_16) + params_prior
        mu_post, log_var_post = torch.chunk(params_post, 2, dim=1)
        
        z16 = self.reparameterize(mu_post, log_var_post)
        kl_losses.append(self.calculate_kl(mu_post, log_var_post, mu_prior, log_var_prior))
        
        h = h_16 + self.z16_to_feat(z16)
        
        # --- Final Reconstruction 32x32 ---
        h_32 = self.dec_upsample_32x32(h) # Upsample to 32x32
        out = self.final_conv(h_32)
        
        return out, kl_losses

    def sample(self, num_samples, device='cpu', temp=1.0):
        """
        Ancestral Sampling from p(z)
        """
        with torch.no_grad():
            # Start from constant
            h = self.decoder_input.expand(num_samples, -1, -1, -1).to(device)
            h = self.dec_tower_4x4(h)
            
            # --- 4x4 ---
            params_prior = self.z4_prior(h)
            mu, log_var = torch.chunk(params_prior, 2, dim=1)
            z4 = self.reparameterize(mu, log_var) * temp # Temperature scaling
            h = h + self.z4_to_feat(z4)
            
            # --- 8x8 ---
            h = self.dec_upsample_8x8(h)
            params_prior = self.z8_prior(h)
            mu, log_var = torch.chunk(params_prior, 2, dim=1)
            z8 = self.reparameterize(mu, log_var) * temp
            h = h + self.z8_to_feat(z8)
            
            # --- 16x16 ---
            h = self.dec_upsample_16x16(h)
            params_prior = self.z16_prior(h)
            mu, log_var = torch.chunk(params_prior, 2, dim=1)
            z16 = self.reparameterize(mu, log_var) * temp
            h = h + self.z16_to_feat(z16)
            
            # --- Output ---
            h = self.dec_upsample_32x32(h)
            out = self.final_conv(h)
            
            # Use DMOL Sampling
            return sample_from_discretized_mix_logistic(out, self.num_mix)

    def calculate_kl(self, mu_q, log_var_q, mu_p, log_var_p):
        """
        KL divergence between two Gaussians: q(z|x) and p(z)
        """
        # kl = 0.5 * (var_q / var_p + (mu_p - mu_q)^2 / var_p - 1 + log_var_p - log_var_q)
        # simplified term-by-term
        kl = 0.5 * (
            torch.exp(log_var_q - log_var_p) +
            (mu_p - mu_q)**2 / torch.exp(log_var_p) -
            1 +
            (log_var_p - log_var_q)
        )
        return torch.sum(kl, dim=[1, 2, 3])
