import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import numpy as np
from .model import NVAE, sample_from_discretized_mix_logistic
from .loss import vae_loss
import torchvision.utils as vutils

def train_vae(config, train_loader, test_loader, device):
    """
    Main training loop for NVAE.
    """
    # 0. Initialize WandB if configured
    if config.get('use_wandb', False):
        wandb.init(project="cifar10-nvae", config=config, name=config.get('run_name', 'nvae_experiment'))
        # Log model gradients and topology
        # wandb.watch(model, log='all') 

    # 1. Initialize Model
    model = NVAE(
        hidden_dim=config.get('hidden_dim', 64),
        latent_dim=config.get('latent_dim', 20),
        num_scales=config.get('num_scales', 3)
    ).to(device)
    
    print(f"Model initialized on {device}")
    
    # 2. Optimizer & Scheduler
    # NVAE paper uses AdamAX or Adam with specific parameters. We'll use AdamW.
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 3e-4))
    
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-5)
    
    # 3. Training Loop
    best_loss = float('inf')
    start_epoch = 0
    
    # Resume logic
    model_save_dir = config.get('model_save_dir', './checkpoints')
    if os.path.exists(model_save_dir):
        # Find all epoch checkpoints
        checkpoints = [f for f in os.listdir(model_save_dir) if f.startswith('nvae_epoch_') and f.endswith('.pth')]
        if checkpoints:
            # Extract epoch numbers
            epochs = []
            for cp in checkpoints:
                try:
                    # format: nvae_epoch_{epoch}.pth
                    ep = int(cp.replace('nvae_epoch_', '').replace('.pth', ''))
                    epochs.append(ep)
                except ValueError:
                    continue
            
            if epochs:
                last_epoch = max(epochs)
                resume_path = os.path.join(model_save_dir, f'nvae_epoch_{last_epoch}.pth')
                print(f"Resuming from checkpoint: {resume_path}")
                
                checkpoint = torch.load(resume_path, map_location=device)
                
                # Check if it's a full checkpoint or just state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] # The saved epoch is the one that finished
                    if 'best_loss' in checkpoint:
                        best_loss = checkpoint['best_loss']
                else:
                    # Old format: just model state_dict
                    model.load_state_dict(checkpoint)
                    start_epoch = last_epoch
                    
                print(f"Resumed training from epoch {start_epoch}")
    
    # KL Annealing parameters
    warmup_epochs = config.get('warmup_epochs', 5)
    
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        epoch_bpd = 0
        
        # Calculate beta for KL annealing
        if epoch < warmup_epochs:
            beta = (epoch + 1) / warmup_epochs
        else:
            beta = 1.0
            
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, kl_losses = model(data)
            
            # Loss calculation
            loss, recon_loss, kl_loss, bpd = vae_loss(recon_batch, data, kl_losses, beta=beta)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (important for deep VAEs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=200)
            
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_bpd += bpd.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.2f}", 
                'bpd': f"{bpd.item():.2f}",
                'beta': f"{beta:.2f}"
            })
            
            # WandB logging (step-wise)
            if batch_idx % 100 == 0 and config.get('use_wandb', False):
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/step_recon": recon_loss.item(),
                    "train/step_kl": kl_loss.item(),
                    "train/step_bpd": bpd.item(),
                    "train/beta": beta,
                    "train/lr": optimizer.param_groups[0]['lr']
                })

        # Epoch averages
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        avg_bpd = epoch_bpd / len(train_loader)
        
        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"Train Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | BPD: {avg_bpd:.4f}")
        
        # Validation
        val_loss, val_bpd, val_recon_img, val_orig_img = evaluate(model, test_loader, device, return_images=True)
        print(f"Val Loss:   {val_loss:.4f} | Val BPD: {val_bpd:.4f}")
        
        # Generate samples for visualization
        model.eval()
        with torch.no_grad():
            gen_samples = model.sample(num_samples=16, device=device, temp=0.8)
        
        if config.get('use_wandb', False):
            # Create image grids for WandB
            # make_grid returns (C, H, W) which wandb can handle, or we can permute to (H, W, C)
            # Normalization ensures values are in valid range for visualization
            orig_grid = vutils.make_grid(val_orig_img, nrow=4, normalize=True)
            recon_grid = vutils.make_grid(val_recon_img, nrow=4, normalize=True)
            gen_grid = vutils.make_grid(gen_samples, nrow=4, normalize=True)

            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/bpd": avg_bpd,
                "train/kl": avg_kl,
                "train/recon": avg_recon,
                "val/loss": val_loss,
                "val/bpd": val_bpd,
                # Log Images
                "images/original": [wandb.Image(orig_grid, caption="Original")],
                "images/reconstructed": [wandb.Image(recon_grid, caption="Reconstructed")],
                "images/generated": [wandb.Image(gen_grid, caption="Generated (T=0.8)")]
            })
            
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(config['model_save_dir'], 'nvae_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path} (Val Loss: {val_loss:.4f})")
            
        # Save epoch checkpoint
        save_path_epoch = os.path.join(config['model_save_dir'], f'nvae_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'config': config
        }, save_path_epoch)
        print(f"💾 Saved epoch checkpoint to {save_path_epoch}")
            
        scheduler.step()

def evaluate(model, test_loader, device, return_images=False):
    model.eval()
    total_loss = 0
    total_bpd = 0
    
    recon_img = None
    orig_img = None
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, kl_losses = model(data)
            loss, _, _, bpd = vae_loss(recon_batch, data, kl_losses, beta=1.0)
            total_loss += loss.item()
            total_bpd += bpd.item()
            
            # Capture first batch for visualization
            if i == 0 and return_images:
                # Take first 16 images
                orig_img = data[:16]
                
                # Check output channels for DMOL vs BCE
                if recon_batch.size(1) > 3:
                    # DMOL: Sample from mixture
                    # recon_batch is [B, 100, H, W]
                    recon_img = sample_from_discretized_mix_logistic(recon_batch[:16], nr_mix=10)
                else:
                    # BCE: Sigmoid
                    recon_img = torch.sigmoid(recon_batch[:16]) # Apply sigmoid to logits
            
    avg_loss = total_loss / len(test_loader)
    avg_bpd = total_bpd / len(test_loader)
    
    if return_images:
        return avg_loss, avg_bpd, recon_img, orig_img
    return avg_loss, avg_bpd

def evaluate_with_importance_sampling(model, test_loader, device, k=1000):
    """
    Evaluates the model using Importance Weighted Sampling (IWELBO).
    This gives a tighter bound on the log-likelihood than the standard ELBO.
    
    Args:
        model: Trained NVAE model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        k: Number of importance samples (NVAE paper uses k=1000)
        
    Returns:
        avg_loss: Negative IWELBO (NLL)
        avg_bpd: Bits per dimension based on IWELBO
    """
    model.eval()
    total_loss = 0
    total_bpd = 0
    total_samples = 0
    
    print(f"Starting Importance Weighted Evaluation (k={k})...")
    
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader, desc="IWELBO Eval")):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Repeat input k times: [N, C, H, W] -> [N*k, C, H, W]
            data_expanded = data.repeat_interleave(k, dim=0)
            
            # Forward pass with expanded batch
            recon_batch, kl_losses = model(data_expanded)
            
            # Calculate UNREDUCED loss (element-wise)
            # We get loss per sample: [N*k]
            # Note: vae_loss returns (total_loss, recon, kl, bpd)
            # We need 'total_loss' which is -ELBO
            loss_unreduced, _, _, _ = vae_loss(recon_batch, data_expanded, kl_losses, beta=1.0, reduction='none')
            
            # Reshape to [N, k]
            loss_unreduced = loss_unreduced.view(batch_size, k)
            
            # IWELBO calculation:
            # log p(x) approx log (1/k sum exp(ELBO_i))
            # ELBO_i = -loss_unreduced_i
            # log p(x) = logsumexp(-loss_unreduced) - log(k)
            # We want to return Negative Log Likelihood (NLL) = -log p(x)
            
            # nll = - (logsumexp(-loss) - log(k))
            # nll = -logsumexp(-loss) + log(k)
            
            iw_elbo = -loss_unreduced # Convert loss to ELBO
            log_likelihood = torch.logsumexp(iw_elbo, dim=1) - np.log(k)
            nll = -log_likelihood # Back to loss (NLL)
            
            # Sum up NLL for the batch
            total_loss += torch.sum(nll).item()
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples
    
    # Calculate BPD
    dims = 32 * 32 * 3
    avg_bpd = avg_loss / (np.log(2) * dims)
    
    return avg_loss, avg_bpd
