import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from .model import NVAE
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
        torch.save(model.state_dict(), save_path_epoch)
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
                recon_img = torch.sigmoid(recon_batch[:16]) # Apply sigmoid to logits
            
    avg_loss = total_loss / len(test_loader)
    avg_bpd = total_bpd / len(test_loader)
    
    if return_images:
        return avg_loss, avg_bpd, recon_img, orig_img
    return avg_loss, avg_bpd
