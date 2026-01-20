import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from .model import NVAE
from .loss import vae_loss

def train_vae(config, train_loader, test_loader, device):
    """
    Main training loop for NVAE.
    """
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
                'loss': loss.item(), 
                'bpd': bpd.item(),
                'beta': beta
            })
            
            # WandB logging (step-wise)
            if batch_idx % 100 == 0 and config.get('use_wandb', False):
                wandb.log({
                    "train_step_loss": loss.item(),
                    "train_step_recon": recon_loss.item(),
                    "train_step_kl": kl_loss.item(),
                    "train_step_bpd": bpd.item(),
                    "beta": beta
                })

        # Epoch averages
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        avg_bpd = epoch_bpd / len(train_loader)
        
        print(f"Epoch {epoch+1} Average: Loss={avg_loss:.4f}, BPD={avg_bpd:.4f}, KL={avg_kl:.4f}")
        
        # Validation
        val_loss, val_bpd = evaluate(model, test_loader, device)
        
        if config.get('use_wandb', False):
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_bpd": avg_bpd,
                "val_loss": val_loss,
                "val_bpd": val_bpd,
                "lr": optimizer.param_groups[0]['lr']
            })
            
        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(config['model_save_dir'], 'nvae_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            
        scheduler.step()

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_bpd = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, kl_losses = model(data)
            loss, _, _, bpd = vae_loss(recon_batch, data, kl_losses, beta=1.0)
            total_loss += loss.item()
            total_bpd += bpd.item()
            
    avg_loss = total_loss / len(test_loader)
    avg_bpd = total_bpd / len(test_loader)
    
    return avg_loss, avg_bpd
