"""
Training utilities for IsingTransformer.
"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_ising_transformer(
    model,
    loader,
    num_steps: int = 10_000,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = 'mps',
    plot: bool = True
):
    """
    Train IsingTransformer with learning rate scheduling and progress tracking.
    
    Args:
        model: IsingTransformer instance
        loader: DataLoader for training data
        num_steps: Total training steps
        lr: Initial learning rate
        weight_decay: Weight decay for AdamW
        device: Device to train on
        plot: Whether to plot learning curve
    
    Returns:
        dict with training history
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_steps
    )
    
    model.train()
    
    history = {
        'step': [],
        'loss': [],
        'lr': [],
    }
    
    step = 0
    pbar = tqdm(total=num_steps, desc='Training')
    
    while step < num_steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Record
            history['step'].append(step)
            history['loss'].append(loss.item())
            history['lr'].append(scheduler.get_last_lr()[0])
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
            step += 1
            if step >= num_steps:
                break
    
    pbar.close()
    
    print(f"\n✓ Training finished!")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Min loss: {min(history['loss']):.4f} at step {history['step'][history['loss'].index(min(history['loss']))]}")
    
    if plot:
        plot_training_curve(history)
    
    return history


def plot_training_curve(history):
    """Plot training loss and learning rate."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Loss curve
    ax1.plot(history['step'], history['loss'], alpha=0.6, lw=1, label='Training Loss')
    
    # Smooth loss (moving average)
    window = 50
    if len(history['loss']) > window:
        import numpy as np
        smooth_loss = np.convolve(
            history['loss'], 
            np.ones(window)/window, 
            mode='valid'
        )
        ax1.plot(
            history['step'][window-1:], 
            smooth_loss, 
            color='red', 
            lw=2, 
            label=f'Smoothed (window={window})'
        )
    
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_title('IsingGPT Training Curve', fontsize=15, pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # Learning rate
    ax2.plot(history['step'], history['lr'], color='green', lw=2)
    ax2.set_xlabel('Training Step', fontsize=13)
    ax2.set_ylabel('Learning Rate', fontsize=13)
    ax2.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

