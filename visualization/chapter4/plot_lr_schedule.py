#!/usr/bin/env python3
"""
Figure 4.XX: Learning Rate Schedule Visualization

This script trains a chaotic speaker recognition model and plots the 
cosine annealing learning rate schedule over 100 epochs.

Usage:
    # Full training with real data
    python plot_lr_schedule.py --data_dir /path/to/librispeech --epochs 100 --output fig4_lr_schedule.png
    
    # Quick visualization without training (theoretical curve only)
    python plot_lr_schedule.py --theoretical_only --output fig4_lr_schedule.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
import json

# Setup imports
def setup_imports():
    current_file = Path(__file__).resolve()
    paths_to_try = [
        '/scratch/project_2003370/yueyao/Model',
        str(current_file.parent),
    ]
    for path in paths_to_try:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

setup_imports()

# Try to import project modules
try:
    from data.dataset_loader import create_speaker_dataloaders
    from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
    HAS_PROJECT = True
except ImportError:
    HAS_PROJECT = False
    print("Warning: Project modules not found, will use theoretical curve only")


def train_and_record_lr(data_dir, num_epochs=100, batch_size=32, 
                        lr_max=0.005, lr_min=1e-6, save_checkpoint=False):
    """
    Train the chaotic model and record learning rates at each epoch.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders with CORRECT parameter names
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_speaker_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.7,
        val_split=0.15,
        num_workers=0,
        max_length=3.0,
        sample_rate=16000
    )
    
    num_speakers = train_loader.dataset.num_speakers
    print(f"Number of speakers: {num_speakers}")
    print(f"Training batches: {len(train_loader)}")
    
    # Create model
    model_config = {
        'num_speakers': num_speakers,
        'chaotic_system': 'lorenz',
        'embedding_dim': 10,
        'speaker_embedding_dim': 128,
        'classifier_type': 'cosine',
        'temperature': 30.0,
        'margin': 0.35,
        'evolution_time': 0.5,
        'time_step': 0.01,
    }
    
    print("Creating model...")
    model = ChaoticSpeakerRecognitionNetwork(model_config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer with AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_max,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Create cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr_min
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    lr_history = []
    train_loss_history = []
    val_acc_history = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Initial LR: {lr_max}, Final LR: {lr_min}")
    print("="*60)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Training phase
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (audio, targets) in enumerate(train_loader):
            audio = audio.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(audio, targets)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = criterion(logits, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
        
        avg_train_loss = total_loss / total_samples
        train_loss_history.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, targets in val_loader:
                audio = audio.to(device)
                targets = targets.to(device)
                
                outputs = model(audio)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = correct / total
        val_acc_history.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"LR: {current_lr:.6f} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_checkpoint:
                checkpoint_path = f'best_model_epoch{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
    
    print("="*60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    
    return lr_history, train_loss_history, val_acc_history


def compute_theoretical_lr(num_epochs=100, lr_max=0.005, lr_min=1e-6):
    """Compute theoretical cosine annealing learning rate schedule."""
    epochs = np.arange(num_epochs)
    lrs = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epochs / num_epochs))
    return epochs, lrs


def plot_lr_schedule(lr_history=None, train_loss_history=None, val_acc_history=None,
                     num_epochs=100, lr_max=0.005, lr_min=1e-6,
                     output_path='fig4_lr_schedule.png', show_training_curves=True):
    """Plot learning rate schedule with optional training curves."""
    
    epochs_theo, lrs_theo = compute_theoretical_lr(num_epochs, lr_max, lr_min)
    
    if show_training_curves and train_loss_history is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes = [ax]
    
    # === Plot 1: Learning Rate Schedule ===
    ax1 = axes[0]
    
    ax1.plot(epochs_theo, lrs_theo, 'b-', linewidth=2, label='Cosine Annealing Schedule')
    
    if lr_history is not None:
        epochs_actual = np.arange(len(lr_history))
        ax1.scatter(epochs_actual, lr_history, c='red', s=15, alpha=0.5, 
                   label='Recorded LR', zorder=5)
    
    ax1.axhline(y=lr_max, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=lr_min, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.annotate(f'$\\eta_{{max}}$ = {lr_max}', xy=(5, lr_max), fontsize=10, va='bottom')
    ax1.annotate(f'$\\eta_{{min}}$ = {lr_min:.0e}', xy=(5, lr_min*10), fontsize=10, va='bottom')
    
    formula = r'$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos\frac{t\pi}{T})$'
    ax1.text(0.5, 0.85, formula, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title('Cosine Annealing Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, num_epochs])
    ax1.set_yscale('log')
    ax1.set_ylim([lr_min * 0.5, lr_max * 2])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # === Plot 2: Training Loss (if available) ===
    if show_training_curves and train_loss_history is not None:
        ax2 = axes[1]
        epochs_train = np.arange(len(train_loss_history))
        
        ax2.plot(epochs_train, train_loss_history, 'g-', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
        ax2.set_xlim([0, num_epochs])
        ax2.grid(True, alpha=0.3)
        
        if len(train_loss_history) > 10:
            window = 5
            smoothed = np.convolve(train_loss_history, np.ones(window)/window, mode='valid')
            ax2.plot(np.arange(window-1, len(train_loss_history)), smoothed, 
                    'darkgreen', linewidth=2, label='Smoothed')
            ax2.legend(fontsize=9)
    
    # === Plot 3: Validation Accuracy (if available) ===
    if show_training_curves and val_acc_history is not None:
        ax3 = axes[2]
        epochs_val = np.arange(len(val_acc_history))
        
        ax3.plot(epochs_val, np.array(val_acc_history) * 100, 'orange', linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax3.set_title('Validation Accuracy Curve', fontsize=13, fontweight='bold')
        ax3.set_xlim([0, num_epochs])
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3)
        
        best_epoch = np.argmax(val_acc_history)
        best_acc = val_acc_history[best_epoch] * 100
        ax3.scatter([best_epoch], [best_acc], c='red', s=100, marker='*', zorder=5)
        ax3.annotate(f'Best: {best_acc:.1f}%\n(Epoch {best_epoch+1})', 
                    xy=(best_epoch, best_acc), xytext=(best_epoch+10, best_acc-10),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate Learning Rate Schedule Visualization')
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2',
                       help='Path to LibriSpeech dataset')
    parser.add_argument('--output', type=str, default='fig4_lr_schedule.png',
                       help='Output file path')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr_max', type=float, default=0.005,
                       help='Maximum learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                       help='Minimum learning rate')
    parser.add_argument('--theoretical_only', action='store_true',
                       help='Only plot theoretical curve without training')
    parser.add_argument('--save_checkpoint', action='store_true',
                       help='Save model checkpoints during training')
    parser.add_argument('--load_history', type=str, default=None,
                       help='Load training history from JSON file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Learning Rate Schedule Visualization Generator")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"LR Max: {args.lr_max}")
    print(f"LR Min: {args.lr_min}")
    
    lr_history = None
    train_loss_history = None
    val_acc_history = None
    
    if args.load_history:
        print(f"\nLoading training history from: {args.load_history}")
        with open(args.load_history, 'r') as f:
            history = json.load(f)
        lr_history = history.get('lr_history')
        train_loss_history = history.get('train_loss_history')
        val_acc_history = history.get('val_acc_history')
        
    elif not args.theoretical_only:
        if HAS_PROJECT and os.path.exists(args.data_dir):
            print(f"\nTraining with real data from: {args.data_dir}")
            
            lr_history, train_loss_history, val_acc_history = train_and_record_lr(
                data_dir=args.data_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr_max=args.lr_max,
                lr_min=args.lr_min,
                save_checkpoint=args.save_checkpoint
            )
            
            history_path = args.output.replace('.png', '_history.json')
            with open(history_path, 'w') as f:
                json.dump({
                    'lr_history': lr_history,
                    'train_loss_history': train_loss_history,
                    'val_acc_history': val_acc_history,
                    'config': {
                        'epochs': args.epochs,
                        'lr_max': args.lr_max,
                        'lr_min': args.lr_min,
                        'batch_size': args.batch_size,
                    }
                }, f, indent=2)
            print(f"Training history saved to: {history_path}")
        else:
            print("\nProject modules or data not available, using theoretical curve only")
    else:
        print("\nUsing theoretical curve only (--theoretical_only)")
    
    show_curves = lr_history is not None and train_loss_history is not None
    
    plot_lr_schedule(
        lr_history=lr_history,
        train_loss_history=train_loss_history,
        val_acc_history=val_acc_history,
        num_epochs=args.epochs,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        output_path=args.output,
        show_training_curves=show_curves
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()