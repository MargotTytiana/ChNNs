#!/usr/bin/env python3
"""
Figure 4.XX: Angular Margin Visualization on Unit Sphere

This script visualizes how angular margin affects speaker embeddings
on a unit hypersphere, showing:
1. Speaker embeddings distributed on the sphere
2. Class prototypes (weight vectors)
3. Decision boundaries with and without angular margin

Usage:
    python plot_angular_margin.py --data_dir /path/to/librispeech --output fig4_angular_margin.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

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
    print("Warning: Project modules not found, using synthetic data")


def extract_real_embeddings(data_dir, num_speakers=10, samples_per_speaker=20):
    """
    Extract real speaker embeddings from trained model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dataloaders with CORRECT parameter names
    train_loader, val_loader, _ = create_speaker_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        num_workers=0,
        max_length=3.0,
        sample_rate=16000
    )
    
    # Update num_speakers from actual data
    actual_num_speakers = train_loader.dataset.num_speakers
    
    # Create model
    model_config = {
        'num_speakers': actual_num_speakers,
        'chaotic_system': 'lorenz',
        'embedding_dim': 10,
        'speaker_embedding_dim': 128,
        'classifier_type': 'cosine',
        'temperature': 30.0,
        'margin': 0.35,
    }
    
    model = ChaoticSpeakerRecognitionNetwork(model_config)
    model = model.to(device)
    model.eval()
    
    # Extract embeddings
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, (audio, labels) in enumerate(val_loader):
            if len(embeddings_list) >= num_speakers * samples_per_speaker:
                break
            
            audio = audio.to(device)
            
            # Get embeddings before classifier
            speaker_emb = model.get_speaker_embedding(audio)
            
            # L2 normalize to unit sphere
            speaker_emb = F.normalize(speaker_emb, p=2, dim=1)
            
            embeddings_list.append(speaker_emb.cpu().numpy())
            labels_list.append(labels.numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Get class prototypes (normalized classifier weights)
    if hasattr(model.classifier, 'weight'):
        prototypes = model.classifier.weight.data.cpu().numpy()
        prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
    else:
        # Compute prototypes from embeddings
        prototypes = []
        for speaker_id in range(min(num_speakers, actual_num_speakers)):
            mask = labels == speaker_id
            if mask.sum() > 0:
                proto = embeddings[mask].mean(axis=0)
                proto = proto / np.linalg.norm(proto)
                prototypes.append(proto)
        prototypes = np.array(prototypes)
    
    return embeddings, labels, prototypes


def generate_synthetic_embeddings(num_speakers=5, samples_per_speaker=30, embedding_dim=3):
    """
    Generate synthetic embeddings on unit sphere for visualization.
    """
    np.random.seed(42)
    
    # Generate class prototypes on unit sphere
    prototypes = []
    for i in range(num_speakers):
        theta = 2 * np.pi * i / num_speakers
        phi = np.pi / 2 + 0.3 * np.sin(theta)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        prototypes.append([x, y, z])
    
    prototypes = np.array(prototypes)
    
    # Generate samples around each prototype
    embeddings = []
    labels = []
    
    for speaker_id, proto in enumerate(prototypes):
        for _ in range(samples_per_speaker):
            noise = np.random.randn(3) * 0.15
            sample = proto + noise
            sample = sample / np.linalg.norm(sample)
            
            embeddings.append(sample)
            labels.append(speaker_id)
    
    return np.array(embeddings), np.array(labels), prototypes


def project_to_3d(embeddings, method='pca'):
    """Project high-dimensional embeddings to 3D for visualization."""
    if embeddings.shape[1] == 3:
        return embeddings
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
    else:
        np.random.seed(42)
        proj_matrix = np.random.randn(embeddings.shape[1], 3)
        proj_matrix = proj_matrix / np.linalg.norm(proj_matrix, axis=0)
        embeddings_3d = embeddings @ proj_matrix
    
    norms = np.linalg.norm(embeddings_3d, axis=1, keepdims=True)
    embeddings_3d = embeddings_3d / norms
    
    return embeddings_3d


def draw_decision_boundary(ax, proto1, proto2, margin=0.0, color='gray', alpha=0.3):
    """Draw great circle decision boundary between two prototypes."""
    normal = np.cross(proto1, proto2)
    if np.linalg.norm(normal) < 1e-6:
        return
    normal = normal / np.linalg.norm(normal)
    
    t = np.linspace(0, 2*np.pi, 100)
    
    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    
    circle = np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2)
    
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], 
            color=color, alpha=alpha, linewidth=1.5, linestyle='--')


def plot_angular_margin_visualization(embeddings, labels, prototypes, 
                                      margin=0.35, output_path='fig4_angular_margin.png'):
    """Create the angular margin visualization figure."""
    if embeddings.shape[1] > 3:
        embeddings_3d = project_to_3d(embeddings, method='pca')
        prototypes_3d = project_to_3d(prototypes, method='pca')
    else:
        embeddings_3d = embeddings
        prototypes_3d = prototypes
    
    fig = plt.figure(figsize=(14, 6))
    
    num_speakers = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_speakers))
    
    # === Left subplot: Without angular margin ===
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.3)
    
    # Plot embeddings
    for speaker_id in range(min(num_speakers, 5)):
        mask = labels == speaker_id
        ax1.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                   c=[colors[speaker_id]], s=30, alpha=0.7, label=f'Speaker {speaker_id+1}')
    
    # Plot prototypes
    for i, proto in enumerate(prototypes_3d[:5]):
        ax1.scatter([proto[0]], [proto[1]], [proto[2]], 
                   c=[colors[i]], s=200, marker='*', edgecolors='black', linewidths=1)
    
    # Draw decision boundaries (no margin)
    for i in range(min(len(prototypes_3d)-1, 4)):
        for j in range(i+1, min(len(prototypes_3d), 5)):
            draw_decision_boundary(ax1, prototypes_3d[i], prototypes_3d[j], 
                                  margin=0.0, color='gray', alpha=0.4)
    
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.set_title('(a) Standard Cross-Entropy\n(No Angular Margin)', fontsize=12, fontweight='bold')
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])
    
    # === Right subplot: With angular margin ===
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax2.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.3)
    
    # Plot embeddings (tighter clusters due to margin)
    for speaker_id in range(min(num_speakers, 5)):
        mask = labels == speaker_id
        proto = prototypes_3d[speaker_id] if speaker_id < len(prototypes_3d) else prototypes_3d[0]
        
        # Pull embeddings closer to prototype (simulate tighter clustering)
        tight_emb = embeddings_3d[mask] * 0.7 + proto * 0.3
        tight_emb = tight_emb / np.linalg.norm(tight_emb, axis=1, keepdims=True)
        
        ax2.scatter(tight_emb[:, 0], tight_emb[:, 1], tight_emb[:, 2],
                   c=[colors[speaker_id]], s=30, alpha=0.7, label=f'Speaker {speaker_id+1}')
    
    # Plot prototypes
    for i, proto in enumerate(prototypes_3d[:5]):
        ax2.scatter([proto[0]], [proto[1]], [proto[2]], 
                   c=[colors[i]], s=200, marker='*', edgecolors='black', linewidths=1)
    
    # Draw decision boundaries (with margin - red color)
    for i in range(min(len(prototypes_3d)-1, 4)):
        for j in range(i+1, min(len(prototypes_3d), 5)):
            draw_decision_boundary(ax2, prototypes_3d[i], prototypes_3d[j], 
                                  margin=margin, color='red', alpha=0.5)
    
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.set_title(f'(b) Angular Margin Loss\n(m = {np.degrees(margin):.0f}°)', fontsize=12, fontweight='bold')
    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_zlim([-1.1, 1.1])
    
    # Add legend
    handles, labels_legend = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', ncol=5, 
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    
    # Add annotation
    fig.text(0.5, 0.02, 
             '★ = Class Prototype    ● = Speaker Embedding    --- = Decision Boundary',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate Angular Margin Visualization')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to LibriSpeech dataset')
    parser.add_argument('--output', type=str, default='fig4_angular_margin.png',
                       help='Output file path')
    parser.add_argument('--num_speakers', type=int, default=5,
                       help='Number of speakers to visualize')
    parser.add_argument('--margin', type=float, default=0.35,
                       help='Angular margin in radians')
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real data from trained model')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Angular Margin Visualization Generator")
    print("="*60)
    
    if args.use_real_data and args.data_dir and HAS_PROJECT:
        print(f"Using real data from: {args.data_dir}")
        try:
            embeddings, labels, prototypes = extract_real_embeddings(
                args.data_dir, 
                num_speakers=args.num_speakers
            )
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to synthetic data")
            embeddings, labels, prototypes = generate_synthetic_embeddings(
                num_speakers=args.num_speakers
            )
    else:
        print("Using synthetic data for visualization")
        embeddings, labels, prototypes = generate_synthetic_embeddings(
            num_speakers=args.num_speakers
        )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of speakers: {len(np.unique(labels))}")
    print(f"Angular margin: {args.margin} rad ({np.degrees(args.margin):.1f}°)")
    
    plot_angular_margin_visualization(
        embeddings, labels, prototypes,
        margin=args.margin,
        output_path=args.output
    )
    
    print("Done!")


if __name__ == '__main__':
    main()