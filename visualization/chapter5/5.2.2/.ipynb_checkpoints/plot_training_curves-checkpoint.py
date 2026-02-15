#!/usr/bin/env python3
"""
Plot Training Curves from ChaoNet JSON results.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


import matplotlib.font_manager as fm
from pathlib import Path
current_dir = Path(__file__).parent
font_path = current_dir / '..'/ '..' / 'Helvetica.ttf'
fm.fontManager.addfont(font_path)

# Set up matplotlib for academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Helvetica', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})


def load_and_plot(json_path, output_path='training_curves.pdf'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract from dynamics_history
    if 'dynamics_history' in data:
        epochs = []
        val_acc = []
        for item in data['dynamics_history']:
            epochs.append(item['epoch'])
            val_acc.append(item['metrics']['accuracy'] * 100)  # Convert to %
        
        print(f"Found {len(val_acc)} epochs")
        print(f"Best accuracy: {max(val_acc):.2f}% at epoch {epochs[np.argmax(val_acc)]}")
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, val_acc, color='#2596be', linestyle='-', linewidth=2, marker='o', markersize=3, alpha=0.7)
        
        # Mark best
        best_idx = np.argmax(val_acc)
        plt.scatter([epochs[best_idx]], [val_acc[best_idx]], color='#e28743', s=100, zorder=5, marker='*')
        plt.axhline(y=val_acc[best_idx], color='#e28743', linestyle='--', linewidth=1.5, alpha=1.0)
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Validation Accuracy (%)', fontsize=14)
        plt.title(f'ChaoNet Training Curve (Best: {val_acc[best_idx]:.2f}%)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved to: {output_path}")
    else:
        print("Error: dynamics_history not found")


if __name__ == '__main__':
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'lorenz_full_chaotic_run_0_results.json'
    output = sys.argv[2] if len(sys.argv) > 2 else 'training_curves.pdf'
    load_and_plot(json_file, output)