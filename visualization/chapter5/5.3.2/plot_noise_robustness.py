#!/usr/bin/env python3
"""Plot noise robustness: Accuracy vs SNR for all models and noise types."""

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

# Data from experiments
SNR_LABELS = ['Clean', '20dB', '15dB', '10dB', '5dB', '0dB']
X_POS = [0, 1, 2, 3, 4, 5]  # Equal spacing

# Accuracy data
DATA = {
    'ChaoNet': {
        'Gaussian': [94.92, 77.12, 55.93, 38.14, 22.03, 11.86],
        'Babble':   [94.92, 75.42, 52.54, 33.05, 16.95, 8.47],
        'Cafe':     [94.92, 77.12, 55.08, 35.59, 18.64, 10.17],
        'Street':   [94.92, 83.90, 66.95, 44.07, 24.58, 12.71],
    },
    'Mel-MLP': {
        'Gaussian': [94.92, 41.53, 22.03, 11.86, 5.93, 5.08],
        'Babble':   [94.92, 33.05, 13.56, 6.78, 5.08, 4.24],
        'Cafe':     [94.92, 38.14, 16.10, 8.47, 5.08, 5.08],
        'Street':   [94.92, 62.71, 38.98, 21.19, 9.32, 5.08],
    },
    'MFCC-MLP': {
        'Gaussian': [99.15, 38.14, 16.95, 11.02, 6.78, 5.08],
        'Babble':   [99.15, 23.73, 10.17, 5.93, 5.08, 5.08],
        'Cafe':     [99.15, 33.05, 13.56, 7.63, 5.08, 5.08],
        'Street':   [99.15, 55.93, 33.05, 16.10, 7.63, 5.08],
    },
}

COLORS = {'ChaoNet': '#7eacce', 'Mel-MLP': '#ebac34', 'MFCC-MLP': '#a4bfa1'}
MARKERS = {'ChaoNet': 'o', 'Mel-MLP': 's', 'MFCC-MLP': '^'}
NOISE_TYPES = ['Gaussian', 'Babble', 'Cafe', 'Street']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, noise in enumerate(NOISE_TYPES):
    ax = axes[idx]
    for model in DATA:
        ax.plot(X_POS, DATA[model][noise], 
                color=COLORS[model], marker=MARKERS[model],
                linewidth=2, markersize=8, label=model)
    
    ax.set_xlabel('SNR Level', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{noise} Noise', fontsize=13, fontweight='bold')
    ax.set_xticks(X_POS)
    ax.set_xticklabels(SNR_LABELS, fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Noise Robustness Comparison', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('noise_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('noise_robustness.pdf', bbox_inches='tight')
print('Saved: noise_robustness.png/pdf')