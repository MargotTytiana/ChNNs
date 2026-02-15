#!/usr/bin/env python3
"""
Precompute optimal phase space embedding parameters from dataset.
Run this ONCE before training to determine optimal tau and dimension.

This version saves raw data and generates visualization plots.

Usage:
    python precompute_embedding_params_with_plot.py \
        --data_dir /path/to/librispeech \
        --output params.yaml \
        --n_samples 100 \
        --plot
"""

import os
import sys
import argparse
import json
import numpy as np
import yaml
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import random

import librosa
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches


class EmbeddingParameterEstimator:
    """Estimate optimal embedding parameters from audio samples."""
    
    def __init__(self, sample_rate: int = 16000, max_delay: int = 50):
        self.sample_rate = sample_rate
        self.max_delay = max_delay
    
    def estimate_tau_mutual_info(self, signal: np.ndarray, n_bins: int = 50) -> int:
        """
        Estimate optimal delay using mutual information.
        Returns first local minimum of I(tau).
        """
        max_len = 10000
        if len(signal) > max_len:
            signal = signal[:max_len]
        
        mutual_info = np.zeros(self.max_delay)
        
        for tau in range(1, self.max_delay):
            x1 = signal[:-tau]
            x2 = signal[tau:]
            
            hist_2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
            p_joint = hist_2d / hist_2d.sum()
            p_x1 = p_joint.sum(axis=1)
            p_x2 = p_joint.sum(axis=0)
            
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if p_joint[i, j] > 1e-10 and p_x1[i] > 1e-10 and p_x2[j] > 1e-10:
                        mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_x1[i] * p_x2[j]))
            
            mutual_info[tau] = mi
        
        for i in range(2, len(mutual_info) - 1):
            if mutual_info[i] < mutual_info[i-1] and mutual_info[i] < mutual_info[i+1]:
                return i
        
        diff = np.diff(mutual_info[1:])
        return np.argmin(diff) + 2
    
    def estimate_tau_autocorr(self, signal: np.ndarray) -> int:
        """
        Estimate optimal delay using autocorrelation.
        """
        max_len = 10000
        if len(signal) > max_len:
            signal = signal[:max_len]
        
        n = len(signal)
        signal = signal - np.mean(signal)
        
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1:n-1+self.max_delay]
        autocorr = autocorr / autocorr[0]
        
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                return i
        
        e_threshold = 1.0 / np.e
        for i in range(1, len(autocorr)):
            if autocorr[i] < e_threshold:
                return i
        
        return self.max_delay // 4
    
    def estimate_dimension_fnn(self, signal: np.ndarray, tau: int, 
                                max_dim: int = 15, threshold: float = 0.05) -> int:
        """
        Estimate optimal embedding dimension using False Nearest Neighbors.
        """
        from sklearn.neighbors import NearestNeighbors
        
        max_len = 5000
        if len(signal) > max_len:
            signal = signal[:max_len]
        
        for dim in range(2, max_dim + 1):
            n_points = len(signal) - (dim) * tau
            if n_points < 100:
                return dim - 1
            
            embedded_d = np.zeros((n_points, dim))
            embedded_d1 = np.zeros((n_points, dim + 1))
            
            for i in range(dim):
                embedded_d[:, i] = signal[i*tau:i*tau + n_points]
            for i in range(dim + 1):
                embedded_d1[:, i] = signal[i*tau:i*tau + n_points]
            
            nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_d)
            distances, indices = nbrs.kneighbors(embedded_d)
            
            n_false = 0
            n_test = min(500, n_points)
            test_indices = np.random.choice(n_points, n_test, replace=False)
            
            for idx in test_indices:
                nn_idx = indices[idx, 1]
                dist_d = distances[idx, 1]
                if dist_d < 1e-10:
                    continue
                dist_d1 = np.linalg.norm(embedded_d1[idx] - embedded_d1[nn_idx])
                if (dist_d1 - dist_d) / dist_d > 15.0:
                    n_false += 1
            
            fnn_fraction = n_false / n_test
            
            if fnn_fraction < threshold:
                return dim
        
        return max_dim
    
    def estimate_all(self, signal: np.ndarray, method: str = 'mutual_info') -> Dict:
        """Estimate both tau and dimension for a signal."""
        if method == 'mutual_info':
            tau = self.estimate_tau_mutual_info(signal)
        else:
            tau = self.estimate_tau_autocorr(signal)
        
        dim = self.estimate_dimension_fnn(signal, tau)
        
        return {
            'tau': tau,
            'dimension': dim,
            'tau_ms': tau / self.sample_rate * 1000
        }


def load_audio_files(data_dir: str, n_samples: int, sample_rate: int = 16000) -> List[np.ndarray]:
    """Load random audio samples from LibriSpeech-style directory."""
    data_path = Path(data_dir)
    all_files = list(data_path.rglob("*.flac"))
    
    if not all_files:
        all_files = list(data_path.rglob("*.wav"))
    
    if not all_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    selected_files = random.sample(all_files, min(n_samples, len(all_files)))
    
    print(f"Loading {len(selected_files)} audio files...")
    
    audios = []
    for f in tqdm(selected_files, desc="Loading audio"):
        try:
            audio, sr = librosa.load(f, sr=sample_rate)
            if len(audio) > sample_rate:
                audios.append(audio)
        except Exception as e:
            warnings.warn(f"Failed to load {f}: {e}")
    
    return audios


def compute_statistics(values: List[float]) -> Dict:
    """Compute statistics for a list of values."""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75))
    }


def plot_embedding_params(tau_values: List[int], dim_values: List[int], 
                          tau_stats: Dict, dim_stats: Dict,
                          selected_tau: int, selected_dim: int,
                          output_prefix: str = 'embedding_params'):
    """
    Generate visualization plots from real data.
    """
    # Set academic plot style
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    tau_data = np.array(tau_values)
    dim_data = np.array(dim_values)
    
    # Color scheme
    tau_color = '#E15759'
    dim_color = '#59A14F'
    highlight_color = '#2E86AB'
    
    # ========================================
    # Figure 1: Distribution plots only
    # ========================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # --- Left: Time Delay Distribution ---
    bins_tau = np.arange(0, int(tau_stats['max']) + 5, 3)
    n, bins, patches = ax1.hist(tau_data, bins=bins_tau, color=tau_color, 
                                 edgecolor='white', alpha=0.8, linewidth=1.2)
    
    ax1.axvline(x=tau_stats['median'], color='#666666', linestyle='--', linewidth=2, 
                label=f'Median = {tau_stats["median"]:.0f}')
    ax1.axvline(x=selected_tau, color=highlight_color, linestyle='-', linewidth=2.5, 
                label=f'Selected $\\tau$ = {selected_tau}')
    
    stats_text = (f'Mean = {tau_stats["mean"]:.1f} samples\n'
                  f'Std = {tau_stats["std"]:.1f}\n'
                  f'Range = [{int(tau_stats["min"])}, {int(tau_stats["max"])}]')
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                       edgecolor='#cccccc', alpha=0.9))
    
    ax1.set_xlabel('Time Delay $\\tau$ (samples)')
    ax1.set_ylabel('Number of Utterances')
    ax1.set_title('(a) Time Delay Distribution', pad=12)
    ax1.legend(loc='upper right', framealpha=0.9, bbox_to_anchor=(0.98, 0.72))
    ax1.set_xlim(0, int(tau_stats['max']) + 5)
    ax1.set_ylim(0, max(n) * 1.25)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # --- Right: Embedding Dimension Distribution ---
    dim_unique, dim_counts = np.unique(dim_data, return_counts=True)
    bars = ax2.bar(dim_unique, dim_counts, color=dim_color, edgecolor='white', 
                   alpha=0.8, linewidth=1.2, width=0.6)
    
    # Add percentage labels on bars
    total = len(dim_data)
    for bar, count in zip(bars, dim_counts):
        height = bar.get_height()
        pct = count / total * 100
        ax2.annotate(f'{pct:.0f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.axvline(x=dim_stats['median'], color='#666666', linestyle='--', linewidth=2,
                label=f'Median = {dim_stats["median"]:.0f}')
    ax2.legend(loc='upper right')
    
    # Add annotation for selected dimension
    # ax2.annotate(f'Selected $d_e$ = {selected_dim}\n(Takens\' theorem)', 
    #              xy=(dim_unique.max() + 0.3, max(dim_counts) * 0.7), 
    #              fontsize=10, color=highlight_color, fontweight='bold',
    #              bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4FD', 
    #                        edgecolor=highlight_color, alpha=0.9))
    
    stats_text2 = (f'Mean: {dim_stats["mean"]:.2f}\n'
                   f'FNN threshold: 5%')
    ax2.text(0.03, 0.97, stats_text2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                       edgecolor='#cccccc', alpha=0.9))
    
    ax2.set_xlabel('Embedding Dimension $d_e$')
    ax2.set_ylabel('Number of Utterances')
    ax2.set_title('(b) Embedding Dimension Distribution', pad=12)
    ax2.legend(loc='center right', framealpha=0.9)
    ax2.set_xlim(dim_unique.min() - 0.5, dim_unique.max() + 1)
    ax2.set_ylim(0, max(dim_counts) * 1.3)
    ax2.set_xticks(dim_unique)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    fig1.savefig(f'{output_prefix}_distribution.pdf', format='pdf')
    fig1.savefig(f'{output_prefix}_distribution.png', format='png', dpi=300)
    print(f"Distribution plot saved: {output_prefix}_distribution.pdf/png")
    
    plt.close(fig1)
    
    # ========================================
    # Figure 2: Combined with ablation heatmap
    # ========================================
    # Ablation study data (you need to update these with actual results)
    accuracy_data = np.array([
        [95.76, 94.00],   # tau=10: d_e=4, d_e=10
        [95.76, 97.46],   # tau=12: d_e=4, d_e=10
    ])
    tau_ablation = [10, 12]
    de_ablation = [4, 10]
    
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # --- Panel (a): Time Delay Distribution ---
    ax1 = axes[0]
    n, bins, patches = ax1.hist(tau_data, bins=bins_tau, color=tau_color, 
                                 edgecolor='white', alpha=0.8, linewidth=1.2)
    ax1.axvline(x=tau_stats['median'], color='#666666', linestyle='--', linewidth=2, 
                label=f'Median = {tau_stats["median"]:.0f}')
    ax1.axvline(x=selected_tau, color=highlight_color, linestyle='-', linewidth=2.5, 
                label=f'Selected $\\tau$ = {selected_tau}')
    
    stats_text = f'$n$ = {len(tau_data)}\nMean = {tau_stats["mean"]:.1f}\nStd = {tau_stats["std"]:.1f}'
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                       edgecolor='#cccccc', alpha=0.9))
    
    ax1.set_xlabel('Time Delay $\\tau$ (samples)')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Time Delay Distribution', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9, bbox_to_anchor=(0.98, 0.75))
    ax1.set_xlim(0, int(tau_stats['max']) + 5)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # --- Panel (b): Embedding Dimension Distribution ---
    ax2 = axes[1]
    bars = ax2.bar(dim_unique, dim_counts, color=dim_color, edgecolor='white', 
                   alpha=0.8, linewidth=1.2, width=0.6)
    
    for bar, count in zip(bars, dim_counts):
        height = bar.get_height()
        pct = count / total * 100
        ax2.annotate(f'{pct:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.axvline(x=dim_stats['median'], color='#666666', linestyle='--', linewidth=2,
                label=f'Median = {dim_stats["median"]:.0f}')
    
    ax2.annotate(f'Selected $d_e$ = {selected_dim}\n(Takens\')', 
                 xy=(dim_unique.max() + 0.2, max(dim_counts) * 0.65), 
                 fontsize=9, color=highlight_color, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F4FD', 
                           edgecolor=highlight_color, alpha=0.9))
    
    stats_text2 = f'$n$ = {len(dim_data)}\nMean = {dim_stats["mean"]:.2f}'
    ax2.text(0.03, 0.97, stats_text2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                       edgecolor='#cccccc', alpha=0.9))
    
    ax2.set_xlabel('Embedding Dimension $d_e$')
    ax2.set_ylabel('Count')
    ax2.set_title('(b) Dimension Distribution', pad=10)
    ax2.legend(loc='center right', framealpha=0.9, fontsize=9)
    ax2.set_xlim(dim_unique.min() - 0.5, dim_unique.max() + 1)
    ax2.set_ylim(0, max(dim_counts) * 1.3)
    ax2.set_xticks(dim_unique)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # --- Panel (c): Ablation Heatmap ---
    ax3 = axes[2]
    im = ax3.imshow(accuracy_data, cmap=plt.cm.RdYlGn, aspect='auto', vmin=93, vmax=98)
    
    cbar = fig2.colorbar(im, ax=ax3, shrink=0.9, pad=0.02)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=10)
    
    ax3.set_xticks(np.arange(len(de_ablation)))
    ax3.set_yticks(np.arange(len(tau_ablation)))
    ax3.set_xticklabels([f'{d}' for d in de_ablation], fontsize=11)
    ax3.set_yticklabels([f'{t}' for t in tau_ablation], fontsize=11)
    
    for i in range(len(tau_ablation)):
        for j in range(len(de_ablation)):
            acc = accuracy_data[i, j]
            text_color = 'white' if acc > 96.5 or acc < 94.5 else 'black'
            
            if acc == accuracy_data.max():
                text = f'{acc:.2f}%'
                fontweight = 'bold'
                rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=2.5, edgecolor='#2E86AB', 
                                           facecolor='none')
                ax3.add_patch(rect)
            else:
                text = f'{acc:.2f}%'
                fontweight = 'normal'
            
            ax3.text(j, i, text, ha='center', va='center', 
                    color=text_color, fontsize=12, fontweight=fontweight)
    
    ax3.set_xlabel('Embedding Dimension $d_e$')
    ax3.set_ylabel('Time Delay $\\tau$')
    ax3.set_title('(c) Ablation Study', pad=10)
    
    plt.tight_layout()
    
    fig2.savefig(f'{output_prefix}_combined.pdf', format='pdf')
    fig2.savefig(f'{output_prefix}_combined.png', format='png', dpi=300)
    print(f"Combined plot saved: {output_prefix}_combined.pdf/png")
    
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description='Precompute embedding parameters with visualization')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to audio dataset')
    parser.add_argument('--output', type=str, default='embedding_params.yaml',
                        help='Output YAML file')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of audio samples to analyze')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--method', type=str, default='mutual_info',
                        choices=['mutual_info', 'autocorr'],
                        help='Method for tau estimation')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--selected_tau', type=int, default=12,
                        help='Selected tau value for visualization')
    parser.add_argument('--selected_dim', type=int, default=10,
                        help='Selected dimension value for visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EMBEDDING PARAMETER PRECOMPUTATION")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Method: {args.method}")
    print()
    
    # Load audio samples
    audios = load_audio_files(args.data_dir, args.n_samples, args.sample_rate)
    print(f"Loaded {len(audios)} valid audio samples")
    
    # Estimate parameters for each sample
    estimator = EmbeddingParameterEstimator(sample_rate=args.sample_rate)
    
    tau_values = []
    dim_values = []
    
    print("\nEstimating parameters...")
    for audio in tqdm(audios, desc="Processing"):
        try:
            result = estimator.estimate_all(audio, method=args.method)
            tau_values.append(result['tau'])
            dim_values.append(result['dimension'])
        except Exception as e:
            warnings.warn(f"Estimation failed: {e}")
    
    # Compute statistics
    tau_stats = compute_statistics(tau_values)
    dim_stats = compute_statistics(dim_values)
    
    # Determine recommended values
    recommended_tau = int(np.round(tau_stats['median']))
    recommended_dim = int(np.ceil(dim_stats['q75']))
    
    # Prepare output
    results = {
        'recommended': {
            'fixed_delay': recommended_tau,
            'embedding_dim': recommended_dim,
            'delay_ms': recommended_tau / args.sample_rate * 1000
        },
        'statistics': {
            'tau': tau_stats,
            'dimension': dim_stats
        },
        'metadata': {
            'n_samples': len(audios),
            'method': args.method,
            'sample_rate': args.sample_rate,
            'data_dir': args.data_dir
        },
        # Save raw data for reproducibility
        'raw_data': {
            'tau_values': tau_values,
            'dim_values': dim_values
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTau (delay) statistics:")
    print(f"  Mean:   {tau_stats['mean']:.2f} samples ({tau_stats['mean']/args.sample_rate*1000:.3f} ms)")
    print(f"  Median: {tau_stats['median']:.2f} samples")
    print(f"  Std:    {tau_stats['std']:.2f}")
    print(f"  Range:  [{tau_stats['min']}, {tau_stats['max']}]")
    
    print(f"\nDimension statistics:")
    print(f"  Mean:   {dim_stats['mean']:.2f}")
    print(f"  Median: {dim_stats['median']:.2f}")
    print(f"  Std:    {dim_stats['std']:.2f}")
    print(f"  Range:  [{dim_stats['min']}, {dim_stats['max']}]")
    
    print(f"\n*** RECOMMENDED VALUES ***")
    print(f"  fixed_delay: {recommended_tau}")
    print(f"  embedding_dim: {recommended_dim}")
    
    # Save to YAML file
    with open(args.output, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\nResults saved to: {args.output}")
    
    # Save raw data to JSON for easier loading
    # Convert numpy types to native Python types for JSON serialization
    output_path = Path(args.output)
    json_output = output_path.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump({
            'tau_values': [int(x) for x in tau_values],
            'dim_values': [int(x) for x in dim_values],
            'tau_stats': {k: float(v) for k, v in tau_stats.items()},
            'dim_stats': {k: float(v) for k, v in dim_stats.items()}
        }, f, indent=2)
    print(f"Raw data saved to: {json_output}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating visualization plots...")
        output_prefix = str(output_path.with_suffix(''))
        plot_embedding_params(
            tau_values, dim_values, 
            tau_stats, dim_stats,
            args.selected_tau, args.selected_dim,
            output_prefix
        )
    
    # Print config snippet
    print("\n" + "="*60)
    print("ADD TO chaotic_config.yaml:")
    print("="*60)
    print(f"""
model:
  phase_space:
    embedding_dim: {recommended_dim}
    fixed_delay: {recommended_tau}    # {recommended_tau/args.sample_rate*1000:.2f}ms at {args.sample_rate}Hz
    delay_method: "none"              # Skip estimation, use fixed values
""")


if __name__ == "__main__":
    main()