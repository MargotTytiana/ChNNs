#!/usr/bin/env python3
"""
Plot embedding parameters distribution from precomputed data.

Usage:
    # From JSON data file (preferred - uses real data)
    python plot_embedding_params.py --json params.json --plot
    
    # Or just generate plots if JSON exists
    python plot_embedding_params.py --json params.json
"""
from matplotlib import colors
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from pathlib import Path
import matplotlib.font_manager as fm

current_dir = Path(__file__).parent
font_path = current_dir / '..' / 'Helvetica.ttf'
if font_path.exists():
    fm.fontManager.addfont(str(font_path))

def setup_plot_style():
    """Set academic plot style."""
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica', 'DejaVu Serif'],
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


def plot_distribution_only(tau_data, dim_data, tau_stats, dim_stats,
                           selected_tau, selected_dim, output_prefix):
    """Generate distribution plots only (2 panels)."""
    setup_plot_style()
    
    tau_color = '#8da9c4'
    dim_color = '#8ab8a8'
    highlight_color = '#595959'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # --- Left: Time Delay Distribution ---
    bins_tau = np.arange(0, int(tau_stats['max']) + 5, 3)
    n, bins, patches = ax1.hist(tau_data, bins=bins_tau, color=tau_color, 
                                 edgecolor='white', alpha=0.8, linewidth=1.2)
    
    ax1.axvline(x=tau_stats['median'], color='#c77f76', linestyle='--', linewidth=2, 
                label=f'Median = {tau_stats["median"]:.0f}')
    ax1.axvline(x=selected_tau, color=highlight_color, linestyle='-', linewidth=2.5, 
                label=f'Selected $\\tau$ = {selected_tau}')
    
    stats_text = (f'Mean = {tau_stats["mean"]:.1f} samples\n'
                  f'Std = {tau_stats["std"]:.1f}\n'
                  f'Range = [{int(tau_stats["min"])}, {int(tau_stats["max"])}]')
    ax1.text(0.67, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
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
    
    # ax2.annotate(f'Selected $d_e$ = {selected_dim}\n(Takens\' theorem)', 
    #              xy=(dim_unique.max() + 0.3, max(dim_counts) * 0.7), 
    #              fontsize=10, color=highlight_color, fontweight='bold',
    #              bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4FD', 
    #                        edgecolor=highlight_color, alpha=0.9))
    
    stats_text2 = (f'Mean = {dim_stats["mean"]:.2f}\n'
                   f'FNN threshold = 5%')
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
    
    fig.savefig(f'{output_prefix}_distribution.pdf', format='pdf')
    fig.savefig(f'{output_prefix}_distribution.png', format='png', dpi=300)
    print(f"Saved: {output_prefix}_distribution.pdf/png")
    
    return fig


def plot_combined_with_ablation(tau_data, dim_data, tau_stats, dim_stats,
                                 selected_tau, selected_dim, 
                                 ablation_data, output_prefix):
    """Generate combined plot with ablation heatmap (3 panels)."""
    setup_plot_style()
    
    tau_color = '#8ab8a8'
    dim_color = '#8da9c4'
    highlight_color = '#595959'
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # --- Panel (a): Time Delay Distribution ---
    ax1 = axes[0]
    bins_tau = np.arange(0, int(tau_stats['max']) + 5, 3)
    n, bins, patches = ax1.hist(tau_data, bins=bins_tau, color=tau_color, 
                                 edgecolor='white', alpha=0.8, linewidth=1.2)
    ax1.axvline(x=tau_stats['median'], color='#c77f76', linestyle='--', linewidth=1.2, 
                label=f'Median = {tau_stats["median"]:.0f}')
    ax1.axvline(x=selected_tau, color=highlight_color, linestyle='-', linewidth=1.5, 
                label=f'Selected $\\tau$ = {selected_tau}')
    
    stats_text = f'Mean = {tau_stats["mean"]:.1f}\nStd = {tau_stats["std"]:.1f}\nRange = [{int(tau_stats["min"])}, {int(tau_stats["max"])}]'
    ax1.text(0.7, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                       edgecolor='#cccccc'))
    
    ax1.set_xlabel('Time Delay $\\tau$ (samples)')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Time Delay Distribution', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9, bbox_to_anchor=(0.98, 0.75))
    ax1.set_xlim(0, int(tau_stats['max']) + 5)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # --- Panel (b): Embedding Dimension Distribution ---
    ax2 = axes[1]
    dim_unique, dim_counts = np.unique(dim_data, return_counts=True)
    bars = ax2.bar(dim_unique, dim_counts, color=dim_color, edgecolor='white', 
                   alpha=0.8, linewidth=1.2, width=0.6)
    
    total = len(dim_data)
    for bar, count in zip(bars, dim_counts):
        height = bar.get_height()
        pct = count / total * 100
        ax2.annotate(f'{pct:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.axvline(x=dim_stats['median'], color='#595959', linestyle='--', linewidth=1.2,
                label=f'Median = {dim_stats["median"]:.0f}')
    # ax2.legend(loc='upper right')
    # ax2.annotate(f'Selected $d_e$ = {selected_dim}\n(Takens\')', 
    #              xy=(dim_unique.max() + 0.2, max(dim_counts) * 0.65), 
    #              fontsize=9, color=highlight_color, fontweight='bold',
    #              bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F4FD', 
    #                        edgecolor=highlight_color, alpha=0.9))
    
    stats_text2 = f'Mean = {dim_stats["mean"]:.2f}\nFNN threshold = 5%'
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
    accuracy_matrix = np.array(ablation_data['accuracy_matrix'])
    tau_ablation = ablation_data['tau_values']
    de_ablation = ablation_data['de_values']

    colors_list = ["#e29c98", "#fcfdf5", "#8ab8a8"]
    my_gradient = LinearSegmentedColormap.from_list("custom_rdylgn", colors_list, N=256)
    im = ax3.imshow(accuracy_matrix, cmap=my_gradient, aspect='auto', vmin=93, vmax=98)
    
    cbar = fig.colorbar(im, ax=ax3, shrink=0.9, pad=0.02, alpha=0.6)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=10)
    
    ax3.set_xticks(np.arange(len(de_ablation)))
    ax3.set_yticks(np.arange(len(tau_ablation)))
    ax3.set_xticklabels([f'{d}' for d in de_ablation], fontsize=11)
    ax3.set_yticklabels([f'{t}' for t in tau_ablation], fontsize=11)
    
    for i in range(len(tau_ablation)):
        for j in range(len(de_ablation)):
            acc = accuracy_matrix[i, j]
            text_color = 'black'
            
            # --- 新增：给所有格子加上白色细边框 ---
            # facecolor='none' 表示空心，edgecolor='white' 表示白边
            base_rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='none')
            ax3.add_patch(base_rect)
            # ----------------------------------
            
            if acc == accuracy_matrix.max():
                text = f'{acc:.2f}%'
                fontweight = 'bold'
            else:
                text = f'{acc:.2f}%'
                fontweight = 'normal'
            
            ax3.text(j, i, text, ha='center', va='center', 
                    color=text_color, fontsize=12, fontweight=fontweight)
    
    ax3.set_xlabel('Embedding Dimension $d_e$')
    ax3.set_ylabel('Time Delay $\\tau$')
    ax3.set_title('(c) Ablation Study', pad=10)
    
    plt.tight_layout()
    
    fig.savefig(f'{output_prefix}_combined.pdf', format='pdf')
    fig.savefig(f'{output_prefix}_combined.png', format='png', dpi=300)
    print(f"Saved: {output_prefix}_combined.pdf/png")
    
    return fig

def plot_ablation_only(ablation_data, output_prefix):
    """Generate ablation heatmap only (1 panel)."""
    setup_plot_style()
    
    # 单独画图，尺寸设小一点
    fig, ax = plt.subplots(figsize=(6, 5))
    
    accuracy_matrix = np.array(ablation_data['accuracy_matrix'])
    tau_ablation = ablation_data['tau_values']
    de_ablation = ablation_data['de_values']

    # --- 使用您选定的配色方案 1 (清透薄荷风) ---
    colors_list = ["#e29c98", "#fcfdf5", "#8ab8a8"]
    my_gradient = LinearSegmentedColormap.from_list("custom_rdylgn", colors_list, N=256)
    
    im = ax.imshow(accuracy_matrix, cmap=my_gradient, aspect='auto', vmin=93, vmax=98)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.04, alpha=0.6)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=10)
    
    ax.set_xticks(np.arange(len(de_ablation)))
    ax.set_yticks(np.arange(len(tau_ablation)))
    ax.set_xticklabels([f'{d}' for d in de_ablation], fontsize=11)
    ax.set_yticklabels([f'{t}' for t in tau_ablation], fontsize=11)
    
    for i in range(len(tau_ablation)):
        for j in range(len(de_ablation)):
            acc = accuracy_matrix[i, j]
            text_color = 'black'
            
            # 白色细边框
            base_rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='none')
            ax.add_patch(base_rect)

            if acc == accuracy_matrix.max():
                text = f'{acc:.2f}%'
                fontweight = 'bold'
            else:
                text = f'{acc:.2f}%'
                fontweight = 'normal'
            
            ax.text(j, i, text, ha='center', va='center', 
                    color=text_color, fontsize=12, fontweight=fontweight)
    
    ax.set_xlabel('Embedding Dimension $d_e$')
    ax.set_ylabel('Time Delay $\\tau$')
    ax.set_title('Ablation Study Results', pad=10)
    
    plt.tight_layout()
    
    # 保存为 _heatmap.pdf/.png
    fig.savefig(f'{output_prefix}_heatmap.pdf', format='pdf')
    fig.savefig(f'{output_prefix}_heatmap.png', format='png', dpi=300)
    print(f"Saved: {output_prefix}_heatmap.pdf/png")
    
    return fig
    
def main():
    parser = argparse.ArgumentParser(description='Plot embedding parameters from data')
    parser.add_argument('--json', type=str, required=True,
                        help='JSON file with raw data (tau_values, dim_values)')
    parser.add_argument('--output', type=str, default='embedding_params',
                        help='Output file prefix')
    parser.add_argument('--selected_tau', type=int, default=12,
                        help='Selected tau value to highlight')
    parser.add_argument('--selected_dim', type=int, default=10,
                        help='Selected dimension value to highlight')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined plot with ablation heatmap')
    
    args = parser.parse_args()
    
    # Load data from JSON
    print(f"Loading data from: {args.json}")
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    tau_values = np.array(data['tau_values'])
    dim_values = np.array(data['dim_values'])
    tau_stats = data['tau_stats']
    dim_stats = data['dim_stats']
    
    print(f"Loaded {len(tau_values)} tau values and {len(dim_values)} dim values")
    print(f"Tau stats: mean={tau_stats['mean']:.2f}, median={tau_stats['median']:.0f}")
    print(f"Dim stats: mean={dim_stats['mean']:.2f}, median={dim_stats['median']:.0f}")
    
    # Generate distribution plot
    plot_distribution_only(tau_values, dim_values, tau_stats, dim_stats,
                           args.selected_tau, args.selected_dim, args.output)
    
    # Generate combined plot if requested
    if args.combined:
        # Ablation study data (hardcoded from experiments)
        ablation_data = {
            'accuracy_matrix': [
                [95.76, 94.00],   # tau=10: d_e=4, d_e=10
                [95.76, 97.46],   # tau=12: d_e=4, d_e=10
                [97.46, 97.46],   # tau=30: d_e=4, d_e=10
            ],
            'tau_values': [10, 12, 30],
            'de_values': [4, 10]
        }
        plot_combined_with_ablation(tau_values, dim_values, tau_stats, dim_stats,
                                     args.selected_tau, args.selected_dim,
                                     ablation_data, args.output)
        plot_ablation_only(ablation_data, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
