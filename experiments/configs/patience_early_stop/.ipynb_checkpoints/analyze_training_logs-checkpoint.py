#!/usr/bin/env python3
"""
Training Log Analysis and Patience Optimization Script

This script analyzes training logs to:
1. Visualize training/validation curves (Loss, Accuracy, EER)
2. Simulate early stopping with different patience values
3. Recommend optimal patience settings
4. Generate analysis report

Usage:
    python analyze_training_logs.py --log_file experiment_state.json
    python analyze_training_logs.py --checkpoint latest.pth
    python analyze_training_logs.py --log_dir ./outputs/chaotic/experiments/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from pathlib import Path

current_dir = Path(__file__).parent
font_path = current_dir / '..' / 'Helvetica.ttf'
if font_path.exists():
    fm.fontManager.addfont(str(font_path))

# Set academic plot style
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Helvetica', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.0,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Try to import torch for checkpoint loading
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Cannot load .pth checkpoints.")


@dataclass
class TrainingHistory:
    """Container for training history data."""
    epochs: List[int]
    train_losses: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    val_eer: Optional[List[float]] = None
    learning_rates: Optional[List[float]] = None
    
    @property
    def num_epochs(self) -> int:
        return len(self.epochs)


class TrainingLogParser:
    """Parse training logs from various formats."""
    
    @staticmethod
    def load_from_json(filepath: str) -> TrainingHistory:
        """Load training history from experiment_state.json or training_history.json."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"[DEBUG] JSON top-level keys: {list(data.keys())}")
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_eer = None
        
        # Try multiple possible JSON structures
        
        # Structure 1: Direct keys (experiment_state.json format)
        if 'train_losses' in data:
            train_losses = data.get('train_losses', [])
            val_metrics = data.get('val_metrics', [])
            if val_metrics:
                val_losses = [m.get('loss', 0) if isinstance(m, dict) else 0 for m in val_metrics]
                val_accuracies = [m.get('accuracy', 0) if isinstance(m, dict) else 0 for m in val_metrics]
                if isinstance(val_metrics[0], dict) and 'eer' in val_metrics[0]:
                    val_eer = [m.get('eer', 0) for m in val_metrics]
        
        # Structure 2: Nested under 'history' key
        elif 'history' in data:
            history = data['history']
            train_losses = history.get('train_loss', history.get('loss', []))
            val_losses = history.get('val_loss', [])
            val_accuracies = history.get('val_accuracy', history.get('val_acc', []))
            val_eer = history.get('val_eer', None)
        
        # Structure 3: Nested under 'training_history' key
        elif 'training_history' in data:
            history = data['training_history']
            train_losses = history.get('train_losses', history.get('train_loss', []))
            val_losses = history.get('val_losses', history.get('val_loss', []))
            val_accuracies = history.get('val_accuracies', history.get('val_accuracy', []))
            val_eer = history.get('val_eer', None)
        
        # Structure 4: Epoch-based list format
        elif 'epochs' in data and isinstance(data['epochs'], list):
            for epoch_data in data['epochs']:
                if isinstance(epoch_data, dict):
                    train_losses.append(epoch_data.get('train_loss', 0))
                    val_losses.append(epoch_data.get('val_loss', 0))
                    val_accuracies.append(epoch_data.get('val_accuracy', epoch_data.get('val_acc', 0)))
        
        # Structure 5: Results format from chaotic_training_results.json
        elif 'results' in data:
            print("[DEBUG] Found 'results' key - this appears to be a summary file, not training history")
            print("[DEBUG] Looking for experiment_state.json or checkpoint files instead")
            raise ValueError(
                "This file contains experiment results summary, not training history.\n"
                "Please use one of these instead:\n"
                "  - experiment_state.json (in experiment output directory)\n"
                "  - latest.pth or checkpoint_epoch_*.pth (checkpoint files)\n"
                "  - TensorBoard logs directory"
            )
        
        # Structure 6: Direct loss/accuracy arrays
        elif 'loss' in data or 'train_loss' in data:
            train_losses = data.get('train_loss', data.get('loss', []))
            val_losses = data.get('val_loss', [])
            val_accuracies = data.get('val_accuracy', data.get('val_acc', data.get('accuracy', [])))
        
        print(f"[DEBUG] Parsed data - train_losses: {len(train_losses)}, val_losses: {len(val_losses)}, val_accuracies: {len(val_accuracies)}")
        
        # Validate we have data
        if not train_losses and not val_losses and not val_accuracies:
            print(f"[DEBUG] Full JSON structure (first 2000 chars):")
            print(json.dumps(data, indent=2)[:2000])
            raise ValueError(
                "Could not find training history data in JSON file.\n"
                "Expected keys: 'train_losses', 'val_metrics', 'history', or 'epochs'\n"
                f"Found keys: {list(data.keys())}"
            )
        
        # Use the longest available list to determine epochs
        max_len = max(len(train_losses), len(val_losses), len(val_accuracies))
        epochs = list(range(max_len))
        
        # Pad shorter lists if needed
        if len(train_losses) < max_len:
            train_losses.extend([train_losses[-1] if train_losses else 0] * (max_len - len(train_losses)))
        if len(val_losses) < max_len:
            val_losses.extend([val_losses[-1] if val_losses else 0] * (max_len - len(val_losses)))
        if len(val_accuracies) < max_len:
            val_accuracies.extend([val_accuracies[-1] if val_accuracies else 0] * (max_len - len(val_accuracies)))
        
        return TrainingHistory(
            epochs=epochs,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            val_eer=val_eer
        )
    
    @staticmethod
    def load_from_checkpoint(filepath: str) -> TrainingHistory:
        """Load training history from PyTorch checkpoint."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required to load .pth files")
        
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        if 'experiment_state' in checkpoint:
            state = checkpoint['experiment_state']
        else:
            state = checkpoint
        
        train_losses = state.get('train_losses', [])
        val_metrics = state.get('val_metrics', [])
        
        epochs = list(range(len(train_losses)))
        val_losses = [m.get('loss', 0) if isinstance(m, dict) else 0 for m in val_metrics]
        val_accuracies = [m.get('accuracy', 0) if isinstance(m, dict) else 0 for m in val_metrics]
        
        val_eer = None
        if val_metrics and isinstance(val_metrics[0], dict) and 'eer' in val_metrics[0]:
            val_eer = [m.get('eer', 0) for m in val_metrics]
        
        return TrainingHistory(
            epochs=epochs,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            val_eer=val_eer
        )
    
    @staticmethod
    def load_from_tensorboard(log_dir: str) -> TrainingHistory:
        """Load training history from TensorBoard logs."""
        try:
            from tensorboard.backend.event_processing import event_accumulator
        except ImportError:
            raise RuntimeError("tensorboard package required for TensorBoard logs")
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Extract scalar data
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        if 'Train/loss' in ea.Tags()['scalars']:
            train_losses = [s.value for s in ea.Scalars('Train/loss')]
        if 'Validation/loss' in ea.Tags()['scalars']:
            val_losses = [s.value for s in ea.Scalars('Validation/loss')]
        if 'Validation/accuracy' in ea.Tags()['scalars']:
            val_accuracies = [s.value for s in ea.Scalars('Validation/accuracy')]
        
        epochs = list(range(len(train_losses)))
        
        return TrainingHistory(
            epochs=epochs,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies
        )
    
    @classmethod
    def auto_load(cls, path: str) -> TrainingHistory:
        """Automatically detect format and load training history."""
        path = Path(path)
        
        if path.suffix == '.json':
            return cls.load_from_json(str(path))
        elif path.suffix == '.pth':
            return cls.load_from_checkpoint(str(path))
        elif path.is_dir():
            # Try to find experiment_state.json or latest.pth
            json_file = path / 'experiment_state.json'
            pth_file = path / 'checkpoints' / 'latest.pth'
            
            if json_file.exists():
                return cls.load_from_json(str(json_file))
            elif pth_file.exists():
                return cls.load_from_checkpoint(str(pth_file))
            else:
                # Try TensorBoard
                return cls.load_from_tensorboard(str(path))
        else:
            raise ValueError(f"Unknown file format: {path}")


class EarlyStoppingSimulator:
    """Simulate early stopping with different patience values."""
    
    def __init__(self, history: TrainingHistory, metric: str = 'accuracy'):
        """
        Initialize simulator.
        
        Args:
            history: Training history data
            metric: Metric to monitor ('accuracy', 'loss', 'eer')
        """
        self.history = history
        self.metric = metric
        
        # Get metric values
        if metric == 'accuracy':
            self.values = history.val_accuracies
            self.higher_is_better = True
        elif metric == 'loss':
            self.values = history.val_losses
            self.higher_is_better = False
        elif metric == 'eer':
            if history.val_eer is None:
                raise ValueError("EER not available in training history")
            self.values = history.val_eer
            self.higher_is_better = False
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def simulate(self, patience: int) -> Dict[str, Any]:
        """
        Simulate early stopping with given patience.
        
        Args:
            patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary with simulation results
        """
        best_value = self.values[0]
        best_epoch = 0
        epochs_without_improvement = 0
        stop_epoch = len(self.values) - 1
        
        for epoch, value in enumerate(self.values):
            if self.higher_is_better:
                is_improvement = value > best_value
            else:
                is_improvement = value < best_value
            
            if is_improvement:
                best_value = value
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                stop_epoch = epoch
                break
        
        # Calculate final metric at stop epoch
        final_value = self.values[stop_epoch]
        
        # Calculate potential improvement if trained longer
        actual_best = max(self.values) if self.higher_is_better else min(self.values)
        actual_best_epoch = (
            self.values.index(max(self.values)) if self.higher_is_better 
            else self.values.index(min(self.values))
        )
        
        return {
            'patience': patience,
            'stop_epoch': stop_epoch,
            'best_epoch': best_epoch,
            'best_value': best_value,
            'final_value': final_value,
            'actual_best_value': actual_best,
            'actual_best_epoch': actual_best_epoch,
            'missed_improvement': (
                (actual_best - best_value) if self.higher_is_better 
                else (best_value - actual_best)
            ),
            'epochs_saved': len(self.values) - 1 - stop_epoch,
            'stopped_early': stop_epoch < len(self.values) - 1
        }
    
    def analyze_patience_range(
        self, 
        patience_values: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple patience values.
        
        Args:
            patience_values: List of patience values to test
            
        Returns:
            List of simulation results for each patience value
        """
        if patience_values is None:
            total_epochs = len(self.values)
            patience_values = [
                5, 10, 15, 20, 25, 30, 40, 50,
                int(total_epochs * 0.1),
                int(total_epochs * 0.2),
                int(total_epochs * 0.3),
                int(total_epochs * 0.5)
            ]
            patience_values = sorted(set(p for p in patience_values if p > 0))
        
        results = []
        for patience in patience_values:
            results.append(self.simulate(patience))
        
        return results


class TrainingAnalyzer:
    """Comprehensive training analysis and visualization."""
    
    def __init__(self, history: TrainingHistory):
        self.history = history
    
    def find_overfitting_point(self) -> Tuple[int, float]:
        """
        Find the epoch where overfitting begins.
        
        Returns:
            Tuple of (epoch, gap between train and val loss)
        """
        train_losses = np.array(self.history.train_losses)
        val_losses = np.array(self.history.val_losses)
        
        # Calculate gap
        gaps = val_losses - train_losses
        
        # Find where gap starts consistently increasing
        window = 5
        if len(gaps) < window * 2:
            return len(gaps) - 1, gaps[-1]
        
        # Calculate moving average of gap increase
        gap_changes = np.diff(gaps)
        
        for i in range(len(gap_changes) - window):
            if np.mean(gap_changes[i:i+window]) > 0:
                return i, gaps[i]
        
        return len(gaps) - 1, gaps[-1]
    
    def find_plateau_regions(
        self, 
        values: List[float], 
        threshold: float = 0.001,
        min_length: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Find plateau regions in metric curve.
        
        Args:
            values: Metric values
            threshold: Change threshold to consider as plateau
            min_length: Minimum length of plateau region
            
        Returns:
            List of (start_epoch, end_epoch) tuples
        """
        values = np.array(values)
        plateaus = []
        
        i = 0
        while i < len(values) - min_length:
            # Check if region is plateau
            region = values[i:i+min_length]
            if np.std(region) < threshold:
                # Extend plateau
                end = i + min_length
                while end < len(values) and abs(values[end] - np.mean(region)) < threshold:
                    end += 1
                plateaus.append((i, end - 1))
                i = end
            else:
                i += 1
        
        return plateaus
    
    def calculate_convergence_rate(self) -> Dict[str, float]:
        """Calculate convergence rate metrics."""
        val_accs = np.array(self.history.val_accuracies)
        
        if len(val_accs) == 0:
            return {
                'final_accuracy': 0.0,
                'best_accuracy': 0.0,
                'best_epoch': 0,
                'epochs_to_90pct': 0,
                'epochs_to_95pct': 0,
                'convergence_efficiency': 0.0
            }
        
        # Final accuracy
        final_acc = val_accs[-1]
        best_acc = np.max(val_accs)
        best_epoch = int(np.argmax(val_accs))
        
        # Epochs to reach 90% of best
        target = 0.9 * best_acc
        epochs_to_90 = np.where(val_accs >= target)[0]
        epochs_to_90 = int(epochs_to_90[0]) if len(epochs_to_90) > 0 else len(val_accs)
        
        # Epochs to reach 95% of best
        target = 0.95 * best_acc
        epochs_to_95 = np.where(val_accs >= target)[0]
        epochs_to_95 = int(epochs_to_95[0]) if len(epochs_to_95) > 0 else len(val_accs)
        
        return {
            'final_accuracy': float(final_acc),
            'best_accuracy': float(best_acc),
            'best_epoch': best_epoch,
            'epochs_to_90pct': epochs_to_90,
            'epochs_to_95pct': epochs_to_95,
            'convergence_efficiency': best_epoch / len(val_accs) if len(val_accs) > 0 else 0.0
        }
    
    def recommend_patience(self, safety_margin: int = 5) -> Dict[str, Any]:
        """
        Recommend optimal patience value.
        
        Args:
            safety_margin: Additional epochs to add for safety
            
        Returns:
            Recommendation dictionary
        """
        val_accs = np.array(self.history.val_accuracies)
        best_epoch = np.argmax(val_accs)
        total_epochs = len(val_accs)
        
        # Find fluctuation level
        fluctuation = np.std(np.diff(val_accs))
        
        # Find plateau regions
        plateaus = self.find_plateau_regions(val_accs.tolist())
        max_plateau_length = max(
            (end - start for start, end in plateaus), 
            default=5
        )
        
        # Calculate recommended patience
        base_patience = max(10, max_plateau_length + safety_margin)
        
        # Adjust for fluctuation
        if fluctuation > 0.02:
            base_patience = int(base_patience * 1.5)
        
        # Ensure we wouldn't miss the best epoch
        min_required = total_epochs - best_epoch
        
        # Final recommendation
        recommended = max(base_patience, min(min_required + safety_margin, total_epochs // 2))
        
        return {
            'recommended_patience': recommended,
            'reasoning': {
                'best_epoch': best_epoch,
                'total_epochs': total_epochs,
                'max_plateau_length': max_plateau_length,
                'fluctuation_level': fluctuation,
                'safety_margin': safety_margin
            },
            'alternative_values': {
                'conservative': min(recommended + 10, total_epochs),
                'aggressive': max(10, recommended - 5)
            }
        }


class TrainingVisualizer:
    """Visualize training curves and analysis results."""
    
    def __init__(self, history: TrainingHistory, figsize: Tuple[int, int] = (14, 10)):
        self.history = history
        self.figsize = figsize
        self.colors = {
            'train': '#2E86AB',
            'val': '#E94F37',
            'best': '#6b3e3c',
            'stop': '#F4B942'
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot comprehensive training curves."""
        num_plots = 2 + (1 if self.history.val_eer else 0)
        fig, axes = plt.subplots(1, num_plots, figsize=self.figsize)
        
        epochs = self.history.epochs
        
        # Plot 1: Loss curves
        ax1 = axes[0]
        ax1.plot(epochs, self.history.train_losses, 
                 label='Train Loss', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history.val_losses, 
                 label='Val Loss', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_loss_epoch = np.argmin(self.history.val_losses)
        ax1.axvline(x=best_loss_epoch, color=self.colors['best'], 
                    linestyle='--', alpha=0.7, label=f'Best Loss (Epoch {best_loss_epoch})')
        
        # Plot 2: Accuracy curve
        ax2 = axes[1]
        ax2.plot(epochs, self.history.val_accuracies, 
                 label='Val Accuracy', color=self.colors['val'], linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_acc_epoch = np.argmax(self.history.val_accuracies)
        ax2.axvline(x=best_acc_epoch, color=self.colors['best'], 
                    linestyle='--', alpha=0.7)
        ax2.scatter([best_acc_epoch], [self.history.val_accuracies[best_acc_epoch]], 
                    color=self.colors['best'], s=100, zorder=5,
                    label=f'Best Acc: {self.history.val_accuracies[best_acc_epoch]:.4f}')
        ax2.legend(fontsize=10)
        
        # Plot 3: EER if available
        if self.history.val_eer:
            ax3 = axes[2]
            ax3.plot(epochs, self.history.val_eer, 
                     label='Val EER', color=self.colors['val'], linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('EER (%)', fontsize=12)
            ax3.set_title('Validation EER', fontsize=14)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Mark best epoch
            best_eer_epoch = np.argmin(self.history.val_eer)
            ax3.axvline(x=best_eer_epoch, color=self.colors['best'], 
                        linestyle='--', alpha=0.7)
            ax3.scatter([best_eer_epoch], [self.history.val_eer[best_eer_epoch]], 
                        color=self.colors['best'], s=100, zorder=5,
                        label=f'Best EER: {self.history.val_eer[best_eer_epoch]:.4f}')
            ax3.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_patience_analysis(
        self, 
        simulation_results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """Plot early stopping simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        patience_values = [r['patience'] for r in simulation_results]
        stop_epochs = [r['stop_epoch'] for r in simulation_results]
        best_values = [r['best_value'] for r in simulation_results]
        missed_improvements = [r['missed_improvement'] for r in simulation_results]
        epochs_saved = [r['epochs_saved'] for r in simulation_results]
        
        # Plot 1: Stop epoch vs Patience
        ax1 = axes[0, 0]
        ax1.bar(range(len(patience_values)), stop_epochs, color=self.colors['train'])
        ax1.set_xticks(range(len(patience_values)))
        ax1.set_xticklabels(patience_values, rotation=45)
        ax1.set_xlabel('Patience', fontsize=12)
        ax1.set_ylabel('Stop Epoch', fontsize=12)
        ax1.set_title('Training Stop Epoch by Patience', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Best metric achieved vs Patience
        ax2 = axes[0, 1]
        ax2.plot(patience_values, best_values, 
                 'o-', color=self.colors['val'], linewidth=2, markersize=8)
        ax2.set_xlabel('Patience', fontsize=12)
        ax2.set_ylabel('Best Metric Value', fontsize=12)
        ax2.set_title('Best Metric Achieved by Patience', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Highlight optimal
        optimal_idx = np.argmax(best_values)
        ax2.scatter([patience_values[optimal_idx]], [best_values[optimal_idx]], 
                    color=self.colors['best'], s=150, zorder=5,
                    label=f'Optimal: P={patience_values[optimal_idx]}')
        ax2.legend(fontsize=10, loc='upper left')
        
        # Plot 3: Missed improvement vs Patience
        ax3 = axes[1, 0]
        colors = [self.colors['best'] if m < 0.01 else self.colors['stop'] 
                  for m in missed_improvements]
        ax3.bar(range(len(patience_values)), missed_improvements, color=colors)
        ax3.set_xticks(range(len(patience_values)))
        ax3.set_xticklabels(patience_values, rotation=45)
        ax3.set_xlabel('Patience', fontsize=12)
        ax3.set_ylabel('Missed Improvement', fontsize=12)
        ax3.set_title('Potential Improvement Missed', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Epochs saved vs Patience
        ax4 = axes[1, 1]
        ax4.bar(range(len(patience_values)), epochs_saved, color=self.colors['train'])
        ax4.set_xticks(range(len(patience_values)))
        ax4.set_xticklabels(patience_values, rotation=45)
        ax4.set_xlabel('Patience', fontsize=12)
        ax4.set_ylabel('Epochs Saved', fontsize=12)
        ax4.set_title('Training Time Saved (Epochs)', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Patience analysis saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_early_stopping_overlay(
        self,
        simulation_results: List[Dict[str, Any]],
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ):
        """Plot training curve with early stopping points overlaid."""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        epochs = self.history.epochs
        
        if metric == 'accuracy':
            values = self.history.val_accuracies
            ylabel = 'Validation Accuracy'
        elif metric == 'loss':
            values = self.history.val_losses
            ylabel = 'Validation Loss'
        else:
            values = self.history.val_eer
            ylabel = 'Validation EER'
        
        # Plot main curve
        ax.plot(epochs, values, color='#c45a65', 
                linewidth=2, label='Val ' + metric.capitalize())
        
        # Mark early stopping points for different patience values
        colors = plt.cm.viridis(np.linspace(0, 1, len(simulation_results)))
        
        for result, color in zip(simulation_results, colors):
            stop_epoch = result['stop_epoch']
            patience = result['patience']
            
            if result['stopped_early']:
                ax.axvline(x=stop_epoch, color='#63a1f2', linestyle='--', 
                           alpha=0.6, label=f'P={patience} (Stop: {stop_epoch})')
                ax.scatter([stop_epoch], [values[stop_epoch]], 
                           color='#63a1f2', s=80, zorder=5)
        
        # Mark actual best
        if metric == 'accuracy':
            best_epoch = np.argmax(values)
        else:
            best_epoch = np.argmin(values)
        
        ax.axvline(x=best_epoch, color='#5a9239', 
                   linewidth=1.5, linestyle='-', alpha=0.8)
        ax.scatter([best_epoch], [values[best_epoch]], 
                   color='#5a9239', s=150, zorder=6,
                   marker='*', label=f'Actual Best (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title('Early Stopping Comparison', fontsize=14)
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Early stopping overlay saved to: {save_path}")
        
        plt.show()
        return fig


def generate_report(
    history: TrainingHistory,
    simulation_results: List[Dict[str, Any]],
    recommendation: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Generate comprehensive analysis report."""
    
    lines = [
        "=" * 70,
        "TRAINING LOG ANALYSIS REPORT",
        "=" * 70,
        "",
        "1. TRAINING OVERVIEW",
        "-" * 40,
        f"   Total epochs trained: {len(history.epochs)}",
        f"   Final train loss: {history.train_losses[-1]:.6f}",
        f"   Final val loss: {history.val_losses[-1]:.6f}",
        f"   Best val accuracy: {max(history.val_accuracies):.4f} (Epoch {np.argmax(history.val_accuracies)})",
        "",
    ]
    
    if history.val_eer:
        lines.append(f"   Best val EER: {min(history.val_eer):.4f} (Epoch {np.argmin(history.val_eer)})")
        lines.append("")
    
    lines.extend([
        "2. PATIENCE SIMULATION RESULTS",
        "-" * 40,
        f"   {'Patience':<10} {'Stop Epoch':<12} {'Best Value':<12} {'Missed':<12} {'Saved':<10}",
        "   " + "-" * 56,
    ])
    
    for r in simulation_results:
        lines.append(
            f"   {r['patience']:<10} {r['stop_epoch']:<12} {r['best_value']:<12.4f} "
            f"{r['missed_improvement']:<12.4f} {r['epochs_saved']:<10}"
        )
    
    lines.extend([
        "",
        "3. RECOMMENDATION",
        "-" * 40,
        f"   Recommended patience: {recommendation['recommended_patience']}",
        f"   Conservative option: {recommendation['alternative_values']['conservative']}",
        f"   Aggressive option: {recommendation['alternative_values']['aggressive']}",
        "",
        "   Reasoning:",
        f"   - Best epoch observed at: {recommendation['reasoning']['best_epoch']}",
        f"   - Maximum plateau length: {recommendation['reasoning']['max_plateau_length']}",
        f"   - Metric fluctuation level: {recommendation['reasoning']['fluctuation_level']:.4f}",
        "",
        "4. INSIGHTS & SUGGESTIONS",
        "-" * 40,
    ])
    
    # Generate insights
    best_patience_result = max(simulation_results, key=lambda x: x['best_value'])
    if best_patience_result['patience'] < len(history.epochs) // 2:
        lines.append("   [!] Model converges early - consider smaller patience for efficiency")
    else:
        lines.append("   [!] Model benefits from longer training - use large patience or disable early stopping")
    
    if recommendation['reasoning']['fluctuation_level'] > 0.02:
        lines.append("   [!] High metric fluctuation detected - use larger patience to avoid premature stopping")
    
    # Check for overfitting
    train_final = history.train_losses[-1]
    val_final = history.val_losses[-1]
    if val_final > train_final * 1.5:
        lines.append("   [!] Overfitting detected - consider regularization or smaller patience")
    
    lines.extend([
        "",
        "=" * 70,
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training logs and optimize patience settings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    parser.add_argument('--log_file', type=str, help='Path to experiment_state.json')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint .pth file')
    parser.add_argument('--log_dir', type=str, help='Path to experiment directory')
    
    # Analysis options
    parser.add_argument('--metric', type=str, default='accuracy',
                        choices=['accuracy', 'loss', 'eer'],
                        help='Metric to monitor for early stopping')
    parser.add_argument('--patience_values', type=int, nargs='+',
                        help='Specific patience values to test')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                        help='Directory for output files')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Determine input source
    input_path = args.log_file or args.checkpoint or args.log_dir
    if not input_path:
        parser.error("Must specify --log_file, --checkpoint, or --log_dir")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training history
    print("Loading training history...")
    try:
        history = TrainingLogParser.auto_load(input_path)
    except ValueError as e:
        print(f"\nError loading training history: {e}")
        sys.exit(1)
    
    print(f"Loaded {history.num_epochs} epochs of training data")
    
    # Validate we have enough data
    if history.num_epochs == 0:
        print("\nError: No training data found in the file.")
        print("Please check that the file contains training history data.")
        print("\nSupported formats:")
        print("  1. experiment_state.json with 'train_losses' and 'val_metrics' keys")
        print("  2. PyTorch checkpoint (.pth) with 'experiment_state' key")
        print("  3. TensorBoard log directory")
        sys.exit(1)
    
    if history.num_epochs < 3:
        print(f"\nWarning: Only {history.num_epochs} epochs found. Need at least 3 epochs for meaningful analysis.")
    
    # Run analysis
    print("\nRunning analysis...")
    analyzer = TrainingAnalyzer(history)
    
    # Get convergence metrics
    convergence = analyzer.calculate_convergence_rate()
    print(f"\nConvergence Analysis:")
    print(f"  Best accuracy: {convergence['best_accuracy']:.4f} at epoch {convergence['best_epoch']}")
    print(f"  Epochs to 90% of best: {convergence['epochs_to_90pct']}")
    print(f"  Epochs to 95% of best: {convergence['epochs_to_95pct']}")
    
    # Simulate early stopping
    print("\nSimulating early stopping with different patience values...")
    simulator = EarlyStoppingSimulator(history, metric=args.metric)
    
    patience_values = args.patience_values
    simulation_results = simulator.analyze_patience_range(patience_values)
    
    # Get recommendation
    recommendation = analyzer.recommend_patience()
    print(f"\nRecommended patience: {recommendation['recommended_patience']}")
    
    # Generate report
    report = generate_report(
        history, 
        simulation_results, 
        recommendation,
        output_path=str(output_dir / 'analysis_report.txt')
    )
    print("\n" + report)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")
        visualizer = TrainingVisualizer(history)
        
        # Training curves
        visualizer.plot_training_curves(
            save_path=str(output_dir / 'training_curves.pdf')
        )
        
        # Patience analysis
        visualizer.plot_patience_analysis(
            simulation_results,
            save_path=str(output_dir / 'patience_analysis.pdf')
        )
        
        # Early stopping overlay
        visualizer.plot_early_stopping_overlay(
            simulation_results,
            metric=args.metric,
            save_path=str(output_dir / 'early_stopping_overlay.pdf')
        )
    
    # Save simulation results
    results_file = output_dir / 'simulation_results.json'
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable({
            'simulation_results': simulation_results,
            'recommendation': recommendation,
            'convergence_metrics': convergence
        }), f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()