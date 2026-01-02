#!/usr/bin/env python3
"""
Experiment 5: Feature Visualization and Analysis

This script visualizes and analyzes the feature representations learned by
different models using:
- t-SNE visualization
- UMAP visualization
- Silhouette score (cluster quality)
- Inter/Intra class distance ratio
- Feature distribution analysis

Usage:
    python scripts/run_experiment5_visualization.py --all
    python scripts/run_experiment5_visualization.py --model mel_mlp
    python scripts/run_experiment5_visualization.py --quick
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# Setup Imports
# ============================================================
def setup_imports():
    """Setup Python path for project imports."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    paths = [
        str(project_root),
        str(project_root / 'experiments'),
        str(project_root / 'models'),
        str(project_root / 'features'),
        str(project_root / 'data'),
        str(project_root / 'utils'),
        str(project_root / 'evaluation'),
        str(project_root / 'core')
    ]
    
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root

PROJECT_ROOT = setup_imports()

# Import project modules
from dataset_loader import create_speaker_dataloaders
from hybrid_models import TraditionalMLPBaseline
from reproducibility import set_seed

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not available, using t-SNE only")


# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    # Data settings
    'data_dir': '/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2',
    'sample_rate': 16000,
    'max_length': 3.0,
    'batch_size': 32,
    
    # Visualization settings
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    'umap_n_neighbors': 15,
    'umap_min_dist': 0.1,
    
    # Analysis settings
    'num_samples_for_viz': 500,  # Max samples for visualization
    
    # Output settings
    'output_dir': './outputs/experiment5_visualization',
    'seed': 42,
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 150,
}


# ============================================================
# Logger Setup
# ============================================================
def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# Feature Extractor
# ============================================================
class FeatureExtractor:
    """Extract intermediate features from models."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.features = {}
        self.hooks = []
    
    def _get_activation(self, name: str):
        """Hook function to capture activations."""
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.features[name] = output[0].detach().cpu()
            else:
                self.features[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self, layer_names: List[str] = None):
        """Register forward hooks on specified layers."""
        self.hooks = []
        
        # Default: hook on the layer before classifier
        for name, module in self.model.named_modules():
            # Look for common embedding layer patterns
            if any(key in name.lower() for key in ['embedding', 'features', 'encoder', 'backbone']):
                hook = module.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)
            # Hook on the last layer before classifier
            elif 'classifier' in name.lower() and '0' in name:
                # Get the layer just before classifier
                pass
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_features(
        self, 
        dataloader: DataLoader,
        max_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataloader.
        
        Returns:
            features: (N, D) feature matrix
            labels: (N,) label array
        """
        self.model.eval()
        self.model.to(self.device)
        
        all_features = []
        all_labels = []
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", leave=False):
                if len(batch) == 3:
                    audio, labels, _ = batch
                else:
                    audio, labels = batch[0], batch[1]
                
                audio = audio.to(self.device)
                
                # Forward pass
                outputs = self.model(audio)
                
                # Get features (before final classification)
                # Try to get intermediate features from model
                if hasattr(self.model, 'get_embeddings'):
                    features = self.model.get_embeddings(audio)
                elif hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(audio)
                else:
                    # Use output logits as features
                    if isinstance(outputs, tuple):
                        features = outputs[0]
                    else:
                        features = outputs
                
                # Flatten if needed
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
                total_samples += audio.size(0)
                if max_samples and total_samples >= max_samples:
                    break
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        if max_samples and len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        return features, labels


# ============================================================
# Visualization Functions
# ============================================================
class FeatureVisualizer:
    """Visualize feature representations."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_tsne(
        self, 
        features: np.ndarray,
        perplexity: int = 30,
        n_iter: int = 1000
    ) -> np.ndarray:
        """Compute t-SNE embedding."""
        if not HAS_SKLEARN:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(features) - 1),
            n_iter=n_iter,
            random_state=self.config['seed'],
            init='pca'
        )
        
        embedding = tsne.fit_transform(features_scaled)
        return embedding
    
    def compute_umap(
        self,
        features: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> np.ndarray:
        """Compute UMAP embedding."""
        if not HAS_UMAP:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(features) - 1),
            min_dist=min_dist,
            random_state=self.config['seed']
        )
        
        embedding = reducer.fit_transform(features_scaled)
        return embedding
    
    def plot_embedding(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str,
        filename: str
    ):
        """Plot 2D embedding with color-coded labels."""
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique labels
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Create colormap
        if num_classes <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        elif num_classes <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        
        # Plot each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=f'Speaker {label}',
                alpha=0.6,
                s=30
            )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        # Add legend (only if not too many classes)
        if num_classes <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_class_separation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        title: str,
        filename: str
    ):
        """Plot inter/intra class distance distribution."""
        if not HAS_MATPLOTLIB:
            return
        
        unique_labels = np.unique(labels)
        
        intra_distances = []
        inter_distances = []
        
        # Compute pairwise distances (sample for efficiency)
        num_samples = min(100, len(features))
        indices = np.random.choice(len(features), num_samples, replace=False)
        
        for i in indices:
            for j in indices:
                if i >= j:
                    continue
                
                dist = np.linalg.norm(features[i] - features[j])
                
                if labels[i] == labels[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)
        
        # Plot distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(intra_distances, bins=50, alpha=0.5, label='Intra-class', color='blue', density=True)
        ax.hist(inter_distances, bins=50, alpha=0.5, label='Inter-class', color='red', density=True)
        
        ax.axvline(np.mean(intra_distances), color='blue', linestyle='--', 
                   label=f'Intra mean: {np.mean(intra_distances):.2f}')
        ax.axvline(np.mean(inter_distances), color='red', linestyle='--',
                   label=f'Inter mean: {np.mean(inter_distances):.2f}')
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.config['plot_dpi'])
        plt.close()
        
        return str(filepath)


# ============================================================
# Feature Analysis
# ============================================================
class FeatureAnalyzer:
    """Analyze feature quality metrics."""
    
    @staticmethod
    def compute_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score (cluster quality)."""
        if not HAS_SKLEARN:
            return 0.0
        
        try:
            # Standardize
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            score = silhouette_score(features_scaled, labels)
            return float(score)
        except Exception as e:
            print(f"Silhouette score error: {e}")
            return 0.0
    
    @staticmethod
    def compute_calinski_harabasz(features: np.ndarray, labels: np.ndarray) -> float:
        """Compute Calinski-Harabasz index (higher is better)."""
        if not HAS_SKLEARN:
            return 0.0
        
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            score = calinski_harabasz_score(features_scaled, labels)
            return float(score)
        except Exception as e:
            print(f"CH score error: {e}")
            return 0.0
    
    @staticmethod
    def compute_davies_bouldin(features: np.ndarray, labels: np.ndarray) -> float:
        """Compute Davies-Bouldin index (lower is better)."""
        if not HAS_SKLEARN:
            return float('inf')
        
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            score = davies_bouldin_score(features_scaled, labels)
            return float(score)
        except Exception as e:
            print(f"DB score error: {e}")
            return float('inf')
    
    @staticmethod
    def compute_class_separation_ratio(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute inter-class / intra-class distance ratio.
        Higher ratio = better separation.
        """
        unique_labels = np.unique(labels)
        
        # Compute class centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = np.mean(features[mask], axis=0)
        
        # Intra-class: average distance to centroid
        intra_distances = []
        for label in unique_labels:
            mask = labels == label
            centroid = centroids[label]
            distances = np.linalg.norm(features[mask] - centroid, axis=1)
            intra_distances.extend(distances)
        
        avg_intra = np.mean(intra_distances)
        
        # Inter-class: average distance between centroids
        inter_distances = []
        centroid_list = list(centroids.values())
        for i in range(len(centroid_list)):
            for j in range(i + 1, len(centroid_list)):
                dist = np.linalg.norm(centroid_list[i] - centroid_list[j])
                inter_distances.append(dist)
        
        avg_inter = np.mean(inter_distances) if inter_distances else 0
        
        ratio = avg_inter / (avg_intra + 1e-8)
        
        return {
            'avg_intra_distance': float(avg_intra),
            'avg_inter_distance': float(avg_inter),
            'separation_ratio': float(ratio)
        }
    
    @staticmethod
    def compute_feature_statistics(features: np.ndarray) -> Dict[str, Any]:
        """Compute basic feature statistics."""
        return {
            'num_samples': features.shape[0],
            'feature_dim': features.shape[1],
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'sparsity': float(np.mean(np.abs(features) < 0.01))  # % near-zero values
        }


# ============================================================
# Model Factory
# ============================================================
def create_model(
    model_type: str,
    num_speakers: int,
    config: Dict[str, Any],
    device: str
) -> nn.Module:
    """Create model based on type."""
    
    if model_type == 'mel_mlp':
        model = TraditionalMLPBaseline(
            feature_type='mel',
            n_mels=80,
            n_mfcc=40,
            sample_rate=config['sample_rate'],
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
            num_speakers=num_speakers,
            device=device
        )
    elif model_type == 'mfcc_mlp':
        model = TraditionalMLPBaseline(
            feature_type='mfcc',
            n_mels=80,
            n_mfcc=40,
            sample_rate=config['sample_rate'],
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
            num_speakers=num_speakers,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str
) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================
# Main Experiment Runner
# ============================================================
class Experiment5Runner:
    """Runner for Experiment 5: Feature Visualization."""
    
    AVAILABLE_MODELS = ['mel_mlp', 'mfcc_mlp']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            'experiment5',
            str(self.output_dir / 'experiment5.log')
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(config['seed'])
        
        self.visualizer = FeatureVisualizer(self.output_dir, config)
        self.analyzer = FeatureAnalyzer()
        
        self.results = {}
        
        # Data
        self.test_loader = None
        self.num_speakers = 26
    
    def setup_data(self):
        """Load dataset."""
        self.logger.info("Loading dataset...")
        
        _, _, test_loader = create_speaker_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            sample_rate=self.config['sample_rate'],
            max_length=self.config['max_length'],
            train_split=0.7,
            val_split=0.15,
            seed=self.config['seed']
        )
        
        self.test_loader = test_loader
        
        try:
            if hasattr(test_loader.dataset, 'num_classes'):
                self.num_speakers = test_loader.dataset.num_classes
        except:
            pass
        
        self.logger.info(f"Test samples: {len(test_loader.dataset)}")
        self.logger.info(f"Number of speakers: {self.num_speakers}")
    
    def find_checkpoint(self, model_type: str) -> Optional[str]:
        """Find checkpoint for model."""
        # Look in experiment1 output
        search_paths = [
            Path('./outputs/experiment1_unified') / model_type / 'checkpoints' / 'best_model.pth',
            Path('./outputs/experiment1_comparison') / model_type / f'exp1_{model_type}' / 'checkpoints' / 'best_model.pth',
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        # Glob search
        for pattern in [f'**/experiment1*/**/{model_type}/**/best_model.pth',
                       f'**/{model_type}/**/best_model.pth']:
            matches = list(Path('./outputs').glob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def analyze_model(self, model_type: str) -> Dict[str, Any]:
        """Analyze features from a single model."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Analyzing: {model_type.upper()}")
        self.logger.info(f"{'='*60}")
        
        results = {'model_type': model_type}
        
        # Find and load checkpoint
        checkpoint_path = self.find_checkpoint(model_type)
        if not checkpoint_path:
            self.logger.warning(f"No checkpoint found for {model_type}")
            return {'error': 'No checkpoint found'}
        
        self.logger.info(f"Checkpoint: {checkpoint_path}")
        
        # Create and load model
        model = create_model(
            model_type=model_type,
            num_speakers=self.num_speakers,
            config=self.config,
            device=self.device
        )
        model = load_model_checkpoint(model, checkpoint_path, self.device)
        
        # Extract features
        self.logger.info("Extracting features...")
        extractor = FeatureExtractor(model, self.device)
        features, labels = extractor.extract_features(
            self.test_loader,
            max_samples=self.config['num_samples_for_viz']
        )
        
        self.logger.info(f"Features shape: {features.shape}")
        results['feature_shape'] = features.shape
        
        # 1. Feature statistics
        self.logger.info("\n1. Computing feature statistics...")
        stats = self.analyzer.compute_feature_statistics(features)
        results['statistics'] = stats
        self.logger.info(f"   Feature dim: {stats['feature_dim']}")
        self.logger.info(f"   Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
        # 2. Cluster quality metrics
        self.logger.info("\n2. Computing cluster quality metrics...")
        
        silhouette = self.analyzer.compute_silhouette_score(features, labels)
        results['silhouette_score'] = silhouette
        self.logger.info(f"   Silhouette score: {silhouette:.4f}")
        
        ch_score = self.analyzer.compute_calinski_harabasz(features, labels)
        results['calinski_harabasz'] = ch_score
        self.logger.info(f"   Calinski-Harabasz: {ch_score:.2f}")
        
        db_score = self.analyzer.compute_davies_bouldin(features, labels)
        results['davies_bouldin'] = db_score
        self.logger.info(f"   Davies-Bouldin: {db_score:.4f}")
        
        # 3. Class separation
        self.logger.info("\n3. Computing class separation...")
        separation = self.analyzer.compute_class_separation_ratio(features, labels)
        results['separation'] = separation
        self.logger.info(f"   Intra-class dist: {separation['avg_intra_distance']:.4f}")
        self.logger.info(f"   Inter-class dist: {separation['avg_inter_distance']:.4f}")
        self.logger.info(f"   Separation ratio: {separation['separation_ratio']:.4f}")
        
        # 4. t-SNE visualization
        if self.config['save_plots'] and HAS_SKLEARN:
            self.logger.info("\n4. Computing t-SNE...")
            tsne_embedding = self.visualizer.compute_tsne(
                features,
                perplexity=self.config['tsne_perplexity'],
                n_iter=self.config['tsne_n_iter']
            )
            
            if tsne_embedding is not None:
                plot_path = self.visualizer.plot_embedding(
                    tsne_embedding, labels,
                    f't-SNE Visualization: {model_type}',
                    f'{model_type}_tsne.{self.config["plot_format"]}'
                )
                results['tsne_plot'] = plot_path
                self.logger.info(f"   Saved: {plot_path}")
        
        # 5. UMAP visualization
        if self.config['save_plots'] and HAS_UMAP:
            self.logger.info("\n5. Computing UMAP...")
            umap_embedding = self.visualizer.compute_umap(
                features,
                n_neighbors=self.config['umap_n_neighbors'],
                min_dist=self.config['umap_min_dist']
            )
            
            if umap_embedding is not None:
                plot_path = self.visualizer.plot_embedding(
                    umap_embedding, labels,
                    f'UMAP Visualization: {model_type}',
                    f'{model_type}_umap.{self.config["plot_format"]}'
                )
                results['umap_plot'] = plot_path
                self.logger.info(f"   Saved: {plot_path}")
        
        # 6. Class separation plot
        if self.config['save_plots'] and HAS_MATPLOTLIB:
            self.logger.info("\n6. Plotting class separation...")
            plot_path = self.visualizer.plot_class_separation(
                features, labels,
                f'Distance Distribution: {model_type}',
                f'{model_type}_distances.{self.config["plot_format"]}'
            )
            results['distance_plot'] = plot_path
            self.logger.info(f"   Saved: {plot_path}")
        
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run visualization for all models."""
        self.logger.info("=" * 70)
        self.logger.info("EXPERIMENT 5: FEATURE VISUALIZATION AND ANALYSIS")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Save plots: {self.config['save_plots']}")
        
        # Setup data
        self.setup_data()
        
        # Analyze each model
        for model_type in self.AVAILABLE_MODELS:
            try:
                results = self.analyze_model(model_type)
                self.results[model_type] = results
            except Exception as e:
                self.logger.error(f"Error analyzing {model_type}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.results[model_type] = {'error': str(e)}
        
        # Generate summary
        self._generate_summary()
        self._save_results()
        
        return self.results
    
    def run_single_model(self, model_type: str) -> Dict[str, Any]:
        """Analyze a single model."""
        if self.test_loader is None:
            self.setup_data()
        
        results = self.analyze_model(model_type)
        self.results[model_type] = results
        
        self._generate_summary()
        self._save_results()
        
        return results
    
    def _generate_summary(self):
        """Generate summary table."""
        self.logger.info("\n" + "=" * 90)
        self.logger.info("FEATURE ANALYSIS SUMMARY")
        self.logger.info("=" * 90)
        
        # Header
        self.logger.info(
            f"{'Model':<12} {'Silhouette':>12} {'CH Index':>12} "
            f"{'DB Index':>12} {'Sep. Ratio':>12}"
        )
        self.logger.info("-" * 90)
        
        for model_name, results in self.results.items():
            if 'error' in results:
                self.logger.info(f"{model_name:<12} ERROR: {results['error']}")
                continue
            
            silhouette = results.get('silhouette_score', 0)
            ch = results.get('calinski_harabasz', 0)
            db = results.get('davies_bouldin', 0)
            sep = results.get('separation', {}).get('separation_ratio', 0)
            
            self.logger.info(
                f"{model_name:<12} {silhouette:>12.4f} {ch:>12.2f} "
                f"{db:>12.4f} {sep:>12.4f}"
            )
        
        self.logger.info("=" * 90)
        self.logger.info("Note: Silhouette [-1,1] higher=better, CH higher=better, DB lower=better")
    
    def _save_results(self):
        """Save results to JSON."""
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj
        
        output = {
            'experiment': 'Experiment 5: Feature Visualization',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_samples': self.config['num_samples_for_viz'],
                'tsne_perplexity': self.config['tsne_perplexity'],
                'umap_n_neighbors': self.config['umap_n_neighbors']
            },
            'results': convert(self.results)
        }
        
        results_path = self.output_dir / 'visualization_results.json'
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {results_path}")
        
        # List generated plots
        if self.config['save_plots']:
            self.logger.info("\nGenerated plots:")
            for f in self.output_dir.glob(f'*.{self.config["plot_format"]}'):
                self.logger.info(f"  - {f.name}")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5: Feature Visualization and Analysis'
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Analyze all models'
    )
    parser.add_argument(
        '--model', type=str, choices=['mel_mlp', 'mfcc_mlp'],
        help='Analyze a specific model'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick analysis (fewer samples, no UMAP)'
    )
    parser.add_argument(
        '--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'],
        help='Path to dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
        help='Output directory'
    )
    parser.add_argument(
        '--no_plots', action='store_true',
        help='Skip plot generation'
    )
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['save_plots'] = not args.no_plots
    
    if args.quick:
        config['num_samples_for_viz'] = 200
        config['tsne_n_iter'] = 500
    
    # Create runner
    runner = Experiment5Runner(config)
    
    if args.all:
        runner.run_all()
    elif args.model:
        runner.run_single_model(args.model)
    else:
        print("Please specify --all or --model <name>")
        print("Example: python run_experiment5_visualization.py --all")
        print("Example: python run_experiment5_visualization.py --quick --all")
        parser.print_help()


if __name__ == "__main__":
    main()
