#!/usr/bin/env python3
"""
Precompute optimal phase space embedding parameters from dataset.
Run this ONCE before training to determine optimal tau and dimension.

Usage:
    python scripts/precompute_embedding_params.py \
        --data_dir /path/to/librispeech \
        --output params.yaml \
        --n_samples 100
"""

import os
import sys
import argparse
import numpy as np
import yaml
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import random

# Setup imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import librosa
import soundfile as sf


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
        # Limit signal length for efficiency
        max_len = 10000
        if len(signal) > max_len:
            signal = signal[:max_len]
        
        mutual_info = np.zeros(self.max_delay)
        
        for tau in range(1, self.max_delay):
            x1 = signal[:-tau]
            x2 = signal[tau:]
            
            # Compute 2D histogram
            hist_2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
            
            # Normalize to get joint probability
            p_joint = hist_2d / hist_2d.sum()
            
            # Marginal probabilities
            p_x1 = p_joint.sum(axis=1)
            p_x2 = p_joint.sum(axis=0)
            
            # Compute mutual information
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if p_joint[i, j] > 1e-10 and p_x1[i] > 1e-10 and p_x2[j] > 1e-10:
                        mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_x1[i] * p_x2[j]))
            
            mutual_info[tau] = mi
        
        # Find first local minimum
        for i in range(2, len(mutual_info) - 1):
            if mutual_info[i] < mutual_info[i-1] and mutual_info[i] < mutual_info[i+1]:
                return i
        
        # Fallback: return point of maximum decrease
        diff = np.diff(mutual_info[1:])
        return np.argmin(diff) + 2
    
    def estimate_tau_autocorr(self, signal: np.ndarray) -> int:
        """
        Estimate optimal delay using autocorrelation.
        Returns first zero crossing or 1/e decay point.
        """
        # Limit signal length
        max_len = 10000
        if len(signal) > max_len:
            signal = signal[:max_len]
        
        n = len(signal)
        signal = signal - np.mean(signal)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1:n-1+self.max_delay]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first minimum or 1/e crossing
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                return i
        
        # Fallback: 1/e crossing
        e_threshold = 1.0 / np.e
        for i in range(1, len(autocorr)):
            if autocorr[i] < e_threshold:
                return i
        
        return self.max_delay // 4
    
    def estimate_dimension_fnn(self, signal: np.ndarray, tau: int, 
                                max_dim: int = 15, threshold: float = 0.05) -> int:
        """
        Estimate optimal embedding dimension using False Nearest Neighbors.
        Returns dimension where FNN fraction drops below threshold.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Limit signal length
        max_len = 5000
        if len(signal) > max_len:
            signal = signal[:max_len]
        
        for dim in range(2, max_dim + 1):
            n_points = len(signal) - (dim) * tau
            if n_points < 100:
                return dim - 1
            
            # Create embedding for dim and dim+1
            embedded_d = np.zeros((n_points, dim))
            embedded_d1 = np.zeros((n_points, dim + 1))
            
            for i in range(dim):
                embedded_d[:, i] = signal[i*tau:i*tau + n_points]
            for i in range(dim + 1):
                embedded_d1[:, i] = signal[i*tau:i*tau + n_points]
            
            # Find nearest neighbors in d-dimensional space
            nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_d)
            distances, indices = nbrs.kneighbors(embedded_d)
            
            # Count false neighbors
            n_false = 0
            n_test = min(500, n_points)
            test_indices = np.random.choice(n_points, n_test, replace=False)
            
            for idx in test_indices:
                nn_idx = indices[idx, 1]  # Nearest neighbor index
                
                # Distance in d-space
                dist_d = distances[idx, 1]
                if dist_d < 1e-10:
                    continue
                
                # Distance in (d+1)-space
                dist_d1 = np.linalg.norm(embedded_d1[idx] - embedded_d1[nn_idx])
                
                # Check if false neighbor (distance increases significantly)
                if (dist_d1 - dist_d) / dist_d > 15.0:  # Rtol threshold
                    n_false += 1
            
            fnn_fraction = n_false / n_test
            
            if fnn_fraction < threshold:
                return dim
        
        return max_dim
    
    def estimate_all(self, signal: np.ndarray, method: str = 'mutual_info') -> Dict:
        """Estimate both tau and dimension for a signal."""
        # Estimate tau
        if method == 'mutual_info':
            tau = self.estimate_tau_mutual_info(signal)
        else:
            tau = self.estimate_tau_autocorr(signal)
        
        # Estimate dimension
        dim = self.estimate_dimension_fnn(signal, tau)
        
        return {
            'tau': tau,
            'dimension': dim,
            'tau_ms': tau / self.sample_rate * 1000  # Convert to milliseconds
        }


def load_audio_files(data_dir: str, n_samples: int, sample_rate: int = 16000) -> List[np.ndarray]:
    """Load random audio samples from LibriSpeech-style directory."""
    audio_files = []
    
    # Find all .flac files
    data_path = Path(data_dir)
    all_files = list(data_path.rglob("*.flac"))
    
    if not all_files:
        all_files = list(data_path.rglob("*.wav"))
    
    if not all_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    # Random sample
    selected_files = random.sample(all_files, min(n_samples, len(all_files)))
    
    print(f"Loading {len(selected_files)} audio files...")
    
    audios = []
    for f in tqdm(selected_files, desc="Loading audio"):
        try:
            audio, sr = librosa.load(f, sr=sample_rate)
            if len(audio) > sample_rate:  # At least 1 second
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


def main():
    parser = argparse.ArgumentParser(description='Precompute embedding parameters')
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
    recommended_dim = int(np.ceil(dim_stats['q75']))  # Use 75th percentile for safety
    
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
    
    # Save to file
    with open(args.output, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\nResults saved to: {args.output}")
    
    # Also print config snippet
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