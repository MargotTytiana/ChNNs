"""
MLSA Comparison Validation Experiment
======================================

Compare three versions of MLSA:
1. Original - The baseline that achieved Fisher=1.43
2. Improved - Added RQA, CorrDim, voiced selection (performed worse)
3. Refined - Keep effective improvements, remove harmful ones

Goal: Find the best MLSA configuration for speaker recognition.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
import time
import os
import json

warnings.filterwarnings('ignore')


# =============================================================================
# MLSA Version 1: Original (Baseline - Fisher=1.43)
# =============================================================================

class OriginalMLSAExtractor:
    """
    Original MLSA that achieved Fisher Score = 1.4352.
    This is our baseline to beat.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.scale_factors = [1, 2, 4, 8, 16]
        self.embedding_dim = 7
        self.delay = 16
        self.sample_rate = sample_rate
    
    @property
    def feature_dim(self) -> int:
        return 8 + len(self.scale_factors)  # 13
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        dt = 1.0 / self.sample_rate
        scale_lles = []
        
        for scale in self.scale_factors:
            if scale > 1:
                n_samples = len(audio) // scale
                if n_samples < 500:
                    scale_lles.append(np.nan)
                    continue
                downsampled = np.array([
                    np.mean(audio[i*scale:(i+1)*scale]) 
                    for i in range(n_samples)
                ])
            else:
                downsampled = audio.copy()
            
            lle = self._calculate_lle(downsampled, dt * scale)
            scale_lles.append(lle)
        
        valid_lles = [l for l in scale_lles if np.isfinite(l)]
        
        if len(valid_lles) >= 2:
            features = [
                np.mean(valid_lles),
                np.std(valid_lles),
                np.max(valid_lles),
                np.min(valid_lles),
                valid_lles[0] if len(valid_lles) > 0 else 0,
                valid_lles[-1] if len(valid_lles) > 1 else 0,
                valid_lles[0] - valid_lles[-1] if len(valid_lles) > 1 else 0,
                len(valid_lles),
            ]
            
            for i in range(len(self.scale_factors)):
                if i < len(scale_lles) and np.isfinite(scale_lles[i]):
                    features.append(scale_lles[i])
                else:
                    features.append(0)
        else:
            features = [0] * self.feature_dim
        
        return np.array(features)
    
    def _calculate_lle(self, data: np.ndarray, dt: float) -> float:
        n_points = len(data) - (self.embedding_dim - 1) * self.delay
        if n_points < 100:
            return np.nan
        
        embedded = np.zeros((n_points, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = data[i * self.delay : i * self.delay + n_points]
        
        try:
            tree = KDTree(embedded)
        except:
            return np.nan
        
        theiler = max(20, self.delay * 2)
        max_iter = min(n_points // 4, 100)
        divergences = []
        
        for i in range(min(500, n_points - max_iter)):
            distances, indices = tree.query(embedded[i], k=10)
            
            j_nearest = None
            d_nearest = np.inf
            
            for j, d in zip(indices[1:], distances[1:]):
                if abs(j - i) > theiler and j < n_points - max_iter:
                    if d < d_nearest and d > 1e-10:
                        j_nearest = j
                        d_nearest = d
                        break
            
            if j_nearest is None:
                continue
            
            for k in range(1, max_iter):
                if i + k < n_points and j_nearest + k < n_points:
                    dist = np.linalg.norm(embedded[i + k] - embedded[j_nearest + k])
                    if dist > 1e-10:
                        divergences.append((k, np.log(dist)))
        
        if len(divergences) < 50:
            return np.nan
        
        times = np.array([d[0] for d in divergences])
        log_divs = np.array([d[1] for d in divergences])
        
        unique_times = np.unique(times)[:20]
        avg_divs = [np.mean(log_divs[times == t]) for t in unique_times]
        
        if len(avg_divs) < 5:
            return np.nan
        
        x = unique_times[:len(avg_divs)] * dt
        y = np.array(avg_divs)
        
        valid = np.isfinite(y)
        x, y = x[valid], y[valid]
        
        if len(x) < 3:
            return np.nan
        
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]
    
    def get_feature_names(self) -> List[str]:
        names = ['LLE_mean', 'LLE_std', 'LLE_max', 'LLE_min', 
                 'LLE_first', 'LLE_last', 'LLE_decay', 'n_valid_scales']
        for scale in self.scale_factors:
            names.append(f'LLE_scale_{scale}')
        return names


# =============================================================================
# MLSA Version 2: Refined (Keep good improvements, remove bad ones)
# =============================================================================

class RefinedMLSAExtractor:
    """
    Refined MLSA based on experimental analysis:
    - KEEPS: Multi-scale LLE (effective)
    - KEEPS: Adaptive Theiler window
    - KEEPS: RANSAC-style robust fitting
    - ADDS: pitch_stability, zcr_variability (were in top 10)
    - REMOVES: RQA features (not helpful)
    - REMOVES: CorrDim features (not helpful)
    - REMOVES: Voiced segment selection (loses info)
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.scale_factors = [1, 2, 4, 8, 16]
        self.embedding_dim = 5  # Reduced from 7
        self.base_delay = 16
        self.sample_rate = sample_rate
    
    @property
    def feature_dim(self) -> int:
        return 11 + 3  # 11 LLE features + 3 signal features = 14
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        features = []
        
        # Core: Multi-scale LLE
        lle_features = self._extract_multiscale_lle(audio)
        features.extend(lle_features)
        
        # Complementary: Simple signal features
        signal_features = self._extract_signal_features(audio)
        features.extend(signal_features)
        
        features = np.array(features)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _extract_multiscale_lle(self, audio: np.ndarray) -> List[float]:
        scale_lles = []
        
        for scale in self.scale_factors:
            if scale > 1:
                n_samples = len(audio) // scale
                if n_samples < 500:
                    scale_lles.append(np.nan)
                    continue
                downsampled = np.array([
                    np.mean(audio[i*scale:(i+1)*scale]) 
                    for i in range(n_samples)
                ])
            else:
                downsampled = audio.copy()
            
            scaled_delay = max(1, self.base_delay // max(1, scale // 2))
            
            lle = self._calculate_lle_robust(
                downsampled, 
                delay=scaled_delay,
                dt=scale / self.sample_rate
            )
            scale_lles.append(lle)
        
        valid_lles = [l for l in scale_lles if np.isfinite(l)]
        
        features = []
        
        # Individual scale LLEs
        for lle in scale_lles:
            features.append(lle if np.isfinite(lle) else 0.0)
        
        # Aggregated statistics
        if len(valid_lles) >= 2:
            features.extend([
                np.mean(valid_lles),
                np.std(valid_lles),
                np.max(valid_lles),
                np.min(valid_lles),
                np.max(valid_lles) - np.min(valid_lles),
                valid_lles[0] - valid_lles[-1] if len(valid_lles) > 1 else 0,
            ])
        else:
            features.extend([0.0] * 6)
        
        return features
    
    def _calculate_lle_robust(self, data: np.ndarray, delay: int, dt: float) -> float:
        n_points = len(data) - (self.embedding_dim - 1) * delay
        
        if n_points < 200:
            return np.nan
        
        embedded = np.zeros((n_points, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = data[i * delay : i * delay + n_points]
        
        # Normalize
        embedded_mean = np.mean(embedded, axis=0)
        embedded_std = np.std(embedded, axis=0)
        embedded_std[embedded_std < 1e-10] = 1.0
        embedded_norm = (embedded - embedded_mean) / embedded_std
        
        try:
            tree = KDTree(embedded_norm)
        except:
            return np.nan
        
        # Adaptive Theiler window
        theiler = self._estimate_theiler_window(data, delay)
        
        max_iter = min(n_points // 4, 150)
        n_reference = min(400, n_points - max_iter - theiler)
        
        if n_reference < 50:
            return np.nan
        
        divergences = [[] for _ in range(max_iter)]
        reference_indices = np.linspace(0, n_points - max_iter - 1, n_reference, dtype=int)
        
        for i in reference_indices:
            distances, indices = tree.query(embedded_norm[i], k=20)
            
            j_nearest = None
            d_nearest = np.inf
            
            for j, d in zip(indices[1:], distances[1:]):
                if abs(j - i) > theiler and j < n_points - max_iter:
                    if 1e-10 < d < d_nearest:
                        j_nearest = j
                        d_nearest = d
                        break
            
            if j_nearest is None or d_nearest < 1e-10:
                continue
            
            for k in range(1, max_iter):
                idx_i = i + k
                idx_j = j_nearest + k
                
                if idx_i < n_points and idx_j < n_points:
                    dist = np.linalg.norm(embedded_norm[idx_i] - embedded_norm[idx_j])
                    if dist > 1e-10:
                        divergences[k].append(np.log(dist / d_nearest))
        
        avg_divs = []
        times = []
        
        for k in range(1, max_iter):
            if len(divergences[k]) >= 10:
                avg_divs.append(np.median(divergences[k]))
                times.append(k)
        
        if len(avg_divs) < 5:
            return np.nan
        
        times = np.array(times) * dt
        avg_divs = np.array(avg_divs)
        
        # RANSAC-style fitting
        n_fit = min(len(times), max(5, len(times) // 3))
        best_slope = 0
        best_r2 = -np.inf
        
        for start in range(min(5, len(times) - n_fit)):
            end = start + n_fit
            x = times[start:end]
            y = avg_divs[start:end]
            
            if len(x) < 3:
                continue
            
            try:
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.polyval(coeffs, x)
                
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_slope = coeffs[0]
            except:
                continue
        
        return best_slope
    
    def _estimate_theiler_window(self, data: np.ndarray, delay: int) -> int:
        max_lag = min(500, len(data) // 4)
        
        data_centered = data - np.mean(data)
        autocorr = np.correlate(data_centered[:max_lag*2], data_centered[:max_lag*2], mode='full')
        autocorr = autocorr[len(autocorr)//2:][:max_lag]
        
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] <= 0:
                return max(delay * 2, i)
            if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                return max(delay * 2, i)
        
        return max(delay * 2, 50)
    
    def _extract_signal_features(self, audio: np.ndarray) -> List[float]:
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        n_frames = (len(audio) - frame_length) // hop_length
        
        if n_frames < 5:
            return [0.5, 0.5, 0.5]
        
        # Pitch stability
        try:
            min_period = int(self.sample_rate / 400)
            max_period = int(self.sample_rate / 50)
            
            autocorr = np.correlate(audio[:4000], audio[:4000], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            peaks, _ = find_peaks(autocorr[min_period:max_period])
            
            if len(peaks) > 1:
                peak_distances = np.diff(peaks)
                pitch_stability = 1.0 / (1.0 + np.std(peak_distances) / (np.mean(peak_distances) + 1e-10))
            else:
                pitch_stability = 0.5
        except:
            pitch_stability = 0.5
        
        # ZCR variability
        zcr_frames = []
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zcr_frames.append(zcr)
        
        zcr_variability = np.std(zcr_frames) / (np.mean(zcr_frames) + 1e-10) if len(zcr_frames) > 1 else 0.5
        
        # Energy entropy
        energies = []
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energies.append(np.sum(frame ** 2))
        
        energies = np.array(energies)
        if np.sum(energies) > 0:
            energies = energies / np.sum(energies)
            energies = energies[energies > 1e-10]
            energy_entropy = entropy(energies) / np.log(len(energies) + 1)
        else:
            energy_entropy = 0.5
        
        return [pitch_stability, zcr_variability, energy_entropy]
    
    def get_feature_names(self) -> List[str]:
        names = []
        for scale in self.scale_factors:
            names.append(f'LLE_scale_{scale}')
        names.extend(['LLE_mean', 'LLE_std', 'LLE_max', 'LLE_min', 'LLE_range', 'LLE_decay'])
        names.extend(['pitch_stability', 'zcr_variability', 'energy_entropy'])
        return names


# =============================================================================
# MLSA Version 3: Hybrid (Original LLE + Signal Features)
# =============================================================================

class HybridMLSAExtractor:
    """
    Hybrid approach: Original LLE algorithm + helpful signal features.
    
    Uses exact original LLE calculation (proven effective)
    plus the signal features that showed importance.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.original = OriginalMLSAExtractor(sample_rate=sample_rate)
        self.sample_rate = sample_rate
    
    @property
    def feature_dim(self) -> int:
        return self.original.feature_dim + 3  # 13 + 3 = 16
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        # Get original LLE features (proven effective)
        original_features = self.original.extract(audio)
        
        # Add signal features
        signal_features = self._extract_signal_features(audio)
        
        return np.concatenate([original_features, signal_features])
    
    def _extract_signal_features(self, audio: np.ndarray) -> np.ndarray:
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        n_frames = (len(audio) - frame_length) // hop_length
        
        if n_frames < 5:
            return np.array([0.5, 0.5, 0.5])
        
        # Pitch stability
        try:
            min_period = int(self.sample_rate / 400)
            max_period = int(self.sample_rate / 50)
            
            autocorr = np.correlate(audio[:4000], audio[:4000], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            peaks, _ = find_peaks(autocorr[min_period:max_period])
            
            if len(peaks) > 1:
                peak_distances = np.diff(peaks)
                pitch_stability = 1.0 / (1.0 + np.std(peak_distances) / (np.mean(peak_distances) + 1e-10))
            else:
                pitch_stability = 0.5
        except:
            pitch_stability = 0.5
        
        # ZCR variability
        zcr_frames = []
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zcr_frames.append(zcr)
        
        zcr_variability = np.std(zcr_frames) / (np.mean(zcr_frames) + 1e-10) if len(zcr_frames) > 1 else 0.5
        
        # Energy entropy
        energies = []
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energies.append(np.sum(frame ** 2))
        
        energies = np.array(energies)
        if np.sum(energies) > 0:
            energies = energies / np.sum(energies)
            energies = energies[energies > 1e-10]
            energy_entropy = entropy(energies) / np.log(len(energies) + 1)
        else:
            energy_entropy = 0.5
        
        return np.array([pitch_stability, zcr_variability, energy_entropy])
    
    def get_feature_names(self) -> List[str]:
        names = self.original.get_feature_names()
        names.extend(['pitch_stability', 'zcr_variability', 'energy_entropy'])
        return names


# =============================================================================
# Traditional Feature Extractors
# =============================================================================

class MFCCExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.n_mfcc = 13
        self.sample_rate = sample_rate
    
    @property
    def feature_dim(self) -> int:
        return self.n_mfcc * 4
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfcc)
        
        return np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
        ])


class CombinedExtractor:
    """MFCC + Best MLSA variant."""
    
    def __init__(self, mlsa_extractor, sample_rate: int = 16000):
        self.mfcc = MFCCExtractor(sample_rate)
        self.mlsa = mlsa_extractor
        self.sample_rate = sample_rate
    
    @property
    def feature_dim(self) -> int:
        return self.mfcc.feature_dim + self.mlsa.feature_dim
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        mfcc_feat = self.mfcc.extract(audio)
        mlsa_feat = self.mlsa.extract(audio)
        return np.concatenate([mfcc_feat, mlsa_feat])


# =============================================================================
# Dataset Loader
# =============================================================================

class DatasetLoader:
    def __init__(self, data_dir: str, max_samples: int = 20, 
                 max_duration: float = 3.0, sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.max_duration = max_duration
        self.sample_rate = sample_rate
    
    def load(self) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        print(f"Loading from: {self.data_dir}")
        
        flac_files = list(self.data_dir.rglob("*.flac"))
        print(f"Found {len(flac_files)} FLAC files")
        
        speaker_files = defaultdict(list)
        for f in flac_files:
            speaker_id = f.stem.split('-')[0]
            speaker_files[speaker_id].append(f)
        
        sorted_speakers = sorted(speaker_files.keys())[:26]
        print(f"Using {len(sorted_speakers)} speakers")
        
        speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted_speakers)}
        
        audios = []
        labels = []
        
        for spk_id in sorted_speakers:
            files = speaker_files[spk_id][:self.max_samples]
            
            for f in files:
                try:
                    audio, _ = librosa.load(f, sr=self.sample_rate, duration=self.max_duration)
                    if len(audio) >= self.sample_rate * 0.5:
                        audios.append(audio)
                        labels.append(speaker_to_idx[spk_id])
                except:
                    pass
        
        print(f"Loaded {len(audios)} samples")
        return audios, np.array(labels), sorted_speakers


# =============================================================================
# Evaluation
# =============================================================================

class Evaluator:
    @staticmethod
    def fisher_score(features: np.ndarray, labels: np.ndarray) -> float:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        overall_mean = np.mean(features, axis=0)
        between, within = 0, 0
        
        for label in unique_labels:
            mask = labels == label
            class_features = features[mask]
            class_mean = np.mean(class_features, axis=0)
            n = np.sum(mask)
            
            between += n * np.sum((class_mean - overall_mean) ** 2)
            within += np.sum((class_features - class_mean) ** 2)
        
        return between / within if within > 1e-10 else 0.0
    
    @staticmethod
    def silhouette(features: np.ndarray, labels: np.ndarray) -> float:
        try:
            return silhouette_score(features, labels)
        except:
            return 0.0
    
    @staticmethod
    def knn_accuracy(features: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
        try:
            knn = KNeighborsClassifier(n_neighbors=k)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return np.mean(cross_val_score(knn, features, labels, cv=cv))
        except:
            return 0.0
    
    @staticmethod
    def lda_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
        try:
            lda = LinearDiscriminantAnalysis()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return np.mean(cross_val_score(lda, features, labels, cv=cv))
        except:
            return 0.0


# =============================================================================
# Main Experiment
# =============================================================================

class MLSAComparisonExperiment:
    def __init__(self, data_dir: str, output_dir: str = "outputs"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create all MLSA variants
        self.mlsa_variants = {
            'MLSA_Original': OriginalMLSAExtractor(),
            'MLSA_Refined': RefinedMLSAExtractor(),
            'MLSA_Hybrid': HybridMLSAExtractor(),
        }
        
        # Full extractor set
        self.extractors = {
            'MFCC': MFCCExtractor(),
            **self.mlsa_variants,
            'MFCC+Original': CombinedExtractor(OriginalMLSAExtractor()),
            'MFCC+Refined': CombinedExtractor(RefinedMLSAExtractor()),
            'MFCC+Hybrid': CombinedExtractor(HybridMLSAExtractor()),
        }
    
    def run(self, max_samples: int = 15) -> Dict:
        print("\n" + "="*70)
        print("MLSA COMPARISON EXPERIMENT")
        print("Comparing Original vs Refined vs Hybrid MLSA")
        print("="*70)
        
        # Load data
        print("\n[Step 1] Loading Dataset...")
        loader = DatasetLoader(self.data_dir, max_samples=max_samples)
        audios, labels, speakers = loader.load()
        print(f"  {len(audios)} samples, {len(np.unique(labels))} speakers")
        
        # Extract features
        print("\n[Step 2] Extracting Features...")
        all_features = {}
        times = {}
        
        for name, extractor in self.extractors.items():
            print(f"\n  Extracting {name}...")
            start = time.time()
            
            features = []
            for i, audio in enumerate(audios):
                try:
                    feat = extractor.extract(audio)
                    features.append(feat)
                except:
                    features.append(np.zeros(extractor.feature_dim))
                
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(audios)}...")
            
            elapsed = time.time() - start
            times[name] = elapsed
            
            features = np.array(features)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            all_features[name] = features
            
            print(f"  {name}: shape={features.shape}, time={elapsed:.1f}s")
        
        # Normalize
        print("\n[Step 3] Normalizing...")
        scaler = StandardScaler()
        for name in all_features:
            all_features[name] = scaler.fit_transform(all_features[name])
        
        # Evaluate
        print("\n[Step 4] Evaluating...")
        results = {}
        
        for name, features in all_features.items():
            fisher = Evaluator.fisher_score(features, labels)
            silhouette = Evaluator.silhouette(features, labels)
            knn = Evaluator.knn_accuracy(features, labels)
            lda = Evaluator.lda_accuracy(features, labels)
            
            results[name] = {
                'dim': features.shape[1],
                'fisher': fisher,
                'silhouette': silhouette,
                'knn': knn,
                'lda': lda,
                'time': times[name]
            }
            
            print(f"\n  {name}:")
            print(f"    Fisher={fisher:.4f}, Silhouette={silhouette:.4f}")
            print(f"    K-NN={knn*100:.2f}%, LDA={lda*100:.2f}%")
        
        # Visualization
        print("\n[Step 5] Creating Visualizations...")
        self._create_visualization(results, all_features, labels)
        
        # Save
        self._save_results(results)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _create_visualization(self, results: Dict, features: Dict, labels: np.ndarray):
        fig = plt.figure(figsize=(18, 12))
        
        # 1. MLSA Variants Comparison
        ax1 = fig.add_subplot(2, 3, 1)
        mlsa_names = ['MLSA_Original', 'MLSA_Refined', 'MLSA_Hybrid']
        metrics = ['fisher', 'knn', 'lda']
        
        x = np.arange(len(mlsa_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[n][metric] * (100 if metric != 'fisher' else 1) for n in mlsa_names]
            bars = ax1.bar(x + i*width, values, width, label=metric.upper())
            
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + width/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
        
        ax1.set_ylabel('Score')
        ax1.set_title('MLSA Variants Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([n.replace('MLSA_', '') for n in mlsa_names])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Combined Features Comparison
        ax2 = fig.add_subplot(2, 3, 2)
        combined_names = ['MFCC', 'MFCC+Original', 'MFCC+Refined', 'MFCC+Hybrid']
        knn_vals = [results[n]['knn'] * 100 for n in combined_names]
        lda_vals = [results[n]['lda'] * 100 for n in combined_names]
        
        x = np.arange(len(combined_names))
        ax2.bar(x - 0.2, knn_vals, 0.4, label='K-NN', color='steelblue')
        ax2.bar(x + 0.2, lda_vals, 0.4, label='LDA', color='darkorange')
        
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Combined Features Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace('MFCC+', '+') for n in combined_names], fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Fisher Score Comparison
        ax3 = fig.add_subplot(2, 3, 3)
        all_names = list(results.keys())
        fisher_vals = [results[n]['fisher'] for n in all_names]
        colors = ['green' if 'Hybrid' in n else 'blue' if 'Original' in n else 'orange' if 'Refined' in n else 'gray' for n in all_names]
        
        bars = ax3.barh(range(len(all_names)), fisher_vals, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(all_names)))
        ax3.set_yticklabels(all_names, fontsize=8)
        ax3.set_xlabel('Fisher Score')
        ax3.set_title('Fisher Discriminant Ratio')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Highlight best
        best_idx = np.argmax(fisher_vals)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(1.0)
        
        # 4. PCA - Original MLSA
        ax4 = fig.add_subplot(2, 3, 4)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(features['MLSA_Original'])
        scatter = ax4.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab20', alpha=0.6, s=20)
        ax4.set_title('MLSA Original - PCA')
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('PC2')
        ax4.grid(True, alpha=0.3)
        
        # 5. PCA - Hybrid MLSA
        ax5 = fig.add_subplot(2, 3, 5)
        proj = pca.fit_transform(features['MLSA_Hybrid'])
        ax5.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab20', alpha=0.6, s=20)
        ax5.set_title('MLSA Hybrid - PCA')
        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC2')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Find best performers
        best_mlsa = max(['MLSA_Original', 'MLSA_Refined', 'MLSA_Hybrid'], 
                       key=lambda x: results[x]['fisher'])
        best_combined = max(['MFCC+Original', 'MFCC+Refined', 'MFCC+Hybrid'],
                           key=lambda x: results[x]['knn'])
        
        summary = f"""
EXPERIMENT SUMMARY
==================

MLSA VARIANTS:
  Original:  Fisher={results['MLSA_Original']['fisher']:.4f}, K-NN={results['MLSA_Original']['knn']*100:.1f}%
  Refined:   Fisher={results['MLSA_Refined']['fisher']:.4f}, K-NN={results['MLSA_Refined']['knn']*100:.1f}%
  Hybrid:    Fisher={results['MLSA_Hybrid']['fisher']:.4f}, K-NN={results['MLSA_Hybrid']['knn']*100:.1f}%
  
  Best MLSA: {best_mlsa}

COMBINED WITH MFCC:
  MFCC only:     K-NN={results['MFCC']['knn']*100:.1f}%
  MFCC+Original: K-NN={results['MFCC+Original']['knn']*100:.1f}%
  MFCC+Refined:  K-NN={results['MFCC+Refined']['knn']*100:.1f}%
  MFCC+Hybrid:   K-NN={results['MFCC+Hybrid']['knn']*100:.1f}%
  
  Best Combined: {best_combined}

RECOMMENDATION:
  Use {best_mlsa} for standalone MLSA
  Use {best_combined} for combined features
"""
        
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'mlsa_comparison_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'mlsa_comparison_results.pdf', bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    
    def _save_results(self, results: Dict):
        output_path = self.output_dir / 'mlsa_comparison_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"  Saved to: {output_path}")
    
    def _print_summary(self, results: Dict):
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        print("\n" + "-"*80)
        print(f"{'Method':<20} {'Dim':>5} {'Fisher':>10} {'Silhouette':>12} {'K-NN':>10} {'LDA':>10}")
        print("-"*80)
        
        for name, r in results.items():
            print(f"{name:<20} {r['dim']:>5} {r['fisher']:>10.4f} {r['silhouette']:>12.4f} "
                  f"{r['knn']*100:>9.2f}% {r['lda']*100:>9.2f}%")
        
        print("-"*80)
        
        # Find best
        mlsa_only = {k: v for k, v in results.items() if k.startswith('MLSA_')}
        best_mlsa = max(mlsa_only.items(), key=lambda x: x[1]['fisher'])
        
        combined = {k: v for k, v in results.items() if k.startswith('MFCC+')}
        best_combined = max(combined.items(), key=lambda x: x[1]['knn'])
        
        print(f"\n*** BEST MLSA VARIANT: {best_mlsa[0]} (Fisher={best_mlsa[1]['fisher']:.4f}) ***")
        print(f"*** BEST COMBINED: {best_combined[0]} (K-NN={best_combined[1]['knn']*100:.2f}%) ***")
        
        # Improvement analysis
        orig_fisher = results['MLSA_Original']['fisher']
        
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        
        for name in ['MLSA_Refined', 'MLSA_Hybrid']:
            fisher_diff = results[name]['fisher'] - orig_fisher
            knn_diff = (results[name]['knn'] - results['MLSA_Original']['knn']) * 100
            print(f"\n{name} vs Original:")
            print(f"  Fisher: {fisher_diff:+.4f}")
            print(f"  K-NN:   {knn_diff:+.2f}%")


# =============================================================================
# Main
# =============================================================================

def find_dataset():
    paths = [
        "/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2",
        os.environ.get('DATA_DIR', ''),
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


if __name__ == "__main__":
    print("="*70)
    print("MLSA Comparison: Original vs Refined vs Hybrid")
    print("="*70)
    
    data_dir = find_dataset()
    
    if not data_dir:
        print("\nDataset not found. Set DATA_DIR environment variable.")
        exit(1)
    
    print(f"\nDataset: {data_dir}")
    
    output_dir = os.environ.get('OUTPUT_DIR', 
        "/scratch/project_2003370/yueyao/Model/scripts/outputs/mlsa_experiments")
    
    experiment = MLSAComparisonExperiment(data_dir, output_dir)
    
    start = time.time()
    results = experiment.run(max_samples=15)
    
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print("\nExperiment complete!")