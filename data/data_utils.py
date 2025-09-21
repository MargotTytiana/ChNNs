"""
Basic Data Utility Functions for Chaotic Speaker Recognition Project
Provides essential data handling, validation, and transformation utilities
for audio processing and machine learning workflows.
"""

import os
import re
import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import random
import numpy as np
from collections import defaultdict, Counter

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    librosa = None
    sf = None

# Scientific computing
try:
    import scipy.io.wavfile as wavfile
    from scipy import signal
    import pandas as pd
    HAS_SCIPY_PANDAS = True
except ImportError:
    HAS_SCIPY_PANDAS = False
    wavfile = None
    signal = None
    pd = None

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress indicator
    def tqdm(iterable, *args, **kwargs):
        return iterable


class DataValidator:
    """Comprehensive data validation utilities for audio and metadata"""
    
    @staticmethod
    def is_valid_audio_file(file_path: Union[str, Path]) -> bool:
        """Check if file is a valid audio file"""
        if not HAS_AUDIO_LIBS:
            # Fallback to extension check
            valid_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aiff'}
            return Path(file_path).suffix.lower() in valid_extensions
        
        try:
            # Try to load audio metadata without loading the full file
            info = sf.info(str(file_path))
            return info.frames > 0 and info.samplerate > 0
        except Exception:
            return False
    
    @staticmethod
    def validate_audio_properties(file_path: Union[str, Path], 
                                min_duration: float = 0.1,
                                max_duration: float = 300.0,
                                min_sample_rate: int = 8000,
                                max_sample_rate: int = 96000) -> Dict[str, Any]:
        """
        Validate audio file properties and return detailed information
        
        Args:
            file_path: Path to audio file
            min_duration: Minimum required duration in seconds
            max_duration: Maximum allowed duration in seconds
            min_sample_rate: Minimum required sample rate
            max_sample_rate: Maximum allowed sample rate
            
        Returns:
            Dictionary with validation results and audio properties
        """
        result = {
            'is_valid': False,
            'file_path': str(file_path),
            'file_size': 0,
            'duration': 0.0,
            'sample_rate': 0,
            'channels': 0,
            'format': 'unknown',
            'errors': [],
            'warnings': []
        }
        
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            result['errors'].append('File does not exist')
            return result
        
        # Check file size
        result['file_size'] = file_path.stat().st_size
        if result['file_size'] == 0:
            result['errors'].append('File is empty')
            return result
        
        if not HAS_AUDIO_LIBS:
            result['errors'].append('Audio libraries not available')
            return result
        
        try:
            # Get audio file info
            info = sf.info(str(file_path))
            
            result['duration'] = info.frames / info.samplerate
            result['sample_rate'] = info.samplerate
            result['channels'] = info.channels
            result['format'] = info.format
            
            # Validate duration
            if result['duration'] < min_duration:
                result['errors'].append(f'Duration {result["duration"]:.2f}s < minimum {min_duration}s')
            elif result['duration'] > max_duration:
                result['errors'].append(f'Duration {result["duration"]:.2f}s > maximum {max_duration}s')
            
            # Validate sample rate
            if result['sample_rate'] < min_sample_rate:
                result['errors'].append(f'Sample rate {result["sample_rate"]} < minimum {min_sample_rate}')
            elif result['sample_rate'] > max_sample_rate:
                result['errors'].append(f'Sample rate {result["sample_rate"]} > maximum {max_sample_rate}')
            
            # Check for mono audio (recommended for speaker recognition)
            if result['channels'] > 1:
                result['warnings'].append(f'Multi-channel audio ({result["channels"]} channels) - consider mono conversion')
            
            # Additional format warnings
            if 'MP3' in result['format'].upper():
                result['warnings'].append('MP3 format detected - lossless formats recommended for research')
            
            # Set validity
            result['is_valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f'Failed to read audio file: {str(e)}')
        
        return result
    
    @staticmethod
    def validate_speaker_label(label: str, 
                             allowed_pattern: str = r'^[a-zA-Z0-9_-]+$',
                             min_length: int = 1,
                             max_length: int = 50) -> Dict[str, Any]:
        """
        Validate speaker label format and content
        
        Args:
            label: Speaker label to validate
            allowed_pattern: Regex pattern for allowed characters
            min_length: Minimum label length
            max_length: Maximum label length
            
        Returns:
            Validation result dictionary
        """
        result = {
            'is_valid': True,
            'label': label,
            'errors': [],
            'warnings': []
        }
        
        # Check type
        if not isinstance(label, str):
            result['errors'].append(f'Label must be string, got {type(label)}')
            result['is_valid'] = False
            return result
        
        # Check length
        if len(label) < min_length:
            result['errors'].append(f'Label too short: {len(label)} < {min_length}')
        elif len(label) > max_length:
            result['errors'].append(f'Label too long: {len(label)} > {max_length}')
        
        # Check pattern
        if not re.match(allowed_pattern, label):
            result['errors'].append(f'Label contains invalid characters: "{label}"')
        
        # Check for potential issues
        if label.startswith(('_', '-')) or label.endswith(('_', '-')):
            result['warnings'].append('Label starts or ends with special character')
        
        if '--' in label or '__' in label:
            result['warnings'].append('Label contains consecutive special characters')
        
        # Set final validity based on errors
        result['is_valid'] = len(result['errors']) == 0
        
        return result
    
    @staticmethod
    def validate_dataset_structure(dataset_path: Union[str, Path],
                                 expected_structure: str = "librispeech") -> Dict[str, Any]:
        """
        Validate the structure of a speaker recognition dataset
        
        Args:
            dataset_path: Path to the dataset directory
            expected_structure: Expected dataset structure type
                              ("librispeech" for LibriSpeech-style structure)
            
        Returns:
            Validation results with detailed statistics
        """
        dataset_path = Path(dataset_path)
        result = {
            'is_valid': False,
            'dataset_path': str(dataset_path),
            'speaker_count': 0,
            'file_count': 0,
            'valid_files': 0,
            'errors': [],
            'warnings': [],
            'speakers': {}
        }
        
        if not dataset_path.exists():
            result['errors'].append(f"Dataset path does not exist: {dataset_path}")
            return result
        
        if not dataset_path.is_dir():
            result['errors'].append(f"Dataset path is not a directory: {dataset_path}")
            return result
        
        # LibriSpeech-style dataset structure
        if expected_structure == "librispeech":
            # Look for subdirectories like dev-clean, dev-other, etc.
            subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            
            if not subdirs:
                result['errors'].append("No subdirectories found in LibriSpeech dataset")
                return result
            
            # Process each subdirectory (e.g., dev-clean, dev-other)
            for subdir in subdirs:
                # Find speaker directories
                speaker_dirs = [d for d in subdir.iterdir() if d.is_dir()]
                
                for speaker_dir in speaker_dirs:
                    speaker_id = speaker_dir.name
                    
                    # Find chapter directories for this speaker
                    chapter_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
                    
                    if not chapter_dirs:
                        result['warnings'].append(f"Speaker {speaker_id} has no chapter directories")
                        continue
                    
                    # Count audio files in all chapter directories
                    audio_files = []
                    for chapter_dir in chapter_dirs:
                        chapter_files = [f for f in chapter_dir.iterdir() if DataValidator.is_valid_audio_file(f)]
                        audio_files.extend(chapter_files)
                    
                    # Initialize speaker entry if not exists
                    if speaker_id not in result['speakers']:
                        result['speakers'][speaker_id] = {
                            'file_count': 0,
                            'valid_files': 0,
                            'chapters': {}
                        }
                    
                    # Update counts
                    result['speakers'][speaker_id]['file_count'] += len(audio_files)
                    result['speakers'][speaker_id]['valid_files'] += len(audio_files)  # Assuming all are valid for now
                    
                    # Store chapter information
                    for chapter_dir in chapter_dirs:
                        chapter_id = chapter_dir.name
                        chapter_files = [f for f in chapter_dir.iterdir() if DataValidator.is_valid_audio_file(f)]
                        result['speakers'][speaker_id]['chapters'][chapter_id] = len(chapter_files)
                    
                    # Update global counts
                    result['file_count'] += len(audio_files)
                    result['valid_files'] += len(audio_files)
                    
                    # Check for minimum files per speaker
                    if len(audio_files) < 3:
                        result['warnings'].append(f"Speaker {speaker_id} has only {len(audio_files)} files (minimum 3 recommended)")
            
            # Count unique speakers
            result['speaker_count'] = len(result['speakers'])
        
        # Flat structure (speaker directories with audio files directly inside)
        elif expected_structure == "speaker_per_folder":
            speaker_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            result['speaker_count'] = len(speaker_dirs)
            
            for speaker_dir in speaker_dirs:
                speaker_name = speaker_dir.name
                audio_files = [f for f in speaker_dir.iterdir() if DataValidator.is_valid_audio_file(f)]
                
                result['speakers'][speaker_name] = {
                    'file_count': len(audio_files),
                    'valid_files': len(audio_files)  # Assuming all are valid for now
                }
                
                result['file_count'] += len(audio_files)
                result['valid_files'] += len(audio_files)
                
                # Check for minimum files per speaker
                if len(audio_files) < 3:
                    result['warnings'].append(f"Speaker {speaker_name} has only {len(audio_files)} files (minimum 3 recommended)")
        
        else:
            result['errors'].append(f"Unknown dataset structure: {expected_structure}")
            return result
        
        # Additional validation based on dataset structure
        if result['speaker_count'] < 2:
            result['errors'].append(f"Dataset must contain at least 2 speakers, found {result['speaker_count']}")
        
        if result['file_count'] < 10:
            result['warnings'].append(f"Dataset has only {result['file_count']} files (more data recommended)")
        
        result['is_valid'] = len(result['errors']) == 0
        
        return result
    
    @staticmethod
    def find_audio_files(dataset_path: Union[str, Path], 
                        pattern: str = "**/*.flac") -> Dict[str, List[Path]]:
        """
        Find all audio files in a dataset organized in LibriSpeech style
        
        Args:
            dataset_path: Path to the dataset directory
            pattern: Glob pattern to match audio files
            
        Returns:
            Dictionary mapping speaker IDs to lists of audio file paths
        """
        dataset_path = Path(dataset_path)
        speaker_files = defaultdict(list)
        
        # Find all audio files matching the pattern
        audio_files = list(dataset_path.glob(pattern))
        
        # Group by speaker ID (extracted from file path)
        for file_path in audio_files:
            # Extract speaker ID from path (assuming structure: .../speaker_id/chapter_id/file.flac)
            parts = file_path.parts
            if len(parts) >= 3:
                speaker_id = parts[-3]  # Third from last part should be speaker ID
                speaker_files[speaker_id].append(file_path)
        
        return dict(speaker_files)


class DataTransformer:
    """Data transformation and augmentation utilities"""
    
    @staticmethod
    def resample_audio(audio: np.ndarray, 
                      original_sr: int, 
                      target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Input audio signal
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio signal
        """
        if original_sr == target_sr:
            return audio
        
        if not HAS_SCIPY_PANDAS:
            raise ImportError("scipy is required for audio resampling")
        
        # Calculate number of samples in resampled audio
        num_samples = int(len(audio) * target_sr / original_sr)
        
        # Resample using scipy's signal.resample
        resampled_audio = signal.resample(audio, num_samples)
        
        return resampled_audio.astype(np.float32)
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, 
                       method: str = "peak",
                       target_level: float = -3.0) -> np.ndarray:
        """
        Normalize audio to specified level
        
        Args:
            audio: Input audio signal
            method: Normalization method ("peak", "rms", "loudness")
            target_level: Target level in dB
            
        Returns:
            Normalized audio signal
        """
        if method == "peak":
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                scaling_factor = 10**(target_level / 20) / peak
                audio = audio * scaling_factor
        elif method == "rms" and HAS_SCIPY_PANDAS:
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                scaling_factor = 10**(target_level / 20) / rms
                audio = audio * scaling_factor
        elif method == "loudness" and HAS_AUDIO_LIBS:
            # Loudness normalization (EBU R128)
            try:
                # Calculate integrated loudness
                meter = pyln.Meter(original_sr)  # pyln is pyloudnorm
                loudness = meter.integrated_loudness(audio.reshape(-1, 1))
                
                # Apply loudness normalization
                audio = pyln.normalize.loudness(audio, loudness, target_level)
            except (ImportError, AttributeError):
                warnings.warn("pyloudnorm not available, falling back to RMS normalization")
                audio = DataTransformer.normalize_audio(audio, "rms", target_level)
        
        return audio
    
    @staticmethod
    def split_audio(audio: np.ndarray, 
                   sample_rate: int,
                   segment_duration: float = 3.0,
                   overlap: float = 0.0) -> List[np.ndarray]:
        """
        Split audio into fixed-length segments
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List of audio segments
        """
        segment_length = int(segment_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_size = segment_length - overlap_samples
        
        if step_size <= 0:
            raise ValueError("Overlap must be less than segment duration")
        
        segments = []
        for start in range(0, len(audio) - segment_length + 1, step_size):
            end = start + segment_length
            segment = audio[start:end]
            segments.append(segment)
        
        return segments


class DataLoader:
    """Data loading and caching utilities"""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_audio(self, 
                  file_path: Union[str, Path],
                  sample_rate: Optional[int] = None,
                  mono: bool = True,
                  cache: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load audio file with optional resampling and caching
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate (None to keep original)
            mono: Convert to mono if True
            cache: Enable caching of loaded audio
            
        Returns:
            Tuple of (audio_data, actual_sample_rate)
        """
        file_path = Path(file_path)
        
        # Generate cache key
        cache_key = None
        if cache and self.cache_dir:
            key_str = f"{file_path}_{sample_rate}_{mono}"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            
            # Check if cached version exists
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    return cached_data['audio'], cached_data['sample_rate']
                except Exception as e:
                    warnings.warn(f"Failed to load cached audio: {e}")
        
        # Load audio from file
        if not HAS_AUDIO_LIBS:
            raise ImportError("librosa and soundfile are required for audio loading")
        
        try:
            audio, original_sr = librosa.load(str(file_path), sr=None, mono=mono)
        except Exception as e:
            raise IOError(f"Failed to load audio file {file_path}: {e}")
        
        # Resample if needed
        actual_sr = original_sr
        if sample_rate and sample_rate != original_sr:
            audio = DataTransformer.resample_audio(audio, original_sr, sample_rate)
            actual_sr = sample_rate
        
        # Cache the result
        if cache and self.cache_dir and cache_key:
            try:
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump({'audio': audio, 'sample_rate': actual_sr}, f)
            except Exception as e:
                warnings.warn(f"Failed to cache audio: {e}")
        
        return audio, actual_sr
    
    def batch_load_audio(self,
                        file_paths: List[Union[str, Path]],
                        sample_rate: Optional[int] = None,
                        mono: bool = True,
                        cache: bool = True,
                        max_workers: int = 4) -> List[Tuple[np.ndarray, int]]:
        """
        Load multiple audio files in parallel
        
        Args:
            file_paths: List of audio file paths
            sample_rate: Target sample rate
            mono: Convert to mono
            cache: Enable caching
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (audio_data, sample_rate) tuples
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.load_audio, path, sample_rate, mono, cache): path
                for path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Failed to load {path}: {e}")
                    results.append((np.array([]), 0))
        
        return results


class DatasetSplitter:
    """Utilities for splitting datasets into train/validation/test sets"""
    
    @staticmethod
    def create_speaker_independent_split(speaker_files: Dict[str, List[Path]],
                                       train_ratio: float = 0.7,
                                       val_ratio: float = 0.15,
                                       test_ratio: float = 0.15,
                                       min_files_per_split: int = 1,
                                       random_seed: int = 42) -> Dict[str, Dict[str, List[Path]]]:
        """
        Create speaker-independent dataset splits
        
        Args:
            speaker_files: Dictionary mapping speaker IDs to file paths
            train_ratio: Proportion of speakers for training
            val_ratio: Proportion of speakers for validation
            test_ratio: Proportion of speakers for testing
            min_files_per_split: Minimum files required per speaker
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train, validation, and test splits
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Filter speakers with enough files
        valid_speakers = {
            speaker: files for speaker, files in speaker_files.items()
            if len(files) >= min_files_per_split
        }
        
        if len(valid_speakers) < 3:
            raise ValueError(f"Need at least 3 speakers with {min_files_per_split}+ files, found {len(valid_speakers)}")
        
        # Set random seed
        random.seed(random_seed)
        speakers = list(valid_speakers.keys())
        random.shuffle(speakers)
        
        # Calculate split sizes
        n_speakers = len(speakers)
        n_train = int(n_speakers * train_ratio)
        n_val = int(n_speakers * val_ratio)
        n_test = n_speakers - n_train - n_val
        
        # Adjust if test set is too small
        if n_test < 1:
            n_test = 1
            n_train = n_speakers - n_val - n_test
        
        # Create splits
        train_speakers = speakers[:n_train]
        val_speakers = speakers[n_train:n_train+n_val]
        test_speakers = speakers[n_train+n_val:]
        
        # Build result dictionary
        result = {
            'train': {s: valid_speakers[s] for s in train_speakers},
            'val': {s: valid_speakers[s] for s in val_speakers},
            'test': {s: valid_speakers[s] for s in test_speakers}
        }
        
        return result
    
    @staticmethod
    def create_file_based_split(speaker_files: Dict[str, List[Path]],
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15,
                              min_files_per_speaker: int = 5,
                              random_seed: int = 42) -> Dict[str, Dict[str, List[Path]]]:
        """
        Create file-based dataset splits (speakers appear in all splits)
        
        Args:
            speaker_files: Dictionary mapping speaker IDs to file paths
            train_ratio: Proportion of files for training
            val_ratio: Proportion of files for validation
            test_ratio: Proportion of files for testing
            min_files_per_speaker: Minimum files required per speaker
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train, validation, and test splits
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Filter speakers with enough files
        valid_speakers = {
            speaker: files for speaker, files in speaker_files.items()
            if len(files) >= min_files_per_speaker
        }
        
        if len(valid_speakers) == 0:
            raise ValueError(f"No speakers with {min_files_per_speaker}+ files")
        
        # Set random seed
        random.seed(random_seed)
        
        result = {'train': {}, 'val': {}, 'test': {}}
        
        for speaker, files in valid_speakers.items():
            # Shuffle files for this speaker
            random.shuffle(files)
            
            # Calculate split sizes
            n_files = len(files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            n_test = n_files - n_train - n_val
            
            # Adjust if test set is too small
            if n_test < 1:
                n_test = 1
                n_val = max(0, n_val - 1)
            
            # Split files
            train_files = files[:n_train]
            val_files = files[n_train:n_train+n_val]
            test_files = files[n_train+n_val:]
            
            # Add to results
            result['train'][speaker] = train_files
            result['val'][speaker] = val_files
            result['test'][speaker] = test_files
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Example usage of DataValidator
    validator = DataValidator()
    
    # Test speaker label validation
    test_labels = ["speaker1", "invalid speaker", "s", "very_long_speaker_name_that_exceeds_limit", 123]
    for label in test_labels:
        result = validator.validate_speaker_label(label)
        print(f"Label: {label}, Valid: {result['is_valid']}, Errors: {result['errors']}")
    
    # Test LibriSpeech dataset structure validation
    dataset_path = Path("../../devDataset/dev-clean/LibriSpeech/dev-clean")
    if dataset_path.exists():
        result = validator.validate_dataset_structure(dataset_path, expected_structure="librispeech")
        print(f"Dataset valid: {result['is_valid']}")
        print(f"Speakers: {result['speaker_count']}, Files: {result['file_count']}")
        
        # Find all audio files
        audio_files = validator.find_audio_files(dataset_path)
        print(f"Found {sum(len(files) for files in audio_files.values())} audio files for {len(audio_files)} speakers")