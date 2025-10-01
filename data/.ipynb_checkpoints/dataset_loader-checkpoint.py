"""
LibriSpeech Dataset Loader for Chaotic Speaker Recognition Project
Integrated with existing audio_preprocessor and data_utils modules.
Optimized for LibriSpeech format with development and training focus.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TYPE_CHECKING
from collections import defaultdict, Counter
import random
import numpy as np
import sys
from pathlib import Path

# Setup project imports
try:
    from setup_imports import setup_project_imports
    setup_project_imports()
except ImportError:
    # Fallback method
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from features.traditional_features import MelSpectrogramExtractor, MFCCExtractor
from audio_preprocessor import AudioPreprocessingPipeline, create_preprocessing_pipeline
from data_utils import DataValidator, DataTransformer, DatasetSplitter
HAS_PROJECT_MODULES = True


try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    Dataset = None
    DataLoader = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
import warnings

def custom_audio_collate_fn(batch: List[Tuple[torch.Tensor, int]], 
                           max_length: int = 48000,  # 3 seconds at 16kHz
                           pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自定义音频collate函数，处理不同长度的音频数据
    
    Args:
        batch: List of (audio_tensor, label) tuples
        max_length: 最大音频长度（样本数）
        pad_value: 填充值
        
    Returns:
        Tuple of (padded_audio_batch, labels_batch)
    """
    audios, labels = zip(*batch)
    
    # 转换为tensor
    processed_audios = []
    for audio in audios:
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.float()
        else:
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # 确保音频是1D
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
            if audio_tensor.dim() > 1:  # 如果还是多维，取第一个通道
                audio_tensor = audio_tensor[0]
        
        # 截断过长的音频
        if len(audio_tensor) > max_length:
            audio_tensor = audio_tensor[:max_length]
        
        # 填充过短的音频
        if len(audio_tensor) < max_length:
            pad_length = max_length - len(audio_tensor)
            audio_tensor = F.pad(audio_tensor, (0, pad_length), value=pad_value)
        
        processed_audios.append(audio_tensor)
    
    # 堆叠成批次
    audio_batch = torch.stack(processed_audios, dim=0)
    labels_batch = torch.tensor(labels, dtype=torch.long)
    
    return audio_batch, labels_batch


def custom_feature_collate_fn(batch: List[Tuple[torch.Tensor, int]], 
                             max_time_frames: int = 300,  # 最大时间帧数
                             pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自定义特征collate函数，处理不同时间维度的特征数据（如Mel频谱图、MFCC）
    
    Args:
        batch: List of (feature_tensor, label) tuples where features are [feature_dim, time]
        max_time_frames: 最大时间帧数
        pad_value: 填充值
        
    Returns:
        Tuple of (padded_features_batch, labels_batch)
    """
    features, labels = zip(*batch)
    
    processed_features = []
    for feat in features:
        if isinstance(feat, torch.Tensor):
            feat_tensor = feat.float()
        else:
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
        
        # 确保特征是2D: [feature_dim, time]
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(0)
        elif feat_tensor.dim() > 2:
            feat_tensor = feat_tensor.squeeze()
            if feat_tensor.dim() > 2:
                feat_tensor = feat_tensor.view(feat_tensor.shape[0], -1)
        
        feature_dim, time_frames = feat_tensor.shape
        
        # 截断时间维度
        if time_frames > max_time_frames:
            feat_tensor = feat_tensor[:, :max_time_frames]
            time_frames = max_time_frames
        
        # 填充时间维度
        if time_frames < max_time_frames:
            pad_length = max_time_frames - time_frames
            feat_tensor = F.pad(feat_tensor, (0, pad_length), value=pad_value)
        
        processed_features.append(feat_tensor)
    
    # 堆叠成批次: [batch_size, feature_dim, time]
    features_batch = torch.stack(processed_features, dim=0)
    labels_batch = torch.tensor(labels, dtype=torch.long)
    
    return features_batch, labels_batch


def smart_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    智能collate函数，自动检测数据类型并应用相应的处理
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # 检查第一个样本来确定数据类型
    sample_data, _ = batch[0]
    if isinstance(sample_data, torch.Tensor):
        sample_tensor = sample_data
    else:
        sample_tensor = torch.tensor(sample_data)
    
    if sample_tensor.dim() == 1:
        # 原始音频数据
        return custom_audio_collate_fn(batch, max_length=48000)
    elif sample_tensor.dim() == 2:
        # 特征数据（Mel/MFCC）
        return custom_feature_collate_fn(batch, max_time_frames=300)
    else:
        raise ValueError(f"Unsupported tensor dimension: {sample_tensor.dim()}")


        
class LibriSpeechChaoticDataset:
    """
    LibriSpeech Dataset Loader for Chaotic Speaker Recognition
    
    Integrates with existing audio preprocessing pipeline and data utilities.
    Designed specifically for speaker recognition tasks using chaotic neural networks.
    Supports the standard LibriSpeech directory structure and provides efficient
    data loading with comprehensive preprocessing capabilities.
    """
    
    def __init__(self,
                 dataset_path: str = "/dataset/train-clean-100/LibriSpeech/train-clean-100/",
                 subsets: Optional[List[str]] = None,
                 preprocessing_config: Optional[Dict[str, Any]] = None,
                 min_samples_per_speaker: int = 1,
                 max_samples_per_speaker: Optional[int] = None,
                 target_num_speakers: Optional[int] = None,  # 添加这个参数
                 validation_config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[str] = None,
                 random_seed: int = 42):
        """
        Initialize LibriSpeech dataset for chaotic speaker recognition
        
        Args:
            dataset_path: Root path to LibriSpeech dataset
            subsets: List of subsets to load (None for auto-detection)
            preprocessing_config: Configuration for audio preprocessing pipeline
            min_samples_per_speaker: Minimum audio files per speaker
            max_samples_per_speaker: Maximum audio files per speaker (None for unlimited)
            validation_config: Configuration for data validation
            cache_dir: Directory for preprocessing cache
            random_seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.subsets = subsets
        self.min_samples_per_speaker = min_samples_per_speaker
        self.max_samples_per_speaker = max_samples_per_speaker
        self.random_seed = random_seed
        self.target_num_speakers = target_num_speakers
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Initialize preprocessing pipeline
        self.preprocessing_pipeline = self._create_preprocessing_pipeline(
            preprocessing_config, cache_dir
        )
        
        # Initialize data validator
        self.validator = DataValidator() if HAS_PROJECT_MODULES else None
        self.validation_config = validation_config or self._get_default_validation_config()
        
        # Dataset containers
        self.audio_files = []           # List of file metadata
        self.speaker_labels = []        # Corresponding speaker label indices
        self.speaker_to_idx = {}        # Speaker ID -> index mapping
        self.idx_to_speaker = {}        # Index -> speaker ID mapping
        self.num_speakers = 0
        
        # Load and validate dataset
        self._load_and_validate_dataset()
    
    def _create_preprocessing_pipeline(self, 
                                     config: Optional[Dict[str, Any]], 
                                     cache_dir: Optional[str]) -> Optional['AudioPreprocessingPipeline']:
        """Create audio preprocessing pipeline"""
        if not HAS_PROJECT_MODULES:
            return None
        
        # Default preprocessing configuration for speaker recognition
        default_config = {
            'loader': {
                'target_sample_rate': 16000,
                'mono': True,
                'normalize_on_load': False
            },
            'normalizer': {
                'method': 'rms',
                'target_level': 0.1,
                'clip_threshold': 1.0
            },
            'silence_trimmer': {
                'top_db': 30,
                'margin_seconds': 0.1
            },
            'noise_reducer': {
                'enabled': False  # Disabled by default for cleaner datasets
            },
            'augmenter': {
                'enabled': False,  # Enable during training if needed
                'augmentation_prob': 0.3
            }
        }
        
        # Merge with user config
        if config:
            # Deep merge configurations
            merged_config = {}
            for key in set(default_config.keys()) | set(config.keys()):
                if key in default_config and key in config:
                    if isinstance(default_config[key], dict) and isinstance(config[key], dict):
                        merged_config[key] = {**default_config[key], **config[key]}
                    else:
                        merged_config[key] = config[key]
                elif key in config:
                    merged_config[key] = config[key]
                else:
                    merged_config[key] = default_config[key]
            final_config = merged_config
        else:
            final_config = default_config
        
        return create_preprocessing_pipeline(config=final_config, cache_dir=cache_dir)
    
    def _get_default_validation_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            'min_duration': 0.5,      # Minimum audio duration in seconds
            'max_duration': 30.0,     # Increased maximum audio duration to 30 seconds
            'min_sample_rate': 8000,  # Minimum sample rate
            'max_sample_rate': 48000, # Maximum sample rate
        }
    
    def _load_and_validate_dataset(self) -> None:
        """Load LibriSpeech dataset and validate structure"""
        print(f"Loading LibriSpeech dataset from: {self.dataset_path}")
        
        # First, check if the path contains LibriSpeech structure
        if not self._has_librispeech_structure():
            print(f"Warning: Path {self.dataset_path} doesn't appear to have LibriSpeech structure")
            print("Trying to find audio files in any subdirectory structure...")
            
            # Try to find audio files in any structure
            all_speaker_files = self._find_audio_files_anywhere()
            
            if not all_speaker_files:
                raise ValueError(f"No audio files found in {self.dataset_path}")
                
            print(f"Found {sum(len(files) for files in all_speaker_files.values())} audio files "
                  f"from {len(all_speaker_files)} speakers in non-standard structure")
        else:
            # Validate dataset structure using data_utils
            if HAS_PROJECT_MODULES and self.validator:
                validation_result = self.validator.validate_dataset_structure(
                    self.dataset_path, expected_structure="librispeech"
                )
                
                if not validation_result['is_valid']:
                    warnings.warn(f"Dataset validation warnings: {validation_result['errors']}")
                
                if validation_result['warnings']:
                    warnings.warn(f"Dataset warnings: {validation_result['warnings']}")
                
                print(f"Dataset validation: {validation_result['speaker_count']} speakers, "
                      f"{validation_result['file_count']} files")
            
            # Auto-detect subsets if not specified
            if self.subsets is None:
                self.subsets = self._detect_subsets()
            
            print(f"Loading subsets: {self.subsets}")
            
            # Collect speaker files using data_utils
            if HAS_PROJECT_MODULES and self.validator:
                all_speaker_files = {}
                for subset in self.subsets:
                    subset_path = self._get_subset_path(subset)
                    if subset_path.exists():
                        subset_files = self.validator.find_audio_files(
                            subset_path, pattern="**/*.flac"
                        )
                        # Merge with existing speaker files
                        for speaker_id, files in subset_files.items():
                            if speaker_id not in all_speaker_files:
                                all_speaker_files[speaker_id] = []
                            all_speaker_files[speaker_id].extend(files)
            else:
                # Fallback manual detection
                all_speaker_files = self._manual_file_detection()
        
        # Filter and process speakers
        self._process_speaker_files(all_speaker_files)
        
        if self.num_speakers < 2:
            raise ValueError(f"Need at least 2 speakers, found {self.num_speakers}")
            
        print(f"Dataset loaded: {len(self.audio_files)} samples from {self.num_speakers} speakers")
    
    def _has_librispeech_structure(self) -> bool:
        """Check if the path has LibriSpeech-like structure"""
        # Check for common LibriSpeech directory names
        librispeech_dirs = {'train-clean-100', 'train-clean-360', 'train-other-500',
                           'dev-clean', 'dev-other', 'test-clean', 'test-other'}
        
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name in librispeech_dirs:
                return True
                
        # Also check if we have speaker directories directly
        speaker_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        if speaker_dirs:
            # Check if directories look like speaker IDs (numeric)
            for speaker_dir in speaker_dirs[:5]:  # Check first 5
                if speaker_dir.name.isdigit():
                    # Check if these directories contain audio files directly
                    audio_files = list(speaker_dir.rglob("*.flac"))
                    if audio_files:
                        return True
                    
        return False
    
    def _find_audio_files_anywhere(self) -> Dict[str, List[Path]]:
        """Find audio files anywhere in the directory structure"""
        speaker_files = defaultdict(list)
        
        # Find all FLAC files recursively
        flac_files = list(self.dataset_path.rglob("*.flac"))
        
        if not flac_files:
            # Try WAV files if no FLAC files found
            flac_files = list(self.dataset_path.rglob("*.wav"))
        
        # Group by directory name (use parent directory as speaker ID)
        for file_path in flac_files:
            # Use the parent directory name as speaker ID
            speaker_id = file_path.parent.name
            
            # If parent is a chapter directory, use grandparent as speaker ID
            if file_path.parent.parent != self.dataset_path and file_path.parent.parent.is_dir():
                speaker_id = file_path.parent.parent.name
                
            speaker_files[speaker_id].append(file_path)
            
        return dict(speaker_files)
    
    def _detect_subsets(self) -> List[str]:
        """Auto-detect available subsets in dataset"""
        detected_subsets = []
        
        # Check for common LibriSpeech subset names
        common_subsets = {'train-clean-100', 'train-clean-360', 'train-other-500',
                         'dev-clean', 'dev-other', 'test-clean', 'test-other'}
        
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name in common_subsets:
                detected_subsets.append(item.name)
        
        # If no standard subsets found, check for any directories
        if not detected_subsets:
            for item in self.dataset_path.iterdir():
                if item.is_dir():
                    detected_subsets.append(item.name)
            
        # If no subsets found, assume direct speaker structure
        if not detected_subsets:
            detected_subsets = ['direct']
            
        return detected_subsets
    
    def _get_subset_path(self, subset_name: str) -> Path:
        """Get path to subset directory"""
        if subset_name == 'direct':
            return self.dataset_path
        else:
            return self.dataset_path / subset_name
    
    def _manual_file_detection(self) -> Dict[str, List[Path]]:
        """Manual file detection fallback when data_utils not available"""
        speaker_files = defaultdict(list)
        
        for subset in self.subsets:
            subset_path = self._get_subset_path(subset)
            
            if not subset_path.exists():
                warnings.warn(f"Subset path not found: {subset_path}")
                continue
            
            # Find audio files in LibriSpeech structure
            for speaker_dir in subset_path.iterdir():
                if not speaker_dir.is_dir():
                    continue
                
                speaker_id = speaker_dir.name
                
                # Look for chapter directories
                chapter_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
                
                if chapter_dirs:
                    # Standard LibriSpeech structure with chapter directories
                    for chapter_dir in chapter_dirs:
                        # Find .flac files in chapter directory
                        for audio_file in chapter_dir.glob("*.flac"):
                            speaker_files[speaker_id].append(audio_file)
                else:
                    # Non-standard structure: audio files directly in speaker directory
                    for audio_file in speaker_dir.glob("*.flac"):
                        speaker_files[speaker_id].append(audio_file)
        
        return dict(speaker_files)
    
    def _process_speaker_files(self, speaker_files: Dict[str, List[Path]]) -> None:
        """Process and filter speaker files"""
        # Validate and filter speakers
        valid_speakers = {}
        
        for speaker_id, files in speaker_files.items():
            # Apply minimum sample requirement
            if len(files) < self.min_samples_per_speaker:
                continue
            
            # Apply maximum sample limit if specified
            processed_files = files
            if self.max_samples_per_speaker and len(files) > self.max_samples_per_speaker:
                processed_files = random.sample(files, self.max_samples_per_speaker)
            
            # Validate individual files if validator available
            if HAS_PROJECT_MODULES and self.validator:
                validated_files = []
                for file_path in processed_files:
                    validation_config = self.validation_config.copy()
                    if 'required_format' in validation_config:
                        del validation_config['required_format']
                    
                    validation_result = self.validator.validate_audio_properties(
                        file_path, **validation_config
                    )
                    if validation_result['is_valid']:
                        validated_files.append(file_path)
                    elif validation_result['errors']:
                        duration_errors = [e for e in validation_result['errors'] if 'Duration' in e]
                        if duration_errors and 'max_duration' in self.validation_config:
                            warnings.warn(f"Audio file {file_path} exceeds duration limit but will be included: {duration_errors}")
                            validated_files.append(file_path)
                        else:
                            warnings.warn(f"Invalid audio file {file_path}: {validation_result['errors']}")
                
                if len(validated_files) >= self.min_samples_per_speaker:
                    valid_speakers[speaker_id] = validated_files
            else:
                valid_speakers[speaker_id] = processed_files
        
        # CRITICAL FIX: Limit number of speakers BEFORE creating label mapping
        speaker_ids = sorted(valid_speakers.keys())
        
        # If we have a target number of speakers from config, limit here
        # This ensures labels stay in [0, num_speakers-1] range
        if hasattr(self, 'target_num_speakers') and self.target_num_speakers:
            if len(speaker_ids) > self.target_num_speakers:
                print(f"Limiting speakers from {len(speaker_ids)} to {self.target_num_speakers}")
                speaker_ids = speaker_ids[:self.target_num_speakers]
                # Filter valid_speakers to only include selected speakers
                valid_speakers = {spk_id: valid_speakers[spk_id] for spk_id in speaker_ids}
        
        # Create speaker mappings with continuous labels [0, num_speakers-1]
        self.speaker_to_idx = {speaker_id: idx for idx, speaker_id in enumerate(speaker_ids)}
        self.idx_to_speaker = {idx: speaker_id for speaker_id, idx in self.speaker_to_idx.items()}
        self.num_speakers = len(speaker_ids)
        
        print(f"Created label mapping for {self.num_speakers} speakers: labels in [0, {self.num_speakers-1}]")
        
        # Build sample list with metadata
        for speaker_id, files in valid_speakers.items():
            speaker_idx = self.speaker_to_idx[speaker_id]
            
            for file_path in files:
                file_metadata = self._extract_file_metadata(file_path, speaker_id)
                self.audio_files.append(file_metadata)
                self.speaker_labels.append(speaker_idx)
    
    def _extract_file_metadata(self, file_path: Path, speaker_id: str) -> Dict[str, Any]:
        """Extract metadata from LibriSpeech file path structure"""
        # LibriSpeech structure: subset/speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
        parts = file_path.parts
        
        metadata = {
            'file_path': str(file_path),
            'speaker_id': speaker_id,
            'file_name': file_path.stem,
            'subset': 'unknown',
            'chapter_id': 'unknown',
            'utterance_id': 'unknown'
        }
        
        # Extract chapter information
        if len(parts) >= 2:
            metadata['chapter_id'] = parts[-2]
        
        # Extract subset information
        if len(parts) >= 4:
            metadata['subset'] = parts[-4]
        
        # Extract utterance ID from filename
        if '-' in file_path.stem:
            filename_parts = file_path.stem.split('-')
            if len(filename_parts) >= 3:
                metadata['utterance_id'] = filename_parts[2]
        
        return metadata
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.audio_files)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Get a single sample with preprocessing
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (processed_audio, speaker_label, metadata)
        """
        if index < 0 or index >= len(self.audio_files):
            raise IndexError(f"Index {index} out of range for dataset size {len(self.audio_files)}")
        
        file_metadata = self.audio_files[index]
        speaker_label = self.speaker_labels[index]
        file_path = file_metadata['file_path']
        
        # Process audio through preprocessing pipeline
        try:
            if self.preprocessing_pipeline:
                result = self.preprocessing_pipeline.process_file(file_path)
                audio, sample_rate = result
                
                # Handle segmented audio (take first segment if multiple)
                if isinstance(audio, list):
                    audio = audio[0] if audio else np.zeros(16000)
            else:
                # Fallback: basic loading
                import librosa
                audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            
        except Exception as e:
            warnings.warn(f"Failed to process audio file {file_path}: {str(e)}")
            # Return silence as fallback
            audio = np.zeros(16000)
            sample_rate = 16000
        
        # Update metadata with processing info
        enhanced_metadata = {
            **file_metadata,
            'speaker_label': speaker_label,
            'audio_length': len(audio) / sample_rate,
            'sample_rate': sample_rate,
            'processed': True
        }
        
        return audio, speaker_label, enhanced_metadata
    
    def get_speaker_name(self, speaker_index: int) -> str:
        """Get speaker ID from index"""
        return self.idx_to_speaker.get(speaker_index, f"unknown_{speaker_index}")
    
    def get_speaker_index(self, speaker_id: str) -> int:
        """Get speaker index from ID"""
        return self.speaker_to_idx.get(speaker_id, -1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        speaker_counts = Counter(self.speaker_labels)
        
        stats = {
            'dataset_path': str(self.dataset_path),
            'subsets': self.subsets,
            'total_samples': len(self.audio_files),
            'num_speakers': self.num_speakers,
            'samples_per_speaker': {
                self.idx_to_speaker[idx]: count 
                for idx, count in speaker_counts.items()
            },
            'min_samples_per_speaker': min(speaker_counts.values()) if speaker_counts else 0,
            'max_samples_per_speaker': max(speaker_counts.values()) if speaker_counts else 0,
            'mean_samples_per_speaker': np.mean(list(speaker_counts.values())) if speaker_counts else 0,
            'std_samples_per_speaker': np.std(list(speaker_counts.values())) if speaker_counts else 0,
        }
        
        # Add preprocessing statistics if available
        if self.preprocessing_pipeline:
            preprocessing_stats = self.preprocessing_pipeline.get_statistics()
            stats['preprocessing'] = preprocessing_stats
        
        return stats
    
    def create_data_splits(self, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          split_method: str = 'file_based',
                          min_files_per_split: int = 1) -> Dict[str, 'LibriSpeechChaoticDataset']:
        """
        Create dataset splits using data_utils
        
        Args:
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
            split_method: 'speaker_independent' or 'file_based'
            min_files_per_split: Minimum files required per split
            
        Returns:
            Dictionary with train, val, test dataset splits
        """
        if not HAS_PROJECT_MODULES:
            raise ImportError("data_utils module required for dataset splitting")
        
        # Group files by speaker
        speaker_files = defaultdict(list)
        for file_metadata, label in zip(self.audio_files, self.speaker_labels):
            speaker_id = self.idx_to_speaker[label]
            speaker_files[speaker_id].append(Path(file_metadata['file_path']))
        
        # Create splits using DatasetSplitter
        splitter = DatasetSplitter()
        
        if split_method == 'speaker_independent':
            splits = splitter.create_speaker_independent_split(
                speaker_files, train_ratio, val_ratio, test_ratio,
                min_files_per_split, self.random_seed
            )
        elif split_method == 'file_based':
            splits = splitter.create_file_based_split(
                speaker_files, train_ratio, val_ratio, test_ratio,
                min_files_per_split * 3,  # Need more files for file-based splitting
                self.random_seed
            )
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        # Create new dataset instances for each split
        result_datasets = {}
        
        for split_name, split_speaker_files in splits.items():
            # Create split dataset
            split_dataset = self._create_split_dataset(split_speaker_files, split_name)
            result_datasets[split_name] = split_dataset
        
        return result_datasets
    
    def _create_split_dataset(self, split_speaker_files, split_name):
        split_dataset = LibriSpeechChaoticDataset.__new__(LibriSpeechChaoticDataset)
        split_dataset.__dict__.update(self.__dict__)
        
        # ❌ 删除这些会重新创建映射的代码：
        # split_speaker_ids = sorted(split_speaker_files.keys())
        # split_dataset.speaker_to_idx = {speaker_id: idx for idx, speaker_id in enumerate(split_speaker_ids)}
        # split_dataset.idx_to_speaker = {idx: speaker_id for speaker_id, idx in split_dataset.speaker_to_idx.items()}
        # split_dataset.num_speakers = len(split_speaker_ids)
        
        # ✅ 使用父数据集的映射（已有，保持不变）
        split_dataset.speaker_to_idx = self.speaker_to_idx
        split_dataset.idx_to_speaker = self.idx_to_speaker  
        split_dataset.num_speakers = self.num_speakers
        
        # 重建样本列表
        split_dataset.audio_files = []
        split_dataset.speaker_labels = []
        
        for speaker_id, file_paths in split_speaker_files.items():
            speaker_idx = self.speaker_to_idx[speaker_id]  # 使用父数据集的映射
            
            for file_path in file_paths:
                # 找到对应的metadata
                file_metadata = None
                for metadata in self.audio_files:
                    if metadata['file_path'] == str(file_path):
                        file_metadata = metadata.copy()
                        break
                
                if file_metadata:
                    split_dataset.audio_files.append(file_metadata)
                    split_dataset.speaker_labels.append(speaker_idx)
        
        # 不打印错误的标签范围信息
        print(f"分割数据集 {split_name}: {len(split_dataset.audio_files)} 样本, "
              f"{len(set(split_dataset.speaker_labels))} 说话人")
        
        return split_dataset
    
    def create_pytorch_dataset(self, 
                              transform: Optional[Callable] = None,
                              return_metadata: bool = False) -> 'LibriSpeechPyTorchDataset':
        """Create PyTorch-compatible dataset wrapper"""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for PyTorch dataset creation")
        
        return LibriSpeechPyTorchDataset(self, transform, return_metadata)
    
    def enable_augmentation(self, augmentation_config: Optional[Dict[str, Any]] = None) -> None:
        """Enable audio augmentation in preprocessing pipeline"""
        if not self.preprocessing_pipeline:
            warnings.warn("No preprocessing pipeline available for augmentation")
            return
        
        # Default augmentation config for training
        default_aug_config = {
            'augmenter': {
                'enabled': True,
                'augmentation_prob': 0.3,
                'augmentation_config': {
                    'time_stretch': {'rate_range': (0.9, 1.1), 'prob': 0.2},
                    'pitch_shift': {'semitone_range': (-1, 1), 'prob': 0.2},
                    'noise_injection': {'noise_factor_range': (0.001, 0.005), 'prob': 0.1},
                    'volume_change': {'gain_range': (-3, 3), 'prob': 0.3}
                }
            }
        }
        
        # Use provided config or default
        aug_config = augmentation_config or default_aug_config
        
        # Update preprocessing pipeline config
        print("Enabling audio augmentation for training...")
        self.preprocessing_pipeline.config.update(aug_config)
        # Rebuild pipeline with new config
        self.preprocessing_pipeline.components = self.preprocessing_pipeline._build_pipeline()
    
    def save_dataset_info(self, output_path: str) -> None:
        """Save comprehensive dataset information"""
        dataset_info = {
            'configuration': {
                'dataset_path': str(self.dataset_path),
                'subsets': self.subsets,
                'min_samples_per_speaker': self.min_samples_per_speaker,
                'max_samples_per_speaker': self.max_samples_per_speaker,
                'random_seed': self.random_seed,
                'validation_config': self.validation_config
            },
            'statistics': self.get_statistics(),
            'speaker_mapping': {
                'speaker_to_idx': self.speaker_to_idx,
                'idx_to_speaker': self.idx_to_speaker
            },
            'preprocessing_config': (
                self.preprocessing_pipeline.get_pipeline_config() 
                if self.preprocessing_pipeline else None
            ),
            'created_at': str(np.datetime64('now'))
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        print(f"Dataset information saved to: {output_path}")


class LibriSpeechPyTorchDataset(Dataset if HAS_TORCH else object):
    """PyTorch-compatible wrapper for LibriSpeechChaoticDataset"""
    
    def __init__(self, 
                 chaotic_dataset: LibriSpeechChaoticDataset,
                 transform: Optional[Callable] = None,
                 return_metadata: bool = False):
        """
        Initialize PyTorch dataset wrapper
        
        Args:
            chaotic_dataset: LibriSpeechChaoticDataset instance
            transform: Optional transform function for audio data
            return_metadata: Whether to return metadata
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LibriSpeechPyTorchDataset")
        
        self.dataset = chaotic_dataset
        self.transform = transform
        self.return_metadata = return_metadata
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        """Get item as PyTorch tensors"""
        audio, label, metadata = self.dataset[index]
        
        # Convert to PyTorch tensors
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply transform if provided
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        
        if self.return_metadata:
            return audio_tensor, label_tensor, metadata
        else:
            return audio_tensor, label_tensor
    
    @property
    def num_classes(self) -> int:
        """Get number of speaker classes"""
        return self.dataset.num_speakers


def create_chaotic_speaker_dataset(
    dataset_path: str = "../../dataset/train-clean-100/LibriSpeech/train-clean-100/",
    preprocessing_config: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = "./cache/librispeech",
    target_num_speakers: Optional[int] = None,  # 添加这个参数
    **kwargs
) -> LibriSpeechChaoticDataset:
    """
    Convenience function to create LibriSpeech dataset for chaotic speaker recognition
    
    Args:
        dataset_path: Path to LibriSpeech dataset
        preprocessing_config: Audio preprocessing configuration
        cache_dir: Cache directory for processed audio
        **kwargs: Additional dataset parameters
        
    Returns:
        Configured LibriSpeechChaoticDataset instance
    """
    # Default preprocessing optimized for chaotic speaker recognition
    if preprocessing_config is None:
        preprocessing_config = {
            'loader': {
                'target_sample_rate': 16000,
                'mono': True
            },
            'normalizer': {
                'method': 'rms',
                'target_level': 0.1
            },
            'silence_trimmer': {
                'top_db': 25,
                'margin_seconds': 0.05
            }
        }
    
    return LibriSpeechChaoticDataset(
        dataset_path=dataset_path,
        preprocessing_config=preprocessing_config,
        cache_dir=cache_dir,
        target_num_speakers=target_num_speakers,  # 传递参数
        **kwargs
    )


def create_pytorch_dataloaders(
    dataset: LibriSpeechChaoticDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    **split_kwargs
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits
    
    Args:
        dataset: LibriSpeechChaoticDataset instance
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        train_transform: Transform function for training data
        **split_kwargs: Arguments for dataset splitting
        
    Returns:
        Dictionary with DataLoader instances for each split
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for DataLoader creation")
    
    # Create dataset splits
    splits = dataset.create_data_splits(**split_kwargs)
    
    dataloaders = {}
    
    for split_name, split_dataset in splits.items():
        # Use transform only for training data
        transform = train_transform if split_name == 'train' else None
        
        # Create PyTorch dataset
        pytorch_dataset = split_dataset.create_pytorch_dataset(
            transform=transform, 
            return_metadata=False
        )
        
        # Create DataLoader
        shuffle = (split_name == 'train')
        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == 'train')
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders

def create_speaker_dataloaders_with_collate(
    data_dir: str,
    batch_size: int = 32,
    sample_rate: int = 16000,
    max_length: float = 3.0,
    num_workers: int = 0,  # 设置为0避免多进程问题
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    collate_fn=None,
    target_num_speakers: Optional[int] = None  # 添加这个参数
):
    """创建说话人识别的 DataLoader，支持自定义collate函数"""
    print(f"尝试从 {data_dir} 加载数据集...")
    
    # 检查路径是否存在
    if not os.path.exists(data_dir):
        print(f"数据路径不存在: {data_dir}")
        print("使用模拟数据进行训练...")
        return _create_simple_mock_dataloaders_with_collate(batch_size, sample_rate, seed, collate_fn)
    
    # 如果路径存在，尝试加载真实数据
    try:
        dataset = create_chaotic_speaker_dataset(
            dataset_path=data_dir,
            min_samples_per_speaker=2,
            max_samples_per_speaker=30,
            target_num_speakers=target_num_speakers 
        )
        
        # 创建数据分割
        splits = dataset.create_data_splits(
            train_ratio=train_split,
            val_ratio=val_split,
            test_ratio=1.0 - train_split - val_split
        )
        
        # 如果没有指定collate函数，使用智能collate函数
        if collate_fn is None:
            collate_fn = smart_collate_fn
        
        # 创建dataloaders
        dataloaders = {}
        for split_name, split_dataset in splits.items():
            pytorch_dataset = split_dataset.create_pytorch_dataset(return_metadata=False)
            
            dataloader = DataLoader(
                pytorch_dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                num_workers=num_workers,
                pin_memory=False,  # 避免CUDA相关问题
                drop_last=(split_name == 'train'),
                collate_fn=collate_fn  # 使用自定义collate函数
            )
            dataloaders[split_name] = dataloader
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test']
        
    except Exception as e:
        print(f"真实数据加载失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return _create_simple_mock_dataloaders_with_collate(batch_size, sample_rate, seed, collate_fn)



def _create_simple_mock_dataloaders_with_collate(batch_size: int, sample_rate: int, seed: int, collate_fn=None):
    """创建简单的模拟数据加载器，支持自定义collate函数"""
    import torch
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleMockDataset(Dataset):
        def __init__(self, num_samples: int, num_classes: int = 10):
            self.num_samples = num_samples
            self.num_classes = num_classes
            torch.manual_seed(seed)
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            speaker_id = idx % self.num_classes
            
            # 创建不同长度的音频来测试collate函数
            length_variation = torch.randint(sample_rate//2, sample_rate*2, (1,)).item()
            
            # 为每个说话人创建不同的信号特征
            base_freq = 200 + speaker_id * 100
            t = torch.linspace(0, length_variation/sample_rate, length_variation)
            
            # 生成有特征的音频
            audio = torch.sin(2 * torch.pi * base_freq * t)
            audio += 0.3 * torch.sin(2 * torch.pi * base_freq * 1.5 * t)
            audio += 0.1 * torch.randn_like(audio)
            
            # 归一化
            audio = audio / torch.max(torch.abs(audio))
            
            return audio.float(), torch.tensor(speaker_id, dtype=torch.long)
    
    # 如果没有指定collate函数，使用智能collate函数
    if collate_fn is None:
        collate_fn = smart_collate_fn
    
    # 创建数据集
    train_dataset = SimpleMockDataset(800, 10)
    val_dataset = SimpleMockDataset(160, 10)
    test_dataset = SimpleMockDataset(160, 10)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"模拟数据集创建成功: 10个说话人，训练集 {len(train_dataset)} 样本（支持变长音频）")
    return train_loader, val_loader, test_loader
    
# 修改 create_speaker_dataloaders 函数
def create_speaker_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    sample_rate: int = 16000,
    max_length: float = 3.0,
    num_workers: int = 0,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    target_num_speakers: Optional[int] = None  # 添加这个参数
):
    """修改后的create_speaker_dataloaders函数，自动使用smart_collate_fn"""
    return create_speaker_dataloaders_with_collate(
        data_dir=data_dir,
        batch_size=batch_size,
        sample_rate=sample_rate,
        max_length=max_length,
        num_workers=num_workers,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        collate_fn=smart_collate_fn,
        target_num_speakers=target_num_speakers  # 传递参数
    )

# 特定用途的collate函数创建器
def create_mel_collate_fn(n_mels: int = 80, max_time_frames: int = 300):
    """创建Mel频谱图专用的collate函数"""
    def mel_collate_fn(batch):
        return custom_feature_collate_fn(batch, max_time_frames=max_time_frames)
    return mel_collate_fn


def create_mfcc_collate_fn(n_mfcc: int = 13, max_time_frames: int = 300):
    """创建MFCC专用的collate函数"""
    def mfcc_collate_fn(batch):
        return custom_feature_collate_fn(batch, max_time_frames=max_time_frames)
    return mfcc_collate_fn


def create_audio_collate_fn(max_length_seconds: float = 3.0, sample_rate: int = 16000):
    """创建原始音频专用的collate函数"""
    max_length = int(max_length_seconds * sample_rate)
    def audio_collate_fn(batch):
        return custom_audio_collate_fn(batch, max_length=max_length)
    return audio_collate_fn


# 测试collate函数的功能
def test_collate_functions():
    """测试各种collate函数"""
    print("测试自定义collate函数...")
    
    # 测试音频collate函数
    print("1. 测试音频collate函数")
    test_audio_batch = [
        (torch.randn(24000), 0),  # 1.5s
        (torch.randn(48000), 1),  # 3.0s  
        (torch.randn(16000), 0),  # 1.0s
    ]
    
    try:
        audio_batch, labels = custom_audio_collate_fn(test_audio_batch)
        print(f"✓ 音频collate成功: {audio_batch.shape}, {labels.shape}")
    except Exception as e:
        print(f"✗ 音频collate失败: {e}")
    
    # 测试特征collate函数
    print("2. 测试特征collate函数")
    test_feature_batch = [
        (torch.randn(80, 200), 0),  # 80 mel bins, 200 frames
        (torch.randn(80, 150), 1),  # 80 mel bins, 150 frames
        (torch.randn(80, 350), 0),  # 80 mel bins, 350 frames
    ]
    
    try:
        feature_batch, labels = custom_feature_collate_fn(test_feature_batch)
        print(f"✓ 特征collate成功: {feature_batch.shape}, {labels.shape}")
    except Exception as e:
        print(f"✗ 特征collate失败: {e}")
    
    # 测试智能collate函数
    print("3. 测试智能collate函数")
    try:
        audio_batch, labels = smart_collate_fn(test_audio_batch)
        print(f"✓ 智能collate (音频)成功: {audio_batch.shape}, {labels.shape}")
        
        feature_batch, labels = smart_collate_fn(test_feature_batch)
        print(f"✓ 智能collate (特征)成功: {feature_batch.shape}, {labels.shape}")
    except Exception as e:
        print(f"✗ 智能collate失败: {e}")

        
# Example usage and testing
if __name__ == "__main__":
    test_collate_functions()
    print("LibriSpeech Chaotic Speaker Recognition Dataset Test")
    print("=" * 60)
    
    # Test dataset creation
    dataset_path = "../../dataset/train-clean-100/LibriSpeech/train-clean-100/"
    
    try:
        # Create dataset with project integration
        dataset = create_chaotic_speaker_dataset(
            dataset_path=dataset_path,
            min_samples_per_speaker=3,
            max_samples_per_speaker=20,  # Limit for testing
            cache_dir="./cache/test"
        )
        
        # Display dataset statistics
        stats = dataset.get_statistics()
        print("Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Speakers: {stats['num_speakers']}")
        print(f"  Samples per speaker: {stats['mean_samples_per_speaker']:.1f} ± {stats['std_samples_per_speaker']:.1f}")
        
        # Test sample loading
        print(f"\nTesting sample loading:")
        for i in range(min(3, len(dataset))):
            audio, label, metadata = dataset[i]
            print(f"  Sample {i}: Audio {audio.shape}, Speaker: {metadata['speaker_id']}, "
                  f"Duration: {metadata['audio_length']:.2f}s")
        
        # Test data splitting
        print(f"\nTesting dataset splitting:")
        splits = dataset.create_data_splits(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            split_method='file_based'
        )
        
        for split_name, split_dataset in splits.items():
            split_stats = split_dataset.get_statistics()
            print(f"  {split_name}: {split_stats['total_samples']} samples, "
                  f"{split_stats['num_speakers']} speakers")
        
        # Test PyTorch integration
        if HAS_TORCH:
            print(f"\nTesting PyTorch DataLoaders:")
            dataloaders = create_pytorch_dataloaders(
                dataset, 
                batch_size=4, 
                num_workers=0,  # Avoid multiprocessing in testing
                train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
            )
            
            for split_name, dataloader in dataloaders.items():
                try:
                    batch = next(iter(dataloader))
                    audio_batch, label_batch = batch
                    print(f"  {split_name}: Audio batch {audio_batch.shape}, "
                          f"Label batch {label_batch.shape}")
                except Exception as e:
                    print(f"  {split_name}: Error loading batch - {e}")
        
        # Save dataset information
        dataset.save_dataset_info("dataset_info_chaotic.json")
        
        print(f"\nIntegrated LibriSpeech dataset test completed successfully!")
        
    except FileNotFoundError:
        print(f"Dataset path not found: {dataset_path}")
        print("Please ensure the LibriSpeech dataset is available at the specified path.")
    except ImportError as e:
        print(f"Missing required modules: {e}")
        print("Please ensure audio_preprocessor.py and data_utils.py are available.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()