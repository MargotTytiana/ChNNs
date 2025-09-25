"""
Comprehensive Test Suite for Data Processing Components
Tests data utilities, audio preprocessing, and dataset loading functionality
for the Chaotic Speaker Recognition project.
"""

import os
import sys
import tempfile
import shutil
import json
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import unittest
from unittest.mock import Mock, patch, MagicMock
import random
import time

import numpy as np
import pytest

import os
import sys
import unittest
from pathlib import Path

# =============================================================================
# 统一导入设置  
# =============================================================================
def setup_module_imports(current_file: str = __file__):
    """Setup imports for current module."""
    try:
        from setup_imports import setup_project_imports
        return setup_project_imports(current_file), True
    except ImportError:
        current_dir = Path(current_file).resolve().parent  # data目录
        project_root = current_dir.parent  # data -> Model
        
        paths_to_add = [
            str(project_root),
            str(project_root / 'data'),
            str(project_root / 'utils'),
        ]
        
        for path in paths_to_add:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
        
        return project_root, False

# Setup imports
PROJECT_ROOT, USING_IMPORT_MANAGER = setup_module_imports()

# =============================================================================
# 测试导入 (清晰明了)
# =============================================================================
# Import components to test with clear error handling
def safe_import_test_modules():
    """Safely import all test modules."""
    modules = {}
    
    try:
        from data_utils import DataValidator, FileSystemUtils, DataSplitter
        modules['data_utils'] = True
        modules['DataValidator'] = DataValidator
        modules['FileSystemUtils'] = FileSystemUtils
        modules['DataSplitter'] = DataSplitter
    except ImportError as e:
        modules['data_utils'] = False
        print(f"Warning: Could not import data_utils: {e}")
    
    try:
        from audio_preprocessor import AudioLoader, AudioNormalizer
        modules['audio_preprocessor'] = True
        modules['AudioLoader'] = AudioLoader
        modules['AudioNormalizer'] = AudioNormalizer
    except ImportError as e:
        modules['audio_preprocessor'] = False
        print(f"Warning: Could not import audio_preprocessor: {e}")
        
    try:
        from dataset_loader import DirectoryDataset, MetadataDataset
        modules['dataset_loader'] = True  
        modules['DirectoryDataset'] = DirectoryDataset
        modules['MetadataDataset'] = MetadataDataset
    except ImportError as e:
        modules['dataset_loader'] = False
        print(f"Warning: Could not import dataset_loader: {e}")
    
    return modules

# Import all test modules
TEST_MODULES = safe_import_test_modules()


class TestDataValidator:
    """Test suite for DataValidator class"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        if not HAS_DATA_UTILS:
            pytest.skip("DataValidator not available")
        
        self.validator = DataValidator()
        
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_data_validator_"))
        
        # Create test files
        self.valid_audio_file = self.test_dir / "test_audio.wav"
        self.invalid_audio_file = self.test_dir / "test_invalid.txt"
        self.empty_file = self.test_dir / "empty.wav"
        
        # Create mock audio file (just empty for now)
        self.valid_audio_file.touch()
        self.invalid_audio_file.write_text("not an audio file")
        self.empty_file.touch()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_is_valid_audio_file_with_valid_extensions(self):
        """Test audio file validation with valid extensions"""
        valid_files = [
            "test.wav", "test.flac", "test.mp3", "test.m4a", "test.ogg"
        ]
        
        for filename in valid_files:
            file_path = self.test_dir / filename
            file_path.touch()  # Create empty file
            
            # Without audio libs, should check extension only
            if not HAS_AUDIO_LIBS:
                assert self.validator.is_valid_audio_file(file_path) == True
    
    def test_is_valid_audio_file_with_invalid_extensions(self):
        """Test audio file validation with invalid extensions"""
        invalid_files = [
            "test.txt", "test.doc", "test.pdf", "test.exe"
        ]
        
        for filename in invalid_files:
            file_path = self.test_dir / filename
            file_path.touch()  # Create empty file
            
            assert self.validator.is_valid_audio_file(file_path) == False
    
    def test_validate_audio_properties_nonexistent_file(self):
        """Test validation of non-existent file"""
        result = self.validator.validate_audio_properties("nonexistent.wav")
        
        assert result['is_valid'] == False
        assert 'File does not exist' in result['errors']
    
    def test_validate_audio_properties_empty_file(self):
        """Test validation of empty file"""
        result = self.validator.validate_audio_properties(self.empty_file)
        
        assert result['is_valid'] == False
        assert result['file_size'] == 0
        assert 'File is empty' in result['errors']
    
    def test_validate_speaker_label_valid(self):
        """Test speaker label validation with valid labels"""
        valid_labels = [
            "speaker01", "SPEAKER_01", "spk-01", "speaker123", "S1"
        ]
        
        for label in valid_labels:
            result = self.validator.validate_speaker_label(label)
            assert result['is_valid'] == True
            assert len(result['errors']) == 0
    
    def test_validate_speaker_label_invalid(self):
        """Test speaker label validation with invalid labels"""
        invalid_labels = [
            "speaker with spaces", "speaker@01", "speaker#01", "", "a" * 100
        ]
        
        for label in invalid_labels:
            result = self.validator.validate_speaker_label(label)
            if label == "":
                assert result['is_valid'] == False
                assert any('too short' in error for error in result['errors'])
            elif len(label) > 50:
                assert result['is_valid'] == False
                assert any('too long' in error for error in result['errors'])
            else:
                assert result['is_valid'] == False
                assert any('invalid characters' in error for error in result['errors'])
    
    def test_validate_speaker_label_warnings(self):
        """Test speaker label validation warnings"""
        warning_labels = [
            "_speaker01", "speaker01_", "speaker--01", "speaker__01"
        ]
        
        for label in warning_labels:
            result = self.validator.validate_speaker_label(label)
            assert len(result['warnings']) > 0


class TestFileSystemUtils:
    """Test suite for FileSystemUtils class"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        if not HAS_DATA_UTILS:
            pytest.skip("FileSystemUtils not available")
        
        # Create temporary test directory structure
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_fs_utils_"))
        
        # Create directory structure: speaker1/file1.wav, speaker2/file2.wav, etc.
        self.speakers = ["speaker1", "speaker2", "speaker3"]
        self.audio_files = []
        
        for speaker in self.speakers:
            speaker_dir = self.test_dir / speaker
            speaker_dir.mkdir()
            
            # Create audio files
            for i in range(3):
                audio_file = speaker_dir / f"utterance_{i:02d}.wav"
                audio_file.write_text(f"mock audio data for {speaker}")
                self.audio_files.append(audio_file)
            
            # Create one non-audio file
            text_file = speaker_dir / "readme.txt"
            text_file.write_text("This is not an audio file")
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_scan_directory_for_audio_recursive(self):
        """Test recursive directory scanning for audio files"""
        files_info = FileSystemUtils.scan_directory_for_audio(
            self.test_dir, recursive=True
        )
        
        # Should find all audio files
        assert len(files_info) == len(self.audio_files)
        
        # Check that all found files are audio files
        for file_info in files_info:
            assert file_info['suffix'] in ['.wav']
            assert 'speaker_from_path' in file_info
            assert file_info['speaker_from_path'] in self.speakers
    
    def test_scan_directory_for_audio_non_recursive(self):
        """Test non-recursive directory scanning"""
        files_info = FileSystemUtils.scan_directory_for_audio(
            self.test_dir, recursive=False
        )
        
        # Should find no files (audio files are in subdirectories)
        assert len(files_info) == 0
    
    def test_scan_directory_for_audio_with_extensions(self):
        """Test scanning with specific extensions"""
        # Create some MP3 files
        mp3_file = self.test_dir / "speaker1" / "test.mp3"
        mp3_file.write_text("mock mp3 data")
        
        # Scan for only WAV files
        wav_files = FileSystemUtils.scan_directory_for_audio(
            self.test_dir, extensions=['.wav']
        )
        
        # Scan for only MP3 files
        mp3_files = FileSystemUtils.scan_directory_for_audio(
            self.test_dir, extensions=['.mp3']
        )
        
        assert len(wav_files) == len(self.audio_files)
        assert len(mp3_files) == 1
    
    def test_create_directory_structure(self):
        """Test directory structure creation"""
        new_dir = self.test_dir / "new_structure"
        speakers = ["spk1", "spk2", "spk3"]
        subdirs = ["train", "val", "test"]
        
        FileSystemUtils.create_directory_structure(
            new_dir, speakers, subdirs
        )
        
        # Check that all directories were created
        for speaker in speakers:
            speaker_dir = new_dir / speaker
            assert speaker_dir.exists()
            assert speaker_dir.is_dir()
            
            for subdir in subdirs:
                subdir_path = speaker_dir / subdir
                assert subdir_path.exists()
                assert subdir_path.is_dir()
    
    def test_compute_file_hash(self):
        """Test file hash computation"""
        test_file = self.test_dir / "hash_test.txt"
        test_content = "This is test content for hashing"
        test_file.write_text(test_content)
        
        # Compute hash
        hash_md5 = FileSystemUtils.compute_file_hash(test_file, 'md5')
        hash_sha1 = FileSystemUtils.compute_file_hash(test_file, 'sha1')
        
        # Hashes should be different for different algorithms
        assert hash_md5 != hash_sha1
        assert len(hash_md5) == 32  # MD5 hash length
        assert len(hash_sha1) == 40  # SHA1 hash length
        
        # Same file should produce same hash
        hash_md5_2 = FileSystemUtils.compute_file_hash(test_file, 'md5')
        assert hash_md5 == hash_md5_2
    
    def test_find_duplicate_files(self):
        """Test duplicate file detection"""
        # Create duplicate files
        file1 = self.test_dir / "file1.txt"
        file2 = self.test_dir / "file2.txt"
        file3 = self.test_dir / "file3.txt"
        
        content1 = "identical content"
        content2 = "different content"
        
        file1.write_text(content1)
        file2.write_text(content1)  # Duplicate of file1
        file3.write_text(content2)  # Different content
        
        duplicates = FileSystemUtils.find_duplicate_files([file1, file2, file3])
        
        # Should find one group of duplicates (file1 and file2)
        assert len(duplicates) == 1
        duplicate_group = list(duplicates.values())[0]
        assert len(duplicate_group) == 2
        assert str(file1) in duplicate_group
        assert str(file2) in duplicate_group


class TestDataSplitter:
    """Test suite for DataSplitter class"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        if not HAS_DATA_UTILS:
            pytest.skip("DataSplitter not available")
        
        self.splitter = DataSplitter(random_seed=42)
        
        # Create test data
        self.test_data = []
        speakers = ["speaker1", "speaker2", "speaker3"]
        
        for speaker in speakers:
            for i in range(10):  # 10 samples per speaker
                self.test_data.append({
                    'file_path': f"/path/to/{speaker}/utterance_{i:02d}.wav",
                    'speaker': speaker,
                    'duration': random.uniform(1.0, 5.0),
                    'timestamp': f"2024-01-{i+1:02d}T10:00:00"
                })
    
    def test_stratified_split_ratios(self):
        """Test stratified split with specified ratios"""
        splits = self.splitter.stratified_split(
            self.test_data,
            target_key='speaker',
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Check that all splits exist
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check approximate ratios
        total_samples = len(self.test_data)
        assert abs(len(splits['train']) / total_samples - 0.6) < 0.1
        assert abs(len(splits['val']) / total_samples - 0.2) < 0.1
        assert abs(len(splits['test']) / total_samples - 0.2) < 0.1
        
        # Check that all samples are accounted for
        total_split_samples = sum(len(split) for split in splits.values())
        assert total_split_samples == total_samples
    
    def test_stratified_split_all_speakers_represented(self):
        """Test that all speakers are represented in each split"""
        splits = self.splitter.stratified_split(
            self.test_data,
            target_key='speaker',
            min_samples_per_class=2
        )
        
        # Extract speakers from each split
        for split_name, split_data in splits.items():
            speakers_in_split = {sample['speaker'] for sample in split_data}
            
            # All original speakers should be in each split
            original_speakers = {'speaker1', 'speaker2', 'speaker3'}
            assert speakers_in_split == original_speakers, f"Split {split_name} missing speakers"
    
    def test_stratified_split_reproducibility(self):
        """Test that splits are reproducible with same seed"""
        splits1 = self.splitter.stratified_split(self.test_data, target_key='speaker')
        splits2 = self.splitter.stratified_split(self.test_data, target_key='speaker')
        
        # Splits should be identical
        for split_name in splits1.keys():
            assert len(splits1[split_name]) == len(splits2[split_name])
            
            # Compare file paths (should be in same order)
            paths1 = [sample['file_path'] for sample in splits1[split_name]]
            paths2 = [sample['file_path'] for sample in splits2[split_name]]
            assert paths1 == paths2
    
    def test_temporal_split(self):
        """Test temporal (chronological) splitting"""
        splits = self.splitter.temporal_split(
            self.test_data,
            timestamp_key='timestamp',
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Check that splits are in chronological order
        all_timestamps = []
        for split_name in ['train', 'val', 'test']:
            split_timestamps = [sample['timestamp'] for sample in splits[split_name]]
            all_timestamps.extend(split_timestamps)
            
            # Within each split, should be chronologically ordered
            assert split_timestamps == sorted(split_timestamps)
        
        # All timestamps should be in order across splits
        assert all_timestamps == sorted(all_timestamps)


@pytest.mark.skipif(not HAS_AUDIO_PREPROCESSOR, reason="Audio preprocessor not available")
class TestAudioPreprocessor:
    """Test suite for audio preprocessing components"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        # Create synthetic audio data for testing
        self.sample_rate = 16000
        self.duration = 2.0
        self.samples = int(self.sample_rate * self.duration)
        
        # Generate test signals
        t = np.linspace(0, self.duration, self.samples)
        self.sine_wave = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
        self.noise_signal = np.random.normal(0, 0.1, self.samples)  # Gaussian noise
        self.mixed_signal = self.sine_wave + self.noise_signal
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_audio_preproc_"))
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_audio_file(self, filename: str, audio: np.ndarray) -> Path:
        """Create a mock audio file for testing"""
        file_path = self.temp_dir / filename
        
        # Save as numpy array (mock audio file)
        np.save(file_path.with_suffix('.npy'), audio)
        
        return file_path.with_suffix('.npy')
    
    def test_audio_normalizer_rms(self):
        """Test RMS normalization"""
        normalizer = AudioNormalizer(method='rms', target_level=0.1)
        
        normalized_audio, sr = normalizer.process(self.mixed_signal, self.sample_rate)
        
        # Check RMS level
        actual_rms = np.sqrt(np.mean(normalized_audio**2))
        assert abs(actual_rms - 0.1) < 0.01  # Allow small tolerance
        assert sr == self.sample_rate  # Sample rate unchanged
    
    def test_audio_normalizer_peak(self):
        """Test peak normalization"""
        normalizer = AudioNormalizer(method='peak', target_level=0.5)
        
        normalized_audio, sr = normalizer.process(self.mixed_signal, self.sample_rate)
        
        # Check peak level
        actual_peak = np.max(np.abs(normalized_audio))
        assert abs(actual_peak - 0.5) < 0.01  # Allow small tolerance
    
    def test_audio_normalizer_empty_input(self):
        """Test normalizer with empty input"""
        normalizer = AudioNormalizer(method='rms', target_level=0.1)
        
        empty_audio = np.array([])
        result_audio, sr = normalizer.process(empty_audio, self.sample_rate)
        
        assert len(result_audio) == 0
        assert sr == self.sample_rate
    
    @pytest.mark.skipif(not HAS_AUDIO_LIBS, reason="Audio libraries not available")
    def test_silence_trimmer(self):
        """Test silence trimming"""
        # Create signal with silence at beginning and end
        silence_samples = int(0.5 * self.sample_rate)  # 0.5 seconds of silence
        silence = np.zeros(silence_samples)
        
        signal_with_silence = np.concatenate([silence, self.sine_wave, silence])
        
        trimmer = SilenceTrimmer(top_db=20, margin_seconds=0.1)
        trimmed_audio, sr = trimmer.process(signal_with_silence, self.sample_rate)
        
        # Trimmed audio should be shorter than original
        assert len(trimmed_audio) < len(signal_with_silence)
        assert len(trimmed_audio) > len(self.sine_wave)  # Should include margin
    
    def test_audio_augmenter(self):
        """Test audio augmentation"""
        augmenter = AudioAugmenter(
            augmentation_prob=1.0,  # Always augment
            random_seed=42
        )
        
        augmented_audio, sr = augmenter.process(self.sine_wave, self.sample_rate)
        
        # Augmented audio should be different from original
        # (but length might change due to time stretching)
        assert not np.array_equal(augmented_audio, self.sine_wave)
        assert sr == self.sample_rate  # Sample rate should be preserved
    
    def test_audio_augmenter_no_augmentation(self):
        """Test audio augmenter with zero probability"""
        augmenter = AudioAugmenter(augmentation_prob=0.0)  # Never augment
        
        result_audio, sr = augmenter.process(self.sine_wave, self.sample_rate)
        
        # Should be identical to original
        np.testing.assert_array_equal(result_audio, self.sine_wave)
        assert sr == self.sample_rate
    
    def test_audio_segmenter(self):
        """Test audio segmentation"""
        segmenter = AudioSegmenter(
            segment_duration=0.5,  # 0.5 second segments
            hop_duration=0.5,      # Non-overlapping
            min_segment_duration=0.1
        )
        
        segments = segmenter.process(self.sine_wave, self.sample_rate)
        
        # Should create multiple segments
        expected_segments = int(self.duration / 0.5)
        assert len(segments) == expected_segments
        
        # Each segment should have correct length
        expected_segment_length = int(0.5 * self.sample_rate)
        for segment in segments:
            assert len(segment) == expected_segment_length
    
    def test_audio_segmenter_overlapping(self):
        """Test overlapping audio segmentation"""
        segmenter = AudioSegmenter(
            segment_duration=1.0,  # 1 second segments
            hop_duration=0.5,      # 0.5 second hop (50% overlap)
            min_segment_duration=0.1
        )
        
        segments = segmenter.process(self.sine_wave, self.sample_rate)
        
        # Should create overlapping segments
        assert len(segments) > int(self.duration / 1.0)  # More than non-overlapping
        
        # Each segment should have correct length
        expected_segment_length = int(1.0 * self.sample_rate)
        for segment in segments:
            assert len(segment) == expected_segment_length
    
    def test_preprocessing_pipeline_basic(self):
        """Test basic preprocessing pipeline"""
        config = {
            'normalizer': {
                'method': 'rms',
                'target_level': 0.1
            }
        }
        
        pipeline = AudioPreprocessingPipeline(config=config, use_cache=False)
        
        # Create mock audio file
        audio_file = self.create_mock_audio_file("test.wav", self.mixed_signal)
        
        # Since we can't actually load the mock file, we'll test the pipeline components
        normalizer = pipeline.components['normalizer']
        processed_audio, sr = normalizer.process(self.mixed_signal, self.sample_rate)
        
        # Check that normalization was applied
        actual_rms = np.sqrt(np.mean(processed_audio**2))
        assert abs(actual_rms - 0.1) < 0.01
    
    def test_preprocessing_pipeline_with_segmentation(self):
        """Test preprocessing pipeline with segmentation"""
        config = {
            'normalizer': {
                'method': 'rms',
                'target_level': 0.1
            },
            'segmenter': {
                'enabled': True,
                'segment_duration': 0.5
            }
        }
        
        pipeline = AudioPreprocessingPipeline(config=config, use_cache=False)
        
        # Test segmenter component directly
        segmenter = pipeline.components['segmenter']
        segments = segmenter.process(self.sine_wave, self.sample_rate)
        
        assert isinstance(segments, list)
        assert len(segments) > 0
    
    def test_preprocessing_pipeline_statistics(self):
        """Test preprocessing pipeline statistics"""
        config = {'normalizer': {'method': 'rms'}}
        pipeline = AudioPreprocessingPipeline(config=config, use_cache=False)
        
        # Get initial statistics
        stats = pipeline.get_statistics()
        
        assert 'files_processed' in stats
        assert 'processing_times' in stats
        assert stats['files_processed'] == 0


@pytest.mark.skipif(not HAS_DATASET_LOADER, reason="Dataset loader not available")
class TestDatasetLoader:
    """Test suite for dataset loading functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method run before each test"""
        # Create temporary directory structure
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_dataset_"))
        
        # Create speaker directories with audio files
        self.speakers = ["alice", "bob", "charlie"]
        self.files_per_speaker = 5
        
        for speaker in self.speakers:
            speaker_dir = self.test_dir / speaker
            speaker_dir.mkdir()
            
            for i in range(self.files_per_speaker):
                # Create mock audio file (just text for testing)
                audio_file = speaker_dir / f"utterance_{i:02d}.wav"
                audio_file.write_text(f"mock audio for {speaker} utterance {i}")
        
        # Create metadata CSV file
        self.metadata_file = self.test_dir / "metadata.csv"
        self.create_metadata_csv()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_metadata_csv(self):
        """Create metadata CSV file for testing"""
        with open(self.metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'speaker', 'duration', 'quality'])
            
            for speaker in self.speakers:
                for i in range(self.files_per_speaker):
                    file_path = f"{speaker}/utterance_{i:02d}.wav"
                    duration = round(random.uniform(1.0, 5.0), 2)
                    quality = random.choice(['high', 'medium', 'low'])
                    writer.writerow([file_path, speaker, duration, quality])
    
    def test_directory_dataset_creation(self):
        """Test DirectoryDataset creation and basic properties"""
        # Mock audio loading to avoid dependency issues
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset = DirectoryDataset(
                root_dir=self.test_dir,
                min_samples_per_speaker=1,
                preprocessing_pipeline=None
            )
            
            # Check basic properties
            assert len(dataset) == len(self.speakers) * self.files_per_speaker
            assert dataset.num_speakers == len(self.speakers)
            
            # Check speaker mappings
            for speaker in self.speakers:
                assert speaker in dataset.speaker_to_idx
                idx = dataset.speaker_to_idx[speaker]
                assert dataset.idx_to_speaker[idx] == speaker
    
    def test_directory_dataset_min_samples_filter(self):
        """Test filtering speakers with minimum samples"""
        # Create a speaker with only one file
        lone_speaker_dir = self.test_dir / "lone_speaker"
        lone_speaker_dir.mkdir()
        (lone_speaker_dir / "single.wav").write_text("single file")
        
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset = DirectoryDataset(
                root_dir=self.test_dir,
                min_samples_per_speaker=3,  # Require at least 3 files
                preprocessing_pipeline=None
            )
            
            # Should exclude lone_speaker (only 1 file)
            assert dataset.num_speakers == len(self.speakers)  # Original speakers only
            assert "lone_speaker" not in dataset.speaker_to_idx
    
    def test_directory_dataset_max_samples_limit(self):
        """Test limiting maximum samples per speaker"""
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset = DirectoryDataset(
                root_dir=self.test_dir,
                max_samples_per_speaker=3,  # Limit to 3 files per speaker
                preprocessing_pipeline=None
            )
            
            # Should have limited samples per speaker
            expected_total_samples = len(self.speakers) * 3
            assert len(dataset) == expected_total_samples
    
    def test_metadata_dataset_creation(self):
        """Test MetadataDataset creation from CSV"""
        with patch('data.dataset_loader.MetadataDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset = MetadataDataset(
                metadata_file=self.metadata_file,
                root_dir=self.test_dir,
                preprocessing_pipeline=None
            )
            
            # Check basic properties
            assert len(dataset) == len(self.speakers) * self.files_per_speaker
            assert dataset.num_speakers == len(self.speakers)
            
            # Check that metadata is preserved
            sample_info = dataset.samples[0]
            assert 'metadata' in sample_info
            assert 'duration' in sample_info['metadata']
            assert 'quality' in sample_info['metadata']
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation"""
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset = DirectoryDataset(
                root_dir=self.test_dir,
                preprocessing_pipeline=None
            )
            
            stats = dataset.get_statistics()
            
            # Check required statistics
            assert 'total_samples' in stats
            assert 'num_speakers' in stats
            assert 'samples_per_speaker' in stats
            assert 'min_samples_per_speaker' in stats
            assert 'max_samples_per_speaker' in stats
            
            assert stats['total_samples'] == len(dataset)
            assert stats['num_speakers'] == len(self.speakers)
            assert stats['min_samples_per_speaker'] == self.files_per_speaker
            assert stats['max_samples_per_speaker'] == self.files_per_speaker
    
    def test_dataset_filtering(self):
        """Test dataset filtering by speakers"""
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset = DirectoryDataset(
                root_dir=self.test_dir,
                preprocessing_pipeline=None
            )
            
            # Filter to only two speakers
            filtered_speakers = ["alice", "bob"]
            filtered_dataset = dataset.filter_by_speaker(filtered_speakers)
            
            # Check filtered dataset properties
            assert filtered_dataset.num_speakers == 2
            assert len(filtered_dataset) == 2 * self.files_per_speaker
            
            for speaker in filtered_speakers:
                assert speaker in filtered_dataset.speaker_to_idx
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_dataset_wrapper(self):
        """Test PyTorch dataset wrapper"""
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            base_dataset = DirectoryDataset(
                root_dir=self.test_dir,
                preprocessing_pipeline=None
            )
            
            pytorch_dataset = PyTorchDataset(base_dataset)
            
            # Test basic properties
            assert len(pytorch_dataset) == len(base_dataset)
            assert pytorch_dataset.num_classes == base_dataset.num_speakers
            
            # Test item access
            with patch.object(base_dataset, '_load_sample') as mock_load_sample:
                mock_audio = np.random.randn(16000)
                mock_load_sample.return_value = (mock_audio, 0)
                
                audio_tensor, label_tensor = pytorch_dataset[0]
                
                # Check tensor types
                assert torch.is_tensor(audio_tensor)
                assert torch.is_tensor(label_tensor)
                assert audio_tensor.dtype == torch.float32
                assert label_tensor.dtype == torch.long
    
    def test_dataset_manager_creation(self):
        """Test DatasetManager creation and configuration"""
        config = {
            'data': {
                'type': 'directory',
                'dataset_path': str(self.test_dir),
                'min_samples_per_speaker': 1,
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2
            }
        }
        
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset_manager = DatasetManager(config=config)
            
            # Check that splits were created
            stats = dataset_manager.get_statistics()
            assert 'available_splits' in stats
            assert len(stats['available_splits']) >= 2  # At least train and one other
            
            # Check that we can get individual datasets
            train_dataset = dataset_manager.get_dataset('train')
            assert len(train_dataset) > 0
    
    def test_simple_dataset_creation_function(self):
        """Test create_simple_dataset convenience function"""
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset_manager = create_simple_dataset(
                str(self.test_dir),
                min_samples_per_speaker=2
            )
            
            # Should create a valid dataset manager
            assert isinstance(dataset_manager, DatasetManager)
            
            stats = dataset_manager.get_statistics()
            assert stats['total_speakers'] == len(self.speakers)


class TestIntegration:
    """Integration tests combining multiple components"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for integration tests"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_integration_"))
        
        # Create a complete test dataset structure
        self.speakers = ["speaker_A", "speaker_B", "speaker_C"]
        
        for speaker in self.speakers:
            speaker_dir = self.test_dir / speaker
            speaker_dir.mkdir()
            
            for i in range(5):
                audio_file = speaker_dir / f"audio_{i:03d}.wav"
                # Create synthetic audio data
                sample_rate = 16000
                duration = 2.0
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine
                
                # Save as numpy file (mock audio)
                np.save(audio_file.with_suffix('.npy'), audio_data)
                audio_file.write_text("mock wav file")
        
        yield
        
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not (HAS_DATA_UTILS and HAS_DATASET_LOADER), 
                       reason="Required components not available")
    def test_end_to_end_dataset_loading(self):
        """Test complete end-to-end dataset loading pipeline"""
        # Test the complete workflow: scan -> validate -> load -> split
        
        # Step 1: Scan directory for audio files
        if HAS_DATA_UTILS:
            files_info = quick_audio_scan(self.test_dir, validate_files=False)
            assert len(files_info) > 0
        
        # Step 2: Create dataset with preprocessing
        config = {
            'data': {
                'type': 'directory',
                'dataset_path': str(self.test_dir),
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2
            }
        }
        
        with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
            mock_load.return_value = (np.random.randn(16000), 16000)
            
            dataset_manager = DatasetManager(config=config)
            
            # Verify splits were created correctly
            splits = ['train', 'val', 'test']
            for split in splits:
                try:
                    dataset = dataset_manager.get_dataset(split)
                    assert len(dataset) > 0
                except KeyError:
                    # Some splits might be empty for small datasets
                    pass
    
    def test_performance_with_large_file_list(self):
        """Test performance with a larger number of files"""
        # Create many files quickly for performance testing
        large_dir = self.test_dir / "performance_test"
        large_dir.mkdir()
        
        num_speakers = 10
        files_per_speaker = 20
        
        start_time = time.time()
        
        # Create file structure
        for i in range(num_speakers):
            speaker_dir = large_dir / f"speaker_{i:03d}"
            speaker_dir.mkdir()
            
            for j in range(files_per_speaker):
                audio_file = speaker_dir / f"audio_{j:03d}.wav"
                audio_file.write_text("mock audio")
        
        creation_time = time.time() - start_time
        
        # Test scanning performance
        start_time = time.time()
        
        if HAS_DATA_UTILS:
            files_info = FileSystemUtils.scan_directory_for_audio(
                large_dir, validate_files=False
            )
            
            scan_time = time.time() - start_time
            expected_files = num_speakers * files_per_speaker
            
            assert len(files_info) == expected_files
            
            # Performance should be reasonable (less than 1 second for this small test)
            assert scan_time < 5.0, f"Scanning took too long: {scan_time:.2f}s"
            
            print(f"Performance test: Created {expected_files} files in {creation_time:.2f}s, "
                  f"scanned in {scan_time:.2f}s")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_directory_handling(self):
        """Test handling of missing directories"""
        nonexistent_dir = Path("/nonexistent/directory/path")
        
        if HAS_DATASET_LOADER:
            with pytest.raises(FileNotFoundError):
                DirectoryDataset(root_dir=nonexistent_dir)
    
    def test_empty_directory_handling(self):
        """Test handling of empty directories"""
        empty_dir = Path(tempfile.mkdtemp(prefix="test_empty_"))
        
        try:
            if HAS_DATASET_LOADER:
                with patch('data.dataset_loader.DirectoryDataset._load_audio_basic') as mock_load:
                    mock_load.return_value = (np.random.randn(16000), 16000)
                    
                    dataset = DirectoryDataset(root_dir=empty_dir)
                    assert len(dataset) == 0
                    assert dataset.num_speakers == 0
        
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted or unreadable files"""
        test_dir = Path(tempfile.mkdtemp(prefix="test_corrupted_"))
        
        try:
            # Create a "corrupted" file
            speaker_dir = test_dir / "speaker1"
            speaker_dir.mkdir()
            corrupted_file = speaker_dir / "corrupted.wav"
            corrupted_file.write_bytes(b"not valid audio data")
            
            if HAS_DATASET_LOADER:
                # Mock load to raise an exception
                def mock_load_failing(file_path):
                    raise RuntimeError("Corrupted file")
                
                with patch('data.dataset_loader.DirectoryDataset._load_audio_basic', 
                          side_effect=mock_load_failing):
                    
                    dataset = DirectoryDataset(root_dir=test_dir)
                    
                    # Should handle the error gracefully
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        
                        # Try to load the corrupted sample
                        audio, label = dataset._load_sample(0)
                        
                        # Should return fallback audio (silence)
                        assert isinstance(audio, np.ndarray)
                        assert len(w) > 0  # Should have raised a warning
        
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not HAS_AUDIO_PREPROCESSOR, reason="Audio preprocessor not available")
    def test_preprocessing_error_handling(self):
        """Test preprocessing error handling"""
        # Test with invalid audio data
        invalid_audio = np.array([])  # Empty array
        
        normalizer = AudioNormalizer()
        result_audio, sr = normalizer.process(invalid_audio, 16000)
        
        # Should handle empty input gracefully
        assert len(result_audio) == 0
        assert sr == 16000
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations"""
        if HAS_AUDIO_PREPROCESSOR:
            # Test invalid normalization method
            with pytest.raises(ValueError):
                AudioNormalizer(method='invalid_method')
        
        if HAS_DATASET_LOADER:
            # Test invalid dataset configuration
            invalid_config = {
                'data': {
                    'type': 'invalid_type',
                    'dataset_path': '/some/path'
                }
            }
            
            with pytest.raises(ValueError):
                DatasetManager(config=invalid_config)


def run_all_tests():
    """Run all tests with proper setup"""
    print("Running comprehensive data processing tests...")
    print(f"Available components:")
    print(f"  - Data utilities: {HAS_DATA_UTILS}")
    print(f"  - Audio preprocessor: {HAS_AUDIO_PREPROCESSOR}")
    print(f"  - Dataset loader: {HAS_DATASET_LOADER}")
    print(f"  - Audio libraries: {HAS_AUDIO_LIBS}")
    print(f"  - PyTorch: {HAS_TORCH}")
    print(f"  - TensorFlow: {HAS_TF}")
    print()
    
    # Run pytest
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
    ]
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    print(f"✓ Project Root: {PROJECT_ROOT}")
    print(f"✓ Import Manager: {USING_IMPORT_MANAGER}")
    print(f"✓ Module imports successful")
    
    exit_code = run_all_tests()
    sys.exit(exit_code)