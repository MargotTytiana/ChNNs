"""Data package for ChNNs project - Circular Import Free Version"""

# Only import data-related modules - NO MODEL IMPORTS
# This prevents circular dependencies between data and models packages

try:
    from .dataset_loader import LibriSpeechChaoticDataset, create_speaker_dataloaders
    print("  ✓ Dataset loader imported successfully")
except ImportError as e:
    print(f"  ⚠ Dataset loader import warning: {e}")
    LibriSpeechChaoticDataset = None
    create_speaker_dataloaders = None

try:
    from .audio_preprocessor import AudioPreprocessingPipeline, create_preprocessing_pipeline  
except ImportError as e:
    print(f"  ⚠ Audio preprocessor import warning: {e}")
    AudioPreprocessingPipeline = None
    create_preprocessing_pipeline = None

try:
    from .data_utils import DataValidator, DataTransformer, DatasetSplitter
except ImportError as e:
    print(f"  ⚠ Data utilities import warning: {e}")
    DataValidator = None
    DataTransformer = None
    DatasetSplitter = None

# Export ONLY data-related classes
__all__ = [
    'LibriSpeechChaoticDataset',
    'create_speaker_dataloaders',
    'AudioPreprocessingPipeline', 
    'create_preprocessing_pipeline',
    'DataValidator',
    'DataTransformer',
    'DatasetSplitter'
]

print("✓ Data package initialized (circular import free)")
