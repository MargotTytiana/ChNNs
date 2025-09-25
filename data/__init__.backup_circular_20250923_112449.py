"""Data processing package for ChNNs project"""

try:
    from .dataset_loader import LibriSpeechChaoticDataset, create_speaker_dataloaders
    from .audio_preprocessor import AudioPreprocessingPipeline, create_preprocessing_pipeline
    from .data_utils import DataValidator, DataTransformer, DatasetSplitter
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all data components: {e}")
    
    # Define fallbacks
    LibriSpeechChaoticDataset = None
    create_speaker_dataloaders = None
    AudioPreprocessingPipeline = None
    create_preprocessing_pipeline = None
    DataValidator = None
    DataTransformer = None
    DatasetSplitter = None

__all__ = [
    'LibriSpeechChaoticDataset',
    'create_speaker_dataloaders',
    'AudioPreprocessingPipeline', 
    'create_preprocessing_pipeline',
    'DataValidator',
    'DataTransformer',
    'DatasetSplitter'
]