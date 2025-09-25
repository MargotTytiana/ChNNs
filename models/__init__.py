"""
Models subpackage - Contains all neural network architectures and model definitions.
This is like the workshop where all the different types of tools (models) are stored.
"""

# Import key model classes with error handling
# Each try-except block handles a specific model type independently
try:
    from .base_model import BaseModel, ModelConfig, TrainingMetrics
    print("  ✓ Base model classes loaded")
except ImportError as e:
    print(f"  ⚠ Base model import warning: {e}")
    BaseModel = None
    ModelConfig = None
    TrainingMetrics = None

try:
    from .mlp_classifier import MLPClassifier, MLPClassifier, MLPNetwork
    print("  ✓ MLP classifier models loaded")
except ImportError as e:
    print(f"  ⚠ MLP classifier import warning: {e}")
    MLPClassifier = None
    PyTorchMLPClassifier = None
    MLPNetwork = None

try:
    from .hybrid_models import TraditionalMLPBaseline, HybridModelManager
    print("  ✓ Hybrid models loaded")
except ImportError as e:
    print(f"  ⚠ Hybrid models import warning: {e}")
    TraditionalMLPBaseline = None
    HybridModelManager = None

try:
    from .chaotic_network import ChaoticSpeakerRecognitionNetwork, create_chaotic_speaker_network
    print("  ✓ Chaotic network models loaded")
except ImportError as e:
    print(f"  ⚠ Chaotic network import warning: {e}")
    ChaoticSpeakerRecognitionNetwork = None
    create_chaotic_speaker_network = None

try:
    from .model_factory import ModelFactory, ModelRegistry
    print("  ✓ Model factory loaded")
except ImportError as e:
    print(f"  ⚠ Model factory import warning: {e}")
    ModelFactory = None
    ModelRegistry = None

__all__ = [
    'BaseModel', 'ModelConfig', 'TrainingMetrics',
    'MLPClassifier', 'PyTorchMLPClassifier', 'MLPNetwork',
    'TraditionalMLPBaseline', 'HybridModelManager',
    'ChaoticSpeakerRecognitionNetwork', 'create_chaotic_speaker_network',
    'ModelFactory', 'ModelRegistry'
]