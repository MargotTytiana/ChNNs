"""Models package for the C-HiLAP project."""

try:
    from .base_model import (
        BaseModel, SklearnCompatibleModel, ModelConfig, TrainingMetrics,
        ModelType, TaskType, OptimizationType, create_model_config
    )
    from .mlp_classifier import (
        PyTorchMLPClassifier, SklearnMLPClassifier, MLPNetwork,
        create_mlp_classifier
    )
    from .model_factory import (
        ModelFactory, ModelRegistry, ConfigTemplate, HyperparameterOptimizer
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all model components: {e}")
