"""
Model Factory and Registry System for C-HiLAP Project.

This module implements a comprehensive model factory system using the Factory
design pattern, providing unified model creation, registration, configuration
management, and hyperparameter optimization capabilities.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
import json
import copy
from typing import Dict, List, Tuple, Optional, Union, Any, Type, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import logging

# Hyperparameter optimization
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some hyperparameter optimization features will be limited.")

import os
import sys
from pathlib import Path

# 立即修复导入
def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent  # models -> Model
    paths = [str(model_dir), str(model_dir/'models'), str(model_dir/'core')]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

# 现在直接导入，不用相对导入
from base_model import BaseModel, ModelConfig, ModelType

# Import specific model implementations
try:
    from mlp_classifier import create_mlp_classifier, MLPClassifier, SklearnMLPClassifier
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    warnings.warn("MLP classifier not available.")


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""
    pass


class ModelRegistrationError(Exception):
    """Exception raised when model registration fails."""
    pass


@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    name: str
    model_class: Type[BaseModel]
    model_type: ModelType
    supported_tasks: List[TaskType]
    backend: str = "pytorch"
    description: str = ""
    default_config: Optional[Dict[str, Any]] = None
    hyperparameter_space: Optional[Dict[str, Any]] = None
    requires_gpu: bool = False
    memory_requirement: str = "low"  # low, medium, high
    training_time: str = "fast"      # fast, medium, slow
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not issubclass(self.model_class, BaseModel):
            raise ModelRegistrationError(f"Model class {self.model_class} must inherit from BaseModel")


class ConfigTemplate:
    """Template system for model configurations."""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default configuration templates."""
        # Basic classifier template
        self.templates['basic_classifier'] = {
            'model_type': 'classifier',
            'task_type': 'binary_classification',
            'hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'dropout_rate': 0.2,
            'early_stopping_patience': 10
        }
        
        # Deep classifier template
        self.templates['deep_classifier'] = {
            'model_type': 'classifier',
            'task_type': 'multiclass_classification',
            'hidden_dims': [256, 128, 64, 32],
            'learning_rate': 0.0001,
            'batch_size': 64,
            'num_epochs': 200,
            'dropout_rate': 0.3,
            'early_stopping_patience': 20
        }
        
        # Fast prototype template
        self.templates['fast_prototype'] = {
            'model_type': 'classifier',
            'task_type': 'binary_classification',
            'hidden_dims': [32],
            'learning_rate': 0.01,
            'batch_size': 64,
            'num_epochs': 20,
            'dropout_rate': 0.1,
            'early_stopping_patience': 5
        }
        
        # Chaotic network template (placeholder)
        self.templates['chaotic_network'] = {
            'model_type': 'classifier',
            'task_type': 'binary_classification',
            'hidden_dims': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 150,
            'chaos_parameters': {
                'lorenz_sigma': 10.0,
                'lorenz_rho': 28.0,
                'lorenz_beta': 8.0/3.0
            }
        }
    
    def register_template(self, name: str, template: Dict[str, Any]):
        """Register a new configuration template."""
        self.templates[name] = template.copy()
        logging.info(f"Registered configuration template: {name}")
    
    def get_template(self, name: str) -> Dict[str, Any]:
        """Get a configuration template."""
        if name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Template '{name}' not found. Available templates: {available}")
        
        return copy.deepcopy(self.templates[name])
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return list(self.templates.keys())
    
    def save_templates(self, filepath: str):
        """Save templates to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.templates, f, indent=2, default=str)
        logging.info(f"Templates saved to {filepath}")
    
    def load_templates(self, filepath: str):
        """Load templates from JSON file."""
        with open(filepath, 'r') as f:
            loaded_templates = json.load(f)
        self.templates.update(loaded_templates)
        logging.info(f"Templates loaded from {filepath}")


class ModelRegistry:
    """Registry for managing available model types."""
    
    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models that are available."""
        if MLP_AVAILABLE:
            # PyTorch MLP
            self.register_model(
                name="pytorch_mlp",
                model_class=PyTorchMLPClassifier,
                model_type=ModelType.CLASSIFIER,
                supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
                backend="pytorch",
                description="Multi-layer perceptron using PyTorch backend",
                default_config={
                    'hidden_dims': [128, 64],
                    'learning_rate': 0.001,
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'batch_norm': True
                },
                hyperparameter_space={
                    'hidden_dims': [[64, 32], [128, 64], [256, 128, 64]],
                    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                    'dropout_rate': [0.1, 0.2, 0.3, 0.5],
                    'batch_size': [16, 32, 64, 128]
                },
                requires_gpu=False,
                memory_requirement="low",
                training_time="fast"
            )
            
            # Sklearn MLP
            self.register_model(
                name="sklearn_mlp",
                model_class=SklearnMLPClassifier,
                model_type=ModelType.CLASSIFIER,
                supported_tasks=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
                backend="sklearn",
                description="Multi-layer perceptron using scikit-learn backend",
                default_config={
                    'hidden_dims': [100, 50],
                    'learning_rate': 0.001,
                    'num_epochs': 200
                },
                hyperparameter_space={
                    'hidden_dims': [[50], [100], [100, 50], [200, 100]],
                    'learning_rate': [0.01, 0.001, 0.0001],
                    'num_epochs': [100, 200, 500]
                },
                requires_gpu=False,
                memory_requirement="low",
                training_time="medium"
            )
    
    def register_model(self, name: str, model_class: Type[BaseModel],
                      model_type: ModelType, supported_tasks: List[TaskType],
                      backend: str = "pytorch", description: str = "",
                      default_config: Optional[Dict[str, Any]] = None,
                      hyperparameter_space: Optional[Dict[str, Any]] = None,
                      requires_gpu: bool = False,
                      memory_requirement: str = "low",
                      training_time: str = "fast"):
        """Register a new model type."""
        metadata = ModelMetadata(
            name=name,
            model_class=model_class,
            model_type=model_type,
            supported_tasks=supported_tasks,
            backend=backend,
            description=description,
            default_config=default_config or {},
            hyperparameter_space=hyperparameter_space or {},
            requires_gpu=requires_gpu,
            memory_requirement=memory_requirement,
            training_time=training_time
        )
        
        self._models[name] = metadata
        logging.info(f"Registered model: {name} ({backend} backend)")
    
    def unregister_model(self, name: str):
        """Unregister a model type."""
        if name in self._models:
            del self._models[name]
            logging.info(f"Unregistered model: {name}")
        else:
            warnings.warn(f"Model {name} not found in registry")
    
    def get_model_metadata(self, name: str) -> ModelMetadata:
        """Get metadata for a registered model."""
        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        return self._models[name]
    
    def list_models(self, model_type: Optional[ModelType] = None,
                   task_type: Optional[TaskType] = None,
                   backend: Optional[str] = None) -> List[str]:
        """List available models with optional filtering."""
        models = []
        
        for name, metadata in self._models.items():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
            if task_type and task_type not in metadata.supported_tasks:
                continue
            if backend and metadata.backend != backend:
                continue
            
            models.append(name)
        
        return models
    
    def get_models_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all registered models."""
        summary = {}
        
        for name, metadata in self._models.items():
            summary[name] = {
                'model_type': metadata.model_type.value,
                'supported_tasks': [task.value for task in metadata.supported_tasks],
                'backend': metadata.backend,
                'description': metadata.description,
                'requires_gpu': metadata.requires_gpu,
                'memory_requirement': metadata.memory_requirement,
                'training_time': metadata.training_time
            }
        
        return summary


class HyperparameterOptimizer:
    """Hyperparameter optimization utilities."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
    
    def generate_hyperparameter_grid(self, model_name: str,
                                   custom_space: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate hyperparameter grid for model."""
        metadata = self.model_registry.get_model_metadata(model_name)
        
        # Use custom space if provided, otherwise use default
        param_space = custom_space or metadata.hyperparameter_space
        
        if not param_space:
            warnings.warn(f"No hyperparameter space defined for {model_name}")
            return [{}]
        
        # Generate parameter grid
        if SKLEARN_AVAILABLE:
            grid = list(ParameterGrid(param_space))
        else:
            # Simple manual grid generation
            grid = self._manual_parameter_grid(param_space)
        
        return grid
    
    def _manual_parameter_grid(self, param_space: Dict[str, List]) -> List[Dict[str, Any]]:
        """Manual parameter grid generation when sklearn is not available."""
        import itertools
        
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        grid = []
        for combination in itertools.product(*values):
            grid.append(dict(zip(keys, combination)))
        
        return grid
    
    def suggest_hyperparameters(self, model_name: str, task_complexity: str = "medium") -> Dict[str, Any]:
        """Suggest hyperparameters based on task complexity."""
        metadata = self.model_registry.get_model_metadata(model_name)
        base_config = metadata.default_config.copy()
        
        # Adjust based on complexity
        if task_complexity == "simple":
            base_config.update({
                'hidden_dims': base_config.get('hidden_dims', [64])[:1],  # Shallow network
                'learning_rate': 0.01,
                'num_epochs': 50,
                'dropout_rate': 0.1
            })
        elif task_complexity == "complex":
            current_dims = base_config.get('hidden_dims', [128, 64])
            base_config.update({
                'hidden_dims': [dim * 2 for dim in current_dims] + current_dims,  # Deeper network
                'learning_rate': 0.0001,
                'num_epochs': 300,
                'dropout_rate': 0.3,
                'early_stopping_patience': 25
            })
        
        return base_config
    
    def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray,
                                cv_folds: int = 5, n_iter: int = 10,
                                optimization_method: str = "random",
                                custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using cross-validation.
        
        Args:
            model_name: Name of registered model
            X: Training features
            y: Training labels
            cv_folds: Number of CV folds
            n_iter: Number of iterations for random search
            optimization_method: 'grid' or 'random'
            custom_space: Custom hyperparameter space
            
        Returns:
            Dictionary with best parameters and scores
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("Sklearn not available. Hyperparameter optimization is limited.")
            return self.suggest_hyperparameters(model_name)
        
        metadata = self.model_registry.get_model_metadata(model_name)
        param_space = custom_space or metadata.hyperparameter_space
        
        if not param_space:
            return {"error": "No hyperparameter space defined"}
        
        # Create a wrapper for sklearn compatibility
        model_wrapper = self._create_sklearn_wrapper(model_name, metadata)
        
        try:
            if optimization_method == "grid":
                search = GridSearchCV(
                    model_wrapper,
                    param_space,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
            else:  # random
                search = RandomizedSearchCV(
                    model_wrapper,
                    param_space,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42
                )
            
            search.fit(X, y)
            
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
        
        except Exception as e:
            warnings.warn(f"Hyperparameter optimization failed: {e}")
            return {"error": str(e)}
    
    def _create_sklearn_wrapper(self, model_name: str, metadata: ModelMetadata):
        """Create sklearn-compatible wrapper for hyperparameter optimization."""
        # This is a simplified wrapper - full implementation would need more work
        class ModelWrapper(BaseEstimator):
            def __init__(self, **params):
                self.params = params
                self.model = None
            
            def fit(self, X, y):
                # Create model config with parameters
                config = create_model_config(**self.params)
                self.model = metadata.model_class(config)
                self.model.fit(X, y)
                return self
            
            def predict(self, X):
                return self.model.predict(X)
            
            def score(self, X, y):
                return self.model.score(X, y)
        
        return ModelWrapper()


class ModelFactory:
    """Main factory class for creating and managing models."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.config_templates = ConfigTemplate()
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.registry)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_model(self, model_name: str, config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
                    template_name: Optional[str] = None, **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of registered model
            config: Model configuration (ModelConfig or dict)
            template_name: Name of configuration template to use
            **kwargs: Additional configuration parameters
            
        Returns:
            Model instance
        """
        if not BASE_MODEL_AVAILABLE:
            raise ModelCreationError("Base model system is not available")
        
        # Get model metadata
        try:
            metadata = self.registry.get_model_metadata(model_name)
        except ValueError as e:
            raise ModelCreationError(str(e))
        
        # Build configuration
        final_config = self._build_config(metadata, config, template_name, **kwargs)
        
        # Validate configuration for task
        self._validate_config(metadata, final_config)
        
        # Create model instance
        try:
            model = metadata.model_class(final_config)
            self.logger.info(f"Created model: {model_name}")
            return model
        except Exception as e:
            raise ModelCreationError(f"Failed to create model {model_name}: {e}")
    
    def _build_config(self, metadata: ModelMetadata,
                     config: Optional[Union[ModelConfig, Dict[str, Any]]],
                     template_name: Optional[str],
                     **kwargs) -> ModelConfig:
        """Build final configuration from various sources."""
        # Start with model default config
        final_params = metadata.default_config.copy()
        
        # Apply template if specified
        if template_name:
            try:
                template_params = self.config_templates.get_template(template_name)
                final_params.update(template_params)
            except ValueError as e:
                warnings.warn(str(e))
        
        # Apply provided config
        if config:
            if isinstance(config, ModelConfig):
                config_dict = asdict(config)
            else:
                config_dict = config
            final_params.update(config_dict)
        
        # Apply keyword arguments (highest priority)
        final_params.update(kwargs)
        
        # Create ModelConfig instance
        return create_model_config(**final_params)
    
    def _validate_config(self, metadata: ModelMetadata, config: ModelConfig):
        """Validate configuration against model requirements."""
        # Check task compatibility
        if config.task_type not in metadata.supported_tasks:
            raise ModelCreationError(
                f"Task type {config.task_type} not supported by {metadata.name}. "
                f"Supported tasks: {metadata.supported_tasks}"
            )
        
        # Check model type compatibility
        if config.model_type != metadata.model_type:
            warnings.warn(
                f"Model type mismatch: config has {config.model_type}, "
                f"but {metadata.name} is {metadata.model_type}"
            )
    
    def create_models_batch(self, model_configs: List[Dict[str, Any]]) -> List[BaseModel]:
        """
        Create multiple models in batch.
        
        Args:
            model_configs: List of model configuration dictionaries
                         Each dict should contain 'model_name' and other config params
                         
        Returns:
            List of created model instances
        """
        models = []
        
        for i, model_config in enumerate(model_configs):
            try:
                if 'model_name' not in model_config:
                    raise ModelCreationError(f"Config {i} missing 'model_name'")
                
                model_name = model_config.pop('model_name')
                model = self.create_model(model_name, **model_config)
                models.append(model)
                
            except Exception as e:
                self.logger.error(f"Failed to create model {i}: {e}")
                models.append(None)
        
        return models
    
    def create_comparison_suite(self, task_type: TaskType,
                              include_backends: Optional[List[str]] = None) -> List[BaseModel]:
        """
        Create a suite of models for comparison on a specific task.
        
        Args:
            task_type: Type of task
            include_backends: List of backends to include
            
        Returns:
            List of models suitable for the task
        """
        suitable_models = self.registry.list_models(task_type=task_type)
        
        if include_backends:
            filtered_models = []
            for model_name in suitable_models:
                metadata = self.registry.get_model_metadata(model_name)
                if metadata.backend in include_backends:
                    filtered_models.append(model_name)
            suitable_models = filtered_models
        
        models = []
        for model_name in suitable_models:
            try:
                # Use suggested hyperparameters for fair comparison
                suggested_params = self.hyperparameter_optimizer.suggest_hyperparameters(
                    model_name, task_complexity="medium"
                )
                
                model = self.create_model(model_name, config=suggested_params)
                models.append(model)
                
            except Exception as e:
                self.logger.warning(f"Failed to create model {model_name} for comparison: {e}")
        
        return models
    
    def get_model_recommendations(self, task_type: TaskType,
                                dataset_size: int,
                                feature_dim: int,
                                performance_priority: str = "accuracy") -> List[str]:
        """
        Get model recommendations based on task and data characteristics.
        
        Args:
            task_type: Type of task
            dataset_size: Number of training samples
            feature_dim: Number of features
            performance_priority: 'accuracy', 'speed', or 'memory'
            
        Returns:
            List of recommended model names in order of preference
        """
        suitable_models = self.registry.list_models(task_type=task_type)
        recommendations = []
        
        for model_name in suitable_models:
            metadata = self.registry.get_model_metadata(model_name)
            score = 0
            
            # Score based on dataset size
            if dataset_size < 1000:  # Small dataset
                if metadata.training_time == "fast":
                    score += 2
                if metadata.memory_requirement == "low":
                    score += 1
            elif dataset_size > 10000:  # Large dataset
                if metadata.backend == "pytorch":  # Better for large datasets
                    score += 2
            
            # Score based on feature dimension
            if feature_dim > 100:  # High-dimensional
                if "deep" in model_name or len(metadata.default_config.get('hidden_dims', [])) > 2:
                    score += 1
            
            # Score based on priority
            if performance_priority == "speed":
                if metadata.training_time == "fast":
                    score += 3
                elif metadata.training_time == "medium":
                    score += 1
            elif performance_priority == "memory":
                if metadata.memory_requirement == "low":
                    score += 3
                elif metadata.memory_requirement == "medium":
                    score += 1
            elif performance_priority == "accuracy":
                if metadata.backend == "pytorch":  # Generally more flexible
                    score += 2
                if len(metadata.default_config.get('hidden_dims', [])) >= 2:  # Deeper networks
                    score += 1
            
            recommendations.append((model_name, score))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in recommendations]
    
    def register_custom_model(self, name: str, model_class: Type[BaseModel], **metadata_kwargs):
        """Register a custom model."""
        self.registry.register_model(name, model_class, **metadata_kwargs)
    
    def register_config_template(self, name: str, template: Dict[str, Any]):
        """Register a configuration template."""
        self.config_templates.register_template(name, template)
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models with details."""
        return self.registry.get_models_summary()
    
    def list_config_templates(self) -> List[str]:
        """List all available configuration templates."""
        return self.config_templates.list_templates()
    
    def save_factory_state(self, filepath: str):
        """Save factory state (templates and registry) to file."""
        state = {
            'templates': self.config_templates.templates,
            'models_summary': self.registry.get_models_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Factory state saved to {filepath}")


# Global factory instance
_global_factory: Optional[ModelFactory] = None

def get_model_factory() -> ModelFactory:
    """Get the global model factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = ModelFactory()
    return _global_factory


# Convenience functions
def create_model(model_name: str, **kwargs) -> BaseModel:
    """Convenience function to create a model using the global factory."""
    factory = get_model_factory()
    return factory.create_model(model_name, **kwargs)


def list_models(**filters) -> List[str]:
    """Convenience function to list available models."""
    factory = get_model_factory()
    return factory.registry.list_models(**filters)


def get_model_recommendations(task_type: str, dataset_size: int, 
                            feature_dim: int, **kwargs) -> List[str]:
    """Convenience function to get model recommendations."""
    factory = get_model_factory()
    task_enum = TaskType(task_type) if isinstance(task_type, str) else task_type
    return factory.get_model_recommendations(task_enum, dataset_size, feature_dim, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Factory System...")
    
    # Initialize factory
    factory = ModelFactory()
    
    # List available models
    print("Available models:")
    models_summary = factory.list_available_models()
    for name, info in models_summary.items():
        print(f"  {name}: {info['description']}")
        print(f"    Backend: {info['backend']}, Tasks: {info['supported_tasks']}")
        print(f"    Requirements: {info['memory_requirement']} memory, {info['training_time']} training")
    
    # List configuration templates
    print(f"\nAvailable configuration templates: {factory.list_config_templates()}")
    
    # Test model creation with different methods
    if MLP_AVAILABLE:
        print("\n=== Testing Model Creation ===")
        
        # Method 1: Basic creation
        try:
            model1 = factory.create_model('pytorch_mlp', 
                                        input_dim=20, 
                                        output_dim=2,
                                        hidden_dims=[64, 32])
            print("✓ Created model with basic parameters")
        except Exception as e:
            print(f"✗ Basic creation failed: {e}")
        
        # Method 2: Using template
        try:
            model2 = factory.create_model('sklearn_mlp',
                                        template_name='fast_prototype',
                                        input_dim=20,
                                        output_dim=3)
            print("✓ Created model using template")
        except Exception as e:
            print(f"✗ Template creation failed: {e}")
        
        # Method 3: Batch creation
        model_configs = [
            {'model_name': 'pytorch_mlp', 'input_dim': 10, 'output_dim': 2, 'hidden_dims': [32]},
            {'model_name': 'sklearn_mlp', 'input_dim': 10, 'output_dim': 2, 'template_name': 'basic_classifier'}
        ]
        
        try:
            models_batch = factory.create_models_batch(model_configs)
            successful_models = [m for m in models_batch if m is not None]
            print(f"✓ Created {len(successful_models)}/{len(model_configs)} models in batch")
        except Exception as e:
            print(f"✗ Batch creation failed: {e}")
    
    # Test recommendations
    print("\n=== Testing Model Recommendations ===")
    try:
        recommendations = factory.get_model_recommendations(
            TaskType.BINARY_CLASSIFICATION,
            dataset_size=5000,
            feature_dim=50,
            performance_priority="accuracy"
        )
        print(f"Recommended models for binary classification: {recommendations}")
    except Exception as e:
        print(f"✗ Recommendations failed: {e}")
    
    # Test hyperparameter optimization
    print("\n=== Testing Hyperparameter Optimization ===")
    try:
        suggested_params = factory.hyperparameter_optimizer.suggest_hyperparameters(
            'pytorch_mlp', task_complexity='complex'
        )
        print(f"Suggested hyperparameters for complex task: {suggested_params}")
    except Exception as e:
        print(f"✗ Hyperparameter suggestion failed: {e}")
    
    # Test comparison suite
    if MLP_AVAILABLE:
        print("\n=== Testing Comparison Suite ===")
        try:
            comparison_models = factory.create_comparison_suite(
                TaskType.MULTICLASS_CLASSIFICATION,
                include_backends=['pytorch', 'sklearn']
            )
            print(f"✓ Created comparison suite with {len(comparison_models)} models")
            for i, model in enumerate(comparison_models):
                if model:
                    print(f"  Model {i+1}: {model.__class__.__name__}")
        except Exception as e:
            print(f"✗ Comparison suite creation failed: {e}")
    
    # Test convenience functions
    print("\n=== Testing Convenience Functions ===")
    try:
        available_models = list_models(task_type=TaskType.BINARY_CLASSIFICATION)
        print(f"Available binary classification models: {available_models}")
    except Exception as e:
        print(f"✗ Convenience function failed: {e}")
    
    print("\nModel Factory testing completed!")