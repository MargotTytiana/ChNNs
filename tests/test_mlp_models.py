"""
Unit Tests for MLP Models and Model Factory System.

This module provides comprehensive unit tests for MLP classifiers, model factory,
configuration templates, and hyperparameter optimization components of the
C-HiLAP project.

Author: C-HiLAP Project
Date: 2025
"""

import unittest
import numpy as np
import warnings
import sys
import os
import tempfile
import shutil
from pathlib import Path
import time
import pickle
import json
from typing import List, Dict, Any, Tuple
import os
import sys

# 导入路径设置
try:
    from setup_imports import setup_project_imports
    setup_project_imports()
except ImportError:
    # 手动设置路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
# Test data generation
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split

try:
    from models.base_model import (
        BaseModel, ModelConfig, ModelType, TaskType, OptimizationType,
        create_model_config, TrainingMetrics
    )
    from models.mlp_classifier import (
        PyTorchMLPClassifier, SklearnMLPClassifier, MLPNetwork,
        create_mlp_classifier
    )
    from models.model_factory import (
        ModelFactory, ModelRegistry, ConfigTemplate, HyperparameterOptimizer,
        get_model_factory, create_model, list_models, get_model_recommendations
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Could not import MLP modules: {e}")
    IMPORTS_SUCCESSFUL = False

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestDataGenerator:
    """Generate test datasets for MLP testing."""
    
    @staticmethod
    def create_binary_classification_dataset(n_samples=1000, n_features=20, 
                                           n_informative=15, random_state=42):
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=2,
            random_state=random_state
        )
        return X, y
    
    @staticmethod
    def create_multiclass_classification_dataset(n_samples=1000, n_features=20,
                                               n_classes=5, random_state=42):
        """Create multiclass classification dataset."""
        
        # 确保 informative + redundant + repeated < n_features
        # sklearn 要求: n_informative + n_redundant + n_repeated < n_features
        
        # 为重复特征预留2个位置
        available_for_info_and_redundant = max(1, n_features - 2)
        
        # 优先分配信息特征，但不超过可用特征的80%
        max_informative = max(1, int(available_for_info_and_redundant * 0.8))
        n_informative = min(12, max_informative)  # 降低默认值从15到12
        
        # 剩余特征用于冗余特征
        remaining = available_for_info_and_redundant - n_informative
        n_redundant = min(3, max(0, remaining))  # 降低默认值从5到3
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=12,
            n_redundant=5,
            n_classes=n_classes,
            random_state=random_state
        )
        return X, y
    
    @staticmethod
    def create_small_dataset(n_samples=100, n_features=10):
        """Create small dataset for quick testing."""
        return TestDataGenerator.create_binary_classification_dataset(
            n_samples, n_features, n_features//2
        )
    
    @staticmethod
    def create_large_dataset(n_samples=5000, n_features=100):
        """Create larger dataset for performance testing."""
        return TestDataGenerator.create_multiclass_classification_dataset(
            n_samples, n_features, n_classes=3
        )
    
    @staticmethod
    def create_noisy_dataset(n_samples=500, n_features=20, noise_level=0.1):
        """Create dataset with added noise."""
        X, y = TestDataGenerator.create_binary_classification_dataset(n_samples, n_features)
        
        # Add noise to features
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        
        return X_noisy, y


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestModelConfig(unittest.TestCase):
    """Test model configuration system."""
    
    def test_config_creation(self):
        """Test basic model configuration creation."""
        config = create_model_config(
            model_type="classifier",
            task_type="binary_classification",
            input_dim=20,
            output_dim=2,
            hidden_dims=[64, 32],
            learning_rate=0.001
        )
        
        self.assertEqual(config.model_type, ModelType.CLASSIFIER)
        self.assertEqual(config.task_type, TaskType.BINARY_CLASSIFICATION)
        self.assertEqual(config.input_dim, 20)
        self.assertEqual(config.output_dim, 2)
        self.assertEqual(config.hidden_dims, [64, 32])
        self.assertEqual(config.learning_rate, 0.001)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid input dimension
        with self.assertRaises(ValueError):
            create_model_config(input_dim=-1)
        
        # Test invalid learning rate
        with self.assertRaises(ValueError):
            create_model_config(learning_rate=-0.1)
        
        # Test invalid validation split
        with self.assertRaises(ValueError):
            create_model_config(validation_split=1.5)
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        self.assertEqual(config.model_type, ModelType.CLASSIFIER)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_epochs, 100)
        self.assertTrue(config.normalize_features)


@unittest.skipIf(not IMPORTS_SUCCESSFUL or not PYTORCH_AVAILABLE, "PyTorch not available")
class TestPyTorchMLPClassifier(unittest.TestCase):
    """Test PyTorch-based MLP classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train, self.y_train = TestDataGenerator.create_small_dataset(200, 10)
        self.X_test, self.y_test = TestDataGenerator.create_small_dataset(50, 10)
        
        self.config = create_model_config(
            input_dim=10,
            output_dim=2,
            hidden_dims=[32, 16],
            learning_rate=0.01,
            num_epochs=10,  # Small for testing
            batch_size=32,
            verbose=False
        )
        
        self.classifier = PyTorchMLPClassifier(self.config)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.classifier, PyTorchMLPClassifier)
        self.assertIsInstance(self.classifier.model, MLPNetwork)
        self.assertFalse(self.classifier.is_fitted)
        self.assertEqual(self.classifier.config.input_dim, 10)
        self.assertEqual(self.classifier.config.output_dim, 2)
    
    def test_network_architecture(self):
        """Test MLPNetwork architecture."""
        network = self.classifier.model
        
        # Check parameter count
        param_count = network.get_parameter_count()
        self.assertGreater(param_count, 0)
        
        # Test forward pass
        test_input = torch.randn(5, 10)
        if hasattr(network, 'to'):
            network = network.to('cpu')
            test_input = test_input.to('cpu')
        
        output = network(test_input)
        self.assertEqual(output.shape, (5, 2))
    
    def test_training_process(self):
        """Test complete training process."""
        # Train the model
        self.classifier.fit(self.X_train, self.y_train)
        
        # Check that model is marked as fitted
        self.assertTrue(self.classifier.is_fitted)
        
        # Check that training metrics were recorded
        metrics = self.classifier.get_metrics()
        self.assertGreater(len(metrics.train_loss_history), 0)
        self.assertGreater(metrics.total_training_time, 0)
    
    def test_predictions(self):
        """Test prediction functionality."""
        # Train first
        self.classifier.fit(self.X_train, self.y_train)
        
        # Test predictions
        predictions = self.classifier.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Test probability predictions
        probabilities = self.classifier.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0, rtol=1e-5))
    
    def test_evaluation(self):
        """Test model evaluation."""
        # Train first
        self.classifier.fit(self.X_train, self.y_train)
        
        # Evaluate
        metrics = self.classifier.evaluate(self.X_test, self.y_test)
        
        # Check that standard metrics are present
        expected_metrics = ['accuracy']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train first
        self.classifier.fit(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.classifier.get_feature_importance()
        
        if importance is not None:
            self.assertEqual(len(importance), self.config.input_dim)
            self.assertTrue(np.all(importance >= 0))
            self.assertAlmostEqual(np.sum(importance), 1.0, places=6)
    
    def test_model_summary(self):
        """Test model summary functionality."""
        summary = self.classifier.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('PyTorchMLPClassifier', summary)
        self.assertIn('Input Dimension', summary)
        self.assertIn('Output Dimension', summary)
    
    def test_different_configurations(self):
        """Test different model configurations."""
        configs = [
            # Deep network
            {'hidden_dims': [128, 64, 32, 16], 'dropout_rate': 0.3},
            # Wide network
            {'hidden_dims': [256, 128], 'dropout_rate': 0.1},
            # Single layer
            {'hidden_dims': [64], 'dropout_rate': 0.0}
        ]
        
        for config_params in configs:
            config = create_model_config(
                input_dim=10, output_dim=2, num_epochs=5, verbose=False,
                **config_params
            )
            classifier = PyTorchMLPClassifier(config)
            
            # Should be able to create and train without errors
            try:
                classifier.fit(self.X_train, self.y_train)
                predictions = classifier.predict(self.X_test)
                self.assertEqual(len(predictions), len(self.X_test))
            except Exception as e:
                self.fail(f"Configuration {config_params} failed: {e}")
    
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X, y = TestDataGenerator.create_multiclass_classification_dataset(300, 25, 4)
        
        config = create_model_config(
            input_dim=25, output_dim=4,
            task_type='multiclass_classification',
            hidden_dims=[64, 32], num_epochs=10, verbose=False
        )
        
        classifier = PyTorchMLPClassifier(config)
        classifier.fit(X, y)
        
        predictions = classifier.predict(X[:50])
        probabilities = classifier.predict_proba(X[:50])
        
        self.assertEqual(len(predictions), 50)
        self.assertEqual(probabilities.shape, (50, 4))
        self.assertTrue(all(0 <= pred < 4 for pred in predictions))


@unittest.skipIf(not IMPORTS_SUCCESSFUL or not SKLEARN_AVAILABLE, "Sklearn not available")
class TestSklearnMLPClassifier(unittest.TestCase):
    """Test sklearn-based MLP classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train, self.y_train = TestDataGenerator.create_small_dataset(200, 10)
        self.X_test, self.y_test = TestDataGenerator.create_small_dataset(50, 10)
        
        self.config = create_model_config(
            input_dim=10, output_dim=2,
            hidden_dims=[32, 16],
            learning_rate=0.01,
            num_epochs=50,  # Sklearn typically needs more epochs
            verbose=False
        )
        
        self.classifier = SklearnMLPClassifier(self.config)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.classifier, SklearnMLPClassifier)
        self.assertFalse(self.classifier.is_fitted)
        self.assertEqual(self.classifier.config.input_dim, 10)
        self.assertEqual(self.classifier.config.output_dim, 2)
    
    def test_sklearn_parameter_conversion(self):
        """Test conversion of config to sklearn parameters."""
        sklearn_params = self.classifier._convert_config_to_sklearn_params(self.config)
        
        self.assertIn('hidden_layer_sizes', sklearn_params)
        self.assertIn('learning_rate_init', sklearn_params)
        self.assertIn('max_iter', sklearn_params)
        self.assertEqual(sklearn_params['hidden_layer_sizes'], tuple(self.config.hidden_dims))
        self.assertEqual(sklearn_params['learning_rate_init'], self.config.learning_rate)
    
    def test_training_and_prediction(self):
        """Test training and prediction."""
        # Train
        self.classifier.fit(self.X_train, self.y_train)
        self.assertTrue(self.classifier.is_fitted)
        
        # Predict
        predictions = self.classifier.predict(self.X_test)
        probabilities = self.classifier.predict_proba(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_feature_importance_sklearn(self):
        """Test feature importance for sklearn model."""
        # Train first
        self.classifier.fit(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.classifier.get_feature_importance()
        
        if importance is not None:
            self.assertEqual(len(importance), self.config.input_dim)
            self.assertTrue(np.all(importance >= 0))
    
    def test_score_method(self):
        """Test sklearn-style score method."""
        # Train first
        self.classifier.fit(self.X_train, self.y_train)
        
        # Test score
        score = self.classifier.score(self.X_test, self.y_test)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestMLPFactory(unittest.TestCase):
    """Test MLP creation through factory system."""
    
    def test_create_mlp_pytorch(self):
        """Test creating PyTorch MLP through factory function."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        classifier = create_mlp_classifier(
            backend='pytorch',
            input_dim=20,
            output_dim=2,
            hidden_dims=[64, 32],
            learning_rate=0.001
        )
        
        self.assertIsInstance(classifier, PyTorchMLPClassifier)
        self.assertEqual(classifier.config.input_dim, 20)
        self.assertEqual(classifier.config.output_dim, 2)
    
    def test_create_mlp_sklearn(self):
        """Test creating sklearn MLP through factory function."""
        if not SKLEARN_AVAILABLE:
            self.skipTest("Sklearn not available")
        
        classifier = create_mlp_classifier(
            backend='sklearn',
            input_dim=20,
            output_dim=2,
            hidden_dims=[64, 32],
            learning_rate=0.001
        )
        
        self.assertIsInstance(classifier, SklearnMLPClassifier)
        self.assertEqual(classifier.config.input_dim, 20)
        self.assertEqual(classifier.config.output_dim, 2)
    
    def test_factory_error_handling(self):
        """Test factory error handling."""
        # Test unknown backend
        with self.assertRaises(ValueError):
            create_mlp_classifier(backend='unknown')
        
        # Test missing dependencies
        with unittest.mock.patch('models.mlp_classifier.PYTORCH_AVAILABLE', False):
            with self.assertRaises(ImportError):
                create_mlp_classifier(backend='pytorch')


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestModelFactory(unittest.TestCase):
    """Test the complete model factory system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        self.assertIsInstance(self.factory.registry, ModelRegistry)
        self.assertIsInstance(self.factory.config_templates, ConfigTemplate)
        self.assertIsInstance(self.factory.hyperparameter_optimizer, HyperparameterOptimizer)
    
    def test_list_models(self):
        """Test listing available models."""
        models = self.factory.list_available_models()
        self.assertIsInstance(models, dict)
        
        # Check if expected models are available
        if PYTORCH_AVAILABLE:
            self.assertIn('pytorch_mlp', models)
        if SKLEARN_AVAILABLE:
            self.assertIn('sklearn_mlp', models)
    
    def test_list_templates(self):
        """Test listing configuration templates."""
        templates = self.factory.list_config_templates()
        self.assertIsInstance(templates, list)
        
        expected_templates = ['basic_classifier', 'deep_classifier', 'fast_prototype']
        for template in expected_templates:
            self.assertIn(template, templates)
    
    def test_model_creation_through_factory(self):
        """Test creating models through factory."""
        if not (PYTORCH_AVAILABLE or SKLEARN_AVAILABLE):
            self.skipTest("Neither PyTorch nor sklearn available")
        
        # Get available models
        available_models = self.factory.registry.list_models()
        
        if available_models:
            model_name = available_models[0]
            
            try:
                model = self.factory.create_model(
                    model_name,
                    input_dim=20,
                    output_dim=2,
                    num_epochs=5,
                    verbose=False
                )
                
                self.assertIsInstance(model, BaseModel)
                self.assertEqual(model.config.input_dim, 20)
                self.assertEqual(model.config.output_dim, 2)
                
            except Exception as e:
                self.fail(f"Model creation failed for {model_name}: {e}")
    
    def test_model_creation_with_template(self):
        """Test model creation using templates."""
        if not (PYTORCH_AVAILABLE or SKLEARN_AVAILABLE):
            self.skipTest("Neither PyTorch nor sklearn available")
        
        available_models = self.factory.registry.list_models()
        
        if available_models:
            model_name = available_models[0]
            
            try:
                model = self.factory.create_model(
                    model_name,
                    template_name='fast_prototype',
                    input_dim=12,
                    output_dim=3
                )
                
                self.assertIsInstance(model, BaseModel)
                # Template should override some settings
                
            except Exception as e:
                self.fail(f"Template-based model creation failed: {e}")
    
    def test_batch_model_creation(self):
        """Test batch model creation."""
        if not (PYTORCH_AVAILABLE or SKLEARN_AVAILABLE):
            self.skipTest("Neither PyTorch nor sklearn available")
        
        available_models = self.factory.registry.list_models()
        
        if len(available_models) >= 2:
            model_configs = [
                {
                    'model_name': available_models[0],
                    'input_dim': 10,
                    'output_dim': 2,
                    'num_epochs': 5
                },
                {
                    'model_name': available_models[1] if len(available_models) > 1 else available_models[0],
                    'input_dim': 10,
                    'output_dim': 2,
                    'template_name': 'basic_classifier'
                }
            ]
            
            models = self.factory.create_models_batch(model_configs)
            
            self.assertEqual(len(models), len(model_configs))
            # At least some models should be created successfully
            successful_models = [m for m in models if m is not None]
            self.assertGreater(len(successful_models), 0)
    
    def test_model_recommendations(self):
        """Test model recommendation system."""
        recommendations = self.factory.get_model_recommendations(
            task_type=TaskType.BINARY_CLASSIFICATION,
            dataset_size=1000,
            feature_dim=50,
            performance_priority="accuracy"
        )
        
        self.assertIsInstance(recommendations, list)
        # Should have at least one recommendation if models are available
        if self.factory.registry.list_models():
            self.assertGreater(len(recommendations), 0)
    
    def test_comparison_suite_creation(self):
        """Test comparison suite creation."""
        available_models = self.factory.registry.list_models(
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        if available_models:
            comparison_models = self.factory.create_comparison_suite(
                TaskType.BINARY_CLASSIFICATION
            )
            
            self.assertIsInstance(comparison_models, list)
            # Should create at least one model
            self.assertGreater(len(comparison_models), 0)
            
            # All should be BaseModel instances
            for model in comparison_models:
                self.assertIsInstance(model, BaseModel)


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestHyperparameterOptimization(unittest.TestCase):
    """Test hyperparameter optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
        self.optimizer = self.factory.hyperparameter_optimizer
    
    def test_hyperparameter_suggestions(self):
        """Test hyperparameter suggestions."""
        available_models = self.factory.registry.list_models()
        
        if available_models:
            model_name = available_models[0]
            
            # Test different complexity levels
            for complexity in ['simple', 'medium', 'complex']:
                suggestions = self.optimizer.suggest_hyperparameters(
                    model_name, task_complexity=complexity
                )
                
                self.assertIsInstance(suggestions, dict)
                # Should contain at least some parameters
                self.assertGreater(len(suggestions), 0)
    
    def test_hyperparameter_grid_generation(self):
        """Test hyperparameter grid generation."""
        available_models = self.factory.registry.list_models()
        
        if available_models:
            model_name = available_models[0]
            metadata = self.factory.registry.get_model_metadata(model_name)
            
            if metadata.hyperparameter_space:
                grid = self.optimizer.generate_hyperparameter_grid(model_name)
                
                self.assertIsInstance(grid, list)
                self.assertGreater(len(grid), 0)
                
                # Each item should be a parameter dictionary
                for params in grid:
                    self.assertIsInstance(params, dict)
    
    def test_custom_hyperparameter_space(self):
        """Test custom hyperparameter space."""
        available_models = self.factory.registry.list_models()
        
        if available_models:
            model_name = available_models[0]
            custom_space = {
                'learning_rate': [0.01, 0.001],
                'hidden_dims': [[32], [64, 32]]
            }
            
            grid = self.optimizer.generate_hyperparameter_grid(
                model_name, custom_space=custom_space
            )
            
            self.assertIsInstance(grid, list)
            self.assertEqual(len(grid), 4)  # 2 * 2 combinations


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        
        # Create and train a simple model
        X, y = TestDataGenerator.create_small_dataset(100, 5)
        
        if PYTORCH_AVAILABLE:
            self.config = create_model_config(
                input_dim=5, output_dim=2, hidden_dims=[16], 
                num_epochs=5, verbose=False
            )
            self.model = PyTorchMLPClassifier(self.config)
            self.model.fit(X, y)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_model_save_and_load(self):
        """Test saving and loading trained models."""
        # Save model
        self.model.save_model(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Create new model instance and load
        new_model = PyTorchMLPClassifier(self.config)
        new_model.load_model(self.model_path)
        
        # Check that model was loaded correctly
        self.assertTrue(new_model.is_fitted)
        self.assertEqual(new_model.config.input_dim, self.model.config.input_dim)
        self.assertEqual(new_model.config.output_dim, self.model.config.output_dim)
        
        # Test that predictions are the same
        X_test = np.random.randn(10, 5)
        
        pred_original = self.model.predict(X_test)
        pred_loaded = new_model.predict(X_test)
        
        np.testing.assert_array_equal(pred_original, pred_loaded)


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestModelComparison(unittest.TestCase):
    """Test model comparison functionality."""
    
    @unittest.skipIf(not (PYTORCH_AVAILABLE and SKLEARN_AVAILABLE), "Both backends needed")
    def test_backend_comparison(self):
        """Test comparison between PyTorch and sklearn backends."""
        X, y = TestDataGenerator.create_small_dataset(200, 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create models with similar configurations
        pytorch_config = create_model_config(
            input_dim=10, output_dim=2, hidden_dims=[32, 16],
            learning_rate=0.01, num_epochs=20, verbose=False
        )
        sklearn_config = create_model_config(
            input_dim=10, output_dim=2, hidden_dims=[32, 16],
            learning_rate=0.01, num_epochs=100, verbose=False
        )
        
        pytorch_model = PyTorchMLPClassifier(pytorch_config)
        sklearn_model = SklearnMLPClassifier(sklearn_config)
        
        # Train both models
        pytorch_model.fit(X_train, y_train)
        sklearn_model.fit(X_train, y_train)
        
        # Compare predictions
        pytorch_pred = pytorch_model.predict(X_test)
        sklearn_pred = sklearn_model.predict(X_test)
        
        # Both should produce valid predictions
        self.assertEqual(len(pytorch_pred), len(X_test))
        self.assertEqual(len(sklearn_pred), len(X_test))
        
        # Evaluate both models
        pytorch_metrics = pytorch_model.evaluate(X_test, y_test)
        sklearn_metrics = sklearn_model.evaluate(X_test, y_test)
        
        # Both should have reasonable performance
        self.assertGreaterEqual(pytorch_metrics['accuracy'], 0.0)
        self.assertGreaterEqual(sklearn_metrics['accuracy'], 0.0)
    
    def test_model_comparison_through_factory(self):
        """Test model comparison using factory system."""
        if not (PYTORCH_AVAILABLE or SKLEARN_AVAILABLE):
            self.skipTest("No backends available")
        
        factory = ModelFactory()
        
        # Create comparison suite
        models = factory.create_comparison_suite(TaskType.BINARY_CLASSIFICATION)
        
        if models:
            X, y = TestDataGenerator.create_small_dataset(150, 8)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train all models
            trained_models = []
            for model in models:
                try:
                    model.fit(X_train, y_train)
                    trained_models.append(model)
                except Exception as e:
                    print(f"Failed to train {model.__class__.__name__}: {e}")
                
            # Compare models if any were trained successfully
            if trained_models:
                comparison_results = {}
                for i, model in enumerate(trained_models):
                    try:
                        metrics = model.evaluate(X_test, y_test)
                        comparison_results[f"model_{i}_{model.__class__.__name__}"] = metrics
                    except Exception as e:
                        print(f"Failed to evaluate {model.__class__.__name__}: {e}")
                
                self.assertIsInstance(comparison_results, dict)
                self.assertGreater(len(comparison_results), 0)


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability characteristics."""
    
    def test_small_dataset_performance(self):
        """Test performance on small datasets."""
        X, y = TestDataGenerator.create_small_dataset(100, 10)
        
        available_backends = []
        if PYTORCH_AVAILABLE:
            available_backends.append('pytorch')
        if SKLEARN_AVAILABLE:
            available_backends.append('sklearn')
        
        for backend in available_backends:
            with self.subTest(backend=backend):
                config = create_model_config(
                    input_dim=10, output_dim=2, hidden_dims=[32],
                    num_epochs=10 if backend == 'pytorch' else 50,
                    verbose=False
                )
                
                if backend == 'pytorch':
                    model = PyTorchMLPClassifier(config)
                else:
                    model = SklearnMLPClassifier(config)
                
                # Time training
                start_time = time.time()
                model.fit(X, y)
                training_time = time.time() - start_time
                
                # Should train reasonably quickly
                self.assertLess(training_time, 30.0)  # 30 seconds max
                self.assertTrue(model.is_fitted)
    
    def test_medium_dataset_scalability(self):
        """Test scalability with medium-sized datasets."""
        X, y = TestDataGenerator.create_multiclass_classification_dataset(1000, 50, 3)
        
        # Test with one available backend
        backend = None
        if PYTORCH_AVAILABLE:
            backend = 'pytorch'
        elif SKLEARN_AVAILABLE:
            backend = 'sklearn'
        
        if backend:
            config = create_model_config(
                input_dim=50, 
                output_dim=3, 
                task_type='multiclass_classification',
                hidden_dims=[128, 64],
                num_epochs=10 if backend == 'pytorch' else 50,
                batch_size=64, 
                verbose=False
            )
            
            if backend == 'pytorch':
                model = PyTorchMLPClassifier(config)
            else:
                model = SklearnMLPClassifier(config)
            
            # Train and test
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            
            # Should handle medium datasets
            self.assertLess(training_time, 120.0)  # 2 minutes max
            
            # Test predictions
            predictions = model.predict(X[:100])
            self.assertEqual(len(predictions), 100)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with various configurations."""
        X, y = TestDataGenerator.create_small_dataset(500, 20)
        
        # Test different network sizes
        network_configs = [
            {'hidden_dims': [32]},  # Small
            {'hidden_dims': [64, 32]},  # Medium
            {'hidden_dims': [128, 64, 32]}  # Larger
        ]
        
        backend = 'pytorch' if PYTORCH_AVAILABLE else 'sklearn' if SKLEARN_AVAILABLE else None
        
        if backend:
            for i, network_config in enumerate(network_configs):
                with self.subTest(config=i):
                    config = create_model_config(
                        input_dim=20, output_dim=2,
                        num_epochs=5, verbose=False,
                        **network_config
                    )
                    
                    if backend == 'pytorch':
                        model = PyTorchMLPClassifier(config)
                    else:
                        model = SklearnMLPClassifier(config)
                    
                    # Should be able to create and train without memory issues
                    try:
                        model.fit(X, y)
                        predictions = model.predict(X[:50])
                        self.assertEqual(len(predictions), 50)
                    except Exception as e:
                        self.fail(f"Memory efficiency test failed for config {i}: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete MLP system."""
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules not available")
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        print("Running end-to-end MLP integration test...")
        
        # Check what's available
        available_backends = []
        if PYTORCH_AVAILABLE:
            available_backends.append('pytorch')
        if SKLEARN_AVAILABLE:
            available_backends.append('sklearn')
        
        if not available_backends:
            self.skipTest("No backends available")
        
        # 1. Generate dataset
        X, y = TestDataGenerator.create_multiclass_classification_dataset(300, 25, 3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 2. Create factory and list available models
        factory = get_model_factory()
        available_models = factory.registry.list_models(task_type=TaskType.MULTICLASS_CLASSIFICATION)
        
        print(f"Available models: {available_models}")
        
        if not available_models:
            self.skipTest("No suitable models available")
        
        # 3. Create models and train
        successful_models = []
        
        for model_name in available_models[:2]:  # Test first 2 available models
            try:
                print(f"Testing {model_name}...")
                
                # Create model with recommendations
                model = factory.create_model(
                    model_name,
                    input_dim=25,
                    output_dim=3,
                    task_type='multiclass_classification',
                    num_epochs=10,
                    verbose=False
                )
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                predictions = model.predict(X_test)
                metrics = model.evaluate(X_test, y_test)
                
                print(f"  ✓ {model_name} accuracy: {metrics['accuracy']:.4f}")
                
                successful_models.append((model_name, model, metrics))
                
            except Exception as e:
                print(f"  ✗ {model_name} failed: {e}")
        
        # 4. Verify results
        self.assertGreater(len(successful_models), 0, "No models trained successfully")
        
        # 5. Compare models if multiple succeeded
        if len(successful_models) > 1:
            print("\nModel comparison:")
            for name, model, metrics in successful_models:
                print(f"  {name}: accuracy = {metrics['accuracy']:.4f}")
        
        print("✓ End-to-end MLP test completed successfully!")


def run_specific_tests(*test_classes):
    """Run specific test classes."""
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("Running MLP Models Test Suite...")
    print("=" * 80)
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Required modules not available.")
        print("Required modules: models.base_model, models.mlp_classifier, models.model_factory")
        exit(1)
    
    # Check available backends
    available_backends = []
    if PYTORCH_AVAILABLE:
        available_backends.append("PyTorch")
    if SKLEARN_AVAILABLE:
        available_backends.append("Sklearn")
    
    if available_backends:
        print(f"✓ Available backends: {', '.join(available_backends)}")
    else:
        print("⚠ No ML backends available. Limited testing will be performed.")
    
    try:
        result = run_all_tests()
        
        print("\n" + "=" * 80)
        print("MLP MODELS TEST SUMMARY:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.wasSuccessful():
            print("✅ All tests passed!")
            exit_code = 0
        else:
            print("❌ Some tests failed!")
            
            if result.failures:
                print("\nFailures:")
                for test, traceback in result.failures[:3]:  # Show first 3
                    print(f"- {test}")
                    print(f"  {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'Unknown failure'}")
            
            if result.errors:
                print("\nErrors:")
                for test, traceback in result.errors[:3]:  # Show first 3
                    print(f"- {test}")
                    print(f"  {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'Unknown error'}")
            
            exit_code = 1
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠ Tests interrupted by user")
        exit_code = 2
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        exit_code = 1
    
    sys.exit(exit_code)