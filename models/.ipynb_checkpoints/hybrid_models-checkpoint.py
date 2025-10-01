import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent  # models -> Model
    paths = [
        str(model_dir),
        str(model_dir/'experiments'), 
        str(model_dir/'models'),
        str(model_dir/'features'),
        str(model_dir/'data'),
        str(model_dir/'utils')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

from traditional_features import MelSpectrogramExtractor, MFCCExtractor
from chaotic_features import ChaoticFeatureExtractor
from mlp_classifier import MLPClassifier
from chaotic_network import ChaoticEmbedding, AttractorPooling, SpeakerEmbedding, ChaoticClassifier

class MockFeatureExtractor(nn.Module):
    """Mock feature extractor for testing."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim  # Can be None for auto-adaptation
        self.output_dim = output_dim
        self.linear = None  # Will be initialized on first forward pass
    
    def forward(self, x):
        # Initialize linear layer based on actual input shape
        if self.linear is None:
            if len(x.shape) == 2:
                # [batch_size, features]
                actual_input_dim = x.shape[-1]
            elif len(x.shape) == 3:
                # [batch_size, sequence, features] - flatten sequence dimension
                actual_input_dim = x.shape[1] * x.shape[2]
                x = x.view(x.shape[0], -1)  # Flatten to [batch_size, sequence*features]
            else:
                # Fallback: use last dimension
                actual_input_dim = x.shape[-1]
            
            self.linear = nn.Linear(actual_input_dim, self.output_dim).to(x.device)
        
        # Ensure input matches expected shape for linear layer
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
            
        return self.linear(x)
        

class FeatureDimensionAdapter(nn.Module):
    """
    Adapter layer to match different feature dimensions for model compatibility.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adaptation_type: str = 'linear',
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize dimension adapter.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            adaptation_type: Type of adaptation ('linear', 'mlp', 'attention')
            hidden_dim: Hidden dimension for MLP adaptation
        """
        super(FeatureDimensionAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adaptation_type = adaptation_type
        
        if adaptation_type == 'linear':
            self.adapter = nn.Linear(input_dim, output_dim)
            
        elif adaptation_type == 'mlp':
            hidden_dim = hidden_dim or max(input_dim, output_dim)
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
            
        elif adaptation_type == 'attention':
            self.adapter = nn.MultiheadAttention(
                embed_dim=input_dim, 
                num_heads=8, 
                batch_first=True
            )
            if input_dim != output_dim:
                self.projection = nn.Linear(input_dim, output_dim)
            else:
                self.projection = nn.Identity()
                
        else:
            raise ValueError(f"Unknown adaptation type: {adaptation_type}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Adapt feature dimensions.
        
        Args:
            features: Input features [batch_size, ...] 
            
        Returns:
            Adapted features [batch_size, output_dim]
        """
        if self.adaptation_type in ['linear', 'mlp']:
            # Flatten if necessary
            if len(features.shape) > 2:
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)
            return self.adapter(features)
            
        elif self.adaptation_type == 'attention':
            # Attention-based adaptation
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
                
            attended_features, _ = self.adapter(features, features, features)
            pooled_features = torch.mean(attended_features, dim=1)  # Global average pooling
            
            return self.projection(pooled_features)


class TraditionalChaoticHybrid(nn.Module):
    """
    Hybrid Model: Traditional Features (Mel/MFCC) + Chaotic Network Components
    
    This model combines traditional spectral features with chaotic neural network
    components for enhanced speaker discrimination.
    """
    
    def __init__(
        self,
        # Feature extraction parameters
        feature_type: str = 'mel',  # 'mel' or 'mfcc'
        n_mels: int = 80,
        n_mfcc: int = 13,
        sample_rate: int = 16000,
        
        # Chaotic network parameters  
        chaotic_embedding_dim: int = 64,
        evolution_time: float = 0.3,
        pooling_type: str = 'comprehensive',
        
        # Speaker embedding parameters
        speaker_embedding_dim: int = 128,
        
        # Classification parameters
        num_speakers: int = 100,
        classifier_type: str = 'cosine',
        
        device: str = 'cpu'
    ):
        """
        Initialize Traditional-Chaotic Hybrid Model.
        """
        super(TraditionalChaoticHybrid, self).__init__()
        
        self.feature_type = feature_type
        self.device = device
        
        # Traditional feature extractor
        if feature_type == 'mel':
            if MelSpectrogramExtractor is not None:
                self.feature_extractor = MelSpectrogramExtractor(
                    n_mels=n_mels,
                    sample_rate=sample_rate
                )
                traditional_feature_dim = n_mels
            else:
                self.feature_extractor = MockFeatureExtractor(None, n_mels)
                traditional_feature_dim = n_mels
                
        elif feature_type == 'mfcc':
            if MFCCExtractor is not None:
                self.feature_extractor = MFCCExtractor(
                    n_mfcc=n_mfcc,
                    sample_rate=sample_rate
                )
                traditional_feature_dim = n_mfcc
            else:
                self.feature_extractor = MockFeatureExtractor(None, n_mfcc)
                traditional_feature_dim = n_mfcc
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Dimension adapter to match chaotic embedding input requirements
        self.dimension_adapter = FeatureDimensionAdapter(
            input_dim=traditional_feature_dim,
            output_dim=4,  # Standard chaotic feature dimension
            adaptation_type='mlp'
        )
        
        # Chaotic network components
        if ChaoticEmbedding is not None:
            self.chaotic_embedding = ChaoticEmbedding(
                input_dim=4,
                evolution_time=evolution_time,
                device=device
            )
        else:
            trajectory_dim = int(evolution_time / 0.01) * 3
            self.chaotic_embedding = MockFeatureExtractor(None, trajectory_dim)
        
        if AttractorPooling is not None:
            self.attractor_pooling = AttractorPooling(
                pooling_type=pooling_type,
                device=device
            )
            pooling_output_dim = 5 if pooling_type == 'comprehensive' else 3
        else:
            pooling_output_dim = 5
            self.attractor_pooling = MockFeatureExtractor(None, pooling_output_dim)
        
        # Speaker embedding
        if SpeakerEmbedding is not None:
            self.speaker_embedding = SpeakerEmbedding(
                input_dim=pooling_output_dim,
                embedding_dim=speaker_embedding_dim
            )
        else:
            self.speaker_embedding = MockFeatureExtractor(None, speaker_embedding_dim)
        
        # Classifier
        if ChaoticClassifier is not None:
            self.classifier = ChaoticClassifier(
                embedding_dim=speaker_embedding_dim,
                num_speakers=num_speakers,
                classifier_type=classifier_type
            )
        else:
            # 创建一个能接受两个参数的 mock classifier
            class MockClassifier(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.linear = nn.Linear(input_dim, output_dim)
                
                def forward(self, x, labels=None):
                    # 忽略 labels 参数，只使用输入特征
                    return self.linear(x)
            
            self.classifier = MockClassifier(speaker_embedding_dim, num_speakers)
    
    def forward(
        self, 
        audio: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through traditional-chaotic hybrid model.
        
        Args:
            audio: Input audio tensor
            labels: Speaker labels (for training)
            
        Returns:
            Classification logits
        """
        # Extract traditional features - 统一调用方式
        if isinstance(self.feature_extractor, MockFeatureExtractor):
            traditional_features = self.feature_extractor(audio)
        else:
            # 真实的特征提取器使用 extract 方法，需要转换为numpy
            if isinstance(audio, torch.Tensor):
                # 转换为numpy数组，librosa需要numpy输入
                audio_np = audio.detach().cpu().numpy()
                traditional_features = self.feature_extractor.extract(audio_np)
                # 转换回tensor
                traditional_features = torch.tensor(traditional_features, dtype=torch.float32, device=audio.device)
            else:
                traditional_features = self.feature_extractor.extract(audio)
        
        # Adapt dimensions for chaotic embedding
        adapted_features = self.dimension_adapter(traditional_features)
        
        # Pass through chaotic network components
        if hasattr(self.chaotic_embedding, 'forward'):
            chaotic_trajectories = self.chaotic_embedding(adapted_features)
        else:
            # Handle mock component
            flat_output = self.chaotic_embedding(adapted_features)
            batch_size = flat_output.shape[0]
            trajectory_length = flat_output.shape[1] // 3
            chaotic_trajectories = flat_output.view(batch_size, trajectory_length, 3)
        
        pooled_features = self.attractor_pooling(chaotic_trajectories)
        speaker_embeddings = self.speaker_embedding(pooled_features)
        
        # Classification
        if hasattr(self.classifier, 'forward'):
            logits = self.classifier(speaker_embeddings, labels)
        else:
            logits = self.classifier(speaker_embeddings)
        
        return logits


class ChaoticMLPHybrid(nn.Module):
    """
    Hybrid Model: Chaotic Features + Traditional MLP Classifier
    
    This model uses chaotic feature extraction but employs a traditional 
    MLP classifier instead of the chaotic classifier.
    """
    
    def __init__(
        self,
        # Audio processing parameters
        sample_rate: int = 16000,
        embedding_dim: int = 10,
        
        # Chaotic feature parameters
        mlsa_scales: int = 5,
        rqa_radius_ratio: float = 0.1,
        
        # MLP parameters
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        
        # Classification parameters
        num_speakers: int = 100,
        
        device: str = 'cpu'
    ):
        """
        Initialize Chaotic-MLP Hybrid Model.
        """
        super(ChaoticMLPHybrid, self).__init__()
        
        self.device = device
        
        # Chaotic feature extractor
        if ChaoticFeatureExtractor is not None:
            self.feature_extractor = ChaoticFeatureExtractor(
                sample_rate=sample_rate,
                embedding_dim=embedding_dim,
                mlsa_scales=mlsa_scales,
                rqa_radius_ratio=rqa_radius_ratio,
                device=device
            )
            chaotic_feature_dim = mlsa_scales + 3  # MLSA + RQA features
        else:
            chaotic_feature_dim = 8  # Default feature dimension
            self.feature_extractor = MockFeatureExtractor(None, chaotic_feature_dim)
        
        # Traditional MLP classifier
        if MLPClassifier is not None:
            self.classifier = MLPClassifier(
                input_dim=chaotic_feature_dim,
                hidden_dims=hidden_dims,
                output_dim=num_speakers,
                dropout_rate=dropout_rate,
                activation=activation
            )
        else:
            # Simple MLP implementation
            layers = []
            current_dim = chaotic_feature_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU() if activation == 'relu' else nn.Tanh(),
                    nn.Dropout(dropout_rate)
                ])
                current_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, num_speakers))
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through chaotic-MLP hybrid model.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Classification logits
        """
        # Extract chaotic features
        chaotic_features = self.feature_extractor(audio)
        
        # Classify using MLP
        logits = self.classifier(chaotic_features)
        
        return logits


class TraditionalMLPBaseline(nn.Module):
    """
    Baseline Model: Traditional Features (Mel/MFCC) + MLP Classifier
    
    This serves as the traditional baseline for comparison with chaotic approaches.
    """
    def __init__(
        self,
        # 接受但忽略不相关的参数
        input_dim: Optional[int] = None,  # 添加
        num_classes: Optional[int] = None,  # 添加
        baseline_type: Optional[str] = None,  # 添加
        
        # Feature extraction parameters
        feature_type: str = 'mel',
        n_mels: int = 80,
        n_mfcc: int = 13,
        sample_rate: int = 16000,
        
        # MLP parameters
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        
        # Classification parameters
        num_speakers: int = 100,
        
        device: str = 'cpu',
        
        **kwargs  # 添加这个来接受额外参数
        
    ):
        
        """Initialize Traditional-MLP Baseline Model."""
        super(TraditionalMLPBaseline, self).__init__()
        
        print("=== TraditionalMLPBaseline 初始化开始 ===")
        print(f"接收到的参数: {kwargs}")
        
        # 参数处理
        feature_type = kwargs.get('feature_type', 'mel')
        n_mels = kwargs.get('n_mels', 80)
        n_mfcc = kwargs.get('n_mfcc', 13)
        hidden_dims = kwargs.get('hidden_dims', [256, 128, 64])
        num_speakers = kwargs.get('num_speakers', 100)
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        use_batch_norm = kwargs.get('use_batch_norm', True)
        activation = kwargs.get('activation', 'relu')
        
        print(f"处理后的参数:")
        print(f"  feature_type: {feature_type}")
        print(f"  n_mels: {n_mels}, n_mfcc: {n_mfcc}")
        print(f"  hidden_dims: {hidden_dims}")
        print(f"  num_speakers: {num_speakers}")
        
        self.feature_type = feature_type
        
        # Traditional feature extractor
        print("创建特征提取器...")
        try:
            if feature_type == 'mel':
                if MelSpectrogramExtractor is not None:
                    print("使用真实的 MelSpectrogramExtractor")
                    self.feature_extractor = MelSpectrogramExtractor(n_mels=n_mels)
                    feature_dim = n_mels
                else:
                    print("使用 Mock MelSpectrogramExtractor")
                    self.feature_extractor = MockFeatureExtractor(None, n_mels)
                    feature_dim = n_mels
            else:
                print(f"不支持的特征类型: {feature_type}")
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            print(f"特征提取器创建成功，feature_dim: {feature_dim}")
            
        except Exception as e:
            print(f"特征提取器创建失败: {e}")
            raise
        
        # MLP classifier
        print("创建分类器...")
        try:
            # 强制使用手动构建，跳过 MLPClassifier
            print("使用手动构建的MLP")
            
            layers = []
            current_dim = feature_dim
            
            print(f"开始构建MLP: {current_dim} -> {hidden_dims} -> {num_speakers}")
            
            for i, hidden_dim in enumerate(hidden_dims):
                print(f"添加层 {i+1}: Linear({current_dim}, {hidden_dim})")
                layers.append(nn.Linear(current_dim, hidden_dim))
                
                if use_batch_norm:
                    print(f"添加层 {i+1}: BatchNorm1d({hidden_dim})")
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    
                print(f"添加层 {i+1}: ReLU()")
                layers.append(nn.ReLU())
                
                print(f"添加层 {i+1}: Dropout({dropout_rate})")
                layers.append(nn.Dropout(dropout_rate))
                
                current_dim = hidden_dim
            
            print(f"添加输出层: Linear({current_dim}, {num_speakers})")
            layers.append(nn.Linear(current_dim, num_speakers))
            
            print(f"总共创建了 {len(layers)} 层")
            
            self.classifier = nn.Sequential(*layers)
            
            # 验证分类器
            total_params = sum(p.numel() for p in self.classifier.parameters())
            print(f"分类器参数总数: {total_params:,}")
            
            if total_params == 0:
                print("错误：分类器没有参数！")
                raise ValueError("分类器创建失败")
            
        except Exception as e:
            print(f"分类器创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 验证整个模型
        total_model_params = sum(p.numel() for p in self.parameters())
        print(f"整个模型参数总数: {total_model_params:,}")
        
        if total_model_params == 0:
            print("错误：整个模型没有参数！")
            raise ValueError("模型创建失败")
        
        print("=== TraditionalMLPBaseline 初始化完成 ===")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through traditional-MLP baseline."""
        
        # Extract traditional features
        if self.feature_extractor is None:
            # If no feature extractor, use input directly
            features = audio
        elif isinstance(self.feature_extractor, MockFeatureExtractor):
            features = self.feature_extractor(audio)
        else:
            # Real feature extractor, convert data type
            batch_size = audio.shape[0]
            features_list = []
            
            for i in range(batch_size):
                single_audio = audio[i]
                
                if isinstance(single_audio, torch.Tensor):
                    audio_np = single_audio.detach().cpu().numpy()
                    feats = self.feature_extractor.extract(audio_np)
                    # Convert to tensor: feats shape is [n_mels/n_mfcc, time_steps]
                    feats = torch.tensor(feats, dtype=torch.float32, device=audio.device)
                    
                    # CRITICAL: Pool over time dimension to get fixed-size feature
                    # [n_mels, time_steps] -> [n_mels]
                    if len(feats.shape) == 2:
                        feats = torch.mean(feats, dim=-1)
                else:
                    feats = self.feature_extractor.extract(single_audio)
                    if len(feats.shape) == 2:
                        feats = torch.mean(torch.tensor(feats), dim=-1)
                
                features_list.append(feats)
            
            # Stack into batch: [batch_size, feature_dim]
            features = torch.stack(features_list, dim=0)
        
        # Ensure 2D shape [batch_size, feature_dim] for classifier
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)
        elif len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Classify using MLP
        logits = self.classifier(features)
        
        return logits


class HybridModelManager:
    """
    Manager class for creating and managing different hybrid model configurations.
    """
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create hybrid model based on type and configuration.
        
        Args:
            model_type: Type of hybrid model
            config: Model configuration dictionary
            
        Returns:
            Initialized hybrid model
        """
        if model_type == 'traditional_chaotic':
            return TraditionalChaoticHybrid(**config)
        elif model_type == 'chaotic_mlp':
            return ChaoticMLPHybrid(**config)
        elif model_type == 'traditional_mlp':
            return TraditionalMLPBaseline(**config)
        else:
            raise ValueError(f"Unknown hybrid model type: {model_type}")
    
    @staticmethod
    def get_comparison_configs() -> Dict[str, Dict[str, Any]]:
        """
        Get standard configurations for comparison experiments.
        
        Returns:
            Dictionary of model configurations for comparison
        """
        base_config = {
            'sample_rate': 16000,
            'num_speakers': 100,
            'device': 'cpu'
        }
        
        configs = {
            # Method 1: Mel/MFCC + Chaotic Network
            'mel_chaotic': {
                **base_config,
                'feature_type': 'mel',
                'n_mels': 80,
                'evolution_time': 0.3,
                'pooling_type': 'comprehensive',
                'speaker_embedding_dim': 128,
                'classifier_type': 'cosine'
            },
            
            'mfcc_chaotic': {
                **base_config,
                'feature_type': 'mfcc', 
                'n_mfcc': 13,
                'evolution_time': 0.3,
                'pooling_type': 'comprehensive',
                'speaker_embedding_dim': 128,
                'classifier_type': 'cosine'
            },
            
            # Method 2: Mel/MFCC + MLP
            'mel_mlp': {
                **base_config,
                'feature_type': 'mel',
                'n_mels': 80,
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'use_batch_norm': True
            },
            
            'mfcc_mlp': {
                **base_config,
                'feature_type': 'mfcc',
                'n_mfcc': 13,
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.2,
                'activation': 'relu',
                'use_batch_norm': True
            },
            
            # Method 3: Chaotic Features + MLP
            'chaotic_mlp': {
                **base_config,
                'embedding_dim': 10,
                'mlsa_scales': 5,
                'rqa_radius_ratio': 0.1,
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.2,
                'activation': 'relu'
            }
        }
        
        return configs
    
    @staticmethod
    def create_comparison_models() -> Dict[str, nn.Module]:
        """
        Create all models for comparison experiments.
        
        Returns:
            Dictionary of initialized models
        """
        configs = HybridModelManager.get_comparison_configs()
        models = {}
        
        # Traditional-Chaotic hybrids
        models['mel_chaotic'] = HybridModelManager.create_model(
            'traditional_chaotic', configs['mel_chaotic']
        )
        models['mfcc_chaotic'] = HybridModelManager.create_model(
            'traditional_chaotic', configs['mfcc_chaotic'] 
        )
        
        # Traditional-MLP baselines
        models['mel_mlp'] = HybridModelManager.create_model(
            'traditional_mlp', configs['mel_mlp']
        )
        models['mfcc_mlp'] = HybridModelManager.create_model(
            'traditional_mlp', configs['mfcc_mlp']
        )
        
        # Chaotic-MLP hybrid
        models['chaotic_mlp'] = HybridModelManager.create_model(
            'chaotic_mlp', configs['chaotic_mlp']
        )
        
        return models
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        Get detailed information about a hybrid model.
        
        Args:
            model: Hybrid model instance
            
        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'components': []
        }
        
        # List model components
        for name, module in model.named_children():
            info['components'].append({
                'name': name,
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters())
            })
        
        return info


if __name__ == "__main__":
    # Test hybrid models
    print("Testing Hybrid Models...")
    
    # Test individual models
    print("\n1. Testing Traditional-Chaotic Hybrid (Mel + Chaotic):")
    mel_chaotic_config = {
        'feature_type': 'mel',
        'n_mels': 40,
        'evolution_time': 0.2,
        'speaker_embedding_dim': 64,
        'num_speakers': 10,
        'device': 'cpu'
    }
    
    mel_chaotic_model = TraditionalChaoticHybrid(**mel_chaotic_config)
    print(f"Model created: {type(mel_chaotic_model).__name__}")
    
    # Test forward pass
    batch_size = 4
    audio_length = 800
    test_audio = torch.randn(batch_size, audio_length)
    
    with torch.no_grad():
        logits = mel_chaotic_model(test_audio)
    print(f"Input shape: {test_audio.shape}")
    print(f"Output shape: {logits.shape}")
    
    print("\n2. Testing Chaotic-MLP Hybrid:")
    chaotic_mlp_config = {
        'sample_rate': 16000,
        'embedding_dim': 8,
        'mlsa_scales': 3,
        'hidden_dims': [64, 32],
        'num_speakers': 10,
        'device': 'cpu'
    }
    
    chaotic_mlp_model = ChaoticMLPHybrid(**chaotic_mlp_config)
    print(f"Model created: {type(chaotic_mlp_model).__name__}")
    
    with torch.no_grad():
        logits = chaotic_mlp_model(test_audio)
    print(f"Output shape: {logits.shape}")
    
    print("\n3. Testing Traditional-MLP Baseline:")
    traditional_mlp_config = {
        'feature_type': 'mfcc',
        'n_mfcc': 13,
        'hidden_dims': [64, 32],
        'num_speakers': 10,
        'device': 'cpu'
    }
    
    traditional_mlp_model = TraditionalMLPBaseline(**traditional_mlp_config)
    print(f"Model created: {type(traditional_mlp_model).__name__}")
    
    with torch.no_grad():
        logits = traditional_mlp_model(test_audio)
    print(f"Output shape: {logits.shape}")
    
    print("\n4. Testing Hybrid Model Manager:")
    comparison_models = HybridModelManager.create_comparison_models()
    
    print("Created comparison models:")
    for name, model in comparison_models.items():
        info = HybridModelManager.get_model_info(model)
        print(f"  {name}: {info['total_parameters']:,} parameters")
    
    print("\nHybrid Models test completed!")