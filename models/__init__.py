"""
模型包兼容性层
确保向后兼容性
"""

# 导入增强模型作为默认实现
from .enhanced_chaotic_network import EnhancedChaoticNetwork

# 保持向后兼容性 - 将增强模型作为默认ChaoticNetwork
ChaoticNetwork = EnhancedChaoticNetwork

# 导出所有模型类
from .base_model import BaseModel, SklearnCompatibleModel, ModelConfig
from .chaotic_network import ChaoticSpeakerRecognitionNetwork
from .enhanced_chaotic_network import EnhancedChaoticNetwork
from .hybrid_models import TraditionalChaoticHybrid, ChaoticMLPHybrid, TraditionalMLPBaseline
from .mlp_classifier import MLPClassifier, SklearnMLPClassifier
from .model_factory import ModelFactory, ModelRegistry

__all__ = [
    'BaseModel',
    'SklearnCompatibleModel', 
    'ModelConfig',
    'ChaoticSpeakerRecognitionNetwork',
    'EnhancedChaoticNetwork',
    'ChaoticNetwork',  # 兼容性别名
    'TraditionalChaoticHybrid',
    'ChaoticMLPHybrid', 
    'TraditionalMLPBaseline',
    'MLPClassifier',
    'SklearnMLPClassifier',
    'ModelFactory',
    'ModelRegistry'
]
