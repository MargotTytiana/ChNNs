# enhanced_chaotic_network.py
"""
增强版混沌网络模型
修复了模型容量不足的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedChaoticNetwork(nn.Module):
    """增强容量的混沌网络模型"""
    
    def __init__(self, input_dim=230, num_speakers=251, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        # 增强的特征处理器
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4 - i * 0.1)  # 逐渐减少dropout
            ])
            prev_dim = hidden_dim
        
        self.feature_processor = nn.Sequential(*layers)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dims[-1], num_speakers)
        
        # 初始化
        self._initialize_weights()
        
        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🚀 增强模型参数: {total_params:,}")
        print(f"📊 每说话人参数: {total_params/num_speakers:.0f}")
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_processor(x)
        return self.classifier(features)

# 保持原始类名兼容性
ChaoticNetwork = EnhancedChaoticNetwork
