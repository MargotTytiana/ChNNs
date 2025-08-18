"""
C-HiLAP (Chaotic Hierarchical Attractor Propagation) 主模型架构
混沌神经网络说话人识别项目的核心模型
集成混沌特征提取、洛伦兹振荡器、混沌注意力等组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math

# 导入前面实现的模块
from chaos_features import ChaosFeatureExtractor
from chaotic_layers import (
    LorentzOscillatorLayer,
    ChaoticAttentionMechanism,
    StrangeAttractorPooling,
    ChaosRegularizationLayer
)


class TDNN(nn.Module):
    """
    时延神经网络 (Time Delay Neural Network)
    用作C-HiLAP的基础特征提取backbone
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 context_size: int = 5,
                 dilation: int = 1):
        super(TDNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size = context_size
        self.dilation = dilation

        # 1D卷积实现时延
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=context_size,
            dilation=dilation,
            padding=(context_size - 1) * dilation // 2
        )

        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, output_dim]
        """
        # 转换为卷积格式
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # [batch_size, seq_len, output_dim]
        return x


class StatisticsPooling(nn.Module):
    """
    统计池化层
    计算均值和标准差统计量
    """

    def __init__(self, input_dim: int):
        super(StatisticsPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2  # mean + std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, input_dim * 2]
        """
        mean = torch.mean(x, dim=1)  # [batch_size, input_dim]
        std = torch.std(x, dim=1)  # [batch_size, input_dim]
        return torch.cat([mean, std], dim=1)


class CHiLAPCore(nn.Module):
    """
    C-HiLAP核心模块
    包含混沌嵌入层、分岔控制注意力、奇异吸引子池化
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 chaos_dim: int = 256,
                 num_chaos_layers: int = 3,
                 num_attention_heads: int = 8,
                 num_attractors: int = 4,
                 kolmogorov_entropy: float = 0.7):
        """
        初始化C-HiLAP核心模块

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            chaos_dim: 混沌嵌入维度
            num_chaos_layers: 混沌层数量
            num_attention_heads: 注意力头数
            num_attractors: 吸引子数量
            kolmogorov_entropy: 目标Kolmogorov-Sinai熵
        """
        super(CHiLAPCore, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chaos_dim = chaos_dim
        self.num_chaos_layers = num_chaos_layers

        # 可调节的Kolmogorov-Sinai熵参数
        self.ks_entropy_target = nn.Parameter(torch.tensor(kolmogorov_entropy))

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 混沌嵌入层序列
        self.chaos_layers = nn.ModuleList()
        for i in range(num_chaos_layers):
            chaos_layer = LorentzOscillatorLayer(
                input_dim=hidden_dim if i == 0 else chaos_dim,
                hidden_dim=chaos_dim,
                sigma=10.0 + i * 2.0,  # 递增的混沌参数
                rho=28.0 + i * 5.0,
                beta=8.0 / 3.0 + i * 0.5,
                integration_steps=10 + i * 2
            )
            self.chaos_layers.append(chaos_layer)

        # 分岔控制注意力机制
        self.chaotic_attention = ChaoticAttentionMechanism(
            hidden_dim=chaos_dim,
            num_heads=num_attention_heads,
            bifurcation_param=1.2,
            phase_coupling=0.6
        )

        # 奇异吸引子池化
        self.attractor_pooling = StrangeAttractorPooling(
            input_dim=chaos_dim,
            output_dim=chaos_dim,
            attractor_dim=3,
            num_attractors=num_attractors
        )

        # 混沌正则化
        self.chaos_regularization = ChaosRegularizationLayer(
            feature_dim=chaos_dim,
            lyapunov_target=0.5,
            entropy_weight=0.02
        )

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(chaos_dim) for _ in range(num_chaos_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]

        Returns:
            output: 混沌嵌入向量 [batch_size, chaos_dim]
            loss_dict: 各种损失的字典
        """
        loss_dict = {}

        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # 逐层应用混沌变换
        chaos_state = None
        for i, chaos_layer in enumerate(self.chaos_layers):
            x, chaos_state = chaos_layer(x, chaos_state)
            x = self.layer_norms[i](x)

            # 记录每层的混沌状态用于分析
            loss_dict[f'chaos_state_layer_{i}'] = chaos_state.detach()

        # 分岔控制注意力
        attended_features, sync_loss = self.chaotic_attention(x)
        loss_dict['phase_sync_loss'] = sync_loss

        # 混沌正则化
        regularized_features, reg_loss = self.chaos_regularization(attended_features)
        loss_dict['chaos_regularization_loss'] = reg_loss

        # 奇异吸引子池化
        pooled_output = self.attractor_pooling(regularized_features)

        # 计算Kolmogorov-Sinai熵损失
        ks_entropy_loss = self._compute_ks_entropy_loss(regularized_features)
        loss_dict['ks_entropy_loss'] = ks_entropy_loss

        return pooled_output, loss_dict

    def _compute_ks_entropy_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算Kolmogorov-Sinai熵损失
        鼓励特征表示保持适当的混沌特性
        """
        batch_size, seq_len, feature_dim = features.shape

        if seq_len < 10:
            return torch.tensor(0.0, device=features.device)

        # 估计轨迹的信息熵
        # 使用相邻时间步的距离分布来估计熵
        distances = torch.norm(features[:, 1:] - features[:, :-1], dim=-1)  # [batch_size, seq_len-1]

        # 将距离离散化为区间
        num_bins = 20
        min_dist = torch.min(distances)
        max_dist = torch.max(distances)

        if max_dist <= min_dist:
            return torch.tensor(0.0, device=features.device)

        # 计算直方图
        bin_edges = torch.linspace(min_dist, max_dist, num_bins + 1, device=features.device)
        histograms = []

        for i in range(batch_size):
            hist = torch.histc(distances[i], bins=num_bins, min=min_dist.item(), max=max_dist.item())
            histograms.append(hist)

        histograms = torch.stack(histograms, dim=0)  # [batch_size, num_bins]

        # 计算概率分布
        probs = histograms / (histograms.sum(dim=1, keepdim=True) + 1e-8)

        # 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [batch_size]
        mean_entropy = torch.mean(entropy)

        # 熵损失：鼓励熵接近目标值
        entropy_loss = torch.abs(mean_entropy - self.ks_entropy_target)

        return entropy_loss


class CHiLAPModel(nn.Module):
    """
    完整的C-HiLAP说话人识别模型
    集成混沌特征提取、TDNN backbone、C-HiLAP核心、分类头
    """

    def __init__(self,
                 num_speakers: int,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 embedding_dim: int = 256,
                 hidden_dims: List[int] = [512, 512, 512, 1024, 3000],
                 chaos_config: Optional[Dict] = None):
        """
        初始化完整的C-HiLAP模型

        Args:
            num_speakers: 说话人数量
            sample_rate: 音频采样率
            n_mels: Mel频谱特征数
            embedding_dim: 说话人嵌入维度
            hidden_dims: TDNN各层维度
            chaos_config: 混沌模块配置
        """
        super(CHiLAPModel, self).__init__()

        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim

        # 默认混沌配置
        if chaos_config is None:
            chaos_config = {
                'chaos_dim': 256,
                'num_chaos_layers': 3,
                'num_attention_heads': 8,
                'num_attractors': 4,
                'kolmogorov_entropy': 0.7
            }
        self.chaos_config = chaos_config

        # 混沌特征提取器
        self.chaos_feature_extractor = ChaosFeatureExtractor(
            sample_rate=sample_rate,
            embedding_dim=3,
            delay=10
        )

        # TDNN Backbone
        self.tdnn_layers = nn.ModuleList()

        # Frame-level layers
        input_dim = n_mels
        for i, output_dim in enumerate(hidden_dims[:-2]):
            context = 5 if i == 0 else 3
            dilation = 1 if i < 2 else 2 ** (i - 1)

            self.tdnn_layers.append(
                TDNN(input_dim, output_dim, context_size=context, dilation=dilation)
            )
            input_dim = output_dim

        # Segment-level layers
        stats_dim = input_dim * 2  # 统计池化后的维度

        self.tdnn_layers.append(
            nn.Sequential(
                nn.Linear(stats_dim, hidden_dims[-2]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[-2])
            )
        )

        self.tdnn_layers.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-2], hidden_dims[-1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[-1])
            )
        )

        # 统计池化
        self.stats_pooling = StatisticsPooling(hidden_dims[2])  # 在第3个TDNN层后应用

        # C-HiLAP混沌核心模块
        self.chaos_core = CHiLAPCore(
            input_dim=hidden_dims[-1],
            **chaos_config
        )

        # 特征融合层
        traditional_feat_dim = hidden_dims[-1]
        chaos_feat_dim = chaos_config['chaos_dim']

        # 混沌特征处理
        self.chaos_feat_processor = nn.Sequential(
            nn.Linear(8 + 6 + 4 + 15, 64),  # MLE(8) + RQA(6) + Fractal(4) + PhaseSpace(15)
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # 特征融合
        total_feat_dim = traditional_feat_dim + chaos_feat_dim + 32
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feat_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # 说话人嵌入层
        self.embedding_layer = nn.Linear(embedding_dim, embedding_dim)

        # 分类头
        self.classifier = nn.Linear(embedding_dim, num_speakers)

        # 损失权重
        self.register_buffer('chaos_loss_weight', torch.tensor(0.1))
        self.register_buffer('sync_loss_weight', torch.tensor(0.05))
        self.register_buffer('reg_loss_weight', torch.tensor(0.02))

    def forward(self, audio: torch.Tensor, return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            audio: 音频输入 [batch_size, 1, samples] 或预提取的特征 [batch_size, seq_len, n_mels]
            return_embedding: 是否返回嵌入向量

        Returns:
            包含logits、嵌入、损失等的字典
        """
        output_dict = {}

        # 特征提取
        if audio.dim() == 3 and audio.shape[1] == 1:
            # 原始音频输入
            # 提取传统特征
            mel_features = self._extract_mel_features(audio)

            # 提取混沌特征
            chaos_features_dict = self.chaos_feature_extractor(audio)
            chaos_features_raw = torch.cat([
                chaos_features_dict['mle'].flatten(1),
                chaos_features_dict['rqa'].flatten(1),
                chaos_features_dict['fractal'].flatten(1),
                chaos_features_dict['phase_space'].flatten(1)
            ], dim=1)
        else:
            # 预提取的Mel特征输入
            mel_features = audio
            # 对于预提取特征，使用零填充作为混沌特征的placeholder
            batch_size = audio.shape[0]
            chaos_features_raw = torch.zeros(batch_size, 33, device=audio.device)

        # TDNN特征提取
        x = mel_features
        for i, tdnn_layer in enumerate(self.tdnn_layers[:-2]):
            x = tdnn_layer(x)

        # 统计池化
        pooled_features = self.stats_pooling(x)  # [batch_size, hidden_dim * 2]

        # Segment-level TDNN层
        traditional_features = pooled_features
        for segment_layer in self.tdnn_layers[-2:]:
            traditional_features = segment_layer(traditional_features)

        # C-HiLAP混沌处理
        # 将segment-level特征扩展为序列用于混沌处理
        seq_features = traditional_features.unsqueeze(1).repeat(1, 50, 1)  # [batch_size, 50, hidden_dim]
        chaos_output, loss_dict = self.chaos_core(seq_features)

        # 处理原始混沌特征
        processed_chaos_features = self.chaos_feat_processor(chaos_features_raw)

        # 特征融合
        fused_features = torch.cat([
            traditional_features,  # 传统TDNN特征
            chaos_output,  # C-HiLAP混沌特征
            processed_chaos_features  # 原始混沌特征
        ], dim=1)

        # 最终嵌入
        embedding = self.feature_fusion(fused_features)
        embedding = self.embedding_layer(embedding)

        # 分类
        logits = self.classifier(embedding)

        # 计算总损失
        total_chaos_loss = (
                self.chaos_loss_weight * loss_dict.get('ks_entropy_loss', torch.tensor(0.0)) +
                self.sync_loss_weight * loss_dict.get('phase_sync_loss', torch.tensor(0.0)) +
                self.reg_loss_weight * loss_dict.get('chaos_regularization_loss', torch.tensor(0.0))
        )

        # 输出结果
        output_dict['logits'] = logits
        output_dict['embedding'] = embedding if return_embedding else None
        output_dict['chaos_loss'] = total_chaos_loss
        output_dict['loss_components'] = loss_dict

        return output_dict

    def _extract_mel_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        从原始音频提取Mel频谱特征
        """
        batch_size = audio.shape[0]
        mel_features = []

        for i in range(batch_size):
            # 转换为numpy进行librosa处理
            audio_np = audio[i, 0].cpu().numpy()

            # 提取Mel频谱
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=256,
                n_fft=512
            )

            # 转换为对数刻度
            log_mel = librosa.power_to_db(mel_spec)
            mel_features.append(torch.from_numpy(log_mel.T).float())

        # 填充到相同长度
        max_len = max(feat.shape[0] for feat in mel_features)
        padded_features = []

        for feat in mel_features:
            if feat.shape[0] < max_len:
                padding = torch.zeros(max_len - feat.shape[0], feat.shape[1])
                feat = torch.cat([feat, padding], dim=0)
            padded_features.append(feat)

        return torch.stack(padded_features, dim=0).to(audio.device)

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取说话人嵌入向量

        Args:
            audio: 音频输入

        Returns:
            embedding: 说话人嵌入向量 [batch_size, embedding_dim]
        """
        with torch.no_grad():
            output = self.forward(audio, return_embedding=True)
            return output['embedding']

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        计算两个嵌入向量之间的余弦相似度
        """
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        return torch.sum(emb1_norm * emb2_norm, dim=1)

    def get_chaos_statistics(self) -> Dict[str, float]:
        """
        获取模型的混沌系统统计信息
        """
        stats = {}

        # 获取洛伦兹系统参数
        for i, chaos_layer in enumerate(self.chaos_core.chaos_layers):
            stats[f'lorenz_sigma_layer_{i}'] = chaos_layer.sigma.item()
            stats[f'lorenz_rho_layer_{i}'] = chaos_layer.rho.item()
            stats[f'lorenz_beta_layer_{i}'] = chaos_layer.beta.item()
            stats[f'chaos_strength_layer_{i}'] = chaos_layer.chaos_strength.item()

        # 获取注意力机制参数
        stats['bifurcation_param'] = self.chaos_core.chaotic_attention.bifurcation_param.item()
        stats['phase_coupling'] = self.chaos_core.chaotic_attention.phase_coupling.item()

        # 获取目标熵
        stats['ks_entropy_target'] = self.chaos_core.ks_entropy_target.item()

        return stats


# 模型构建辅助函数
def create_chilap_model(num_speakers: int,
                        model_config: Optional[Dict] = None) -> CHiLAPModel:
    """
    创建C-HiLAP模型的便捷函数

    Args:
        num_speakers: 说话人数量
        model_config: 模型配置字典

    Returns:
        CHiLAPModel实例
    """
    if model_config is None:
        model_config = {
            'sample_rate': 16000,
            'n_mels': 80,
            'embedding_dim': 256,
            'hidden_dims': [512, 512, 512, 1024, 3000],
            'chaos_config': {
                'chaos_dim': 256,
                'num_chaos_layers': 3,
                'num_attention_heads': 8,
                'num_attractors': 4,
                'kolmogorov_entropy': 0.7
            }
        }

    return CHiLAPModel(num_speakers=num_speakers, **model_config)


# 测试和使用示例
if __name__ == "__main__":
    # 模型配置
    num_speakers = 1000
    batch_size = 4
    seq_len = 300
    n_mels = 80

    print("开始C-HiLAP模型测试...")

    # 创建模型
    model = create_chilap_model(num_speakers)

    # 测试输入（预提取的Mel特征）
    test_input = torch.randn(batch_size, seq_len, n_mels)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播测试
    print("\n执行前向传播...")
    output = model(test_input, return_embedding=True)

    print(f"Logits形状: {output['logits'].shape}")
    print(f"嵌入向量形状: {output['embedding'].shape}")
    print(f"混沌损失: {output['chaos_loss'].item():.6f}")

    # 嵌入提取测试
    print("\n测试嵌入提取...")
    embeddings = model.extract_embedding(test_input)
    print(f"提取的嵌入形状: {embeddings.shape}")

    # 相似度计算测试
    emb1 = embeddings[:2]
    emb2 = embeddings[2:4]
    similarity = model.compute_similarity(emb1, emb2)
    print(f"嵌入相似度: {similarity}")

    # 混沌统计信息
    print("\n混沌系统统计:")
    chaos_stats = model.get_chaos_statistics()
    for key, value in chaos_stats.items():
        print(f"  {key}: {value:.4f}")

    print("\nC-HiLAP模型测试完成！")