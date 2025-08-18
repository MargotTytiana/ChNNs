import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# 导入基础模型组件
from ecapa_tdnn import ECAPA_TDNN, SEModule, AttentiveStatsPool, AAMSoftmax
from xvector import XVector


class FeatureFusionModule(nn.Module):
    """
    特征融合模块，用于融合传统声学特征和混沌特征。

    支持三种融合策略：
    1. 简单拼接（concat）
    2. 注意力加权融合（attention）
    3. 门控融合（gating）

    Args:
        acoustic_dim: 传统声学特征的维度
        chaotic_dim: 混沌特征的维度
        output_dim: 输出特征的维度
        fusion_type: 融合策略类型
        bottleneck_dim: 注意力或门控机制的瓶颈维度
    """

    def __init__(
            self,
            acoustic_dim: int,
            chaotic_dim: int,
            output_dim: int,
            fusion_type: str = "attention",
            bottleneck_dim: int = 128
    ):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.chaotic_dim = chaotic_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        # 特征投影层，将不同维度的特征映射到相同的空间
        self.acoustic_proj = nn.Linear(acoustic_dim, output_dim)
        self.chaotic_proj = nn.Linear(chaotic_dim, output_dim)

        if fusion_type == "concat":
            # 简单拼接后的线性投影
            self.fusion_proj = nn.Linear(output_dim * 2, output_dim)
            self.bn = nn.BatchNorm1d(output_dim)

        elif fusion_type == "attention":
            # 注意力加权融合
            self.query_proj = nn.Linear(output_dim, bottleneck_dim)
            self.key_proj = nn.Linear(output_dim, bottleneck_dim)
            self.value_proj = nn.Linear(output_dim, output_dim)
            self.attention_proj = nn.Linear(bottleneck_dim, 1)
            self.bn = nn.BatchNorm1d(output_dim)

        elif fusion_type == "gating":
            # 门控融合
            self.gate_net = nn.Sequential(
                nn.Linear(output_dim * 2, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, output_dim),
                nn.Sigmoid()
            )
            self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
            self,
            acoustic_features: torch.Tensor,
            chaotic_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播函数

        Args:
            acoustic_features: 传统声学特征，形状为 (batch_size, seq_len, acoustic_dim)
            chaotic_features: 混沌特征，形状为 (batch_size, seq_len, chaotic_dim)
                              或 (batch_size, chaotic_dim)

        Returns:
            融合后的特征，形状为 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = acoustic_features.shape

        # 处理混沌特征的维度
        if len(chaotic_features.shape) == 2:
            # 如果混沌特征是全局特征，扩展为序列
            chaotic_features = chaotic_features.unsqueeze(1).expand(-1, seq_len, -1)

        # 投影到相同的特征空间
        acoustic_proj = self.acoustic_proj(acoustic_features)
        chaotic_proj = self.chaotic_proj(chaotic_features)

        if self.fusion_type == "concat":
            # 简单拼接
            concat_features = torch.cat([acoustic_proj, chaotic_proj], dim=2)
            fused = self.fusion_proj(concat_features)
            # 应用批归一化（需要调整维度）
            fused = fused.transpose(1, 2)  # (batch_size, output_dim, seq_len)
            fused = self.bn(fused)
            fused = fused.transpose(1, 2)  # (batch_size, seq_len, output_dim)

        elif self.fusion_type == "attention":
            # 注意力加权融合
            # 计算查询和键
            query = self.query_proj(acoustic_proj)
            key = self.key_proj(chaotic_proj)

            # 计算注意力分数
            energy = torch.tanh(query + key)
            attention = self.attention_proj(energy)
            attention_weights = F.softmax(attention, dim=1)

            # 计算值
            value_a = self.value_proj(acoustic_proj)
            value_c = self.value_proj(chaotic_proj)

            # 加权融合
            fused = value_a + attention_weights * value_c

            # 应用批归一化
            fused = fused.transpose(1, 2)  # (batch_size, output_dim, seq_len)
            fused = self.bn(fused)
            fused = fused.transpose(1, 2)  # (batch_size, seq_len, output_dim)

        elif self.fusion_type == "gating":
            # 门控融合
            concat_features = torch.cat([acoustic_proj, chaotic_proj], dim=2)
            gate = self.gate_net(concat_features)

            # 应用门控
            fused = acoustic_proj * gate + chaotic_proj * (1 - gate)

            # 应用批归一化
            fused = fused.transpose(1, 2)  # (batch_size, output_dim, seq_len)
            fused = self.bn(fused)
            fused = fused.transpose(1, 2)  # (batch_size, seq_len, output_dim)

        return fused


class ChaosEnhancedModel_VariantA(nn.Module):
    """
    混沌特征增强型网络（变体A）

    将混沌特征与传统声学特征融合，然后输入到ECAPA-TDNN骨干网络中

    Args:
        acoustic_dim: 传统声学特征的维度
        chaotic_dim: 混沌特征的维度
        backbone_type: 骨干网络类型，'ecapa_tdnn'或'xvector'
        fusion_type: 特征融合策略
        channels: ECAPA-TDNN的通道数
        emb_dim: 最终嵌入向量的维度
        num_classes: 说话人类别数量（0表示仅提取嵌入）
    """

    def __init__(
            self,
            acoustic_dim: int = 80,
            chaotic_dim: int = 12,
            backbone_type: str = "ecapa_tdnn",
            fusion_type: str = "attention",
            channels: int = 512,
            emb_dim: int = 192,
            num_classes: int = 0
    ):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.chaotic_dim = chaotic_dim
        self.backbone_type = backbone_type
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        # 特征融合模块
        self.fusion_module = FeatureFusionModule(
            acoustic_dim=acoustic_dim,
            chaotic_dim=chaotic_dim,
            output_dim=acoustic_dim,  # 保持与原始声学特征相同的维度
            fusion_type=fusion_type
        )

        # 骨干网络
        if backbone_type == "ecapa_tdnn":
            self.backbone = ECAPA_TDNN(
                input_dim=acoustic_dim,
                channels=channels,
                emb_dim=emb_dim
            )
        elif backbone_type == "xvector":
            self.backbone = XVector(
                input_dim=acoustic_dim,
                num_classes=0,  # 不使用XVector内部的分类器
                emb_dim=emb_dim
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # 分类层（可选）
        if num_classes > 0:
            self.classifier = nn.Linear(emb_dim, num_classes)
        else:
            self.classifier = None

    def forward(
            self,
            acoustic_features: torch.Tensor,
            chaotic_features: torch.Tensor,
            extract_embedding: bool = False
    ) -> torch.Tensor:
        """
        前向传播函数

        Args:
            acoustic_features: 传统声学特征，形状为 (batch_size, seq_len, acoustic_dim)
            chaotic_features: 混沌特征，形状为 (batch_size, seq_len, chaotic_dim)
                              或 (batch_size, chaotic_dim)
            extract_embedding: 是否仅提取嵌入向量

        Returns:
            如果extract_embedding为True，返回嵌入向量
            否则，如果有分类器，返回分类结果；否则返回嵌入向量
        """
        # 融合特征
        fused_features = self.fusion_module(acoustic_features, chaotic_features)

        # 通过骨干网络
        embeddings = self.backbone(fused_features, extract_embedding=True)

        # 返回嵌入向量或分类结果
        if extract_embedding:
            return embeddings

        if self.classifier is not None:
            return self.classifier(embeddings)
        else:
            return embeddings


class MLEFeatureExtractor(nn.Module):
    """
    最大李雅普诺夫指数特征提取器

    将原始特征序列转换为最大李雅普诺夫指数特征
    这是一个可学习的近似实现，用于端到端训练

    Args:
        input_dim: 输入特征维度
        output_dim: 输出MLE特征维度
        window_size: 计算MLE的窗口大小
        embedding_dim: 相空间重构的嵌入维度
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            window_size: int = 20,
            embedding_dim: int = 5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        # 相空间重构参数
        self.delay = nn.Parameter(torch.tensor(2.0))

        # 特征变换网络
        self.transform = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # MLE估计网络
        self.mle_estimator = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_dim),
            nn.Tanh()  # 限制输出范围
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x: 输入特征，形状为 (batch_size, seq_len, input_dim)

        Returns:
            MLE特征，形状为 (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 特征变换
        transformed = self.transform(x)  # (batch_size, seq_len, embedding_dim)

        # 计算每个批次的MLE特征
        mle_features = []

        for b in range(batch_size):
            # 获取当前批次的变换特征
            curr_features = transformed[b]  # (seq_len, embedding_dim)

            # 如果序列长度不足，填充
            if seq_len < self.window_size:
                padding = self.window_size - seq_len
                curr_features = F.pad(curr_features, (0, 0, 0, padding))

            # 计算相空间轨迹的差异
            delay = int(torch.clamp(self.delay, 1, 10))

            # 创建延迟坐标
            trajectory1 = curr_features[:-delay]  # (seq_len-delay, embedding_dim)
            trajectory2 = curr_features[delay:]  # (seq_len-delay, embedding_dim)

            # 计算轨迹差异
            diff = torch.norm(trajectory2 - trajectory1, dim=1)  # (seq_len-delay)

            # 取对数
            log_diff = torch.log(diff + 1e-6)  # 添加小常数避免log(0)

            # 计算平均发散率
            divergence_rate = torch.mean(log_diff)

            # 创建特征向量
            feature_vec = torch.cat([
                torch.tensor([divergence_rate], device=x.device),
                torch.std(log_diff).unsqueeze(0)
            ])

            # 扩展特征向量
            mle_feature = self.mle_estimator(feature_vec.repeat(self.output_dim, 1))
            mle_features.append(mle_feature)

        # 堆叠所有批次的MLE特征
        return torch.stack(mle_features)


class RQAFeatureExtractor(nn.Module):
    """
    递归量化分析（RQA）特征提取器

    将原始特征序列转换为RQA特征
    这是一个可学习的近似实现，用于端到端训练

    Args:
        input_dim: 输入特征维度
        output_dim: 输出RQA特征维度
        embedding_dim: 相空间重构的嵌入维度
        threshold: 递归图的阈值
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            embedding_dim: int = 3,
            threshold: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        # 可学习的阈值参数
        self.threshold = nn.Parameter(torch.tensor(threshold))

        # 特征变换网络
        self.transform = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # RQA特征提取网络
        self.rqa_extractor = nn.Sequential(
            nn.Linear(4, output_dim),  # 4个基本RQA指标
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x: 输入特征，形状为 (batch_size, seq_len, input_dim)

        Returns:
            RQA特征，形状为 (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 特征变换
        transformed = self.transform(x)  # (batch_size, seq_len, embedding_dim)

        # 计算每个批次的RQA特征
        rqa_features = []

        for b in range(batch_size):
            # 获取当前批次的变换特征
            curr_features = transformed[b]  # (seq_len, embedding_dim)

            # 计算距离矩阵
            distances = torch.cdist(curr_features, curr_features)

            # 创建递归图（二值矩阵）
            threshold = torch.sigmoid(self.threshold) * torch.max(distances)
            recurrence_matrix = (distances < threshold).float()

            # 计算RQA指标

            # 1. 递归率 (RR)
            rr = torch.mean(recurrence_matrix)

            # 2. 确定性 (DET)：对角线结构的比例
            diag_structures = 0
            for i in range(1, min(seq_len, 10)):  # 检查对角线
                diag = torch.diagonal(recurrence_matrix, offset=i)
                diag_structures += torch.sum(diag)

            det = diag_structures / (torch.sum(recurrence_matrix) + 1e-6)

            # 3. 层流性 (LAM)：垂直线结构的比例
            vert_structures = 0
            for i in range(seq_len):
                col = recurrence_matrix[:, i]
                runs = torch.diff(
                    torch.cat([torch.tensor([0.0], device=x.device), col, torch.tensor([0.0], device=x.device)]))
                starts = torch.where(runs > 0)[0]
                ends = torch.where(runs < 0)[0]
                lengths = ends - starts
                vert_structures += torch.sum((lengths >= 2).float())

            lam = vert_structures / (torch.sum(recurrence_matrix) + 1e-6)

            # 4. 熵 (ENTR)：对角线长度分布的香农熵
            entr = -torch.sum(recurrence_matrix * torch.log(recurrence_matrix + 1e-6))

            # 创建RQA特征向量
            rqa_metrics = torch.tensor([rr, det, lam, entr], device=x.device)

            # 扩展RQA特征
            rqa_feature = self.rqa_extractor(rqa_metrics.unsqueeze(0))
            rqa_features.append(rqa_feature.squeeze(0))

        # 堆叠所有批次的RQA特征
        return torch.stack(rqa_features)


class ChaoticFeatureExtractor(nn.Module):
    """
    混沌特征提取器

    结合MLE和RQA特征提取器，生成完整的混沌特征

    Args:
        input_dim: 输入特征维度
        output_dim: 输出混沌特征维度
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 分配输出维度
        mle_dim = output_dim // 3
        rqa_dim = output_dim - mle_dim

        # MLE特征提取器
        self.mle_extractor = MLEFeatureExtractor(
            input_dim=input_dim,
            output_dim=mle_dim
        )

        # RQA特征提取器
        self.rqa_extractor = RQAFeatureExtractor(
            input_dim=input_dim,
            output_dim=rqa_dim
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(mle_dim + rqa_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x: 输入特征，形状为 (batch_size, seq_len, input_dim)

        Returns:
            混沌特征，形状为 (batch_size, output_dim)
        """
        # 提取MLE特征
        mle_features = self.mle_extractor(x)

        # 提取RQA特征
        rqa_features = self.rqa_extractor(x)

        # 融合特征
        combined = torch.cat([mle_features, rqa_features], dim=1)
        chaotic_features = self.fusion_layer(combined)

        return chaotic_features


class ChaosEnhancedModelWrapper(nn.Module):
    """
    混沌增强模型包装器

    整合特征提取和模型推理的完整流程

    Args:
        model: 混沌增强模型
        chaotic_extractor: 混沌特征提取器
    """

    def __init__(
            self,
            model: ChaosEnhancedModel_VariantA,
            chaotic_extractor: ChaoticFeatureExtractor
    ):
        super().__init__()
        self.model = model
        self.chaotic_extractor = chaotic_extractor

    def forward(
            self,
            acoustic_features: torch.Tensor,
            extract_embedding: bool = False
    ) -> torch.Tensor:
        """
        前向传播函数

        Args:
            acoustic_features: 声学特征，形状为 (batch_size, seq_len, acoustic_dim)
            extract_embedding: 是否仅提取嵌入向量

        Returns:
            模型输出（嵌入向量或分类结果）
        """
        # 提取混沌特征
        chaotic_features = self.chaotic_extractor(acoustic_features)

        # 通过模型
        output = self.model(
            acoustic_features=acoustic_features,
            chaotic_features=chaotic_features,
            extract_embedding=extract_embedding
        )

        return output


def count_parameters(model: nn.Module) -> int:
    """
    计算模型的参数量

    Args:
        model: PyTorch模型

    Returns:
        可训练参数的数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 示例用法
if __name__ == "__main__":
    # 创建随机输入张量
    batch_size = 2
    seq_len = 200
    acoustic_dim = 80
    chaotic_dim = 12

    acoustic_features = torch.randn(batch_size, seq_len, acoustic_dim)

    # 创建混沌特征提取器
    chaotic_extractor = ChaoticFeatureExtractor(
        input_dim=acoustic_dim,
        output_dim=chaotic_dim
    )

    # 提取混沌特征
    chaotic_features = chaotic_extractor(acoustic_features)
    print(f"混沌特征形状: {chaotic_features.shape}")

    # 创建模型
    model = ChaosEnhancedModel_VariantA(
        acoustic_dim=acoustic_dim,
        chaotic_dim=chaotic_dim,
        backbone_type="ecapa_tdnn",
        fusion_type="attention",
        channels=512,
        emb_dim=192,
        num_classes=10
    )

    # 打印模型参数量
    print(f"模型参数量: {count_parameters(model):,}")

    # 前向传播
    output = model(acoustic_features, chaotic_features)
    print(f"输出形状: {output.shape}")

    # 提取嵌入向量
    embeddings = model(acoustic_features, chaotic_features, extract_embedding=True)
    print(f"嵌入向量形状: {embeddings.shape}")

    # 创建模型包装器
    model_wrapper = ChaosEnhancedModelWrapper(model, chaotic_extractor)

    # 使用包装器进行前向传播
    output = model_wrapper(acoustic_features)
    print(f"包装器输出形状: {output.shape}")

    # 使用包装器提取嵌入向量
    embeddings = model_wrapper(acoustic_features, extract_embedding=True)
    print(f"包装器嵌入向量形状: {embeddings.shape}")