import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union


class ClassifierHead(nn.Module):
    """
    说话人分类器头，将嵌入向量映射到类别空间。

    Args:
        embed_dim: 嵌入向量维度
        num_classes: 类别数量
        use_batchnorm: 是否使用批归一化
    """

    def __init__(
            self,
            embed_dim: int,
            num_classes: int,
            use_batchnorm: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm

        # 分类器层
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

        # 批归一化层（可选）
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            embeddings: 嵌入向量，形状为 (batch_size, embed_dim)

        Returns:
            类别logits，形状为 (batch_size, num_classes)
        """
        if self.use_batchnorm:
            embeddings = self.bn(embeddings)

        logits = self.classifier(embeddings)
        return logits


class CrossEntropyLoss(nn.Module):
    """
    标准交叉熵损失的包装器。

    Args:
        weight: 各类别的权重
        reduction: 损失缩减方式
    """

    def __init__(
            self,
            weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean'
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算交叉熵损失。

        Args:
            logits: 预测的类别logits，形状为 (batch_size, num_classes)
            labels: 真实标签，形状为 (batch_size,)

        Returns:
            交叉熵损失值
        """
        return self.criterion(logits, labels)


class AMSoftmax(nn.Module):
    """
    加性余弦边界Softmax损失(Additive Margin Softmax Loss)。

    该损失函数在余弦相似度空间中添加一个边界，增强类间区分性和类内紧凑性。

    Args:
        embed_dim: 嵌入向量维度
        num_classes: 类别数量
        margin: 余弦边界值
        scale: 缩放因子
    """

    def __init__(
            self,
            embed_dim: int,
            num_classes: int,
            margin: float = 0.35,
            scale: float = 30.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # 类别权重，初始化为随机值
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算AM-Softmax损失。

        Args:
            embeddings: 嵌入向量，形状为 (batch_size, embed_dim)
            labels: 真实标签，形状为 (batch_size,)

        Returns:
            AM-Softmax损失值
        """
        # 归一化嵌入向量和权重
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cosine = F.linear(embeddings_norm, weight_norm)

        # 为目标类别添加边界
        phi = cosine - self.margin

        # 创建one-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # 应用边界
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 应用缩放
        output = output * self.scale

        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)

        return loss


class AAMSoftmax(nn.Module):
    """
    加性角度边界Softmax损失(Additive Angular Margin Softmax Loss)，也称为ArcFace。

    该损失函数在角度空间中添加一个边界，增强类间区分性和类内紧凑性。

    Args:
        embed_dim: 嵌入向量维度
        num_classes: 类别数量
        margin: 角度边界值（弧度）
        scale: 缩放因子
    """

    def __init__(
            self,
            embed_dim: int,
            num_classes: int,
            margin: float = 0.5,
            scale: float = 30.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # 类别权重，初始化为随机值
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算AAM-Softmax损失。

        Args:
            embeddings: 嵌入向量，形状为 (batch_size, embed_dim)
            labels: 真实标签，形状为 (batch_size,)

        Returns:
            AAM-Softmax损失值
        """
        # 归一化嵌入向量和权重
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cosine = F.linear(embeddings_norm, weight_norm)

        # 将余弦值裁剪到有效范围
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 计算角度
        theta = torch.acos(cosine)

        # 为目标类别添加角度边界
        marginal_cosine = torch.cos(theta + self.margin)

        # 创建one-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # 应用边界
        output = (one_hot * marginal_cosine) + ((1.0 - one_hot) * cosine)

        # 应用缩放
        output = output * self.scale

        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数，结合分类损失和混沌正则化损失。

    Args:
        classifier: 分类器头
        classification_loss: 分类损失函数
        chaos_regularization: 混沌正则化损失函数（可选）
        chaos_weight: 混沌正则化损失的权重
    """

    def __init__(
            self,
            classifier: ClassifierHead,
            classification_loss: nn.Module,
            chaos_regularization: Optional[nn.Module] = None,
            chaos_weight: float = 0.1
    ):
        super().__init__()
        self.classifier = classifier
        self.classification_loss = classification_loss
        self.chaos_regularization = chaos_regularization
        self.chaos_weight = chaos_weight

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            chaos_trajectory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算组合损失。

        Args:
            embeddings: 嵌入向量，形状为 (batch_size, embed_dim)
            labels: 真实标签，形状为 (batch_size,)
            chaos_trajectory: 混沌轨迹，用于正则化（可选）

        Returns:
            总损失值和包含各分项损失的字典
        """
        # 计算类别logits
        logits = self.classifier(embeddings)

        # 计算分类损失
        cls_loss = self.classification_loss(embeddings, labels)

        # 初始化损失字典
        loss_dict = {'classification_loss': cls_loss}

        # 计算总损失
        total_loss = cls_loss

        # 如果提供了混沌轨迹和正则化函数，计算混沌正则化损失
        if self.chaos_regularization is not None and chaos_trajectory is not None:
            chaos_loss = self.chaos_regularization(chaos_trajectory)
            loss_dict['chaos_loss'] = chaos_loss
            total_loss = total_loss + self.chaos_weight * chaos_loss

        return total_loss, loss_dict


class PhaseSynchronizationLoss(nn.Module):
    """
    相位同步损失，用于混沌神经网络的正则化。

    Args:
        weight: 损失权重
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def extract_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """
        使用解析信号方法提取瞬时相位。

        Args:
            signal: 形状为 (batch_size, seq_len, dim) 的输入信号

        Returns:
            形状为 (batch_size, seq_len, dim) 的相位张量
        """
        # 实现使用简化方法，因为Hilbert变换在PyTorch中不直接可用

        batch_size, seq_len, dim = signal.shape
        phases = torch.zeros_like(signal)

        for b in range(batch_size):
            for d in range(dim):
                # 转换为numpy进行处理
                sig = signal[b, :, d].cpu().detach().numpy()

                # 计算解析信号（简化）
                sig_mean = np.mean(sig)
                sig_centered = sig - sig_mean

                # 找到过零点作为相位参考点
                zero_crossings = np.where(np.diff(np.signbit(sig_centered)))[0]

                if len(zero_crossings) >= 2:
                    # 估计局部频率
                    avg_period = (zero_crossings[-1] - zero_crossings[0]) / (len(zero_crossings) - 1)

                    # 基于过零点生成相位
                    phase = np.zeros(seq_len)
                    last_crossing = 0
                    phase_val = 0

                    for zc in zero_crossings:
                        phase[last_crossing:zc + 1] = np.linspace(
                            phase_val, phase_val + np.pi, zc - last_crossing + 1
                        )
                        last_crossing = zc + 1
                        phase_val += np.pi

                    # 填充剩余值
                    if last_crossing < seq_len:
                        remaining = seq_len - last_crossing
                        phase[last_crossing:] = np.linspace(
                            phase_val, phase_val + np.pi * remaining / avg_period, remaining
                        )

                    # 包裹到 [-π, π]
                    phase = (phase + np.pi) % (2 * np.pi) - np.pi

                    # 转换回张量
                    phases[b, :, d] = torch.tensor(phase, device=signal.device)

        return phases

    def forward(self, input_signal: torch.Tensor, attractor_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算相位同步损失。

        Args:
            input_signal: 输入信号
            attractor_signal: 吸引子信号（可选，如果不提供则使用input_signal）

        Returns:
            同步损失
        """
        # 如果没有提供吸引子信号，使用输入信号
        if attractor_signal is None:
            attractor_signal = input_signal

        # 提取相位
        input_phase = self.extract_phase(input_signal)
        attractor_phase = self.extract_phase(attractor_signal)

        # 计算相位差
        phase_diff = input_phase - attractor_phase

        # 计算同步指数 (1 - R)
        # 其中R是相位锁定值
        sync_loss = 1.0 - torch.abs(torch.mean(torch.exp(1j * phase_diff.float())))

        return self.weight * sync_loss


class LyapunovRegularizationLoss(nn.Module):
    """
    基于李雅普诺夫稳定性理论的正则化损失。

    Args:
        target_exponent: 目标李雅普诺夫指数
        weight: 正则化项权重
        mode: 'positive'表示混沌，'negative'表示稳定，'target'表示特定值
    """

    def __init__(
            self,
            target_exponent: float = 0.1,
            weight: float = 0.1,
            mode: str = "positive"
    ):
        super().__init__()
        self.target_exponent = target_exponent
        self.weight = weight
        self.mode = mode

    def estimate_lyapunov(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        从轨迹估计最大李雅普诺夫指数。

        Args:
            trajectory: 形状为 (batch_size, seq_len, dim) 的轨迹

        Returns:
            形状为 (batch_size,) 的李雅普诺夫指数
        """
        batch_size, seq_len, dim = trajectory.shape

        # 需要足够长的序列进行估计
        if seq_len < 10:
            return torch.zeros(batch_size, device=trajectory.device)

        # 计算相邻轨迹的发散
        divergence = torch.zeros(batch_size, device=trajectory.device)

        for b in range(batch_size):
            # 使用前半部分作为参考
            ref_points = trajectory[b, :seq_len // 2]

            # 计算连续时间步之间的距离
            diffs = torch.norm(ref_points[1:] - ref_points[:-1], dim=1)

            # 估计指数发散率
            if torch.any(diffs > 0):
                log_diffs = torch.log(diffs + 1e-10)
                divergence[b] = torch.mean(log_diffs)

        return divergence

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算李雅普诺夫正则化损失。

        Args:
            trajectory: 形状为 (batch_size, seq_len, dim) 的轨迹

        Returns:
            正则化损失
        """
        # 估计李雅普诺夫指数
        lyapunov = self.estimate_lyapunov(trajectory)

        if self.mode == "positive":
            # 鼓励混沌动力学（正指数）
            loss = torch.mean(torch.relu(-lyapunov))
        elif self.mode == "negative":
            # 鼓励稳定动力学（负指数）
            loss = torch.mean(torch.relu(lyapunov))
        else:  # "target"
            # 鼓励特定指数值
            loss = torch.mean((lyapunov - self.target_exponent) ** 2)

        return self.weight * loss


class SubCenterArcFace(nn.Module):
    """
    带有子中心的加性角度边界Softmax损失(Sub-center ArcFace)。

    该损失函数为每个类别维护多个子中心，有助于处理类内变异性。

    Args:
        embed_dim: 嵌入向量维度
        num_classes: 类别数量
        k: 每个类别的子中心数量
        margin: 角度边界值（弧度）
        scale: 缩放因子
    """

    def __init__(
            self,
            embed_dim: int,
            num_classes: int,
            k: int = 3,
            margin: float = 0.5,
            scale: float = 30.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.k = k
        self.margin = margin
        self.scale = scale

        # 类别权重，每个类别有k个子中心
        self.weight = nn.Parameter(torch.randn(num_classes * k, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Sub-center ArcFace损失。

        Args:
            embeddings: 嵌入向量，形状为 (batch_size, embed_dim)
            labels: 真实标签，形状为 (batch_size,)

        Returns:
            Sub-center ArcFace损失值
        """
        # 归一化嵌入向量和权重
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 重塑权重以便于处理子中心
        weight_norm = weight_norm.view(self.num_classes, self.k, self.embed_dim)

        # 计算每个样本与所有子中心的余弦相似度
        cosine_all = torch.zeros(embeddings.size(0), self.num_classes, self.k, device=embeddings.device)

        for i in range(self.num_classes):
            cosine_all[:, i] = F.linear(embeddings_norm, weight_norm[i])

        # 对于每个类别，选择最大的余弦相似度
        cosine, _ = torch.max(cosine_all, dim=2)

        # 将余弦值裁剪到有效范围
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 计算角度
        theta = torch.acos(cosine)

        # 为目标类别添加角度边界
        marginal_cosine = torch.cos(theta + self.margin)

        # 创建one-hot编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # 应用边界
        output = (one_hot * marginal_cosine) + ((1.0 - one_hot) * cosine)

        # 应用缩放
        output = output * self.scale

        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)

        return loss


# 示例用法
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 创建伪数据
    batch_size = 16
    embed_dim = 192
    num_classes = 10

    embeddings = torch.randn(batch_size, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 创建分类器头
    classifier = ClassifierHead(embed_dim, num_classes)

    # 创建各种损失函数
    ce_loss = CrossEntropyLoss()
    am_loss = AMSoftmax(embed_dim, num_classes)
    aam_loss = AAMSoftmax(embed_dim, num_classes)
    subcenter_loss = SubCenterArcFace(embed_dim, num_classes)

    # 创建混沌轨迹（用于演示）
    chaos_trajectory = torch.randn(batch_size, 50, 3)

    # 创建混沌正则化损失
    lyap_loss = LyapunovRegularizationLoss()
    phase_loss = PhaseSynchronizationLoss()

    # 创建组合损失
    combined_loss = CombinedLoss(
        classifier=classifier,
        classification_loss=aam_loss,
        chaos_regularization=lyap_loss,
        chaos_weight=0.1
    )

    # 计算各种损失
    logits = classifier(embeddings)
    ce = ce_loss(logits, labels)
    am = am_loss(embeddings, labels)
    aam = aam_loss(embeddings, labels)
    subcenter = subcenter_loss(embeddings, labels)
    lyap = lyap_loss(chaos_trajectory)
    phase = phase_loss(chaos_trajectory)

    # 计算组合损失
    total_loss, loss_dict = combined_loss(embeddings, labels, chaos_trajectory)

    # 打印结果
    print(f"交叉熵损失: {ce.item():.4f}")
    print(f"AM-Softmax损失: {am.item():.4f}")
    print(f"AAM-Softmax损失: {aam.item():.4f}")
    print(f"Sub-center ArcFace损失: {subcenter.item():.4f}")
    print(f"李雅普诺夫正则化损失: {lyap.item():.4f}")
    print(f"相位同步损失: {phase.item():.4f}")
    print(f"组合损失: {total_loss.item():.4f}")
    print(f"损失字典: {loss_dict}")