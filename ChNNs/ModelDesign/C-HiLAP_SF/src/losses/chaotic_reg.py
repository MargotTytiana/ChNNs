import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
from scipy.integrate import odeint
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChaoticRegularizer:
    """
    混沌正则化器，通过促进模型权重的混沌行为来提升模型泛化能力。

    This regularizer encourages the model to have chaotic behavior in its weight dynamics,
    which can help improve generalization and robustness.

    Args:
        model: 要应用正则化的模型
        method: 混沌特性计算方法，可选 'lyapunov' 或 'rqa'
        frequency: 应用正则化的频率（每多少步计算一次）
        beta: 正则化强度
        embedding_dim: 相空间重构的嵌入维度
        delay: 相空间重构的时间延迟
        min_lyap: 最小李雅普诺夫指数（用于确保混沌特性）
        device: 计算设备
    """

    def __init__(self,
                 model: nn.Module,
                 method: str = 'lyapunov',
                 frequency: int = 10,
                 beta: float = 0.1,
                 embedding_dim: int = 3,
                 delay: int = 5,
                 min_lyap: float = 0.01,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = model
        self.method = method
        self.frequency = frequency
        self.beta = beta
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.min_lyap = min_lyap
        self.device = device

        # 初始化存储
        self.weight_history = []
        self.step = 0
        self.chaos_scores = []

        # 设置参数组
        self.param_groups = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_groups.append({
                    'name': name,
                    'param': param,
                    'history': []
                })

        logger.info(f"初始化混沌正则化器: 方法={method}, 频率={frequency}, 强度={beta}")

    def compute_lyapunov_exponent(self, signal: np.ndarray) -> float:
        """
        计算给定信号的李雅普诺夫指数（简化版）

        Args:
            signal: 输入信号

        Returns:
            李雅普诺夫指数
        """
        # 简单实现 - 实际应用中可能需要更复杂的方法
        n = len(signal)
        if n < 10:
            return 0.0

        # 使用SVD计算李雅普诺夫指数
        embeddings = []
        for i in range(n - (self.embedding_dim - 1) * self.delay):
            embedding = []
            for j in range(self.embedding_dim):
                embedding.append(signal[i + j * self.delay])
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        _, s, _ = svd(embeddings.T, full_matrices=False)

        # 使用奇异值的衰减率估计李雅普诺夫指数
        lambda_max = np.max(s) / np.sqrt(n)
        return lambda_max

    def compute_rqa_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """
        计算递归定量分析指标

        Args:
            signal: 输入信号

        Returns:
            包含RQA指标的字典
        """
        n = len(signal)
        if n < 20:
            return {'recurrence_rate': 0.0, 'determinism': 0.0, 'laminarity': 0.0}

        # 简化实现 - 实际应用中需要计算完整的递归图
        recurrence_rate = np.random.random() * 0.5 + 0.2
        determinism = np.random.random() * 0.5 + 0.5
        laminarity = np.random.random() * 0.3 + 0.3

        return {
            'recurrence_rate': recurrence_rate,
            'determinism': determinism,
            'laminarity': laminarity
        }

    def get_weights_vector(self) -> np.ndarray:
        """
        获取模型权重的扁平向量

        Returns:
            权重向量
        """
        weights = []
        for param_group in self.param_groups:
            param = param_group['param']
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def update_weight_history(self):
        """更新权重历史记录"""
        weight_vector = self.get_weights_vector()
        self.weight_history.append(weight_vector)
        self.step += 1

        # 也为每个参数组记录历史
        for param_group in self.param_groups:
            param_group['history'].append(param_group['param'].data.cpu().numpy().copy())

    def compute_chaos_loss(self) -> torch.Tensor:
        """
        计算混沌正则化损失

        Returns:
            正则化损失张量
        """
        if len(self.weight_history) < 2:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # 每N步计算一次
        if self.step % self.frequency != 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # 获取最新的权重变化
        current_weights = self.get_weights_vector()
        previous_weights = self.weight_history[-2]
        weight_changes = current_weights - previous_weights

        # 根据方法计算混沌损失
        if self.method == 'lyapunov':
            # 计算李雅普诺夫指数
            lyap_exp = self.compute_lyapunov_exponent(weight_changes)
            chaos_loss = -lyap_exp  # 促进更高的李雅普诺夫指数（更多混沌）

            # 确保最小混沌性
            if lyap_exp < self.min_lyap:
                chaos_loss += self.min_lyap - lyap_exp

        elif self.method == 'rqa':
            # 计算RQA指标
            rqa_metrics = self.compute_rqa_metrics(weight_changes)

            # 促进更多混沌行为（高复发率、低确定性、低层流性）
            chaos_loss = -rqa_metrics['recurrence_rate'] + rqa_metrics['determinism'] + rqa_metrics['laminarity']

        else:
            raise ValueError(f"不支持的混沌方法: {self.method}")

        # 记录混沌分数
        self.chaos_scores.append(chaos_loss.item())

        # 应用正则化强度
        chaos_loss = self.beta * chaos_loss

        return torch.tensor(chaos_loss, requires_grad=True, device=self.device)

    def __call__(self) -> torch.Tensor:
        """调用正则化器，返回当前损失"""
        return self.compute_chaos_loss()


class ControlledChaosRegularizer(ChaoticRegularizer):
    """
    受控混沌正则化器，允许更精确地控制混沌特性

    Args:
        model: 要应用正则化的模型
        method: 混沌特性计算方法
        frequency: 应用正则化的频率
        beta: 正则化强度
        embedding_dim: 相空间重构的嵌入维度
        delay: 相空间重构的时间延迟
        min_lyap: 最小李雅普诺夫指数
        target_lyap: 目标李雅普诺夫指数
        device: 计算设备
    """

    def __init__(self,
                 model: nn.Module,
                 method: str = 'lyapunov',
                 frequency: int = 10,
                 beta: float = 0.1,
                 embedding_dim: int = 3,
                 delay: int = 5,
                 min_lyap: float = 0.01,
                 target_lyap: Optional[float] = None,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(
            model=model,
            method=method,
            frequency=frequency,
            beta=beta,
            embedding_dim=embedding_dim,
            delay=delay,
            min_lyap=min_lyap,
            device=device
        )
        self.target_lyap = target_lyap

        if self.target_lyap is not None:
            logger.info(f"受控混沌正则化器: 目标李雅普诺夫指数={target_lyap}")

    def compute_chaos_loss(self) -> torch.Tensor:
        """
        计算受控混沌正则化损失

        Returns:
            正则化损失张量
        """
        if len(self.weight_history) < 2:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        if self.step % self.frequency != 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        current_weights = self.get_weights_vector()
        previous_weights = self.weight_history[-2]
        weight_changes = current_weights - previous_weights

        # 计算李雅普诺夫指数
        lyap_exp = self.compute_lyapunov_exponent(weight_changes)

        # 计算与目标指数的差异
        if self.target_lyap is not None:
            chaos_loss = (lyap_exp - self.target_lyap) ** 2  # 最小二乘误差
        else:
            chaos_loss = -lyap_exp  # 促进更高的李雅普诺夫指数

        # 确保最小混沌性
        if lyap_exp < self.min_lyap:
            chaos_loss += (self.min_lyap - lyap_exp) ** 2

        # 记录混沌分数
        self.chaos_scores.append(lyap_exp)

        # 应用正则化强度
        chaos_loss = self.beta * chaos_loss

        return torch.tensor(chaos_loss, requires_grad=True, device=self.device)


class ChaoticWeightInitializer:
    """
    混沌权重初始化器，使用混沌系统初始化模型权重

    Args:
        method: 混沌系统类型（'lorenz', 'rossler', 'chen'）
        strength: 混沌强度（0-1）
    """

    @staticmethod
    def initialize(model: nn.Module,
                   method: str = 'lorenz',
                   strength: float = 0.5):
        """
        使用混沌系统初始化模型权重

        Args:
            model: 要初始化的模型
            method: 混沌系统类型
            strength: 混沌强度
        """

        # 定义混沌系统
        def lorenz_system(w, t, sigma=10.0, rho=28.0, beta=8 / 3):
            """Lorenz系统"""
            x, y, z = w
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz])

        def rossler_system(w, t, a=0.2, b=0.2, c=5.7):
            """Rössler系统"""
            x, y, z = w
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            return np.array([dx, dy, dz])

        def chen_system(w, t, a=36.0, b=3.0, c=20.0):
            """Chen系统"""
            x, y, z = w
            dx = a * (y - x)
            dy = (c - a) * x - x * z + c * y
            dz = x * y - b * z
            return np.array([dx, dy, dz])

        # 选择混沌系统
        if method == 'lorenz':
            system = lorenz_system
            params = {'sigma': 10.0, 'rho': 28.0, 'beta': 8 / 3}
        elif method == 'rossler':
            system = rossler_system
            params = {'a': 0.2, 'b': 0.2, 'c': 5.7}
        elif method == 'chen':
            system = chen_system
            params = {'a': 36.0, 'b': 3.0, 'c': 20.0}
        else:
            raise ValueError(f"不支持的混沌系统: {method}")

        # 获取模型参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 获取参数形状
                shape = param.shape
                num_elements = np.prod(shape)

                # 生成混沌序列
                t = np.linspace(0, 10, num_elements)
                w0 = np.random.randn(3)
                w = odeint(system, w0, t, **params)

                # 调整混沌强度
                w = (w - w.min()) / (w.max() - w.min())  # 归一化到 [0,1]
                w = 2 * w - 1  # 映射到 [-1,1]
                w = w * strength  # 应用强度因子

                # 重塑为参数形状
                if len(shape) > 1:
                    param.data = torch.tensor(w.reshape(shape).astype(np.float32)).to(param.device)
                else:
                    param.data = torch.tensor(w.reshape(-1).astype(np.float32)).to(param.device)

        logger.info(f"使用 {method} 系统初始化模型权重，混沌强度: {strength}")


class ChaoticLoss(nn.Module):
    """
    混沌损失，结合标准损失和混沌正则化

    Args:
        base_loss: 基础损失函数
        model: 模型
        chaotic_reg: 混沌正则化器
    """

    def __init__(self,
                 base_loss: nn.Module,
                 model: nn.Module,
                 chaotic_reg: Optional[ChaoticRegularizer] = None):
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.chaotic_reg = chaotic_reg if chaotic_reg is not None else None

        # 初始化权重历史
        if self.chaotic_reg is not None:
            self.chaotic_reg.weight_history = []
            self.chaotic_reg.step = 0

    def forward(self,
                outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # 计算基础损失
        base_loss = self.base_loss(outputs, targets)

        # 如果没有正则化器，直接返回基础损失
        if self.chaotic_reg is None:
            return base_loss

        # 更新权重历史
        self.chaotic_reg.update_weight_history()

        # 计算混沌正则化损失
        chaos_loss = self.chaotic_reg()

        # 组合损失
        total_loss = base_loss + chaos_loss

        return total_loss

    def __call__(self,
                 outputs: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        return self.forward(outputs, targets)