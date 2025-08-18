"""
混沌神经网络层实现 - 混沌神经网络说话人识别项目
包含洛伦兹振荡器层、混沌注意力机制、奇异吸引子池化等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict
from scipy.integrate import odeint


class LorentzOscillatorLayer(nn.Module):
    """
    洛伦兹混沌振荡器神经网络层
    基于洛伦兹系统的混沌动力学实现神经计算
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 sigma: float = 10.0,
                 rho: float = 28.0,
                 beta: float = 8.0 / 3.0,
                 dt: float = 0.01,
                 integration_steps: int = 10):
        """
        初始化洛伦兹振荡器层

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏状态维度
            sigma, rho, beta: 洛伦兹系统参数
            dt: 数值积分步长
            integration_steps: 积分步数
        """
        super(LorentzOscillatorLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.rho = nn.Parameter(torch.tensor(rho))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.dt = dt
        self.integration_steps = integration_steps

        # 输入到洛伦兹系统的线性变换
        self.input_transform = nn.Linear(input_dim, hidden_dim * 3)  # x, y, z三个维度

        # 混沌系统的权重矩阵
        self.W_x = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_y = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_z = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim * 3, hidden_dim)

        # 初始化混沌状态
        self.register_buffer('chaos_state', torch.randn(1, hidden_dim, 3))

        # 可学习的混沌强度参数
        self.chaos_strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            prev_state: 上一时刻的混沌状态 [batch_size, hidden_dim, 3]

        Returns:
            output: 输出张量 [batch_size, seq_len, hidden_dim]
            final_state: 最终混沌状态 [batch_size, hidden_dim, 3]
        """
        batch_size, seq_len, _ = x.shape

        # 初始化状态
        if prev_state is None:
            chaos_state = self.chaos_state.expand(batch_size, -1, -1).clone()
        else:
            chaos_state = prev_state

        # 输入变换
        input_transformed = self.input_transform(x)  # [batch_size, seq_len, hidden_dim * 3]
        input_transformed = input_transformed.view(batch_size, seq_len, self.hidden_dim, 3)

        outputs = []

        for t in range(seq_len):
            # 当前时刻的输入
            current_input = input_transformed[:, t]  # [batch_size, hidden_dim, 3]

            # 执行洛伦兹动力学积分
            chaos_state = self._integrate_lorenz_system(chaos_state, current_input)

            # 混沌状态到输出的投影
            state_flat = chaos_state.view(batch_size, -1)  # [batch_size, hidden_dim * 3]
            output = self.output_projection(state_flat)  # [batch_size, hidden_dim]
            outputs.append(output)

        # 堆叠输出
        output_tensor = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]

        return output_tensor, chaos_state

    def _integrate_lorenz_system(self, state: torch.Tensor, input_coupling: torch.Tensor) -> torch.Tensor:
        """
        使用四阶Runge-Kutta方法积分洛伦兹系统

        Args:
            state: 当前状态 [batch_size, hidden_dim, 3] (x, y, z)
            input_coupling: 输入耦合项 [batch_size, hidden_dim, 3]

        Returns:
            new_state: 更新后的状态
        """
        for _ in range(self.integration_steps):
            state = self._rk4_step(state, input_coupling)
        return state

    def _rk4_step(self, state: torch.Tensor, input_coupling: torch.Tensor) -> torch.Tensor:
        """四阶Runge-Kutta积分步骤"""
        k1 = self.dt * self._lorenz_derivatives(state, input_coupling)
        k2 = self.dt * self._lorenz_derivatives(state + 0.5 * k1, input_coupling)
        k3 = self.dt * self._lorenz_derivatives(state + 0.5 * k2, input_coupling)
        k4 = self.dt * self._lorenz_derivatives(state + k3, input_coupling)

        new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return new_state

    def _lorenz_derivatives(self, state: torch.Tensor, input_coupling: torch.Tensor) -> torch.Tensor:
        """
        计算洛伦兹系统的导数
        修改版洛伦兹方程：包含神经网络输入耦合
        """
        x, y, z = state[:, :, 0], state[:, :, 1], state[:, :, 2]
        input_x, input_y, input_z = input_coupling[:, :, 0], input_coupling[:, :, 1], input_coupling[:, :, 2]

        # 标准洛伦兹方程 + 神经网络耦合项
        dx = self.sigma * (y - x) + torch.matmul(input_x, self.W_x) * self.chaos_strength
        dy = x * (self.rho - z) - y + torch.matmul(input_y, self.W_y) * self.chaos_strength
        dz = x * y - self.beta * z + torch.matmul(input_z, self.W_z) * self.chaos_strength

        derivatives = torch.stack([dx, dy, dz], dim=2)
        return derivatives


class ChaoticAttentionMechanism(nn.Module):
    """
    混沌注意力机制
    基于分岔控制和相位同步的注意力计算
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 bifurcation_param: float = 1.0,
                 phase_coupling: float = 0.5):
        """
        初始化混沌注意力机制

        Args:
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            bifurcation_param: 分岔参数
            phase_coupling: 相位耦合强度
        """
        super(ChaoticAttentionMechanism, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 分岔控制参数
        self.bifurcation_param = nn.Parameter(torch.tensor(bifurcation_param))
        self.phase_coupling = nn.Parameter(torch.tensor(phase_coupling))

        # Query, Key, Value投影
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)

        # 混沌相位编码器
        self.phase_encoder = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # 相位同步损失权重
        self.register_buffer('sync_loss_weight', torch.tensor(0.01))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]

        Returns:
            output: 输出张量 [batch_size, seq_len, hidden_dim]
            sync_loss: 相位同步损失
        """
        batch_size, seq_len, _ = x.shape

        # 计算Query, Key, Value
        Q = self.query_projection(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key_projection(x)
        V = self.value_projection(x)

        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算混沌注意力权重
        attention_weights, phase_info = self._compute_chaotic_attention(Q, K, mask)

        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # 重塑并投影输出
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        output = self.output_projection(attended_values)

        # 计算相位同步损失
        sync_loss = self._compute_phase_sync_loss(phase_info)

        return output, sync_loss

    def _compute_chaotic_attention(self, Q: torch.Tensor, K: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算基于混沌动力学的注意力权重
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # 标准注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        # 混沌相位编码
        phase_Q = self._compute_phase_encoding(Q)
        phase_K = self._compute_phase_encoding(K)

        # 相位同步项
        phase_sync = self._compute_phase_synchronization(phase_Q, phase_K)

        # 分岔控制的非线性变换
        bifurcation_modulation = self._apply_bifurcation_control(scores)

        # 组合混沌注意力分数
        chaotic_scores = scores + self.phase_coupling * phase_sync + bifurcation_modulation

        # 应用掩码
        if mask is not None:
            chaotic_scores = chaotic_scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(chaotic_scores, dim=-1)

        # 保存相位信息用于损失计算
        phase_info = {
            'phase_Q': phase_Q,
            'phase_K': phase_K,
            'phase_sync': phase_sync
        }

        return attention_weights, phase_info

    def _compute_phase_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入的相位编码
        使用Hilbert变换提取瞬时相位
        """
        # 简化的相位编码：使用复数表示
        phase_features = self.phase_encoder(x.view(-1, x.shape[-1]))
        phase_features = phase_features.view(x.shape)

        # 转换为复数形式计算相位
        real_part = phase_features[..., :phase_features.shape[-1] // 2]
        imag_part = phase_features[..., phase_features.shape[-1] // 2:]

        if imag_part.shape[-1] < real_part.shape[-1]:
            imag_part = F.pad(imag_part, (0, real_part.shape[-1] - imag_part.shape[-1]))
        elif imag_part.shape[-1] > real_part.shape[-1]:
            real_part = F.pad(real_part, (0, imag_part.shape[-1] - real_part.shape[-1]))

        complex_features = torch.complex(real_part, imag_part)
        phases = torch.angle(complex_features)

        return phases

    def _compute_phase_synchronization(self, phase_Q: torch.Tensor, phase_K: torch.Tensor) -> torch.Tensor:
        """
        计算相位同步矩阵
        """
        # 计算相位差矩阵
        phase_diff = phase_Q.unsqueeze(-1) - phase_K.unsqueeze(-2)  # [batch, heads, seq, seq, head_dim]

        # 计算同步强度
        sync_strength = torch.cos(phase_diff).mean(dim=-1)  # [batch, heads, seq, seq]

        return sync_strength

    def _apply_bifurcation_control(self, scores: torch.Tensor) -> torch.Tensor:
        """
        应用分岔控制的非线性调制
        """
        # 使用logistic映射作为分岔系统
        normalized_scores = torch.tanh(scores)
        bifurcation_output = self.bifurcation_param * normalized_scores * (1 - normalized_scores)

        return bifurcation_output

    def _compute_phase_sync_loss(self, phase_info: Dict) -> torch.Tensor:
        """
        计算相位同步损失
        鼓励相关的序列位置保持相位同步
        """
        phase_sync = phase_info['phase_sync']

        # 计算同步一致性损失
        sync_variance = torch.var(phase_sync, dim=-1).mean()
        sync_loss = self.sync_loss_weight * sync_variance

        return sync_loss


class StrangeAttractorPooling(nn.Module):
    """
    奇异吸引子池化层
    基于混沌吸引子的几何结构进行时间维度池化
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 attractor_dim: int = 3,
                 num_attractors: int = 4):
        """
        初始化奇异吸引子池化层

        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            attractor_dim: 吸引子维度
            num_attractors: 吸引子数量
        """
        super(StrangeAttractorPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attractor_dim = attractor_dim
        self.num_attractors = num_attractors

        # 吸引子中心参数
        self.attractor_centers = nn.Parameter(torch.randn(num_attractors, attractor_dim))

        # 输入到吸引子空间的投影
        self.input_to_attractor = nn.Linear(input_dim, attractor_dim)

        # 吸引子权重计算网络
        self.attractor_weight_net = nn.Sequential(
            nn.Linear(attractor_dim, attractor_dim * 2),
            nn.ReLU(),
            nn.Linear(attractor_dim * 2, num_attractors)
        )

        # 输出投影
        self.output_projection = nn.Linear(input_dim * num_attractors, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]

        Returns:
            output: 池化后的输出 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 投影到吸引子空间
        attractor_features = self.input_to_attractor(x)  # [batch_size, seq_len, attractor_dim]

        # 计算每个时间步到各吸引子的距离
        distances = self._compute_attractor_distances(attractor_features)  # [batch_size, seq_len, num_attractors]

        # 基于距离计算吸引子权重
        attractor_weights = self._compute_attractor_weights(attractor_features, distances)

        # 加权池化
        pooled_features = self._perform_attractor_pooling(x, attractor_weights)

        # 输出投影
        output = self.output_projection(pooled_features)

        return output

    def _compute_attractor_distances(self, attractor_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征点到各个吸引子中心的距离
        """
        batch_size, seq_len, _ = attractor_features.shape

        # 扩展维度进行广播计算
        features_expanded = attractor_features.unsqueeze(3)  # [batch, seq, attractor_dim, 1]
        centers_expanded = self.attractor_centers.T.unsqueeze(0).unsqueeze(0)  # [1, 1, attractor_dim, num_attractors]

        # 计算欧几里得距离
        distances = torch.norm(features_expanded - centers_expanded, dim=2)  # [batch, seq, num_attractors]

        return distances

    def _compute_attractor_weights(self, attractor_features: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        基于吸引子动力学计算权重
        """
        batch_size, seq_len, _ = attractor_features.shape

        # 使用神经网络计算基础权重
        base_weights = []
        for t in range(seq_len):
            weights = self.attractor_weight_net(attractor_features[:, t])  # [batch_size, num_attractors]
            base_weights.append(weights)

        base_weights = torch.stack(base_weights, dim=1)  # [batch_size, seq_len, num_attractors]

        # 结合距离信息
        distance_weights = F.softmax(-distances, dim=-1)  # 距离越近权重越大

        # 组合权重
        final_weights = F.softmax(base_weights + distance_weights, dim=-1)

        return final_weights

    def _perform_attractor_pooling(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        执行基于吸引子的加权池化
        """
        batch_size, seq_len, input_dim = x.shape

        # 对每个吸引子进行加权池化
        pooled_per_attractor = []

        for i in range(self.num_attractors):
            # 当前吸引子的权重
            attractor_weight = weights[:, :, i:i + 1]  # [batch_size, seq_len, 1]

            # 加权平均池化
            weighted_features = x * attractor_weight  # [batch_size, seq_len, input_dim]
            pooled_feature = torch.sum(weighted_features, dim=1)  # [batch_size, input_dim]

            pooled_per_attractor.append(pooled_feature)

        # 连接所有吸引子的池化结果
        concatenated_features = torch.cat(pooled_per_attractor, dim=1)  # [batch_size, input_dim * num_attractors]

        return concatenated_features


class ChaosRegularizationLayer(nn.Module):
    """
    混沌正则化层
    通过混沌动力学约束提高模型的鲁棒性
    """

    def __init__(self,
                 feature_dim: int,
                 lyapunov_target: float = 0.5,
                 entropy_weight: float = 0.01):
        """
        初始化混沌正则化层

        Args:
            feature_dim: 特征维度
            lyapunov_target: 目标李雅普诺夫指数
            entropy_weight: 混沌熵权重
        """
        super(ChaosRegularizationLayer, self).__init__()

        self.feature_dim = feature_dim
        self.lyapunov_target = lyapunov_target
        self.entropy_weight = entropy_weight

        # 混沌熵计算网络
        self.entropy_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，计算混沌正则化损失

        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]

        Returns:
            x: 原输入（透传）
            regularization_loss: 混沌正则化损失
        """
        # 估计Kolmogorov-Sinai熵
        ks_entropy = self._estimate_ks_entropy(x)

        # 估计李雅普诺夫指数
        estimated_lyapunov = self._estimate_lyapunov_exponent(x)

        # 计算正则化损失
        entropy_loss = self.entropy_weight * torch.abs(ks_entropy - 0.5)  # 目标熵为0.5
        lyapunov_loss = torch.abs(estimated_lyapunov - self.lyapunov_target)

        regularization_loss = entropy_loss + lyapunov_loss

        return x, regularization_loss

    def _estimate_ks_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        估计Kolmogorov-Sinai熵
        """
        batch_size, seq_len, _ = x.shape

        # 使用神经网络估计熵
        entropy_values = []
        for t in range(seq_len):
            entropy = self.entropy_estimator(x[:, t])  # [batch_size, 1]
            entropy_values.append(entropy)

        entropy_sequence = torch.stack(entropy_values, dim=1)  # [batch_size, seq_len, 1]
        mean_entropy = torch.mean(entropy_sequence)

        return mean_entropy

    def _estimate_lyapunov_exponent(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的李雅普诺夫指数估计
        """
        # 计算相邻时间步的距离变化
        if x.shape[1] < 2:
            return torch.tensor(0.0, device=x.device)

        diff = x[:, 1:] - x[:, :-1]  # [batch_size, seq_len-1, feature_dim]
        distances = torch.norm(diff, dim=-1)  # [batch_size, seq_len-1]

        # 计算距离的对数增长率
        log_distances = torch.log(distances + 1e-8)

        if log_distances.shape[1] < 2:
            return torch.tensor(0.0, device=x.device)

        # 简单的差分近似导数
        lyapunov_approx = torch.mean(log_distances[:, 1:] - log_distances[:, :-1])

        return lyapunov_approx


# 测试和使用示例
if __name__ == "__main__":
    # 测试参数
    batch_size = 4
    seq_len = 100
    input_dim = 128
    hidden_dim = 256

    # 创建测试输入
    test_input = torch.randn(batch_size, seq_len, input_dim)

    print("开始混沌神经网络层测试...")

    # 测试洛伦兹振荡器层
    print("\n测试洛伦兹振荡器层...")
    lorenz_layer = LorentzOscillatorLayer(input_dim, hidden_dim)
    lorenz_output, final_state = lorenz_layer(test_input)
    print(f"洛伦兹层输出形状: {lorenz_output.shape}")
    print(f"最终状态形状: {final_state.shape}")

    # 测试混沌注意力机制
    print("\n测试混沌注意力机制...")
    attention_layer = ChaoticAttentionMechanism(hidden_dim)
    attention_output, sync_loss = attention_layer(lorenz_output)
    print(f"注意力输出形状: {attention_output.shape}")
    print(f"同步损失: {sync_loss.item():.6f}")

    # 测试奇异吸引子池化
    print("\n测试奇异吸引子池化...")
    pooling_layer = StrangeAttractorPooling(hidden_dim, hidden_dim // 2)
    pooled_output = pooling_layer(attention_output)
    print(f"池化输出形状: {pooled_output.shape}")

    # 测试混沌正则化层
    print("\n测试混沌正则化层...")
    regularization_layer = ChaosRegularizationLayer(hidden_dim)
    reg_output, reg_loss = regularization_layer(attention_output)
    print(f"正则化输出形状: {reg_output.shape}")
    print(f"正则化损失: {reg_loss.item():.6f}")

    print("\n所有混沌层测试完成！")