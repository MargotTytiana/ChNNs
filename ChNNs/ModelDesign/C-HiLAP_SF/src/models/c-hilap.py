import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from torchdiffeq import odeint


class ChaoticEmbeddingLayer(nn.Module):
    """
    混沌嵌入层，使用混沌映射转换输入特征

    Args:
        input_dim: 输入特征维度
        embed_dim: 嵌入向量维度
        chaos_type: 混沌映射类型 ('logistic', 'tent', 'henon')
        ks_entropy: 目标Kolmogorov-Sinai熵
    """

    def __init__(
            self,
            input_dim: int,
            embed_dim: int,
            chaos_type: str = "tent",
            ks_entropy: float = 0.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.chaos_type = chaos_type
        self.ks_entropy = ks_entropy

        # 线性投影层
        self.proj = nn.Linear(input_dim, embed_dim)

        # 混沌参数（可学习）
        if chaos_type == "logistic":
            # Logistic映射: x_{n+1} = r * x_n * (1 - x_n)
            self.r = nn.Parameter(torch.tensor(3.9))  # 分岔参数
        elif chaos_type == "tent":
            # 帐篷映射: x_{n+1} = mu * min(x_n, 1-x_n)
            self.mu = nn.Parameter(torch.tensor(1.9))  # 斜率参数
        elif chaos_type == "henon":
            # Henon映射: x_{n+1} = 1 - a*x_n^2 + b*y_n, y_{n+1} = x_n
            self.a = nn.Parameter(torch.tensor(1.4))
            self.b = nn.Parameter(torch.tensor(0.3))
        else:
            raise ValueError(f"未知的混沌类型: {chaos_type}")

        # 混合矩阵
        self.mixing = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)

    def apply_chaos(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用混沌映射

        Args:
            x: 输入张量

        Returns:
            转换后的张量
        """
        # 归一化到适当的范围
        x_norm = torch.sigmoid(x)  # [0, 1]

        if self.chaos_type == "logistic":
            # Logistic映射
            r = torch.sigmoid(self.r) * 4.0  # 缩放到 [0, 4]
            return r * x_norm * (1 - x_norm)

        elif self.chaos_type == "tent":
            # 帐篷映射
            mu = torch.sigmoid(self.mu) * 2.0  # 缩放到 [0, 2]
            return mu * torch.min(x_norm, 1 - x_norm)

        elif self.chaos_type == "henon":
            # Henon映射（简化的1D版本）
            a = torch.sigmoid(self.a) * 2.0  # 缩放到 [0, 2]
            b = torch.sigmoid(self.b)  # 缩放到 [0, 1]

            # 应用映射并重新缩放到 [0, 1]
            x_out = 1 - a * x_norm ** 2 + b * (1 - x_norm)
            return torch.sigmoid(x_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        混沌嵌入层的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            嵌入张量，形状为 (batch_size, seq_len, embed_dim)
        """
        # 线性投影
        x = self.proj(x)

        # 应用混沌变换
        x = self.apply_chaos(x)

        # 应用混合
        x = torch.matmul(x, self.mixing)

        return x


class ChaoticOscillator(nn.Module):
    """
    混沌振荡器基类，实现各种混沌系统

    Args:
        system_type: 混沌系统类型 ('lorenz', 'rossler', 'chen', 'chua')
    """

    def __init__(self, system_type: str = "lorenz"):
        super().__init__()
        self.system_type = system_type

        # 初始化不同系统的参数
        if system_type == "lorenz":
            # Lorenz系统参数
            self.sigma = nn.Parameter(torch.tensor(10.0))
            self.rho = nn.Parameter(torch.tensor(28.0))
            self.beta = nn.Parameter(torch.tensor(8.0 / 3.0))
        elif system_type == "rossler":
            # Rössler系统参数
            self.a = nn.Parameter(torch.tensor(0.2))
            self.b = nn.Parameter(torch.tensor(0.2))
            self.c = nn.Parameter(torch.tensor(5.7))
        elif system_type == "chen":
            # Chen系统参数
            self.a = nn.Parameter(torch.tensor(35.0))
            self.b = nn.Parameter(torch.tensor(3.0))
            self.c = nn.Parameter(torch.tensor(28.0))
        elif system_type == "chua":
            # Chua电路参数
            self.alpha = nn.Parameter(torch.tensor(15.6))
            self.beta = nn.Parameter(torch.tensor(28.0))
            self.m0 = nn.Parameter(torch.tensor(-1.143))
            self.m1 = nn.Parameter(torch.tensor(-0.714))
        else:
            raise ValueError(f"未知的系统类型: {system_type}")

    def forward(self, t, state):
        """
        计算混沌系统的状态导数

        Args:
            t: 时间（自治系统中不使用）
            state: 当前状态 [x, y, z]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt]
        """
        if self.system_type == "lorenz":
            x, y, z = state
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return torch.stack([dx, dy, dz])

        elif self.system_type == "rossler":
            x, y, z = state
            dx = -y - z
            dy = x + self.a * y
            dz = self.b + z * (x - self.c)
            return torch.stack([dx, dy, dz])

        elif self.system_type == "chen":
            x, y, z = state
            dx = self.a * (y - x)
            dy = (self.c - self.a) * x - x * z + self.c * y
            dz = x * y - self.b * z
            return torch.stack([dx, dy, dz])

        elif self.system_type == "chua":
            x, y, z = state
            # Chua二极管非线性
            h_x = self.m1 * x + 0.5 * (self.m0 - self.m1) * (torch.abs(x + 1) - torch.abs(x - 1))

            dx = self.alpha * (y - h_x)
            dy = x - y + z
            dz = -self.beta * y
            return torch.stack([dx, dy, dz])


class BifurcationControlledAttention(nn.Module):
    """
    分岔控制注意力机制，使用混沌分岔参数控制注意力行为

    Args:
        embed_dim: 嵌入向量维度
        num_heads: 注意力头数
        dropout: Dropout概率
        bifurcation_factor: 控制混沌行为的分岔因子
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            bifurcation_factor: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.bifurcation_factor = bifurcation_factor

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # Q, K, V的线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 分岔参数（可学习）
        self.r = nn.Parameter(torch.tensor(3.5))  # 初始值在混沌区域

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def logistic_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用Logistic映射迭代引入混沌

        Args:
            x: 输入张量

        Returns:
            转换后的张量
        """
        # 归一化到 [0, 1] 范围
        x_norm = torch.sigmoid(x)

        # 应用Logistic映射和分岔参数
        r_scaled = self.bifurcation_factor * torch.sigmoid(self.r) * 4.0  # 缩放到 [0, 4]
        return r_scaled * x_norm * (1 - x_norm)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        分岔控制注意力机制的前向传播函数

        Args:
            query: 查询张量，形状为 (batch_size, tgt_len, embed_dim)
            key: 键张量，形状为 (batch_size, src_len, embed_dim)
            value: 值张量，形状为 (batch_size, src_len, embed_dim)
            key_padding_mask: 键的填充掩码
            attn_mask: 注意力权重掩码

        Returns:
            输出张量，形状为 (batch_size, tgt_len, embed_dim)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 应用混沌变换
        q = self.logistic_map(q)
        k = self.logistic_map(k)

        # 重塑为多头注意力
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用掩码（如果提供）
        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        output = torch.matmul(attn_weights, v)

        # 重塑回原始形状
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)

        # 输出投影
        output = self.out_proj(output)

        return output


class StrangeAttractorPooling(nn.Module):
    """
    奇异吸引子池化层，使用混沌系统集成时序信息

    Args:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        system_type: 混沌系统类型
        integration_steps: 积分步数
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            system_type: str = "lorenz",
            integration_steps: int = 20
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.system_type = system_type
        self.integration_steps = integration_steps

        # 输入投影
        self.input_proj = nn.Linear(input_dim, 3)

        # 混沌振荡器
        self.oscillator = ChaoticOscillator(system_type)

        # 输出投影
        self.output_proj = nn.Linear(3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        奇异吸引子池化的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 投影输入到初始条件
        init_conditions = self.input_proj(x[:, 0])  # 使用第一帧作为初始条件

        # 初始化每个批次的状态
        states = init_conditions

        # 使用序列通过时间积分
        for t in range(seq_len):
            # 当前输入
            x_t = self.input_proj(x[:, t])

            # 定义积分的时间跨度
            t_span = torch.linspace(0, 1.0, self.integration_steps, device=x.device)

            # 对每个批次项进行积分
            for b in range(batch_size):
                # 定义带有当前输入作为参数的ODE
                def system_with_input(t, state):
                    # 基础动力学
                    dstate = self.oscillator(t, state)

                    # 添加输入影响
                    input_influence = 0.1 * (x_t[b] - state)

                    return dstate + input_influence

                # 求解ODE
                trajectory = odeint(
                    system_with_input,
                    states[b],
                    t_span,
                    method='rk4'
                )

                # 更新状态
                states[b] = trajectory[-1]

        # 将最终状态投影到输出维度
        output = self.output_proj(states)

        return output


class CHiLAP(nn.Module):
    """
    混沌层级吸引子传播 (C-HiLAP) 模型

    Args:
        input_dim: 输入特征维度
        embed_dim: 嵌入向量维度
        num_classes: 说话人类别数量（0表示仅提取嵌入）
        num_layers: 模型层数
        chaos_type: 混沌系统类型
        ks_entropy: 目标Kolmogorov-Sinai熵
        bifurcation_factor: 控制混沌行为的分岔因子
    """

    def __init__(
            self,
            input_dim: int = 80,
            embed_dim: int = 512,
            num_classes: int = 0,
            num_layers: int = 4,
            chaos_type: str = "lorenz",
            ks_entropy: float = 0.5,
            bifurcation_factor: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        # 输入嵌入
        self.input_embed = ChaoticEmbeddingLayer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            chaos_type="tent",
            ks_entropy=ks_entropy
        )

        # 混沌层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                # 受控混沌振荡器
                'chaos': ControlledChaoticOscillator(
                    input_dim=embed_dim,
                    hidden_dim=embed_dim,
                    system_type=chaos_type,
                    integration_time=1.0,
                    integration_steps=10
                ),
                # 混沌注意力
                'attention': BifurcationControlledAttention(
                    embed_dim=embed_dim,
                    num_heads=8,
                    dropout=0.1,
                    bifurcation_factor=bifurcation_factor
                ),
                # 前馈网络
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(4 * embed_dim, embed_dim)
                ),
                # 层归一化
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim)
            })
            self.layers.append(layer)

        # 奇异吸引子池化
        self.pooling = StrangeAttractorPooling(
            input_dim=embed_dim,
            output_dim=embed_dim,
            system_type=chaos_type
        )

        # 输出投影
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        C-HiLAP模型的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
            extract_embedding: 是否仅提取嵌入向量

        Returns:
            输出张量，形状为 (batch_size, num_classes) 或 (batch_size, embed_dim)
        """
        # 输入嵌入
        x = self.input_embed(x)

        # 处理各层
        for layer in self.layers:
            # 混沌处理
            chaos_out = layer['chaos'](x)
            x = layer['norm1'](x + chaos_out)

            # 自注意力
            attn_out = layer['attention'](x, x, x)
            x = x + attn_out

            # 前馈网络
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        # 奇异吸引子池化
        embedding = self.pooling(x)

        # 如果请求，返回嵌入向量
        if extract_embedding:
            return embedding

        # 分类
        if self.classifier is not None:
            return self.classifier(embedding)
        else:
            return embedding


class ControlledChaoticOscillator(nn.Module):
    """
    受控混沌振荡器，将混沌系统与神经网络集成

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏状态维度
        system_type: 混沌系统类型
        integration_time: 积分时间
        integration_steps: 积分步数
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            system_type: str = "lorenz",
            integration_time: float = 1.0,
            integration_steps: int = 10
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.system_type = system_type
        self.integration_time = integration_time
        self.integration_steps = integration_steps

        # 输入投影
        self.input_proj = nn.Linear(input_dim, 3)

        # 初始化混沌系统
        self.oscillator = ChaoticOscillator(system_type)

        # 控制矩阵（耦合强度）
        self.control_matrix = nn.Parameter(torch.randn(3, 3) * 0.01)

        # 输出投影
        self.output_proj = nn.Linear(3, hidden_dim)

    def system_with_control(self, t, state, control_input):
        """
        带有外部控制输入的混沌系统

        Args:
            t: 时间
            state: 当前状态
            control_input: 外部控制输入

        Returns:
            状态导数
        """
        # 获取基础系统动力学
        dstate = self.oscillator(t, state)

        # 添加控制项
        control_term = torch.matmul(self.control_matrix, control_input)

        return dstate + control_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        受控混沌振荡器的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            输出张量，形状为 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 投影输入到控制信号
        control_input = self.input_proj(x)  # (batch_size, seq_len, 3)

        # 初始化输出张量
        output = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)

        # 处理每个序列步骤
        for b in range(batch_size):
            # 使用第一个控制输入初始化状态
            state = control_input[b, 0]

            for t in range(seq_len):
                # 当前控制输入
                control = control_input[b, t]

                # 积分时间点
                t_span = torch.linspace(0, self.integration_time, self.integration_steps, device=x.device)

                # 定义带有当前控制输入的ODE函数
                def controlled_system(t, y):
                    return self.system_with_control(t, y, control)

                # 求解ODE
                trajectory = odeint(
                    controlled_system,
                    state,
                    t_span,
                    method='rk4',
                    options=dict(step_size=self.integration_time / self.integration_steps)
                )

                # 使用轨迹的最后一点更新状态
                state = trajectory[-1]

                # 投影到输出空间
                output[b, t] = self.output_proj(state)

        return output


class PhaseSynchronizationLoss(nn.Module):
    """
    基于相位同步的损失函数

    Args:
        weight: 损失项权重
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def extract_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """
        使用解析信号方法提取瞬时相位

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

    def forward(self, input_signal: torch.Tensor, attractor_signal: torch.Tensor) -> torch.Tensor:
        """
        计算相位同步损失

        Args:
            input_signal: 输入信号
            attractor_signal: 吸引子信号

        Returns:
            同步损失
        """
        # 提取相位
        input_phase = self.extract_phase(input_signal)
        attractor_phase = self.extract_phase(attractor_signal)

        # 计算相位差
        phase_diff = input_phase - attractor_phase

        # 计算同步指数 (1 - R)
        # 其中R是相位锁定值
        sync_loss = 1.0 - torch.abs(torch.mean(torch.exp(1j * phase_diff.float())))

        return self.weight * sync_loss


class CHiLAPWithLoss(nn.Module):
    """
    带有联合损失函数的C-HiLAP模型

    Args:
        c_hilap: C-HiLAP模型
        num_classes: 说话人类别数量
        sync_weight: 相位同步损失权重
    """

    def __init__(
            self,
            c_hilap: CHiLAP,
            num_classes: int,
            sync_weight: float = 0.1
    ):
        super().__init__()
        self.c_hilap = c_hilap
        self.num_classes = num_classes

        # 分类损失
        self.ce_loss = nn.CrossEntropyLoss()

        # 相位同步损失
        self.sync_loss = PhaseSynchronizationLoss(weight=sync_weight)

        # 说话人分类器
        if c_hilap.classifier is None and num_classes > 0:
            self.classifier = nn.Linear(c_hilap.embed_dim, num_classes)
        else:
            self.classifier = c_hilap.classifier

    def forward(
            self,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        带有损失计算的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
            labels: 标签张量，形状为 (batch_size,)

        Returns:
            如果提供labels:
                (总损失, 损失字典)
            否则:
                模型输出
        """
        # 通过C-HiLAP模型
        embeddings = self.c_hilap(x, extract_embedding=True)

        # 如果没有提供标签，仅返回输出
        if labels is None:
            if self.classifier is not None:
                return self.classifier(embeddings)
            else:
                return embeddings

        # 计算分类损失
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            ce_loss = self.ce_loss(logits, labels)
        else:
            logits = None
            ce_loss = torch.tensor(0.0, device=x.device)

        # 计算相位同步损失（假设我们有中间表示）
        # 注意：在实际实现中，需要修改C-HiLAP模型以返回中间表示
        # 这里只是一个占位符
        sync_loss = torch.tensor(0.0, device=x.device)

        # 总损失
        total_loss = ce_loss + sync_loss

        # 返回总损失和损失字典
        return total_loss, {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'sync_loss': sync_loss
        }


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
    input_dim = 80

    x = torch.randn(batch_size, seq_len, input_dim)

    # 创建C-HiLAP模型
    model = CHiLAP(
        input_dim=input_dim,
        embed_dim=512,
        num_classes=10,
        num_layers=4,
        chaos_type="lorenz",
        ks_entropy=0.5,
        bifurcation_factor=1.0
    )

    # 打印模型参数量
    print(f"模型参数量: {count_parameters(model):,}")

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 提取嵌入向量
    embeddings = model(x, extract_embedding=True)
    print(f"嵌入向量形状: {embeddings.shape}")

    # 创建带有损失函数的模型
    model_with_loss = CHiLAPWithLoss(
        c_hilap=model,
        num_classes=10,
        sync_weight=0.1
    )

    # 创建随机标签
    labels = torch.randint(0, 10, (batch_size,))

    # 计算损失
    loss, loss_dict = model_with_loss(x, labels)
    print(f"总损失: {loss.item():.4f}")
    print(f"分类损失: {loss_dict['ce_loss'].item():.4f}")
    print(f"同步损失: {loss_dict['sync_loss'].item():.4f}")