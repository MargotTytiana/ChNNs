import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from torchdiffeq import odeint

# 导入基础模型组件
from ecapa_tdnn import ECAPA_TDNN, SEModule, AttentiveStatsPool
from xvector import TDNN, StatisticsPooling


class LorentzOscillator(nn.Module):
    """
    洛伦兹振荡器模块，实现经典的洛伦兹混沌系统

    Args:
        sigma: 洛伦兹系统参数 σ
        rho: 洛伦兹系统参数 ρ
        beta: 洛伦兹系统参数 β
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            sigma: float = 10.0,
            rho: float = 28.0,
            beta: float = 8.0 / 3.0,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化洛伦兹系统参数
        if learnable:
            self.sigma = nn.Parameter(torch.tensor(sigma))
            self.rho = nn.Parameter(torch.tensor(rho))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('sigma', torch.tensor(sigma))
            self.register_buffer('rho', torch.tensor(rho))
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        洛伦兹系统的前向传播函数

        Args:
            t: 时间点（在自治系统中不使用，但ODE求解器需要）
            state: 系统状态 [x, y, z]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        return torch.stack([dx, dy, dz])


class ChenLeeOscillator(nn.Module):
    """
    Chen-Lee振荡器模块，实现Chen-Lee混沌系统

    Args:
        a: Chen-Lee系统参数 a
        b: Chen-Lee系统参数 b
        c: Chen-Lee系统参数 c
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            a: float = 5.0,
            b: float = -10.0,
            c: float = -0.38,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化Chen-Lee系统参数
        if learnable:
            self.a = nn.Parameter(torch.tensor(a))
            self.b = nn.Parameter(torch.tensor(b))
            self.c = nn.Parameter(torch.tensor(c))
        else:
            self.register_buffer('a', torch.tensor(a))
            self.register_buffer('b', torch.tensor(b))
            self.register_buffer('c', torch.tensor(c))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Chen-Lee系统的前向传播函数

        Args:
            t: 时间点（在自治系统中不使用，但ODE求解器需要）
            state: 系统状态 [x, y, z]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state

        dx = self.a * y
        dy = self.b * x + self.c * y * z
        dz = -2 * y ** 2

        return torch.stack([dx, dy, dz])


class RosslerOscillator(nn.Module):
    """
    罗斯勒振荡器模块，实现Rössler混沌系统

    Args:
        a: Rössler系统参数 a
        b: Rössler系统参数 b
        c: Rössler系统参数 c
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            a: float = 0.2,
            b: float = 0.2,
            c: float = 5.7,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化Rössler系统参数
        if learnable:
            self.a = nn.Parameter(torch.tensor(a))
            self.b = nn.Parameter(torch.tensor(b))
            self.c = nn.Parameter(torch.tensor(c))
        else:
            self.register_buffer('a', torch.tensor(a))
            self.register_buffer('b', torch.tensor(b))
            self.register_buffer('c', torch.tensor(c))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Rössler系统的前向传播函数

        Args:
            t: 时间点（在自治系统中不使用，但ODE求解器需要）
            state: 系统状态 [x, y, z]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state

        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)

        return torch.stack([dx, dy, dz])


class ControlledChaoticLayer(nn.Module):
    """
    受控混沌层，将混沌振荡器与神经网络集成

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏状态维度
        chaos_type: 混沌系统类型 ('lorenz', 'chen_lee', 'rossler')
        integration_steps: 积分步数
        integration_time: 积分时间
        control_strength: 控制强度
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            chaos_type: str = "lorenz",
            integration_steps: int = 10,
            integration_time: float = 1.0,
            control_strength: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chaos_type = chaos_type
        self.integration_steps = integration_steps
        self.integration_time = integration_time
        self.control_strength = control_strength

        # 初始化混沌系统
        if chaos_type == "lorenz":
            self.oscillator = LorentzOscillator(learnable=True)
        elif chaos_type == "chen_lee":
            self.oscillator = ChenLeeOscillator(learnable=True)
        elif chaos_type == "rossler":
            self.oscillator = RosslerOscillator(learnable=True)
        else:
            raise ValueError(f"未知的混沌系统类型: {chaos_type}")

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, 3)

        # 控制矩阵（耦合强度）
        self.control_matrix = nn.Parameter(torch.randn(3, 3) * 0.01)

        # 输出投影层
        self.output_proj = nn.Linear(3, hidden_dim)

        # 批归一化层
        self.bn = nn.BatchNorm1d(hidden_dim)

    def system_with_control(
            self,
            t: torch.Tensor,
            state: torch.Tensor,
            control_input: torch.Tensor
    ) -> torch.Tensor:
        """
        带有外部控制输入的混沌系统

        Args:
            t: 时间点
            state: 当前状态
            control_input: 外部控制输入

        Returns:
            状态导数
        """
        # 获取基础系统动力学
        dstate = self.oscillator(t, state)

        # 添加控制项
        control_term = self.control_strength * torch.matmul(self.control_matrix, control_input)

        return dstate + control_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        受控混沌层的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, input_dim, seq_len)

        Returns:
            输出张量，形状为 (batch_size, hidden_dim, seq_len)
        """
        batch_size, input_dim, seq_len = x.shape

        # 转置输入以便于处理
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_dim)

        # 投影输入到控制信号
        control_input = self.input_proj(x)  # (batch_size, seq_len, 3)

        # 初始化输出张量
        output = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)

        # 处理每个批次
        for b in range(batch_size):
            # 初始化状态
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

                # 更新状态
                state = trajectory[-1]

                # 投影到输出空间
                output[b, t] = self.output_proj(state)

        # 转置回原始格式
        output = output.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)

        # 应用批归一化
        output = self.bn(output)

        return output


class ChaoticTDNN(nn.Module):
    """
    混沌增强的时延神经网络层

    Args:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        context: 上下文偏移列表
        dilation: 卷积扩张因子
        chaos_type: 混沌系统类型
        chaos_hidden_dim: 混沌隐藏状态维度
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            context: List[int],
            dilation: int = 1,
            chaos_type: str = "lorenz",
            chaos_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 标准TDNN层
        self.tdnn = TDNN(input_dim, output_dim, context, dilation)

        # 混沌层
        if chaos_hidden_dim is None:
            chaos_hidden_dim = output_dim

        self.chaotic_layer = ControlledChaoticLayer(
            input_dim=output_dim,
            hidden_dim=output_dim,
            chaos_type=chaos_type,
            integration_steps=5,
            integration_time=0.5,
            control_strength=0.1
        )

        # 残差连接的投影层（如果维度不匹配）
        self.res_proj = None
        if input_dim != output_dim:
            self.res_proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        混沌TDNN层的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, input_dim, seq_len)

        Returns:
            输出张量，形状为 (batch_size, output_dim, seq_len)
        """
        # 标准TDNN处理
        tdnn_out = self.tdnn(x)

        # 混沌层处理
        chaos_out = self.chaotic_layer(tdnn_out)

        # 计算门控权重
        gate_input = torch.cat([tdnn_out, chaos_out], dim=1)
        gate_weights = self.gate(gate_input)

        # 应用门控
        output = tdnn_out * gate_weights + chaos_out * (1 - gate_weights)

        # 添加残差连接
        if self.res_proj is not None:
            residual = self.res_proj(x)
        else:
            residual = x

        output = output + residual

        return output


class ChaoticXVector(nn.Module):
    """
    混沌增强的X-Vector模型

    Args:
        input_dim: 输入特征维度
        num_classes: 说话人类别数量（0表示仅提取嵌入）
        emb_dim: 嵌入向量维度
        chaos_type: 混沌系统类型
    """

    def __init__(
            self,
            input_dim: int = 40,
            num_classes: int = 0,
            emb_dim: int = 512,
            chaos_type: str = "lorenz"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # 帧级层
        self.frame_layers = nn.Sequential(
            # TDNN1: 上下文 [-2, -1, 0, 1, 2]
            ChaoticTDNN(input_dim, 512, context=[-2, -1, 0, 1, 2], chaos_type=chaos_type),

            # TDNN2: 上下文 [-2, 0, 2]
            ChaoticTDNN(512, 512, context=[-2, 0, 2], chaos_type=chaos_type),

            # TDNN3: 上下文 [-3, 0, 3]
            ChaoticTDNN(512, 512, context=[-3, 0, 3], chaos_type=chaos_type),

            # TDNN4: 上下文 [0]
            ChaoticTDNN(512, 512, context=[0], chaos_type=chaos_type),

            # TDNN5: 上下文 [0]
            ChaoticTDNN(512, 1500, context=[0], chaos_type=chaos_type)
        )

        # 统计池化层
        self.stats_pooling = StatisticsPooling()

        # 段级层
        self.segment_layer1 = nn.Sequential(
            nn.Linear(3000, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        # 嵌入层
        self.embedding_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

        # 输出层（可选）
        if num_classes > 0:
            self.output_layer = nn.Linear(emb_dim, num_classes)
        else:
            self.output_layer = None

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        混沌X-Vector模型的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
            extract_embedding: 是否仅提取嵌入向量

        Returns:
            如果extract_embedding为True，返回嵌入向量
            否则，如果有分类器，返回分类结果；否则返回嵌入向量
        """
        # 转置输入为 (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # 帧级处理
        x = self.frame_layers(x)

        # 统计池化
        x = self.stats_pooling(x)

        # 第一个段级层
        x = self.segment_layer1(x)

        # 嵌入层
        embedding = self.embedding_layer(x)

        # 返回嵌入向量（如果请求）
        if extract_embedding:
            return embedding

        # 输出层（如果存在）
        if self.output_layer is not None:
            output = self.output_layer(embedding)
            return output
        else:
            return embedding


class ChaoticECAPATDNN(nn.Module):
    """
    混沌增强的ECAPA-TDNN模型

    Args:
        input_dim: 输入特征维度
        channels: 卷积通道数
        emb_dim: 嵌入向量维度
        num_classes: 说话人类别数量（0表示仅提取嵌入）
        chaos_type: 混沌系统类型
    """

    def __init__(
            self,
            input_dim: int = 80,
            channels: int = 512,
            emb_dim: int = 192,
            num_classes: int = 0,
            chaos_type: str = "lorenz"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        # 初始1x1卷积
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # 帧级块
        self.layer1 = self._create_chaotic_layer(channels, channels, dilation=1, chaos_type=chaos_type)
        self.layer2 = self._create_chaotic_layer(channels, channels, dilation=2, chaos_type=chaos_type)
        self.layer3 = self._create_chaotic_layer(channels, channels, dilation=3, chaos_type=chaos_type)
        self.layer4 = self._create_chaotic_layer(channels, channels, dilation=4, chaos_type=chaos_type)

        # 多层特征聚合
        self.mfa = nn.Conv1d(channels * 4, channels, kernel_size=1)
        self.bn_mfa = nn.BatchNorm1d(channels)

        # 注意力统计池化
        self.asp = AttentiveStatsPool(channels)

        # 最终嵌入层
        self.fc1 = nn.Linear(channels * 2, channels)
        self.bn_fc1 = nn.BatchNorm1d(channels)
        self.fc2 = nn.Linear(channels, emb_dim)
        self.bn_fc2 = nn.BatchNorm1d(emb_dim)

        # 分类层（可选）
        if num_classes > 0:
            self.classifier = nn.Linear(emb_dim, num_classes)
        else:
            self.classifier = None

    def _create_chaotic_layer(
            self,
            in_channels: int,
            out_channels: int,
            dilation: int,
            chaos_type: str
    ) -> nn.Module:
        """
        创建混沌增强的层

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            dilation: 扩张因子
            chaos_type: 混沌系统类型

        Returns:
            混沌增强的层
        """
        # 创建标准ECAPA-TDNN层
        layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            SEModule(out_channels)
        )

        # 添加混沌层
        chaotic_layer = ControlledChaoticLayer(
            input_dim=out_channels,
            hidden_dim=out_channels,
            chaos_type=chaos_type,
            integration_steps=5,
            integration_time=0.5,
            control_strength=0.1
        )

        # 组合成混沌增强的层
        return nn.Sequential(
            layer,
            chaotic_layer
        )

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        混沌ECAPA-TDNN模型的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
            extract_embedding: 是否仅提取嵌入向量

        Returns:
            如果extract_embedding为True，返回嵌入向量
            否则，如果有分类器，返回分类结果；否则返回嵌入向量
        """
        # 转置输入为 (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 帧级特征提取
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer1_out + layer2_out)
        layer4_out = self.layer4(layer1_out + layer2_out + layer3_out)

        # 多层特征聚合
        x = torch.cat([layer1_out, layer2_out, layer3_out, layer4_out], dim=1)
        x = self.mfa(x)
        x = self.bn_mfa(x)
        x = self.relu(x)

        # 注意力统计池化
        x = self.asp(x)

        # 最终嵌入层
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        # 提取嵌入向量
        if extract_embedding:
            return x

        # 应用批归一化
        x = self.bn_fc2(x)

        # 分类（如果请求）
        if self.classifier is not None:
            return self.classifier(x)

        return x


class LyapunovRegularizationLoss(nn.Module):
    """
    基于李雅普诺夫稳定性理论的正则化损失

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
        从轨迹估计最大李雅普诺夫指数

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
        计算李雅普诺夫正则化损失

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


class ChaosEnhancedModel_VariantB(nn.Module):
    """
    混沌增强模型（变体B）

    在标准骨干网络中注入混沌动力学层

    Args:
        input_dim: 输入特征维度
        backbone_type: 骨干网络类型，'ecapa_tdnn'或'xvector'
        chaos_type: 混沌系统类型
        emb_dim: 嵌入向量维度
        num_classes: 说话人类别数量（0表示仅提取嵌入）
    """

    def __init__(
            self,
            input_dim: int = 80,
            backbone_type: str = "ecapa_tdnn",
            chaos_type: str = "lorenz",
            emb_dim: int = 192,
            num_classes: int = 0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.backbone_type = backbone_type
        self.chaos_type = chaos_type
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        # 创建混沌增强的骨干网络
        if backbone_type == "ecapa_tdnn":
            self.backbone = ChaoticECAPATDNN(
                input_dim=input_dim,
                channels=512,
                emb_dim=emb_dim,
                num_classes=num_classes,
                chaos_type=chaos_type
            )
        elif backbone_type == "xvector":
            self.backbone = ChaoticXVector(
                input_dim=input_dim,
                num_classes=num_classes,
                emb_dim=emb_dim,
                chaos_type=chaos_type
            )
        else:
            raise ValueError(f"未知的骨干网络类型: {backbone_type}")

        # 创建李雅普诺夫正则化损失
        self.lyapunov_reg = LyapunovRegularizationLoss(
            target_exponent=0.1,
            weight=0.01,
            mode="positive"
        )

        # 存储中间轨迹（用于正则化）
        self.trajectories = None

    def forward(
            self,
            x: torch.Tensor,
            extract_embedding: bool = False,
            store_trajectories: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        混沌增强模型的前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
            extract_embedding: 是否仅提取嵌入向量
            store_trajectories: 是否存储中间轨迹（用于正则化）

        Returns:
            如果store_trajectories为False:
                模型输出（嵌入向量或分类结果）
            否则:
                (模型输出, 正则化损失)
        """
        # 通过骨干网络
        output = self.backbone(x, extract_embedding=extract_embedding)

        # 如果不需要存储轨迹，直接返回输出
        if not store_trajectories:
            return output

        # 计算正则化损失（假设轨迹已存储在模型中）
        # 注意：在实际实现中，需要修改骨干网络以返回中间轨迹
        # 这里只是一个占位符
        reg_loss = torch.tensor(0.0, device=x.device)

        return output, reg_loss


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

    # 创建混沌振荡器
    oscillator = LorentzOscillator()
    print(
        f"洛伦兹振荡器参数: σ={oscillator.sigma.item():.1f}, ρ={oscillator.rho.item():.1f}, β={oscillator.beta.item():.1f}")

    # 创建受控混沌层
    chaotic_layer = ControlledChaoticLayer(
        input_dim=input_dim,
        hidden_dim=512,
        chaos_type="lorenz"
    )

    # 测试受控混沌层
    x_transposed = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
    chaotic_output = chaotic_layer(x_transposed)
    print(f"混沌层输出形状: {chaotic_output.shape}")

    # 创建混沌TDNN层
    chaotic_tdnn = ChaoticTDNN(
        input_dim=input_dim,
        output_dim=512,
        context=[-2, -1, 0, 1, 2],
        chaos_type="lorenz"
    )

    # 测试混沌TDNN层
    tdnn_output = chaotic_tdnn(x_transposed)
    print(f"混沌TDNN层输出形状: {tdnn_output.shape}")

    # 创建混沌X-Vector模型
    chaotic_xvector = ChaoticXVector(
        input_dim=input_dim,
        num_classes=10,
        emb_dim=512,
        chaos_type="lorenz"
    )

    # 测试混沌X-Vector模型
    xvector_output = chaotic_xvector(x)
    print(f"混沌X-Vector输出形状: {xvector_output.shape}")

    # 创建混沌ECAPA-TDNN模型
    chaotic_ecapa = ChaoticECAPATDNN(
        input_dim=input_dim,
        channels=512,
        emb_dim=192,
        num_classes=10,
        chaos_type="lorenz"
    )

    # 测试混沌ECAPA-TDNN模型
    ecapa_output = chaotic_ecapa(x)
    print(f"混沌ECAPA-TDNN输出形状: {ecapa_output.shape}")

    # 创建混沌增强模型（变体B）
    model_b = ChaosEnhancedModel_VariantB(
        input_dim=input_dim,
        backbone_type="ecapa_tdnn",
        chaos_type="lorenz",
        emb_dim=192,
        num_classes=10
    )

    # 打印模型参数量
    print(f"变体B模型参数量: {count_parameters(model_b):,}")

    # 测试混沌增强模型
    output_b = model_b(x)
    print(f"变体B输出形状: {output_b.shape}")

    # 提取嵌入向量
    embeddings_b = model_b(x, extract_embedding=True)
    print(f"变体B嵌入向量形状: {embeddings_b.shape}")
