import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp


# 配置参数
class Config:
    # 模型参数
    INPUT_DIM = 80  # 输入特征维度
    HIDDEN_DIM = 512  # 隐藏层维度
    EMBEDDING_DIM = 192  # 嵌入层维度
    NUM_CLASSES = 1211  # VoxCeleb1说话人数量

    # 混沌模块参数
    CHAOS_DIM = 3  # 混沌系统维度
    LORENZ_SIGMA = 10.0  # 洛伦兹系统参数
    LORENZ_RHO = 28.0  # 洛伦兹系统参数
    LORENZ_BETA = 8.0 / 3.0  # 洛伦兹系统参数
    CHAOS_STEP_SIZE = 0.01  # 混沌系统积分步长
    CHAOS_TIME_STEPS = 10  # 混沌演化时间步数

    # 分岔注意力参数
    BIFURCATION_THRESHOLD = 0.5  # 分岔阈值
    ATTENTION_HEADS = 4  # 注意力头数


# 洛伦兹混沌振荡器模块
class LorenzOscillator(nn.Module):
    def __init__(self, input_dim, hidden_dim, chaos_dim=Config.CHAOS_DIM,
                 sigma=Config.LORENZ_SIGMA, rho=Config.LORENZ_RHO, beta=Config.LORENZ_BETA):
        """
        洛伦兹混沌振荡器模块
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏维度
        :param chaos_dim: 混沌系统维度
        :param sigma, rho, beta: 洛伦兹系统参数
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chaos_dim = chaos_dim
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        # 输入到混沌系统的映射
        self.input_to_chaos = nn.Linear(input_dim, chaos_dim)

        # 混沌状态到输出的映射
        self.chaos_to_output = nn.Linear(chaos_dim, hidden_dim)

        # 自适应耦合权重
        self.coupling_weights = nn.Parameter(torch.randn(chaos_dim, chaos_dim))

    def lorenz_system(self, t, state, input_signal):
        """
        洛伦兹系统微分方程
        dx/dt = σ(y - x) + W_x·h_{t-1}
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        """
        x, y, z = state

        # 计算混沌系统的导数
        dx_dt = self.sigma * (y - x) + torch.matmul(self.coupling_weights[0], input_signal)
        dy_dt = x * (self.rho - z) - y + torch.matmul(self.coupling_weights[1], input_signal)
        dz_dt = x * y - self.beta * z + torch.matmul(self.coupling_weights[2], input_signal)

        return [dx_dt, dy_dt, dz_dt]

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 混沌处理后的特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        outputs = []
        current_state = torch.zeros(batch_size, self.chaos_dim).to(x.device)

        for t in range(seq_len):
            # 获取当前时间步的输入
            input_t = x[:, t, :]

            # 将输入映射到混沌系统空间
            input_chaos = self.input_to_chaos(input_t)

            # 使用RK4方法数值求解洛伦兹系统
            for _ in range(Config.CHAOS_TIME_STEPS):
                # 计算k1
                k1 = self.lorenz_system(0, current_state, input_chaos)
                k1 = torch.stack(k1, dim=1)

                # 计算k2
                k2_state = current_state + 0.5 * Config.CHAOS_STEP_SIZE * k1
                k2 = self.lorenz_system(0, k2_state, input_chaos)
                k2 = torch.stack(k2, dim=1)

                # 计算k3
                k3_state = current_state + 0.5 * Config.CHAOS_STEP_SIZE * k2
                k3 = self.lorenz_system(0, k3_state, input_chaos)
                k3 = torch.stack(k3, dim=1)

                # 计算k4
                k4_state = current_state + Config.CHAOS_STEP_SIZE * k3
                k4 = self.lorenz_system(0, k4_state, input_chaos)
                k4 = torch.stack(k4, dim=1)

                # 更新状态
                current_state = current_state + (Config.CHAOS_STEP_SIZE / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 将混沌状态映射到输出空间
            output_t = self.chaos_to_output(current_state)
            outputs.append(output_t)

        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)
        return outputs


# 分岔注意力机制
class BifurcationAttention(nn.Module):
    def __init__(self, input_dim, num_heads=Config.ATTENTION_HEADS,
                 threshold=Config.BIFURCATION_THRESHOLD):
        """
        分岔注意力机制
        :param input_dim: 输入维度
        :param num_heads: 注意力头数
        :param threshold: 分岔阈值
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.threshold = threshold

        # 注意力投影
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # 分岔控制参数
        self.bifurcation_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 注意力加权后的特征 [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.size()

        # 线性投影
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 分割头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用分岔控制
        # 分岔函数: f(r) = r * sin(π * r)
        r = torch.sigmoid(self.bifurcation_param)
        bifurcation_factor = r * torch.sin(np.pi * r)

        # 当分岔因子接近阈值时，系统动态变化加剧
        if torch.abs(bifurcation_factor - self.threshold) < 0.1:
            # 添加随机扰动模拟混沌行为
            scores = scores + 0.05 * torch.randn_like(scores)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)

        return context


# 奇异吸引子池化层
class StrangeAttractorPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        奇异吸引子池化层 - 将时序特征映射到吸引子空间
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 投影层
        self.projection = nn.Linear(input_dim, output_dim)

        # 吸引子参数
        self.attractor_weights = nn.Parameter(torch.randn(output_dim, 3))  # 3维吸引子空间

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 池化后的特征 [batch_size, output_dim]
        """
        # 投影到吸引子空间
        projected = self.projection(x)  # [batch_size, seq_len, output_dim]

        # 计算每个维度与吸引子的距离
        batch_size, seq_len, _ = projected.size()

        # 扩展吸引子参数以匹配批次大小
        attractors = self.attractor_weights.unsqueeze(0).expand(batch_size, -1, -1)

        # 计算每个时间步在吸引子空间中的位置
        attractor_positions = torch.matmul(projected, attractors)  # [batch_size, seq_len, 3]

        # 计算吸引子空间中的能量分布（使用欧氏距离）
        center = torch.mean(attractor_positions, dim=1, keepdim=True)
        distances = torch.norm(attractor_positions - center, dim=2)  # [batch_size, seq_len]

        # 使用距离作为权重进行加权池化
        weights = F.softmax(-distances, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]
        pooled = torch.sum(x * weights, dim=1)  # [batch_size, input_dim]

        # 最终投影到嵌入空间
        return self.projection(pooled)


# 完整的C-HiLAP模型
class CHiLAPModel(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM,
                 embedding_dim=Config.EMBEDDING_DIM, num_classes=Config.NUM_CLASSES):
        """
        混沌层次吸引子传播(C-HiLAP)模型
        """
        super().__init__()

        # 特征提取层
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu1 = nn.PReLU()

        # 混沌嵌入层
        self.chaos_layer = LorenzOscillator(hidden_dim, hidden_dim)

        # 分岔注意力层
        self.attention = BifurcationAttention(hidden_dim)

        # TDNN层
        self.tdnn1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu_tdnn1 = nn.PReLU()

        self.tdnn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu_tdnn2 = nn.PReLU()

        self.tdnn3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(hidden_dim)
        self.prelu_tdnn3 = nn.PReLU()

        # 奇异吸引子池化
        self.pooling = StrangeAttractorPooling(hidden_dim, embedding_dim)

        # 嵌入层
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.bn_fc1 = nn.BatchNorm1d(embedding_dim)
        self.prelu_fc1 = nn.PReLU()

        # 分类器
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 嵌入向量和分类结果
        """
        # 交换维度以适应卷积层 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)

        # 特征提取
        x = self.prelu1(self.bn1(self.conv1(x)))

        # 交换维度以适应混沌层 [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)

        # 混沌处理
        x_chaos = self.chaos_layer(x)

        # 分岔注意力
        x_attended = self.attention(x_chaos)

        # 交换维度以适应TDNN层 [batch_size, hidden_dim, seq_len]
        x_attended = x_attended.transpose(1, 2)

        # TDNN层
        x = self.prelu_tdnn1(self.bn_tdnn1(self.tdnn1(x_attended)))
        x = self.prelu_tdnn2(self.bn_tdnn2(self.tdnn2(x)))
        x = self.prelu_tdnn3(self.bn_tdnn3(self.tdnn3(x)))

        # 交换维度以适应池化层 [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)

        # 奇异吸引子池化
        embedding = self.pooling(x)

        # 嵌入层
        embedding = self.prelu_fc1(self.bn_fc1(self.fc1(embedding)))

        # 分类
        logits = self.classifier(embedding)

        return embedding, logits


# 相位同步损失函数
class PhaseSynchronizationLoss(nn.Module):
    def __init__(self):
        """计算输入信号与混沌吸引子之间的相位同步损失"""
        super().__init__()

    def forward(self, inputs, attractors):
        """
        计算相位同步损失
        :param inputs: 输入特征 [batch_size, seq_len, dim]
        :param attractors: 吸引子状态 [batch_size, seq_len, 3] (3维吸引子空间)
        :return: 相位同步损失
        """
        # 计算输入信号的相位（使用希尔伯特变换）
        batch_size, seq_len, dim = inputs.size()

        # 简化版本：使用FFT近似希尔伯特变换
        inputs_fft = torch.fft.rfft(inputs, dim=1)

        # 创建希尔伯特变换滤波器
        h = torch.zeros(seq_len, device=inputs.device)
        if seq_len % 2 == 0:
            h[0] = h[seq_len // 2] = 1
            h[1:seq_len // 2] = 2
        else:
            h[0] = 1
            h[1:(seq_len + 1) // 2] = 2

        h = h.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, dim)
        inputs_hilbert_fft = inputs_fft * h[:, :seq_len // 2 + 1, :]
        inputs_hilbert = torch.fft.irfft(inputs_hilbert_fft, dim=1)

        # 计算相位
        inputs_phase = torch.atan2(inputs_hilbert, inputs)

        # 计算吸引子的相位（简化：使用角度）
        attractor_phase = torch.atan2(attractors[:, :, 1], attractors[:, :, 0])
        attractor_phase = attractor_phase.unsqueeze(2).expand(-1, -1, dim)

        # 计算相位差
        phase_diff = torch.abs(inputs_phase - attractor_phase)

        # 归一化到[0, π]
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)

        # 计算损失（相位同步程度）
        loss = torch.mean(phase_diff)

        return loss


# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = CHiLAPModel()

    # 生成随机输入
    batch_size = 4
    seq_len = 100
    input_dim = Config.INPUT_DIM
    x = torch.randn(batch_size, seq_len, input_dim)

    # 前向传播
    embedding, logits = model(x)

    print(f"输入形状: {x.shape}")
    print(f"嵌入向量形状: {embedding.shape}")
    print(f"分类输出形状: {logits.shape}")

    # 测试相位同步损失
    phase_loss = PhaseSynchronizationLoss()
    attractors = torch.randn(batch_size, seq_len, 3)
    loss = phase_loss(x, attractors)
    print(f"相位同步损失: {loss.item()}")