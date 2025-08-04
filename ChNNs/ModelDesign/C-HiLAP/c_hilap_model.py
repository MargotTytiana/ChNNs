import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp


# 配置参数
class Config:
    # 模型参数
    INPUT_DIM = 1  # 输入特征维度
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
    ATTENTION_HEADS = 8  # 注意力头数
    MAX_SEQ_LEN = 1000  # 最大序列长度，防止内存溢出


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
        self.coupling_weights = nn.Parameter(torch.randn(chaos_dim, chaos_dim) * 0.1)

    def lorenz_system(self, t, state, input_signal):
        """
        洛伦兹系统微分方程 - 批处理版本
        dx/dt = σ(y - x) + W_x·h_{t-1}
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        """
        # 拆分状态变量
        x = state[:, 0]  # [batch_size]
        y = state[:, 1]  # [batch_size]
        z = state[:, 2]  # [batch_size]

        # 计算混沌系统的导数
        dx_dt = self.sigma * (y - x) + torch.einsum('ij,bj->bi', self.coupling_weights[0:1], input_signal).squeeze(1)
        dy_dt = x * (self.rho - z) - y + torch.einsum('ij,bj->bi', self.coupling_weights[1:2], input_signal).squeeze(1)
        dz_dt = x * y - self.beta * z + torch.einsum('ij,bj->bi', self.coupling_weights[2:3], input_signal).squeeze(1)

        # 组合导数
        derivatives = torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
        return derivatives

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 混沌处理后的特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()

        # 限制序列长度防止内存溢出
        if seq_len > Config.MAX_SEQ_LEN:
            x = x[:, :Config.MAX_SEQ_LEN, :]
            seq_len = Config.MAX_SEQ_LEN

        outputs = []
        current_state = torch.zeros(batch_size, self.chaos_dim, device=x.device)

        for t in range(seq_len):
            # 获取当前时间步的输入
            input_t = x[:, t, :]

            # 将输入映射到混沌系统空间
            input_chaos = self.input_to_chaos(input_t)  # [batch_size, chaos_dim]

            # 使用RK4方法数值求解洛伦兹系统
            k1 = self.lorenz_system(0, current_state, input_chaos)
            k2 = self.lorenz_system(0, current_state + 0.5 * Config.CHAOS_STEP_SIZE * k1, input_chaos)
            k3 = self.lorenz_system(0, current_state + 0.5 * Config.CHAOS_STEP_SIZE * k2, input_chaos)
            k4 = self.lorenz_system(0, current_state + Config.CHAOS_STEP_SIZE * k3, input_chaos)

            # 更新状态
            current_state = current_state + (Config.CHAOS_STEP_SIZE / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 将混沌状态映射到输出空间
            output_t = self.chaos_to_output(current_state)
            outputs.append(output_t)

        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
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

        # 确保输入维度可以被头数整除
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = input_dim // num_heads
        self.threshold = threshold

        # 注意力投影
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # 输出投影
        self.out_proj = nn.Linear(input_dim, input_dim)

        # 分岔控制参数
        self.bifurcation_param = nn.Parameter(torch.tensor(0.5))

        # Dropout层
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 注意力加权后的特征 [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.size()

        # 限制序列长度防止内存溢出
        if seq_len > Config.MAX_SEQ_LEN:
            x = x[:, :Config.MAX_SEQ_LEN, :]
            seq_len = Config.MAX_SEQ_LEN

        # 线性投影
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 分割头 [batch_size, num_heads, seq_len, head_dim]
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
            scores = scores + 0.05 * torch.randn_like(scores, device=scores.device)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)

        # 输出投影
        output = self.out_proj(context)

        return output


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
        self.attractor_weights = nn.Parameter(torch.randn(output_dim, 3) * 0.1)  # 3维吸引子空间

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
        pooled = torch.sum(projected * weights, dim=1)  # [batch_size, output_dim]

        return pooled


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
        self.tdnn1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.bn_tdnn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu_tdnn1 = nn.PReLU()

        self.tdnn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.bn_tdnn2 = nn.BatchNorm1d(hidden_dim)
        self.prelu_tdnn2 = nn.PReLU()

        self.tdnn3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3, padding=3)
        self.bn_tdnn3 = nn.BatchNorm1d(hidden_dim)
        self.prelu_tdnn3 = nn.PReLU()

        # 奇异吸引子池化
        self.pooling = StrangeAttractorPooling(hidden_dim, embedding_dim)

        # 嵌入层
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.bn_fc1 = nn.BatchNorm1d(embedding_dim)
        self.prelu_fc1 = nn.PReLU()

        # 添加dropout
        self.dropout = nn.Dropout(0.1)

        # 分类器
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :return: 嵌入向量和分类结果
        """
        batch_size, seq_len, input_dim = x.size()

        # 限制序列长度防止内存溢出
        if seq_len > Config.MAX_SEQ_LEN:
            x = x[:, :Config.MAX_SEQ_LEN, :]
            seq_len = Config.MAX_SEQ_LEN

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
        embedding = self.dropout(embedding)

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
        batch_size, seq_len, dim = inputs.size()

        # 限制计算规模防止内存溢出
        if seq_len > 100:
            # 只使用部分序列计算相位同步损失
            step = seq_len // 100
            inputs = inputs[:, ::step, :]
            attractors = attractors[:, ::step, :]
            seq_len = inputs.size(1)

        # 简化版本：直接使用余弦相似度代替复杂的相位计算
        # 计算输入和吸引子之间的余弦相似度
        inputs_norm = F.normalize(inputs, p=2, dim=-1)
        attractors_norm = F.normalize(attractors, p=2, dim=-1)

        # 如果维度不匹配，通过线性变换调整吸引子维度
        if attractors.size(-1) != inputs.size(-1):
            # 简单重复或截断以匹配维度
            if attractors.size(-1) < inputs.size(-1):
                repeat_times = inputs.size(-1) // attractors.size(-1)
                remainder = inputs.size(-1) % attractors.size(-1)
                attractors_expanded = attractors.repeat(1, 1, repeat_times)
                if remainder > 0:
                    attractors_expanded = torch.cat([attractors_expanded, attractors[:, :, :remainder]], dim=-1)
                attractors_norm = F.normalize(attractors_expanded, p=2, dim=-1)
            else:
                attractors_norm = F.normalize(attractors[:, :, :inputs.size(-1)], p=2, dim=-1)

        # 计算余弦相似度
        similarity = torch.sum(inputs_norm * attractors_norm, dim=-1)

        # 相位同步损失：1 - 平均余弦相似度
        loss = 1.0 - torch.mean(similarity)

        return loss


# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = CHiLAPModel()

    print("模型结构:")
    print(model)

    # 生成随机输入（使用较小的序列长度进行测试）
    batch_size = 2
    seq_len = 100  # 减少序列长度
    input_dim = Config.INPUT_DIM
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"\n测试前向传播:")
    print(f"输入形状: {x.shape}")

    # 前向传播
    try:
        embedding, logits = model(x)
        print(f"嵌入向量形状: {embedding.shape}")
        print(f"分类输出形状: {logits.shape}")
        print("前向传播成功!")
    except Exception as e:
        print(f"前向传播错误: {e}")

    # 测试混沌振荡器模块
    print(f"\n测试混沌振荡器模块:")
    try:
        oscillator = LorenzOscillator(input_dim, Config.HIDDEN_DIM)
        chaos_output = oscillator(x)
        print(f"混沌输出形状: {chaos_output.shape}")
    except Exception as e:
        print(f"混沌振荡器错误: {e}")

    # 测试分岔注意力模块
    print(f"\n测试分岔注意力模块:")
    try:
        attention = BifurcationAttention(Config.HIDDEN_DIM)
        test_input = torch.randn(batch_size, seq_len, Config.HIDDEN_DIM)
        attention_output = attention(test_input)
        print(f"注意力输出形状: {attention_output.shape}")
    except Exception as e:
        print(f"分岔注意力错误: {e}")

    # 测试奇异吸引子池化
    print(f"\n测试奇异吸引子池化:")
    try:
        pooling = StrangeAttractorPooling(Config.HIDDEN_DIM, Config.EMBEDDING_DIM)
        test_input = torch.randn(batch_size, seq_len, Config.HIDDEN_DIM)
        pooled_output = pooling(test_input)
        print(f"池化输出形状: {pooled_output.shape}")
    except Exception as e:
        print(f"池化错误: {e}")

    # 测试相位同步损失
    print(f"\n测试相位同步损失:")
    try:
        phase_loss = PhaseSynchronizationLoss()
        attractors = torch.randn(batch_size, seq_len, 3)
        loss = phase_loss(x, attractors)
        print(f"相位同步损失: {loss.item()}")
    except Exception as e:
        print(f"相位同步损失错误: {e}")