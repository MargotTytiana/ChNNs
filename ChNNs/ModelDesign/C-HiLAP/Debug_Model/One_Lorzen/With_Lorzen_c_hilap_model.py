import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 配置参数
class Config:
    # 模型参数
    INPUT_DIM = 1  # 输入特征维度
    HIDDEN_DIM = 256  # 减少隐藏层维度
    EMBEDDING_DIM = 128  # 减少嵌入层维度

    # 复杂混沌模块参数 (从c_hilap_model.py导入)
    CHAOS_DIM = 3  # 混沌系统维度
    LORENZ_SIGMA = 10.0  # 洛伦兹系统参数
    LORENZ_RHO = 28.0  # 洛伦兹系统参数
    LORENZ_BETA = 8.0 / 3.0  # 洛伦兹系统参数
    CHAOS_STEP_SIZE = 0.01  # 混沌系统积分步长
    CHAOS_TIME_STEPS = 10  # 混沌演化时间步数

    # 注意力参数
    ATTENTION_HEADS = 4  # 减少注意力头数
    MAX_SEQ_LEN = 16000  # 减少最大序列长度（1秒音频）


# 从c_hilap_model.py导入的复杂混沌模块
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


# 简化注意力机制
class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        """
        简化的注意力机制
        :param input_dim: 输入维度
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, channels, seq_len]
        :return: 注意力加权后的特征 [batch_size, channels, seq_len]
        """
        # 计算注意力权重 [batch_size, 1, seq_len]
        attn_weights = self.attention(x)

        # 应用注意力权重
        return x * attn_weights


# 统计池化层
class StatisticalPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, channels, seq_len]
        :return: 池化后的特征 [batch_size, channels*2]
        """
        # 计算均值和标准差
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)

        # 拼接均值和标准差
        return torch.cat((mean, std), dim=1)


# 完整的C-HiLAP模型（简化版+复杂混沌模块）
class CHiLAPModel(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM,
                 embedding_dim=Config.EMBEDDING_DIM, num_classes=None):
        """
        混沌层次吸引子传播(C-HiLAP)模型 - 简化版+复杂混沌模块
        """
        super().__init__()

        # 若未传入num_classes，可设置一个默认值（但实际使用时必须从数据集获取后传入）
        if num_classes is None:
            raise ValueError("必须指定num_classes（说话人数量），请从数据集获取后传入")

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        # 复杂混沌模块 (替换简化版本)
        self.chaos_layer = LorenzOscillator(hidden_dim, hidden_dim)

        # 注意力层
        self.attention = SimpleAttention(hidden_dim)

        # TDNN层
        self.tdnn_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        # 池化层
        self.pooling = StatisticalPooling()

        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),  # 统计池化输出channels*2
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(0.2)
        )

        # 分类器
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, channels, seq_len] 或 [batch_size, seq_len, channels]
        :return: 嵌入向量和分类结果
        """
        # 检查输入维度并转换为正确的格式 [batch_size, channels, seq_len]
        if x.dim() == 3:
            # 如果是 [batch_size, seq_len, channels] 格式
            if x.size(1) > x.size(2):  # 序列长度应该大于通道数
                x = x.permute(0, 2, 1)  # 转换为 [batch_size, channels, seq_len]

        # 限制序列长度防止内存溢出
        seq_len = x.size(2)
        if seq_len > Config.MAX_SEQ_LEN:
            x = x[:, :, :Config.MAX_SEQ_LEN]

        # 特征提取
        x = self.feature_extractor(x)

        # 转置维度以适应复杂混沌模块 [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)

        # 混沌处理 (复杂模块)
        x = self.chaos_layer(x)

        # 转置回原始维度 [batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)

        # 注意力加权
        x = self.attention(x)

        # TDNN处理
        x = self.tdnn_block(x)

        # 池化
        x = self.pooling(x)

        # 嵌入向量
        embedding = self.embedding(x)

        # 分类
        logits = self.classifier(embedding)

        return embedding, logits


# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = CHiLAPModel(num_classes=10)  # 添加num_classes参数

    print("模型结构:")
    print(model)

    # 生成随机输入（使用较小的序列长度进行测试）
    batch_size = 2
    seq_len = Config.MAX_SEQ_LEN
    # 正确的输入格式：[batch_size, channels, seq_len]
    x = torch.randn(batch_size, 1, seq_len)

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
        import traceback

        traceback.print_exc()
