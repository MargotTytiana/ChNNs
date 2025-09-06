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

    # 简化混沌模块参数
    CHAOS_DIM = 64  # 减少混沌系统维度
    CHAOS_TIME_STEPS = 5  # 减少混沌演化时间步数

    # 注意力参数
    ATTENTION_HEADS = 4  # 减少注意力头数
    MAX_SEQ_LEN = 16000  # 减少最大序列长度（1秒音频）


# 简化混沌激励模块
class ChaoticStimulus(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        简化的混沌激励模块
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        """
        super().__init__()
        self.chaos_transform = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.PReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.PReLU()
        )

        # 混沌扰动参数
        self.chaos_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征 [batch_size, channels, seq_len]
        :return: 混沌处理后的特征 [batch_size, channels, seq_len]
        """
        # 常规特征变换
        transformed = self.chaos_transform(x)

        # 添加混沌扰动
        batch_size, channels, seq_len = transformed.size()
        if self.training:  # 仅在训练时添加混沌扰动
            # 生成与特征相同形状的混沌噪声
            chaos_noise = torch.randn_like(transformed) * self.chaos_factor
            # 应用非线性激活增强混沌特性
            chaos_noise = torch.tanh(chaos_noise)
            transformed = transformed + chaos_noise

        return transformed


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


# 完整的C-HiLAP模型（简化版）
class CHiLAPModel(nn.Module):
    def __init__(self, input_dim=Config.INPUT_DIM, hidden_dim=Config.HIDDEN_DIM,
                 embedding_dim=Config.EMBEDDING_DIM, num_classes=None):
        """
        混沌层次吸引子传播(C-HiLAP)模型 - 简化版
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

        # 混沌激励模块
        self.chaos_layer = ChaoticStimulus(hidden_dim, hidden_dim)

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

        # 混沌处理
        x = self.chaos_layer(x)

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
    model = CHiLAPModel()

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

