import torch
import torch.nn as nn
import torch.nn.functional as F


class ChaoticLSTM(nn.Module):
    """
    混沌长短期记忆网络 (Chaotic Long Short-Term Memory Network)
    用于说话人识别任务的持续学习模型，结合混沌动力学增强时序建模能力

    核心思想：
    1. 使用CNN提取音频的局部特征
    2. 通过LSTM捕获长期时序依赖
    3. 引入混沌激活函数增强模型动态表达能力
    4. 支持动态扩展以适应新说话人

    混沌动力学原理：
    - 使用logistic映射作为混沌激活函数：x_{n+1} = r * x_n * (1 - x_n)
    - 其中r是混沌参数(3.57-4.0时系统处于混沌状态)
    - 混沌系统对初始条件敏感，能够增强模型对语音细微差异的捕捉能力
    """

    def __init__(self, input_dim, hidden_dim, num_speakers, chaos_factor=3.9):
        """
        初始化混沌LSTM模型

        参数:
        input_dim: 输入特征维度 (MFCC特征数量)
        hidden_dim: LSTM隐藏层维度
        num_speakers: 说话人数量
        chaos_factor: 混沌参数 (默认3.9处于混沌边缘)
        """
        super(ChaoticLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.chaos_factor = chaos_factor

        # 1. 特征提取层: 1D卷积网络处理原始音频
        # 输入形状: (batch, 1, seq_len) -> 输出形状: (batch, 128, reduced_seq_len)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=160, stride=80, padding=80),  # 10ms窗口@16kHz
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),  # 0.5ms窗口
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)  # 总降采样4倍
        )

        # 添加适配层解决维度不匹配问题
        # self.adapt_layer = nn.Linear(128, input_dim)

        # 2. 混沌LSTM层: 处理时序特征
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 输入特征维度
            hidden_size=hidden_dim,  # LSTM单元数
            batch_first=True,  # 输入形状为(batch, seq, features)
            bidirectional=False  # 单方向LSTM
        )

        # 3. 混沌参数: 可学习的混沌权重矩阵
        # 每个隐藏单元有独立的混沌参数，增强表达能力
        self.chaos_weight = nn.Parameter(
            torch.randn(hidden_dim) * 0.1 + chaos_factor,  # 初始值围绕混沌因子
            requires_grad=True  # 允许梯度更新
        )

        # 4. 说话人分类层: 全连接层映射到说话人空间
        self.classifier = nn.Linear(hidden_dim, num_speakers)

        # 5. 持续学习相关参数: 用于弹性权重巩固(EWC)
        self.ewc_importance = None  # 存储参数重要性(Fisher信息)
        self.ewc_params = None  # 存储旧任务参数快照

        self.dropout = nn.Dropout(0.5)

    def chaotic_activation(self, x):
        # 使用更稳定的激活方式
        x = torch.tanh(x)  # 先压缩到[-1,1]范围
        return self.chaos_weight * x * (1 - torch.abs(x))  # 更稳定的混沌形式

    def adjust_chaos_during_training(self, epoch, total_epochs):
        # 更平滑的衰减
        decay_factor = 0.9
        new_factor = self.chaos_factor * decay_factor ** (epoch / total_epochs)
        self.chaos_factor = max(3.0, new_factor)  # 避免过低的混沌因子

        # 更稳定的参数更新
        self.chaos_weight.data = (
                self.chaos_weight.data * 0.95 +
                torch.randn_like(self.chaos_weight) * 0.05 * self.chaos_factor
        )

    def forward(self, x):
        """
        前向传播过程

        参数:
        x: 输入音频波形 (batch_size, seq_len)

        返回:
        logits: 说话人分类logits (batch_size, num_speakers)
        """
        # 输入形状: (batch_size, seq_len)
        x = x.unsqueeze(1)  # 添加通道维度: (batch_size, 1, seq_len)

        # 1. 特征提取
        # 输入: (batch, 1, seq_len) -> 输出: (batch, 128, seq_len/8)
        features = self.feature_extractor(x)

        # 调整维度: (batch, channels, reduced_seq) -> (batch, reduced_seq, channels)
        features = features.permute(0, 2, 1)

        # 使用适配层解决维度不匹配问题
        # features = self.adapt_layer(features)  # (batch, reduced_seq, input_dim)

        # 2. LSTM时序处理
        # 输入: (batch, reduced_seq, input_dim) -> 输出: (batch, reduced_seq, hidden_dim)
        lstm_out, _ = self.lstm(features)

        # 添加Dropout
        lstm_out = self.dropout(lstm_out)

        # 3. 应用混沌动力学
        # 使用sigmoid将输出压缩到[0,1]区间，满足logistic映射要求
        # lstm_out_normalized = torch.sigmoid(lstm_out)
        chaotic_out = self.chaotic_activation(lstm_out)

        # 4. 取序列最后一个时间步的输出作为整个序列的表示
        # 形状: (batch_size, hidden_dim)
        last_out = chaotic_out[:, -1, :]

        # 5. 说话人分类
        logits = self.classifier(last_out)
        return logits

    def reset_chaos(self, new_factor=None):
        """
        重置混沌参数，维持模型的混沌特性

        参数:
        new_factor: 可选的新混沌因子，不提供则使用初始化值

        为什么需要定期重置:
        - 混沌系统长期演化可能导致模式固化
        - 重置参数保持系统在混沌边缘状态
        - 增强模型探索新特征的能力
        """
        if new_factor is not None:
            self.chaos_factor = new_factor

        # 在初始值附近添加随机扰动
        self.chaos_weight.data = (
                torch.randn(self.hidden_dim, device=self.chaos_weight.device) * 0.1  # 随机扰动
                + self.chaos_factor  # 基础混沌因子
        )

    def expand_for_new_speakers(self, new_speaker_count):
        """
        扩展分类层以容纳新说话人 (持续学习关键操作)

        参数:
        new_speaker_count: 新增说话人数量

        操作:
        1. 保存旧分类层权重
        2. 创建新的更大的分类层
        3. 初始化新权重并保留旧权重

        注意:
        - 旧说话人的权重保持不变
        - 新说话人的权重随机初始化
        - 不破坏已有知识表示
        """
        # 保存旧分类层权重
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        old_out_features = self.classifier.out_features

        # 创建新分类层 (扩展输出维度)
        self.classifier = nn.Linear(
            self.hidden_dim,
            old_out_features + new_speaker_count
        )

        # 迁移旧权重
        self.classifier.weight.data[:old_out_features] = old_weight
        self.classifier.bias.data[:old_out_features] = old_bias

        # 初始化新说话人的权重 (保持梯度)
        nn.init.kaiming_normal_(
            self.classifier.weight.data[old_out_features:],
            nonlinearity='relu'
        )
        nn.init.constant_(self.classifier.bias.data[old_out_features:], 0.1)

        # 将新分类层移到原设备
        device = old_weight.device
        self.classifier = self.classifier.to(device)

    def register_ewc_params(self, importance, params):
        """
        注册弹性权重巩固(EWC)所需参数
        用于持续学习中防止灾难性遗忘

        参数:
        importance: Fisher信息矩阵 (参数重要性)
        params: 旧任务参数快照
        """
        self.ewc_importance = importance
        self.ewc_params = params

    def get_ewc_loss(self):
        """
        计算EWC正则化损失
        惩罚对重要参数的改变

        返回:
        ewc_loss: EWC正则化项
        """
        if self.ewc_importance is None or self.ewc_params is None:
            return 0.0

        loss = 0.0
        for name, param in self.named_parameters():
            if name in self.ewc_importance:
                # 计算当前参数与旧参数的差异
                param_diff = param - self.ewc_params[name]
                # 加权平方差 (重要性权重)
                loss += (self.ewc_importance[name] * param_diff.pow(2)).sum()

        # 添加数值稳定性处理
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros_like(loss)
        return loss

    def adjust_chaos_during_training(self, epoch, total_epochs):
        """
        训练过程中动态调整混沌参数

        参数:
        epoch: 当前训练轮次
        total_epochs: 总训练轮次

        调整策略:
        - 初始阶段保持高混沌性 (探索)
        - 后期降低混沌性 (稳定)
        """
        # 线性衰减策略
        new_factor = self.chaos_factor * (1.0 - 0.8 * epoch / total_epochs)
        self.chaos_factor = max(3.7, new_factor)  # 保持混沌状态

        # 更新参数但不重置 (允许学习过程调整)
        self.chaos_weight.data = (
                self.chaos_weight.data * 0.9
                + torch.randn_like(self.chaos_weight) * 0.1 * self.chaos_factor
        )