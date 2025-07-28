import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticsPooling(nn.Module):
    """统计池化层：计算时域上的均值和标准差"""

    def forward(self, x):
        mean = torch.mean(x, dim=2)  # 沿时间维度求均值
        std = torch.std(x, dim=2)  # 沿时间维度求标准差
        return torch.cat((mean, std), dim=1)  # 拼接均值和标准差


class XVector(nn.Module):
    def __init__(self, input_dim=39, emb_dim=512, num_classes=1000):
        """
        input_dim: 输入特征的维度（MFCC+Delta特征）
        emb_dim: 说话人嵌入的维度
        num_classes: 说话人的数量
        """
        super(XVector, self).__init__()

        # 帧级别的特征提取（TDNN层）
        self.tdnn = nn.Sequential(
            nn.Conv1d(input_dim, 512, 5, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 1500, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1500)
        )

        self.pooling = StatisticsPooling()

        # 段级别的特征提取
        self.segment_layer = nn.Sequential(
            nn.Linear(3000, emb_dim),  # 统计池化后为1500*2=3000
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim)
        )

        # 分类层
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # 输入x的形状: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # 转换为(batch_size, input_dim, seq_len)
        x = self.tdnn(x)
        x = self.pooling(x)
        x = self.segment_layer(x)
        return self.classifier(x), x  # 返回分类结果和嵌入向量