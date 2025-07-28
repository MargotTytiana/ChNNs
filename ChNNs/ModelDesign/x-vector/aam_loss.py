import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """加性角度间隔损失函数，增强类间差异"""

    def __init__(self, emb_dim, num_classes, margin=0.2, scale=30):
        super(AAMSoftmax, self).__init__()
        self.margin = margin
        self.scale = scale
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        # 归一化权重和嵌入向量
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        embedding_norm = F.normalize(embeddings, p=2, dim=1)
        cosine = F.linear(embedding_norm, weight_norm)

        # 加入角度边界
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        cosine_m = cosine - one_hot * self.margin

        # 缩放
        logits = self.scale * (cosine_m if self.training else cosine)
        loss = F.cross_entropy(logits, labels)

        return loss