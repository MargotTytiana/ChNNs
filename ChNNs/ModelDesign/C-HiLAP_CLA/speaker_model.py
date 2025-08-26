import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class ChaoticSpeakerEmbedding(nn.Module):
    """
    Speaker embedding model using chaotic features

    使用混沌特征的说话人嵌入模型
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int = 256,
            hidden_dims: List[int] = [512, 512],
            dropout_rate: float = 0.2
    ):
        """
        Initialize the chaotic speaker embedding model

        初始化混沌说话人嵌入模型

        Args:
            input_dim (int): Input feature dimension
                            输入特征维度
            embedding_dim (int): Output embedding dimension
                                输出嵌入维度
            hidden_dims (List[int]): List of hidden layer dimensions
                                    隐藏层维度列表
            dropout_rate (float): Dropout rate
                                 丢弃率
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build the network
        # 构建网络
        layers = []

        # Input layer
        # 输入层
        in_dim = input_dim

        # Hidden layers
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # Output embedding layer
        # 输出嵌入层
        layers.append(nn.Linear(in_dim, embedding_dim))
        layers.append(nn.BatchNorm1d(embedding_dim))

        self.embedding_network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding model

        嵌入模型的前向传播

        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
                             输入特征 [batch_size, input_dim]

        Returns:
            torch.Tensor: Speaker embeddings [batch_size, embedding_dim]
                         说话人嵌入 [batch_size, embedding_dim]
        """
        # Pass through the embedding network
        # 通过嵌入网络
        embeddings = self.embedding_network(x)

        # L2 normalize the embeddings
        # L2归一化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ChaoticSpeakerClassifier(nn.Module):
    """
    Speaker classifier using chaotic embeddings

    使用混沌嵌入的说话人分类器
    """

    def __init__(
            self,
            embedding_dim: int,
            num_speakers: int,
            use_angular_margin: bool = True,
            margin: float = 0.2,
            scale: float = 30.0
    ):
        """
        Initialize the speaker classifier

        初始化说话人分类器

        Args:
            embedding_dim (int): Input embedding dimension
                                输入嵌入维度
            num_speakers (int): Number of speakers to classify
                               要分类的说话人数量
            use_angular_margin (bool): Whether to use angular margin (AM-Softmax)
                                      是否使用角度间隔(AM-Softmax)
            margin (float): Margin for AM-Softmax
                           AM-Softmax的间隔
            scale (float): Scale factor for AM-Softmax
                          AM-Softmax的缩放因子
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.use_angular_margin = use_angular_margin
        self.margin = margin
        self.scale = scale

        # Weight for the final classification layer
        # 最终分类层的权重
        self.weight = nn.Parameter(torch.Tensor(num_speakers, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the classifier

        分类器的前向传播

        Args:
            embeddings (torch.Tensor): Speaker embeddings [batch_size, embedding_dim]
                                      说话人嵌入 [batch_size, embedding_dim]
            labels (torch.Tensor, optional): Speaker labels [batch_size]
                                            说话人标签 [batch_size]

        Returns:
            torch.Tensor: Logits [batch_size, num_speakers]
                         逻辑值 [batch_size, num_speakers]
        """
        # Normalize weight
        # 归一化权重
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        # Calculate cosine similarity
        # 计算余弦相似度
        cosine = F.linear(embeddings, normalized_weight)

        if self.use_angular_margin and self.training and labels is not None:
            # Apply angular margin penalty
            # 应用角度间隔惩罚
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

            # Calculate margin penalty
            # 计算间隔惩罚
            margin_cosine = cosine - one_hot * self.margin

            # Apply scale
            # 应用缩放
            logits = self.scale * margin_cosine
        else:
            # Apply scale
            # 应用缩放
            logits = self.scale * cosine

        return logits


class ChaoticSpeakerVerifier(nn.Module):
    """
    Speaker verification model using chaotic embeddings

    使用混沌嵌入的说话人验证模型
    """

    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int = 128,
            dropout_rate: float = 0.1
    ):
        """
        Initialize the speaker verification model

        初始化说话人验证模型

        Args:
            embedding_dim (int): Input embedding dimension
                                输入嵌入维度
            hidden_dim (int): Hidden layer dimension
                             隐藏层维度
            dropout_rate (float): Dropout rate
                                 丢弃率
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Network for similarity scoring
        # 相似度评分网络
        self.similarity_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the verification model

        验证模型的前向传播

        Args:
            embedding1 (torch.Tensor): First speaker embedding [batch_size, embedding_dim]
                                      第一个说话人嵌入 [batch_size, embedding_dim]
            embedding2 (torch.Tensor): Second speaker embedding [batch_size, embedding_dim]
                                      第二个说话人嵌入 [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Similarity scores [batch_size, 1]
                         相似度分数 [batch_size, 1]
        """
        # Concatenate embeddings
        # 连接嵌入
        concat_embedding = torch.cat([embedding1, embedding2], dim=1)

        # Calculate similarity score
        # 计算相似度分数
        similarity = self.similarity_network(concat_embedding)

        return similarity


class ChaoticSpeakerRecognitionSystem(nn.Module):
    """
    Complete chaotic speaker recognition system

    完整的混沌说话人识别系统
    """

    def __init__(
            self,
            input_dim: int = 48000,  # 3 seconds at 16kHz
            chaotic_feature_dim: int = 64,
            chaotic_dim: int = 3,
            trajectory_points: int = 100,
            embedding_dim: int = 256,
            num_speakers: int = 100,
            use_chaotic_embedding: bool = True,
            use_attractor_pooling: bool = True,
            system_type: str = 'lorenz'
    ):
        """
        Initialize the complete speaker recognition system

        初始化完整的说话人识别系统

        Args:
            input_dim (int): Input audio dimension (default: 48000 for 3s at 16kHz)
                            输入音频维度（默认：48000，对应3秒16kHz音频）
            chaotic_feature_dim (int): Dimension of chaotic features
                                      混沌特征的维度
            chaotic_dim (int): Dimension of chaotic system
                              混沌系统的维度
            trajectory_points (int): Number of points in trajectory
                                    轨迹中的点数
            embedding_dim (int): Speaker embedding dimension
                                说话人嵌入维度
            num_speakers (int): Number of speakers
                               说话人数量
            use_chaotic_embedding (bool): Whether to use chaotic embedding
                                         是否使用混沌嵌入
            use_attractor_pooling (bool): Whether to use attractor pooling
                                         是否使用吸引子池化
            system_type (str): Type of chaotic system ('lorenz', 'rossler', 'chua')
                              混沌系统类型（'lorenz', 'rossler', 'chua'）
        """
        super().__init__()
        self.input_dim = input_dim
        self.chaotic_feature_dim = chaotic_feature_dim
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.use_chaotic_embedding = use_chaotic_embedding
        self.use_attractor_pooling = use_attractor_pooling

        # Audio feature extractor (CNN-based)
        # 音频特征提取器（基于CNN）
        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, chaotic_feature_dim)
        )

        # Import required modules only when needed
        # 仅在需要时导入所需模块
        if use_chaotic_embedding:
            from chaotic_embedding import DifferentiableChaoticEmbedding
            self.chaotic_embedding = DifferentiableChaoticEmbedding(
                input_dim=chaotic_feature_dim,
                chaotic_dim=chaotic_dim,
                trajectory_points=trajectory_points,
                system_type=system_type
            )

        if use_attractor_pooling:
            from attractor_pooling import DifferentiableAttractorPooling
            self.attractor_pooling = DifferentiableAttractorPooling(
                chaotic_dim=chaotic_dim,
                output_dim=chaotic_feature_dim
            )

        # Speaker embedding network
        # 说话人嵌入网络
        self.speaker_embedding = ChaoticSpeakerEmbedding(
            input_dim=chaotic_feature_dim,
            embedding_dim=embedding_dim
        )

        # Speaker classifier
        # 说话人分类器
        self.speaker_classifier = ChaoticSpeakerClassifier(
            embedding_dim=embedding_dim,
            num_speakers=num_speakers
        )

        # Speaker verifier
        # 说话人验证器
        self.speaker_verifier = ChaoticSpeakerVerifier(
            embedding_dim=embedding_dim
        )

    def forward(
            self,
            audio: torch.Tensor,  # Changed from features to audio
            labels: Optional[torch.Tensor] = None,
            mode: str = 'identification'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the speaker recognition system

        说话人识别系统的前向传播

        Args:
            audio (torch.Tensor): Input audio [batch_size, audio_length]
                                  输入音频 [batch_size, audio_length]
            labels (torch.Tensor, optional): Speaker labels [batch_size]
                                            说话人标签 [batch_size]
            mode (str): Operation mode ('identification', 'verification', 'embedding')
                       操作模式（'identification'识别, 'verification'验证, 'embedding'嵌入）

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing outputs
                                    包含输出的字典
        """
        batch_size = audio.shape[0]

        # Add channel dimension for CNN processing
        # 为CNN处理添加通道维度
        audio = audio.unsqueeze(1)  # [batch_size, 1, audio_length]

        # Extract features from raw audio
        # 从原始音频中提取特征
        features = self.audio_feature_extractor(audio)  # [batch_size, chaotic_feature_dim]

        # Process features through chaotic embedding if enabled
        # 如果启用，则通过混沌嵌入处理特征
        if self.use_chaotic_embedding:
            trajectories = self.chaotic_embedding(features)

            # Process trajectories through attractor pooling if enabled
            # 如果启用，则通过吸引子池化处理轨迹
            if self.use_attractor_pooling:
                features = self.attractor_pooling(trajectories)

        # Generate speaker embeddings
        # 生成说话人嵌入
        embeddings = self.speaker_embedding(features)

        outputs = {'embeddings': embeddings}

        # Mode-specific operations
        # 特定模式的操作
        if mode == 'identification':
            # Speaker identification
            # 说话人识别
            logits = self.speaker_classifier(embeddings, labels if self.training else None)
            outputs['logits'] = logits

            # Calculate predictions
            # 计算预测结果
            _, predictions = torch.max(logits, dim=1)
            outputs['predictions'] = predictions

        elif mode == 'verification':
            # For verification, we need pairs of embeddings
            # 对于验证，我们需要嵌入对
            if batch_size % 2 != 0:
                raise ValueError("Batch size must be even for verification mode")

            # Split embeddings into pairs
            # 将嵌入分成对
            embedding1 = embeddings[0:batch_size:2]
            embedding2 = embeddings[1:batch_size:2]

            # Calculate similarity scores
            # 计算相似度分数
            similarity = self.speaker_verifier(embedding1, embedding2)
            outputs['similarity'] = similarity

        return outputs


class SpeakerRecognitionLoss(nn.Module):
    """
    Combined loss function for speaker recognition

    说话人识别的组合损失函数
    """

    def __init__(
            self,
            ce_weight: float = 1.0,
            triplet_weight: float = 0.1,
            margin: float = 0.2
    ):
        """
        Initialize the combined loss function

        初始化组合损失函数

        Args:
            ce_weight (float): Weight for cross-entropy loss
                              交叉熵损失的权重
            triplet_weight (float): Weight for triplet loss
                                   三元组损失的权重
            margin (float): Margin for triplet loss
                           三元组损失的间隔
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.margin = margin

        # Cross-entropy loss
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the combined loss

        计算组合损失

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
                                             模型输出
            labels (torch.Tensor): Speaker labels
                                  说话人标签

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss values
                                    包含损失值的字典
        """
        losses = {}

        # Cross-entropy loss
        # 交叉熵损失
        if 'logits' in outputs:
            ce_loss = self.ce_loss(outputs['logits'], labels)
            losses['ce_loss'] = ce_loss

        # Triplet loss
        # 三元组损失
        if self.triplet_weight > 0 and 'embeddings' in outputs:
            triplet_loss = self.calculate_triplet_loss(outputs['embeddings'], labels)
            losses['triplet_loss'] = triplet_loss

        # Calculate total loss
        # 计算总损失
        total_loss = 0
        if 'ce_loss' in losses:
            total_loss += self.ce_weight * losses['ce_loss']
        if 'triplet_loss' in losses:
            total_loss += self.triplet_weight * losses['triplet_loss']

        losses['total_loss'] = total_loss

        return losses

    def calculate_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate triplet loss with online mining

        使用在线挖掘计算三元组损失

        Args:
            embeddings (torch.Tensor): Speaker embeddings
                                      说话人嵌入
            labels (torch.Tensor): Speaker labels
                                  说话人标签

        Returns:
            torch.Tensor: Triplet loss value
                         三元组损失值
        """
        batch_size = embeddings.shape[0]

        # Calculate pairwise distances
        # 计算成对距离
        dist_matrix = torch.cdist(embeddings, embeddings)

        # For each anchor, find the hardest positive and negative
        # 对于每个锚点，找到最难的正样本和负样本
        triplet_loss = torch.tensor(0.0, device=embeddings.device)
        valid_triplets = 0

        for i in range(batch_size):
            # Get distances from anchor to all other samples
            # 获取从锚点到所有其他样本的距离
            anchor_dist = dist_matrix[i]

            # Get positive and negative masks
            # 获取正样本和负样本掩码
            pos_mask = labels == labels[i]
            neg_mask = ~pos_mask

            # Exclude the anchor itself
            # 排除锚点本身
            pos_mask[i] = False

            # Check if we have both positives and negatives
            # 检查我们是否同时有正样本和负样本
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue

            # Find hardest positive (furthest positive)
            # 找到最难的正样本（最远的正样本）
            hardest_pos_dist = torch.max(anchor_dist[pos_mask])

            # Find hardest negative (closest negative)
            # 找到最难的负样本（最近的负样本）
            hardest_neg_dist = torch.min(anchor_dist[neg_mask])

            # Calculate triplet loss with margin
            # 使用间隔计算三元组损失
            loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)

            triplet_loss += loss
            valid_triplets += 1

        # Average over valid triplets
        # 对有效三元组取平均
        if valid_triplets > 0:
            triplet_loss = triplet_loss / valid_triplets

        return triplet_loss


def evaluate_speaker_recognition(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
) -> Dict[str, float]:
    """
    Evaluate speaker recognition performance

    评估说话人识别性能

    Args:
        model (nn.Module): Speaker recognition model
                          说话人识别模型
        test_loader (torch.utils.data.DataLoader): Test data loader
                                                 测试数据加载器
        device (torch.device): Device to run evaluation on
                              运行评估的设备

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
                         评估指标字典
    """
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['audio'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            # 前向传播
            outputs = model(features, mode='identification')
            predictions = outputs['predictions']

            # Update statistics
            # 更新统计信息
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # Store for confusion matrix
            # 存储用于混淆矩阵
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate metrics
    # 计算指标
    accuracy = correct / total if total > 0 else 0

    # Convert to numpy arrays for further analysis
    # 转换为numpy数组以进行进一步分析
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate confusion matrix
    # 计算混淆矩阵
    num_classes = len(np.unique(all_labels))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(len(all_labels)):
        confusion_matrix[all_labels[i], all_predictions[i]] += 1

    # Calculate per-class accuracy
    # 计算每类准确率
    per_class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy.tolist(),
        'confusion_matrix': confusion_matrix.tolist()
    }


def evaluate_speaker_verification(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate speaker verification performance

    评估说话人验证性能

    Args:
        model (nn.Module): Speaker recognition model
                          说话人识别模型
        test_loader (torch.utils.data.DataLoader): Test data loader with pairs
                                                 带有配对的测试数据加载器
        device (torch.device): Device to run evaluation on
                              运行评估的设备
        threshold (float): Decision threshold
                          决策阈值

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
                         评估指标字典
    """
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['audio'].to(device)
            labels = batch['is_same'].to(device)  # Binary labels: 1 if same speaker, 0 otherwise

            # Forward pass
            # 前向传播
            outputs = model(features, mode='verification')
            similarity = outputs['similarity'].squeeze()

            # Store scores and labels
            # 存储分数和标签
            all_scores.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    # 转换为numpy数组
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Calculate metrics
    # 计算指标
    predictions = all_scores >= threshold

    # True/False Positives/Negatives
    # 真/假阳性/阴性
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))

    # Calculate metrics
    # 计算指标
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate EER (Equal Error Rate)
    # 计算EER（等错误率）
    far_list = []  # False Acceptance Rate
    frr_list = []  # False Rejection Rate
    thresholds = np.linspace(0, 1, 100)

    for t in thresholds:
        pred_t = all_scores >= t
        far = np.sum((pred_t == 1) & (all_labels == 0)) / np.sum(all_labels == 0) if np.sum(all_labels == 0) > 0 else 0
        frr = np.sum((pred_t == 0) & (all_labels == 1)) / np.sum(all_labels == 1) if np.sum(all_labels == 1) > 0 else 0
        far_list.append(far)
        frr_list.append(frr)

    # Find the threshold where FAR = FRR (EER)
    # 找到FAR = FRR（EER）的阈值
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    abs_diff = np.abs(far_array - frr_array)
    min_idx = np.argmin(abs_diff)
    eer = (far_list[min_idx] + frr_list[min_idx]) / 2

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eer': eer,
        'threshold': threshold
    }


def plot_embeddings_2d(embeddings: np.ndarray, labels: np.ndarray, title: str = "Speaker Embeddings Visualization"):
    """
    Plot 2D visualization of speaker embeddings

    绘制说话人嵌入的2D可视化

    Args:
        embeddings (np.ndarray): Speaker embeddings
                                说话人嵌入
        labels (np.ndarray): Speaker labels
                            说话人标签
        title (str): Plot title
                    图表标题
    """
    # Use t-SNE to reduce dimensionality to 2D
    # 使用t-SNE将维度降至2D
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    # 绘图
    plt.figure(figsize=(12, 10))

    # Get unique labels
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            color=colors[i],
            label=f"Speaker {label}",
            alpha=0.7
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create a speaker recognition system
    # 创建说话人识别系统
    model = ChaoticSpeakerRecognitionSystem(
        chaotic_feature_dim=64,
        num_speakers=10,
        use_chaotic_embedding=False  # Disable for testing
    )

    # Create random input features
    # 创建随机输入特征
    batch_size = 16
    features = torch.randn(batch_size, 64)
    labels = torch.randint(0, 10, (batch_size,))

    # Forward pass for identification
    # 用于识别的前向传播
    outputs_id = model(features, labels=labels, mode='identification')
    print(f"Identification outputs: {outputs_id.keys()}")
    print(f"Embeddings shape: {outputs_id['embeddings'].shape}")
    print(f"Logits shape: {outputs_id['logits'].shape}")
    print(f"Predictions shape: {outputs_id['predictions'].shape}")

    # Forward pass for verification
    # 用于验证的前向传播
    outputs_ver = model(features, mode='verification')
    print(f"Verification outputs: {outputs_ver.keys()}")
    print(f"Similarity shape: {outputs_ver['similarity'].shape}")

    # Calculate loss
    # 计算损失
    loss_fn = SpeakerRecognitionLoss(ce_weight=1.0, triplet_weight=0.1)
    losses = loss_fn(outputs_id, labels)
    print(f"Losses: {losses}")