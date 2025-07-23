"""
混沌神经网络模型 - 用于说话人识别
基于混沌动力学原理设计，增强模型对说话人特征的捕捉能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import pandas as pd


# 配置日志
def setup_logging(log_file='chaotic_nn.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ChaoticActivation(nn.Module):
    """
    混沌激活函数 - 基于Logistic映射
    数学原理：x_{n+1} = r * x_n * (1 - x_n)
    其中r是控制参数，当r≈4时系统呈现混沌行为
    """

    def __init__(self, r=3.9, iterations=3, learnable_r=True):
        """
        r: Logistic映射参数
        iterations: 混沌迭代次数
        learnable_r: 是否允许r作为可学习参数
        """
        super(ChaoticActivation, self).__init__()
        self.iterations = iterations

        # 初始化r参数
        if learnable_r:
            self.r = nn.Parameter(torch.tensor(float(r)))
        else:
            self.r = r

        # 初始化固定噪声（增强混沌特性）
        self.noise_scale = 0.01

    def forward(self, x):
        """
        前向传播：应用混沌变换

        参数:
        x: 输入张量 (batch_size, features)

        返回:
        混沌激活后的张量
        """
        # 将输入归一化到(0,1)区间
        x_norm = torch.sigmoid(x)

        # 应用Logistic映射
        for _ in range(self.iterations):
            # 添加少量噪声增强混沌特性
            noise = self.noise_scale * torch.randn_like(x_norm)
            x_norm = self.r * (x_norm + noise) * (1 - x_norm)

        # 重新缩放回(-1,1)区间
        return 2 * x_norm - 1

    def lyapunov_exponent(self, num_points=1000):
        """
        计算Lyapunov指数（混沌程度度量）
        正Lyapunov指数表示系统处于混沌状态
        """
        x = np.linspace(0.001, 0.999, num_points)
        r_val = self.r.item() if isinstance(self.r, torch.Tensor) else self.r
        lyap = 0.0

        for _ in range(100):  # 忽略前100次迭代
            x = r_val * x * (1 - x)

        for _ in range(num_points):
            x = r_val * x * (1 - x)
            lyap += np.log(np.abs(r_val * (1 - 2 * x)))

        return lyap / num_points


class ChaoticNeuralNetwork(nn.Module):
    """
    混沌神经网络模型架构
    设计理念：结合传统DNN的表示能力和混沌系统的动态特性
    """

    def __init__(self, input_dim, num_classes,
                 hidden_dims=[512, 256, 128],
                 r_params=[3.9, 3.9, 3.9],
                 dropout_rate=0.3,
                 use_batch_norm=True):
        """
        初始化混沌神经网络

        参数:
        input_dim: 输入特征维度
        num_classes: 说话人数量
        hidden_dims: 隐藏层维度列表
        r_params: 各混沌层的r参数
        dropout_rate: Dropout概率
        use_batch_norm: 是否使用批归一化
        """
        super(ChaoticNeuralNetwork, self).__init__()

        # 验证参数
        assert len(hidden_dims) == len(r_params), "隐藏层和r参数数量必须匹配"

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for i, (h_dim, r_val) in enumerate(zip(hidden_dims, r_params)):
            # 全连接层
            layers.append(nn.Linear(prev_dim, h_dim))

            # 批归一化
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))

            # 混沌激活函数
            layers.append(ChaoticActivation(r=r_val, learnable_r=True))

            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = h_dim

        # 输出层
        self.output_layer = nn.Linear(prev_dim, num_classes)

        # 组合所有层
        self.layers = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

        logging.info(f"初始化混沌神经网络: 输入维度={input_dim}, 输出维度={num_classes}")
        logging.info(f"隐藏层结构: {hidden_dims}")
        logging.info(f"混沌参数r: {r_params}")

    def _init_weights(self):
        """使用Xavier初始化权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, x):
        """前向传播"""
        features = self.layers(x)
        logits = self.output_layer(features)
        return logits

    def get_chaotic_params(self):
        """获取当前混沌参数"""
        chaotic_params = []
        for layer in self.layers:
            if isinstance(layer, ChaoticActivation):
                chaotic_params.append({
                    'r': layer.r.item(),
                    'lyapunov': layer.lyapunov_exponent()
                })
        return chaotic_params


class SpeakerDataset(Dataset):
    """
    说话人特征数据集
    适配特征提取模块的输出
    """

    def __init__(self, feature_metadata, max_frames=300, feature_dim=39):
        """
        初始化数据集

        参数:
        feature_metadata: 特征元数据DataFrame或文件路径
        max_frames: 最大帧数（填充/截断）
        feature_dim: 特征维度（MFCC+Delta+DeltaDelta）
        """
        if isinstance(feature_metadata, str):
            self.metadata = pd.read_csv(feature_metadata)
        else:
            self.metadata = feature_metadata

        self.max_frames = max_frames
        self.feature_dim = feature_dim

        # 构建说话人到标签的映射
        self.speaker_to_idx = {sp: i for i, sp in enumerate(self.metadata['speaker_id'].unique())}
        self.metadata['label'] = self.metadata['speaker_id'].map(self.speaker_to_idx)

        logging.info(f"数据集初始化: 共 {len(self)} 个样本, {len(self.speaker_to_idx)} 个说话人")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        try:
            feature_path = row['feature_path']
            print(f"[DEBUG] 正在加载特征文件: {feature_path}")  # <-- 添加调试输出
            features = np.load(feature_path)

            # 时间轴处理 (截断或填充)
            if features.shape[1] > self.max_frames:
                # 随机裁剪
                start = np.random.randint(0, features.shape[1] - self.max_frames)
                features = features[:, start:start + self.max_frames]
            elif features.shape[1] < self.max_frames:
                # 填充
                pad_width = ((0, 0), (0, self.max_frames - features.shape[1]))
                features = np.pad(features, pad_width, mode='constant')

            # 转换为张量 (特征维度, 时间帧)
            features = torch.FloatTensor(features)

            return {
                'features': features,
                'label': row['label'],
                'speaker_id': row['speaker_id'],
                'file_path': row['file_path']
            }
        except Exception as e:
            logging.error(f"加载特征失败: {row['feature_path']} - {str(e)}")
            # 返回空特征
            return {
                'features': torch.zeros((self.feature_dim, self.max_frames)),
                'label': -1,
                'speaker_id': 'unknown',
                'file_path': 'error'
            }


class ChaoticModelTrainer:
    """
    混沌神经网络训练器
    包含训练、评估和模型保存功能
    """

    def __init__(self, model, train_loader, val_loader,
                 optimizer, device, save_dir="chaotic_model"):
        """
        初始化训练器

        参数:
        model: 混沌神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 训练设备 (cuda/cpu)
        save_dir: 模型保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'chaotic_params': []
        }

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        logging.info(f"训练器初始化: 设备={device}, 保存目录={save_dir}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch}")
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)

            # 时间平均池化 (特征维度, 时间帧) -> (特征维度)
            features_pooled = torch.mean(features, dim=2)

            # 前向传播
            outputs = self.model(features_pooled)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': f"{100 * correct / total:.2f}%"
            })

        # 计算epoch指标
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total

        # 记录训练指标
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)

        # 记录混沌参数
        self.history['chaotic_params'].append(self.model.get_chaotic_params())

        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)

                # 时间平均池化
                features_pooled = torch.mean(features, dim=2)

                # 前向传播
                outputs = self.model(features_pooled)
                loss = self.criterion(outputs, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算验证指标
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100 * correct / total

        # 记录验证指标
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        return val_loss, val_acc

    def train(self, epochs, early_stopping=5):
        """
        训练模型

        参数:
        epochs: 训练轮数
        early_stopping: 早停耐心值
        """
        best_val_acc = 0.0
        epochs_no_improve = 0
        best_model_path = os.path.join(self.save_dir, "best_model.pth")

        logging.info(f"开始训练，共 {epochs} 个epochs")

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 打印epoch结果
            logging.info(f"Epoch {epoch}/{epochs} | "
                         f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | "
                         f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, best_model_path)
                logging.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping:
                    logging.info(f"早停触发，验证准确率连续 {early_stopping} 个epoch未提升")
                    break

        # 保存最终模型
        final_model_path = os.path.join(self.save_dir, "final_model.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, final_model_path)
        logging.info(f"保存最终模型: {final_model_path}")

        # 保存训练历史
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # 转换numpy.ndarray为列表
        for key in self.history:
            if isinstance(self.history[key], list):
                for i, item in enumerate(self.history[key]):
                    if isinstance(item, np.ndarray):
                        self.history[key][i] = item.tolist()

        # 可视化训练过程
        self.plot_training_history()

        return best_val_acc

    def plot_training_history(self):
        """可视化训练历史"""
        plt.figure(figsize=(15, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()

        # 准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('训练和验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()

        # 混沌参数变化
        plt.subplot(2, 1, 2)
        chaotic_r = [[p['r'] for p in cp] for cp in self.history['chaotic_params']]
        chaotic_lyap = [[p['lyapunov'] for p in cp] for cp in self.history['chaotic_params']]

        # 转置以便绘图
        chaotic_r = np.array(chaotic_r).T
        chaotic_lyap = np.array(chaotic_lyap).T

        for i, (r_vals, lyap_vals) in enumerate(zip(chaotic_r, chaotic_lyap)):
            plt.plot(r_vals, label=f'层{i + 1} r参数')
            plt.plot(lyap_vals, '--', label=f'层{i + 1} Lyapunov指数')

        plt.title('混沌参数变化')
        plt.xlabel('Epoch')
        plt.ylabel('值')
        plt.legend()
        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(self.save_dir, "training_history.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"训练历史图已保存: {plot_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    setup_logging()

    # 配置参数
    FEATURE_METADATA = "P:/PycharmProjects/pythonProject1/ChNNs/FeatureExtraction/extracted_features/feature_metadata.csv"  # 替换为实际路径
    SAVE_DIR = "chaotic_model_results"
    BATCH_SIZE = 64
    EPOCHS = 50
    MAX_FRAMES = 300  # 对应3秒音频 (10ms/帧 * 300帧 = 3秒)
    FEATURE_DIM = 39  # 13 MFCC + 13 Delta + 13 DeltaDelta

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 加载特征元数据
    metadata = pd.read_csv(FEATURE_METADATA)
    # 补全绝对路径（适用于 Windows）
    FEATURE_ROOT_DIR = "P:/PycharmProjects/pythonProject1/ChNNs/FeatureExtraction"


    def fix_path(p):
        # 如果已有绝对路径就直接返回
        if os.path.isabs(p):
            return p
        # 替换分隔符并拼接
        return os.path.join(FEATURE_ROOT_DIR, p.replace("\\", os.sep).replace("/", os.sep))


    metadata['feature_path'] = metadata['feature_path'].apply(fix_path)

    # 分割训练集和验证集 (80%训练, 20%验证)
    train_metadata, val_metadata = train_test_split(
        metadata, test_size=0.2, stratify=metadata['speaker_id'], random_state=42
    )

    # 创建数据集
    train_dataset = SpeakerDataset(train_metadata, MAX_FRAMES, FEATURE_DIM)
    val_dataset = SpeakerDataset(val_metadata, MAX_FRAMES, FEATURE_DIM)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 初始化模型
    num_classes = len(train_dataset.speaker_to_idx)
    model = ChaoticNeuralNetwork(
        input_dim=FEATURE_DIM,  # 输入特征维度
        num_classes=num_classes,
        hidden_dims=[512, 256, 128],  # 隐藏层维度
        r_params=[3.9, 3.9, 3.9],  # 各层混沌参数
        dropout_rate=0.3
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 初始化训练器
    trainer = ChaoticModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=SAVE_DIR
    )

    # 开始训练
    best_val_acc = trainer.train(epochs=EPOCHS, early_stopping=10)
    logging.info(f"训练完成，最佳验证准确率: {best_val_acc:.2f}%")