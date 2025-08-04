import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import roc_curve
import gc  # 垃圾回收

# 导入模块（假设在同一目录下）
try:
    from data_loader import SpeakerRecognitionDataset, get_dataloaders
    from c_hilap_model import CHiLAPModel, PhaseSynchronizationLoss
    from chaos_features import ChaosFeatureExtractor, ChaoticFeatureExtractor, MLSAAnalyzer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有模块文件在同一目录下")


# 配置参数
class Config:
    # 训练参数
    EPOCHS = 50  # 减少训练轮数
    LR = 0.001
    WEIGHT_DECAY = 1e-5
    SAVE_INTERVAL = 10
    VAL_INTERVAL = 5
    CHECKPOINT_DIR = "./checkpoints"

    # 损失函数权重
    CE_WEIGHT = 1.0  # 交叉熵损失权重
    SYNC_WEIGHT = 0.0  # 减少相位同步损失权重 # 暂时禁用
    LYAPUNOV_WEIGHT = 0.0  # 减少李雅普诺夫稳定性损失权重 # 暂时禁用

    # 早停参数
    PATIENCE = 10
    MIN_DELTA = 0.001

    # 混沌特征参数
    CHAOS_FEATURE_TYPE = 'mle'  # 使用更简单的特征类型

    # 内存优化参数
    GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数
    MAX_BATCH_SIZE = 16  # 减少批次大小
    ENABLE_MIXED_PRECISION = False  # 启用混合精度训练


# 训练器类
class Trainer:
    def __init__(self, config=Config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = CHiLAPModel().to(self.device)

        # 创建混沌特征提取器（使用更简单的特征类型）
        self.chaos_feature_extractor = ChaoticFeatureExtractor(
            feature_type=config.CHAOS_FEATURE_TYPE
        ).to(self.device)

        # 定义损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.sync_loss = PhaseSynchronizationLoss()

        # 定义优化器
        self.optimizer = optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.chaos_feature_extractor.parameters()}
            ],
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            # verbose=True
        )

        # 混合精度训练
        if config.ENABLE_MIXED_PRECISION and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("启用混合精度训练")
        else:
            self.scaler = None

        # 创建检查点目录
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # 早停计数器
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

    def lyapunov_stability_loss(self, embeddings):
        """
        计算李雅普诺夫稳定性损失 - 鼓励混沌系统的稳定性
        :param embeddings: 模型生成的嵌入向量 [batch_size, embedding_dim]
        :return: 李雅普诺夫稳定性损失
        """
        try:
            # 简化版本：直接使用嵌入向量的方差作为稳定性度量
            # 高方差意味着不稳定
            variance = torch.var(embeddings, dim=1)  # [batch_size]
            loss = torch.mean(variance)
            return loss
        except Exception as e:
            print(f"李雅普诺夫损失计算错误: {e}")
            return torch.tensor(0.0, device=embeddings.device)

    def train_one_epoch(self, dataloader, epoch):
        """
        训练一个epoch
        :param dataloader: 训练数据加载器
        :param epoch: 当前epoch
        :return: 平均训练损失
        """
        self.model.train()
        self.chaos_feature_extractor.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 梯度累积
        accumulation_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        self.optimizer.zero_grad()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (inputs, labels) in progress_bar:
            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 添加特征维度 [batch_size, seq_len, 1]
                inputs = inputs.unsqueeze(2)

                # 使用混合精度训练
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        embeddings, logits = self.model(inputs)

                        # 计算损失
                        ce = self.ce_loss(logits, labels)

                        # 简化的相位同步损失
                        sync_loss_val = torch.tensor(0.0, device=inputs.device)
                        if epoch > 10:  # 后期再加入相位同步损失
                            try:
                                # 创建简化的参考吸引子
                                batch_size, seq_len, _ = inputs.size()
                                # 使用前100个时间步计算损失以节省内存
                                limited_seq_len = min(seq_len, 100)
                                limited_inputs = inputs[:, :limited_seq_len, :]
                                reference_attractors = torch.randn(batch_size, limited_seq_len, 3, device=inputs.device)
                                sync_loss_val = self.sync_loss(limited_inputs, reference_attractors)
                            except Exception as e:
                                print(f"相位同步损失计算错误: {e}")
                                sync_loss_val = torch.tensor(0.0, device=inputs.device)

                        lyapunov_loss_val = self.lyapunov_stability_loss(embeddings)

                        # 联合损失
                        loss = (self.config.CE_WEIGHT * ce +
                                self.config.SYNC_WEIGHT * sync_loss_val +
                                self.config.LYAPUNOV_WEIGHT * lyapunov_loss_val)

                        # 梯度累积
                        loss = loss / accumulation_steps

                    # 反向传播
                    self.scaler.scale(loss).backward()

                    if (i + 1) % accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                else:
                    # 标准训练（不使用混合精度）
                    embeddings, logits = self.model(inputs)

                    # 计算损失
                    ce = self.ce_loss(logits, labels)

                    # 简化的损失计算
                    sync_loss_val = torch.tensor(0.0, device=inputs.device)
                    lyapunov_loss_val = self.lyapunov_stability_loss(embeddings)

                    # 联合损失
                    loss = (self.config.CE_WEIGHT * ce +
                            self.config.LYAPUNOV_WEIGHT * lyapunov_loss_val)

                    # 梯度累积
                    loss = loss / accumulation_steps
                    loss.backward()

                    if (i + 1) % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # 统计
                total_loss += loss.item() * accumulation_steps
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                if i % 10 == 0:  # 减少更新频率
                    progress_bar.set_description(
                        f"Epoch {epoch}, Loss: {total_loss / (i + 1):.4f}, Acc: {100. * correct / total:.2f}%"
                    )

                # 手动垃圾回收
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"内存不足错误在批次 {i}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """
        验证模型性能
        :param dataloader: 验证数据加载器
        :return: 验证损失和准确率
        """
        self.model.eval()
        self.chaos_feature_extractor.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                try:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 添加特征维度 [batch_size, seq_len, 1]
                    inputs = inputs.unsqueeze(2)

                    # 前向传播
                    embeddings, logits = self.model(inputs)

                    # 计算损失
                    ce = self.ce_loss(logits, labels)
                    lyapunov_loss_val = self.lyapunov_stability_loss(embeddings)

                    loss = (self.config.CE_WEIGHT * ce +
                            self.config.LYAPUNOV_WEIGHT * lyapunov_loss_val)

                    total_loss += loss.item()

                    # 统计准确率
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # 内存管理
                    if i % 20 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"验证时内存不足: {e}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

        accuracy = 100. * correct / total
        return total_loss / len(dataloader), accuracy

    def train(self, train_dataloader, val_dataloader):
        """
        完整训练流程
        :param train_dataloader: 训练数据加载器
        :param val_dataloader: 验证数据加载器
        """
        print("开始训练...")

        for epoch in range(1, self.config.EPOCHS + 1):
            try:
                # 训练一个epoch
                train_loss = self.train_one_epoch(train_dataloader, epoch)
                print(f"Epoch {epoch}/{self.config.EPOCHS}, Train Loss: {train_loss:.4f}")

                # 验证
                if epoch % self.config.VAL_INTERVAL == 0:
                    val_loss, accuracy = self.validate(val_dataloader)
                    print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

                    # 更新学习率
                    self.scheduler.step(val_loss)

                    # 早停检查
                    if val_loss < self.best_val_loss - self.config.MIN_DELTA:
                        self.best_val_loss = val_loss
                        self.early_stop_counter = 0
                        # 保存最佳模型
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'chaos_feature_state_dict': self.chaos_feature_extractor.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                            'accuracy': accuracy
                        }, os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth'))
                        print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                    else:
                        self.early_stop_counter += 1
                        if self.early_stop_counter >= self.config.PATIENCE:
                            print(f"早停于第 {epoch} 轮")
                            break

                # 定期保存模型
                if epoch % self.config.SAVE_INTERVAL == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'chaos_feature_state_dict': self.chaos_feature_extractor.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss
                    }, os.path.join(self.config.CHECKPOINT_DIR, f'model_epoch_{epoch}.pth'))

                # 清理内存
                torch.cuda.empty_cache()
                gc.collect()

            except KeyboardInterrupt:
                print("训练被用户中断")
                break
            except Exception as e:
                print(f"训练错误: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        :param checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.chaos_feature_extractor.load_state_dict(checkpoint['chaos_feature_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        print(f"从第 {epoch} 轮加载检查点")
        return epoch


# 简化的评估器类
class SimpleEvaluator:
    def __init__(self, model, chaos_feature_extractor, config=Config):
        """初始化评估器"""
        self.model = model
        self.chaos_feature_extractor = chaos_feature_extractor
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.chaos_feature_extractor.to(self.device)
        self.model.eval()
        self.chaos_feature_extractor.eval()

    def evaluate_accuracy(self, dataloader):
        """
        评估模型准确率
        :param dataloader: 数据加载器
        :return: 准确率
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                try:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 添加特征维度 [batch_size, seq_len, 1]
                    inputs = inputs.unsqueeze(2)

                    _, logits = self.model(inputs)
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # 内存管理
                    if i % 20 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"评估时内存不足: {e}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

        accuracy = 100. * correct / total
        return accuracy


# 内存友好的数据加载器创建函数
def create_small_dataloaders(dataset_name, batch_size=4):
    """创建小批次的数据加载器以节省内存"""
    try:
        train_dataset = SpeakerRecognitionDataset(dataset_name, split="train")
        val_dataset = SpeakerRecognitionDataset(dataset_name, split="val")
        test_dataset = SpeakerRecognitionDataset(dataset_name, split="test")

        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=False),  # 减少工作线程和禁用pin_memory
            "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False),
            "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=False),
        }
    except Exception as e:
        print(f"数据加载器创建错误: {e}")
        return None


# 测试代码
if __name__ == "__main__":
    # 设置较小的批次大小
    batch_size = 4

    print("创建数据加载器...")
    try:
        # 使用内存友好的数据加载器
        dataloaders = create_small_dataloaders("librispeech", batch_size=batch_size)

        if dataloaders is None:
            print("无法创建数据加载器，退出")
            exit(1)

        print(f"训练集批次数: {len(dataloaders['train'])}")
        print(f"验证集批次数: {len(dataloaders['val'])}")
        print(f"测试集批次数: {len(dataloaders['test'])}")

        # 创建训练器
        print("创建训练器...")
        trainer = Trainer(Config)

        # 测试一个批次
        print("测试前向传播...")
        try:
            x, y = next(iter(dataloaders["train"]))
            print(f"输入形状: {x.shape}, 标签形状: {y.shape}")

            x = x.to(trainer.device).unsqueeze(2)
            y = y.to(trainer.device)

            with torch.no_grad():
                embeddings, logits = trainer.model(x)
                print(f"嵌入形状: {embeddings.shape}, 输出形状: {logits.shape}")
                print("前向传播测试成功!")
        except Exception as e:
            print(f"前向传播测试失败: {e}")
            exit(1)

        # 开始训练
        print("开始训练...")
        trainer.train(dataloaders["train"], dataloaders["val"])

        # 创建评估器
        print("创建评估器...")
        evaluator = SimpleEvaluator(trainer.model, trainer.chaos_feature_extractor)

        # 评估模型
        print("评估模型...")
        test_accuracy = evaluator.evaluate_accuracy(dataloaders["test"])
        print(f"测试集准确率: {test_accuracy:.2f}%")

    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback

        traceback.print_exc()