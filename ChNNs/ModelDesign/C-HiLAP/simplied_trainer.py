import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import gc  # 垃圾回收
import math

# 导入模块（假设在同一目录下）
try:
    from simplified_data_loader import SpeakerRecognitionDataset, get_dataloaders
    from simplified_c_hilap_model import CHiLAPModel
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有模块文件在同一目录下")


# 配置参数
class Config:
    # 训练参数
    EPOCHS = 50  # 减少训练轮数
    LR = 0.01  # 初始较大学习率
    LR_DECAY = 0.95  # 学习率衰减因子
    WEIGHT_DECAY = 1e-5
    SAVE_INTERVAL = 10
    VAL_INTERVAL = 1  # 每个epoch都验证
    CHECKPOINT_DIR = "./checkpoints"

    # 优化参数
    WARMUP_EPOCHS = 5  # 学习率预热轮数
    GRAD_CLIP = 1.0  # 梯度裁剪阈值

    # 损失函数权重
    CE_WEIGHT = 1.0  # 交叉熵损失权重

    # 早停参数
    PATIENCE = 10
    MIN_DELTA = 0.001

    # 内存优化参数
    GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
    BATCH_SIZE = 16  # 批次大小
    ENABLE_MIXED_PRECISION = False  # CPU上禁用混合精度
    MAX_SEQ_LEN = 16000  # 1秒音频长度


# 训练器类
class Trainer:
    def __init__(self, config=Config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = CHiLAPModel().to(self.device)

        # 定义损失函数
        self.ce_loss = nn.CrossEntropyLoss()

        # 定义优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        # 学习率预热调度器
        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, epoch / config.WARMUP_EPOCHS)
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
        self.best_val_accuracy = 0.0  # 改为基于准确率早停
        print(f"模型预期输入长度: {config.MAX_SEQ_LEN}")

    def train_one_epoch(self, dataloader, epoch):
        """
        训练一个epoch
        :param dataloader: 训练数据加载器
        :param epoch: 当前epoch
        :return: 平均训练损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0

        # 梯度累积
        accumulation_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        self.optimizer.zero_grad()

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (inputs, labels) in progress_bar:
            try:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 确保输入维度正确 [batch, 1, seq_len]
                if inputs.dim() == 2:  # [batch, seq_len]
                    inputs = inputs.unsqueeze(1)  # 添加通道维度 -> [batch, 1, seq_len]

                # 确保音频长度正确
                if inputs.size(2) > self.config.MAX_SEQ_LEN:
                    inputs = inputs[:, :, :self.config.MAX_SEQ_LEN]
                elif inputs.size(2) < self.config.MAX_SEQ_LEN:
                    pad_len = self.config.MAX_SEQ_LEN - inputs.size(2)
                    inputs = torch.nn.functional.pad(inputs, (0, pad_len), value=0.0)

                # 使用混合精度训练
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        embeddings, logits = self.model(inputs)

                        # 计算损失
                        ce = self.ce_loss(logits, labels)
                        loss = self.config.CE_WEIGHT * ce

                        # 梯度累积
                        loss = loss / accumulation_steps

                    # 反向传播
                    self.scaler.scale(loss).backward()

                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

                    if (i + 1) % accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                else:
                    # 标准训练（不使用混合精度）
                    embeddings, logits = self.model(inputs)

                    # 计算损失
                    ce = self.ce_loss(logits, labels)
                    loss = self.config.CE_WEIGHT * ce

                    # 梯度累积
                    loss = loss / accumulation_steps
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

                    if (i + 1) % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # 统计
                total_loss += loss.item() * accumulation_steps
                total_ce_loss += ce.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                if i % 10 == 0:  # 减少更新频率
                    accuracy = 100. * correct / total
                    avg_loss = total_loss / (i + 1)
                    progress_bar.set_description(
                        f"Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%"
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

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self, dataloader):
        """
        验证模型性能
        :param dataloader: 验证数据加载器
        :return: 验证损失和准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                try:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 确保输入维度正确 [batch, 1, seq_len]
                    if inputs.dim() == 2:  # [batch, seq_len]
                        inputs = inputs.unsqueeze(1)  # 添加通道维度 -> [batch, 1, seq_len]

                    # 确保音频长度正确
                    if inputs.size(2) > self.config.MAX_SEQ_LEN:
                        inputs = inputs[:, :, :self.config.MAX_SEQ_LEN]
                    elif inputs.size(2) < self.config.MAX_SEQ_LEN:
                        pad_len = self.config.MAX_SEQ_LEN - inputs.size(2)
                        inputs = torch.nn.functional.pad(inputs, (0, pad_len), value=0.0)

                    # 前向传播
                    embeddings, logits = self.model(inputs)

                    # 计算损失
                    ce = self.ce_loss(logits, labels)
                    loss = self.config.CE_WEIGHT * ce

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

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, train_dataloader, val_dataloader):
        """
        完整训练流程
        :param train_dataloader: 训练数据加载器
        :param val_dataloader: 验证数据加载器
        """
        print("开始训练...")
        print(f"训练集批次: {len(train_dataloader)}, 验证集批次: {len(val_dataloader)}")

        for epoch in range(1, self.config.EPOCHS + 1):
            try:
                # 应用学习率预热
                if epoch <= self.config.WARMUP_EPOCHS:
                    self.warmup_scheduler.step()

                # 训练一个epoch
                train_loss, train_acc = self.train_one_epoch(train_dataloader, epoch)
                print(f"Epoch {epoch}/{self.config.EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

                # 验证
                val_loss, val_acc = self.validate(val_dataloader)
                print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # 更新学习率
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"当前学习率: {current_lr:.6f}")

                # 早停检查 (基于验证准确率)
                if val_acc > self.best_val_accuracy + self.config.MIN_DELTA:
                    self.best_val_accuracy = val_acc
                    self.early_stop_counter = 0
                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_accuracy': val_acc
                    }, os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth'))
                    print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
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
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_accuracy': train_acc
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
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        print(f"从第 {epoch} 轮加载检查点")
        return epoch


# 评估器类
class Evaluator:
    def __init__(self, model, config=Config):
        """初始化评估器"""
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

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

                    # 确保输入维度正确
                    if inputs.dim() == 2:  # [batch, seq_len]
                        inputs = inputs.unsqueeze(1)  # 添加通道维度 -> [batch, 1, seq_len]

                    # 截断过长的序列
                    if inputs.size(2) > self.config.MAX_SEQ_LEN:
                        inputs = inputs[:, :, :self.config.MAX_SEQ_LEN]

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
                                num_workers=0, pin_memory=False),
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
    batch_size = Config.BATCH_SIZE

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
            print(f"原始输入形状: {x.shape}, 标签形状: {y.shape}")

            # 处理输入维度
            if x.dim() == 2:  # [batch, seq_len]
                x = x.unsqueeze(1)  # 添加通道维度 -> [batch, 1, seq_len]

            # 确保音频长度正确
            if x.size(2) > Config.MAX_SEQ_LEN:
                x = x[:, :, :Config.MAX_SEQ_LEN]
            elif x.size(2) < Config.MAX_SEQ_LEN:
                pad_len = Config.MAX_SEQ_LEN - x.size(2)
                x = torch.nn.functional.pad(x, (0, pad_len), value=0.0)

            print(f"处理后输入形状: {x.shape}")

            x = x.to(trainer.device)
            y = y.to(trainer.device)

            with torch.no_grad():
                embeddings, logits = trainer.model(x)
                print(f"嵌入形状: {embeddings.shape}, 输出形状: {logits.shape}")
                print("前向传播测试成功!")
        except Exception as e:
            print(f"前向传播测试失败: {e}")
            import traceback

            traceback.print_exc()
            exit(1)

        # 开始训练
        print("开始训练...")
        trainer.train(dataloaders["train"], dataloaders["val"])

        # 创建评估器
        print("创建评估器...")
        evaluator = Evaluator(trainer.model)

        # 评估模型
        print("评估模型...")
        test_accuracy = evaluator.evaluate_accuracy(dataloaders["test"])
        print(f"测试集准确率: {test_accuracy:.2f}%")

    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback

        traceback.print_exc()