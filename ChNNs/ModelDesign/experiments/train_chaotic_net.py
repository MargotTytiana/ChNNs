import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import time
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import copy
import random

# 导入自定义模块
from ChaoticLSTM import ChaoticLSTM
from EWC import ElasticWeightConsolidation
from ChNNs.PreProcessData.LibriSpeechDataLoader import LibriSpeechDataset, setup_logging

# 设置日志
setup_logging('training.log')
logger = logging.getLogger(__name__)


class ChaoticTrainer:
    """
    混沌神经网络训练框架
    支持持续学习和说话人识别任务

    核心功能：
    1. 混沌LSTM模型的训练与验证
    2. 弹性权重巩固(EWC)防止灾难性遗忘
    3. 动态添加新说话人
    4. 混沌参数自适应调整
    5. 混合精度训练加速
    """

    def __init__(self, config):
        """
        初始化训练器

        参数:
        config: 配置字典，包含所有训练参数
        """
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"发现GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.warning("未发现GPU，将使用CPU进行训练。训练速度会比较慢。")

        # 相应地调整混合精度训练配置
        self.config['use_amp'] = self.config['use_amp'] and torch.cuda.is_available()

        self.model = None
        self.ewc = None
        self.current_task = 0

        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)

        # 初始化数据
        self._load_data()

        logger.info(f"训练器初始化完成，使用设备: {self.device}")

    def _load_data(self):
        """加载预处理好的数据集"""
        # 读取元数据
        metadata_path = os.path.join(self.config['data_root'], 'metadata_test.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)
        logger.info(f"加载元数据: 共 {len(self.metadata)} 条记录")

        # 创建初始训练集和验证集
        val_subsets = ['dev-clean', 'dev-other']  # 提前定义

        # 创建训练集
        self.train_set = LibriSpeechDataset(
            self.metadata,
            subset=self.config['initial_subset'],
            chunk_size=self.config['chunk_size'],
            augment=True
        )

        # 创建验证集
        val_meta = self.metadata[self.metadata['subset'].isin(val_subsets)]

        # 处理验证集为空的情况
        if len(val_meta) == 0:
            logger.warning("验证集为空，使用部分训练数据作为验证集")
            val_indices = np.random.choice(len(self.train_set), min(1000, len(self.train_set)), replace=False)
            self.val_set = Subset(self.train_set, val_indices)
        else:
            self.val_set = LibriSpeechDataset(
                val_meta,
                chunk_size=self.config['chunk_size'],
                augment=False
            )

        # 采样部分验证数据加速评估
        if len(self.val_set) > 5000:
            val_indices = np.random.choice(len(self.val_set), 5000, replace=False)
            self.val_set = Subset(self.val_set, val_indices)

        logger.info(f"数据加载: 训练集 {len(self.train_set)} 样本, 验证集 {len(self.val_set)} 样本")

        # 检查说话人重叠问题
        train_speakers = set(self.train_set.metadata['speaker_id'])
        val_speakers = set(self.val_set.metadata['speaker_id'])
        common_speakers = train_speakers & val_speakers

        logger.info(f"训练集说话人数量: {len(train_speakers)}")
        logger.info(f"验证集说话人数量: {len(val_speakers)}")
        logger.info(f"共同说话人数量: {len(common_speakers)}")

        if common_speakers:
            logger.warning(f"训练集和验证集有共同说话人: {common_speakers}")

            # 创建不重叠的验证集
            logger.warning("正在创建无重叠的验证集...")
            val_speakers = val_speakers - common_speakers
            if val_speakers:
                val_meta = self.metadata[
                    (self.metadata['subset'].isin(val_subsets)) &
                    (self.metadata['speaker_id'].isin(val_speakers))
                    ]
                self.val_set = LibriSpeechDataset(
                    val_meta,
                    chunk_size=self.config['chunk_size'],
                    augment=False
                )
                logger.info(f"创建无重叠验证集: {len(self.val_set)} 样本")
            else:
                logger.error("无法创建无重叠验证集，使用训练集子集作为验证集")
                val_indices = np.random.choice(len(self.train_set), min(1000, len(self.train_set)), replace=False)
                self.val_set = Subset(self.train_set, val_indices)

        # 统计音频长度
        if 'duration' in self.metadata.columns:
            sample_rate = 16000
            self.metadata['audio_length_samples'] = self.metadata['duration'] * sample_rate
            self.metadata['audio_length_samples'] = self.metadata['audio_length_samples'].astype(int)

            lengths = self.metadata['audio_length_samples']
            min_samples = lengths.min()
            max_samples = lengths.max()
            logger.info(f"转换后样本数统计：最小 {min_samples}，最大 {max_samples}")
        else:
            raise KeyError("元数据中缺少音频时长列（'duration'），无法转换样本数")

    def _init_model(self):
        """初始化混沌LSTM模型"""
        self.model = ChaoticLSTM(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_speakers=len(self.train_set.speaker_to_idx),
            chaos_factor=self.config['chaos_factor']
        ).to(self.device)

        # 打印模型结构
        logger.info(f"模型初始化:\n{self.model}")
        logger.info(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_data_loaders(self):
        """创建数据加载器"""
        # 检查数据集是否为空
        if len(self.train_set) == 0:
            raise ValueError("训练集为空，无法创建数据加载器")
        if len(self.val_set) == 0:
            raise ValueError("验证集为空，无法创建数据加载器")

        num_workers = self.config['num_workers'] if torch.cuda.is_available() else 0
        pin_memory = self.config['pin_memory'] if torch.cuda.is_available() else False

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        logger.info(f"数据加载器创建: 训练批次数 {len(self.train_loader)}, 验证批次数 {len(self.val_loader)}")

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            # verbose=True
        )

        # 混合精度训练
        from torch.amp import GradScaler  # 添加导入语句

        # 根据设备类型初始化GradScaler
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler(enabled=self.config['use_amp'] and device_type == 'cuda')

        logger.info(f"优化器初始化: 学习率 {self.config['lr']}, 权重衰减 {self.config['weight_decay']}")

    def train_epoch(self, epoch):
        """
        训练单个epoch

        返回:
        train_loss: 平均训练损失
        train_acc: 训练准确率
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch + 1}/{self.config['epochs']}")

        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            # 混合精度训练
            with autocast(enabled=self.config['use_amp']):
                # 前向传播
                outputs = self.model(audio)

                # 计算损失
                ce_loss = F.cross_entropy(outputs, labels)

                # 添加EWC正则化
                if self.ewc is not None and self.config['lambda_ewc'] > 0:
                    ewc_loss = self.ewc.penalty(self.model)
                    loss = ce_loss + self.config['lambda_ewc'] * ewc_loss
                else:
                    loss = ce_loss
                    ewc_loss = torch.tensor(0.0)

            # 反向传播和优化
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            if self.config['grad_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新混沌参数
            if batch_idx % self.config['chaos_reset_interval'] == 0:
                self.model.reset_chaos()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%", lr=f"{current_lr:.6f}")

        # 计算本epoch指标
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        epoch_time = time.time() - start_time

        logger.info(
            f"训练 Epoch {epoch + 1} | 耗时: {epoch_time:.1f}s | 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%")

        return train_loss, train_acc

    def validate(self):
        """
        在验证集上评估模型

        返回:
        val_loss: 验证损失
        val_acc: 验证准确率
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(audio)
                loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        logger.info(f"验证结果 | 损失: {val_loss:.4f} | 准确率: {val_acc:.2f}%")
        return val_loss, val_acc

    def train(self):
        """训练主循环"""
        # 初始化模型和优化器
        self._init_model()
        self._create_data_loaders()
        self._init_optimizer()

        # === 添加模型输出维度检查 ===
        num_train_classes = len(self.train_set.speaker_to_idx)
        num_model_classes = self.model.classifier.out_features

        logger.info(f"训练集说话人数: {num_train_classes}")
        logger.info(f"模型分类层输出维度: {num_model_classes}")

        if num_train_classes != num_model_classes:
            logger.error(f"模型输出维度({num_model_classes})与训练集类别数({num_train_classes})不匹配!")
            # 自动修正
            self.model.expand_for_new_speakers(num_train_classes - num_model_classes)
            logger.info(f"已扩展模型输出维度至 {num_train_classes}")
        # ========================

        # 训练历史记录
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [], 'chaos_factor': []
        }

        best_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(self.config['epochs']):
            # 动态调整混沌参数
            self.model.adjust_chaos_during_training(epoch, self.config['epochs'])
            history['chaos_factor'].append(self.model.chaos_factor)

            # 训练和验证
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # 更新学习率
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            logger.info(f"当前学习率: {current_lr:.6f}")

            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(f"best_model_task{self.current_task}.pth")
                epochs_no_improve = 0
                logger.info(f"发现新的最佳模型，准确率: {best_acc:.2f}%")
            else:
                epochs_no_improve += 1
                logger.info(f"验证准确率未提升，连续 {epochs_no_improve} 个epoch")

            # 提前停止检查
            if epochs_no_improve >= self.config['early_stop_patience']:
                logger.info(f"提前停止! {self.config['early_stop_patience']} 个epoch验证准确率未提升")
                break

        # 保存最终模型
        self.save_model(f"final_model_task{self.current_task}.pth")
        self.plot_training_history(history)

        return history

    def add_new_speakers(self, new_subset='dev-clean', sample_size=500):
        """
        添加新说话人到模型 (持续学习)

        步骤:
        1. 添加新说话人到数据集
        2. 扩展模型分类层
        3. 初始化EWC防止遗忘
        """
        # 保存旧模型状态
        old_model_state = copy.deepcopy(self.model.state_dict())
        logger.info("保存旧模型状态用于EWC初始化")

        # 筛选新说话人数据
        new_meta = self.metadata[self.metadata['subset'] == new_subset]

        # 采样部分数据 (如果数据集很大)
        if len(new_meta) > sample_size:
            new_meta = new_meta.sample(sample_size)
            logger.info(f"采样 {sample_size} 个样本用于新说话人")

        # 添加新说话人到训练集
        old_speaker_count = len(self.train_set.speaker_to_idx)
        self.train_set.add_new_speakers(new_meta)
        new_speaker_count = len(self.train_set.speaker_to_idx) - old_speaker_count

        logger.info(f"添加 {new_speaker_count} 个新说话人，总说话人数: {len(self.train_set.speaker_to_idx)}")

        # 扩展模型
        self.model.expand_for_new_speakers(new_speaker_count)
        self.model = self.model.to(self.device)

        # 恢复旧模型参数（但跳过分类层）
        current_state = self.model.state_dict()

        # 只更新非分类层参数
        for name, param in old_model_state.items():
            if not name.startswith('classifier.'):  # 跳过分类层
                current_state[name].copy_(param)

        logger.info("恢复旧模型参数（分类层除外）")

        # 初始化EWC
        self.ewc = ElasticWeightConsolidation(
            model=self.model,
            dataset=self.train_set,
            device=self.device,
            num_samples=self.config['ewc_samples']
        )

        # 更新任务计数器
        self.current_task += 1
        self._create_data_loaders()  # 重新创建数据加载器

        logger.info(f"准备学习任务 {self.current_task}，EWC {'已启用' if self.ewc else '未启用'}")

    def save_model(self, filename):
        """保存模型和训练状态"""
        save_path = os.path.join(self.config['output_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'speaker_mapping': self.train_set.speaker_to_idx,
            'config': self.config,
            'task_id': self.current_task  # 保存当前任务ID
        }, save_path)
        logger.info(f"模型保存到 {save_path}")

    def load_model(self, filename):
        """加载模型和训练状态"""
        load_path = os.path.join(self.config['output_dir'], filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # 初始化模型
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            self._init_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 恢复说话人映射
        self.train_set.speaker_to_idx = checkpoint['speaker_mapping']

        # 恢复任务ID
        self.current_task = checkpoint.get('task_id', 0)

        logger.info(f"从 {load_path} 加载模型 (任务 {self.current_task})")

    def plot_training_history(self, history):
        """可视化训练历史"""
        plt.figure(figsize=(15, 10))

        # 1. 准确率曲线
        plt.subplot(2, 2, 1)
        plt.plot(history['train_acc'], label='训练准确率')
        plt.plot(history['val_acc'], label='验证准确率')
        plt.title('准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True)

        # 2. 损失曲线
        plt.subplot(2, 2, 2)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        # 3. 学习率变化
        plt.subplot(2, 2, 3)
        plt.plot(history['lr'])
        plt.title('学习率')
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.grid(True)
        plt.yscale('log')

        # 4. 混沌因子变化
        plt.subplot(2, 2, 4)
        plt.plot(history['chaos_factor'])
        plt.title('混沌因子')
        plt.xlabel('Epoch')
        plt.ylabel('混沌因子')
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.config['output_dir'], f'training_history_task{self.current_task}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"训练历史图保存到 {plot_path}")


def get_default_config():
    """获取默认训练配置"""
    return {
        'data_root': 'P:/PycharmProjects/pythonProject1/processed_data',  # 预处理数据路径
        'output_dir': 'experiments/chaotic_net',  # 输出目录
        'initial_subset': 'train-clean-100',  # 初始训练子集
        'input_dim': 128,  # 输入特征维度
        'hidden_dim': 256,  # LSTM隐藏层维度
        'chaos_factor': 3.5,  # 初始混沌因子
        'batch_size': 64,  # 批大小
        'epochs': 30,  # 训练轮数
        'lr': 0.001,  # 初始学习率
        'weight_decay': 1e-4,  # 权重衰减
        'grad_clip': 1.0,  # 梯度裁剪阈值
        'num_workers': 4,  # 数据加载线程数
        'use_amp': True,  # 启用混合精度训练
        'chunk_size': 20000,  # 音频块大小 (3秒@16kHz)
        'chaos_reset_interval': 100,  # 重置混沌参数的步数间隔
        'lambda_ewc': 1000.0,  # EWC正则化强度
        'ewc_samples': 500,  # EWC使用的样本数
        'early_stop_patience': 5,  # 提前停止耐心值
        'pin_memory': True,  # 添加这个选项
    }


def continual_learning_example():
    """持续学习示例流程"""
    # 加载配置
    config = get_default_config()

    # 设置随机种子确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 初始化训练器
    trainer = ChaoticTrainer(config)

    # ===== 第一阶段：初始训练 =====
    logger.info("===== 开始初始训练 =====")
    trainer.train()

    # ===== 第二阶段：添加新说话人 =====
    logger.info("\n===== 添加新说话人 =====")
    trainer.add_new_speakers(new_subset='dev-clean', sample_size=1000)

    # 更新配置用于新任务
    config['lr'] = 0.0005  # 降低学习率
    config['epochs'] = 20  # 减少训练轮数

    # 训练新任务
    logger.info("\n===== 训练新任务 =====")
    trainer.train()

    # ===== 第三阶段：添加更多说话人 =====
    logger.info("\n===== 添加更多说话人 =====")
    trainer.add_new_speakers(new_subset='test-clean', sample_size=1500)

    # 训练最终任务
    logger.info("\n===== 训练最终任务 =====")
    trainer.train()

    logger.info("\n===== 持续学习完成 =====")


if __name__ == "__main__":
    continual_learning_example()