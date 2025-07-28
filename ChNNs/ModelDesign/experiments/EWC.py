import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import logging
import matplotlib.pyplot as plt
import os


class ElasticWeightConsolidation:
    """
    弹性权重巩固 (Elastic Weight Consolidation, EWC)
    用于防止神经网络在持续学习中的灾难性遗忘

    核心思想：
    1. 识别对旧任务重要的参数
    2. 计算参数的Fisher信息矩阵作为重要性度量
    3. 添加正则化项惩罚重要参数的变化

    数学原理：
    L(θ) = L_new(θ) + λ/2 * Σ_i F_i (θ_i - θ*_i)^2
    其中:
    - L_new(θ): 新任务损失
    - λ: 正则化强度
    - F_i: 参数θ_i的Fisher信息
    - θ*_i: 旧任务最优参数值

    Fisher信息:
    F_i = E[ (∂L/∂θ_i)^2 ]
    近似为训练数据上的梯度平方期望
    """

    def __init__(self, model, dataset, device, num_samples=200, batch_size=32):
        """
        初始化EWC计算器

        参数:
        model: 训练好的模型 (旧任务)
        dataset: 旧任务数据集
        device: 计算设备 (cuda/cpu)
        num_samples: 用于估计Fisher信息的样本数
        batch_size: 计算批次大小
        """
        self.model = model
        self.device = device
        self.fisher = {}
        self.params = {}
        self.dataset = dataset

        # 计算Fisher信息矩阵
        self.compute_fisher_matrix(dataset, num_samples, batch_size)

        logging.info(f"EWC初始化完成: 计算了{len(self.fisher)}个参数的Fisher信息")

    def compute_fisher_matrix(self, dataset, num_samples, batch_size):
        """
        计算Fisher信息矩阵

        步骤:
        1. 创建旧任务数据子集
        2. 遍历数据计算梯度平方
        3. 平均得到Fisher信息估计
        """
        # 创建数据子集
        subset_size = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # 存储初始模型参数
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        # 初始化Fisher信息存储
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        # 模型设置为评估模式
        self.model.eval()

        # 梯度计算
        total_batches = 0
        for batch in dataloader:
            inputs = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)

            # 反向传播计算梯度
            self.model.zero_grad()
            loss.backward()

            # 累积梯度平方 (Fisher信息近似)
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.fisher:
                    # 添加数值稳定性处理
                    grad = param.grad.data
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        logging.warning(f"梯度包含NaN或Inf值: {name}")
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

                    self.fisher[name] += grad.pow(2)

            total_batches += 1

        # 平均Fisher信息
        for name in self.fisher:
            self.fisher[name] = torch.nan_to_num(self.fisher[name] / total_batches, nan=0.0)

        logging.info(f"Fisher计算完成: 使用{subset_size}个样本, {total_batches}个批次")

    def penalty(self, model):
        """
        计算EWC正则化惩罚项

        参数:
        model: 当前模型 (正在训练新任务)

        返回:
        ewc_loss: 正则化损失值
        """
        loss = 0
        for name, param in model.named_parameters():
            if name in self.params:
                # 计算参数差异
                param_diff = param - self.params[name]

                # 添加数值稳定性处理
                if torch.isnan(param_diff).any() or torch.isinf(param_diff).any():
                    logging.warning(f"参数差异包含NaN或Inf值: {name}")
                    param_diff = torch.nan_to_num(param_diff, nan=0.0)

                # 获取Fisher信息，确保数值稳定性
                fisher_val = self.fisher.get(name, torch.zeros_like(param_diff))
                fisher_val = torch.nan_to_num(fisher_val, nan=0.0)

                # 加权平方差 (Fisher信息作为权重)
                loss += (fisher_val * param_diff.pow(2)).sum()

        # 添加数值稳定性处理
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.zeros_like(loss)
            logging.warning("EWC损失包含NaN或Inf值，已置零")

        return loss

    def consolidate(self, model, optimizer, inputs, labels, lambda_ewc=1000.0):
        """
        应用EWC正则化到优化过程

        参数:
        model: 当前模型
        optimizer: 优化器
        inputs: 输入数据
        labels: 目标标签
        lambda_ewc: EWC正则化强度

        返回:
        total_loss: 总损失 (新任务损失 + EWC损失)
        """
        # 计算新任务损失
        optimizer.zero_grad()
        outputs = model(inputs)
        task_loss = F.cross_entropy(outputs, labels)

        # 计算EWC损失
        ewc_loss = self.penalty(model)

        # 总损失
        total_loss = task_loss + lambda_ewc * ewc_loss

        # 反向传播
        total_loss.backward()

        # 梯度裁剪防止爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        return total_loss.item(), task_loss.item(), ewc_loss.item()

    def get_regularization_term(self, model):
        """
        仅计算EWC正则化项 (不执行梯度更新)

        用于自定义训练循环
        """
        return self.penalty(model)

    def update_model_reference(self, new_model):
        """
        更新EWC参考模型 (用于多任务场景)

        当学习多个连续任务时，合并Fisher信息
        """
        # 合并Fisher信息
        for name in new_model.fisher:
            if name in self.fisher:
                # 取最大值保留最显著的特征
                self.fisher[name] = torch.max(self.fisher[name], new_model.fisher[name])

        # 更新参数快照
        self.params = {n: p.clone().detach() for n, p in new_model.named_parameters() if n in self.params}

    @staticmethod
    def compute_parameter_importance(model, dataloader, device):
        """
        替代方法：直接计算参数重要性 (不使用Fisher信息)

        基于参数在训练过程中的变化幅度
        """
        importance = {}
        initial_params = {n: p.clone().detach() for n, p in model.named_parameters()}

        # 训练模型
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        model.train()

        for batch in dataloader:
            inputs = batch['audio'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        # 计算参数变化
        for name, param in model.named_parameters():
            if name in initial_params:
                change = (param - initial_params[name]).abs().sum()
                importance[name] = change

        return importance

    def visualize_fisher_information(self, top_k=20, save_path="parameter_importance.png"):
        """
        可视化最重要的参数 (用于调试和分析)
        """
        if not self.fisher:
            logging.warning("未计算Fisher信息")
            return

        # 收集所有参数的重要性分数
        fisher_scores = []
        for name, fisher_val in self.fisher.items():
            # 使用平均Fisher信息作为分数
            score = fisher_val.mean().item()
            fisher_scores.append((name, score))

        # 按重要性排序
        fisher_scores.sort(key=lambda x: x[1], reverse=True)

        logging.info(f"\n=== 参数重要性排名 (Top {top_k} 基于Fisher信息) ===")
        logging.info("{:<50} {:<15}".format("参数名称", "重要性分数"))
        logging.info("-" * 65)
        for i, (name, score) in enumerate(fisher_scores[:top_k]):
            logging.info("{:<50} {:<15.6f}".format(name, score))

        # 可视化
        names = [n for n, _ in fisher_scores[:top_k]]
        scores = [s for _, s in fisher_scores[:top_k]]

        plt.figure(figsize=(12, 8))
        plt.barh(names, scores, color='skyblue')
        plt.xlabel('Fisher信息')
        plt.title('参数重要性排名')
        plt.gca().invert_yaxis()  # 最重要的在顶部
        plt.tight_layout()

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"参数重要性图保存到 {save_path}")

    def save(self, path):
        """保存EWC状态到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'fisher': {k: v.cpu() for k, v in self.fisher.items()},
            'params': {k: v.cpu() for k, v in self.params.items()},
            'dataset_info': f"{len(self.dataset)} samples"  # 保存数据集信息
        }
        torch.save(state, path)
        logging.info(f"EWC状态保存到 {path}")

    @classmethod
    def load(cls, path, model, device, dataset=None):
        """从文件加载EWC状态"""
        state = torch.load(path, map_location=device)
        ewc = cls.__new__(cls)
        ewc.fisher = state['fisher']
        ewc.params = state['params']
        ewc.device = device
        ewc.model = model
        ewc.dataset = dataset  # 允许传入新的数据集

        # 将参数移到正确设备
        for name in ewc.fisher:
            ewc.fisher[name] = ewc.fisher[name].to(device)
        for name in ewc.params:
            ewc.params[name] = ewc.params[name].to(device)

        logging.info(f"从 {path} 加载EWC状态")
        return ewc