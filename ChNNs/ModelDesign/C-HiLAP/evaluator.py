import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from .data_loader import SpeakerRecognitionDataset, get_dataloaders
from .c_hilap_model import CHiLAPModel


# 配置参数
class Config:
    # 评估参数
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EMBEDDING_DIM = 192  # 嵌入维度

    # 测试集路径
    TEST_PAIRS_PATH = "/path/to/voxceleb1/test_pairs.txt"  # 测试对文件
    NOISE_TYPES = ["white", "babble", "pink"]  # 噪声类型
    SNR_LEVELS = [-5, 0, 5, 10, 15, 20]  # 信噪比(dB)
    DURATION_LEVELS = [1, 2, 3, 5, 10]  # 不同时长测试(秒)

    # 对抗攻击参数
    ATTACK_TYPES = ["fgsm", "pgd"]  # 攻击类型
    EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05]  # 扰动强度


# 评估器类
class Evaluator:
    def __init__(self, model, config=Config, device=None):
        """
        初始化评估器
        :param model: 待评估的模型
        :param config: 配置参数
        :param device: 计算设备
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute_embeddings(self, dataloader):
        """
        计算数据集的嵌入向量
        :param dataloader: 数据加载器
        :return: 嵌入向量字典 {utterance_id: embedding}
        """
        embeddings = {}

        with torch.no_grad():
            for inputs, labels, utt_ids in tqdm(dataloader, desc="Computing embeddings"):
                inputs = inputs.to(self.device)
                batch_embeddings, _ = self.model(inputs)

                for i, utt_id in enumerate(utt_ids):
                    embeddings[utt_id] = batch_embeddings[i].cpu().numpy()

        return embeddings

    def calculate_eer(self, scores, labels):
        """
        计算等错误率(EER)和最小检测代价函数(minDCF)
        :param scores: 相似度分数数组
        :param labels: 真实标签数组(1为匹配，0为不匹配)
        :return: EER, minDCF, 阈值, FPR, TPR
        """
        # 计算FPR和TPR
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

        # 计算EER
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = interp1d(fpr, thresholds)(eer)

        # 计算minDCF (p-target=0.01, c_miss=1, c_fa=1)
        dcf = 0.01 * fpr + 0.99 * (1 - tpr)
        min_dcf = np.min(dcf)
        min_dcf_threshold = thresholds[np.argmin(dcf)]

        return eer, min_dcf, eer_threshold, fpr, tpr

    def evaluate_voxceleb1(self, pairs_file=None, embeddings=None, dataloader=None):
        """
        评估VoxCeleb1测试集性能
        :param pairs_file: 测试对文件路径
        :param embeddings: 预计算的嵌入向量
        :param dataloader: 数据加载器(用于计算嵌入向量)
        :return: EER, minDCF
        """
        pairs_file = pairs_file or self.config.TEST_PAIRS_PATH

        # 如果没有提供嵌入向量，则计算它们
        if embeddings is None:
            if dataloader is None:
                raise ValueError("Either embeddings or dataloader must be provided")
            embeddings = self.compute_embeddings(dataloader)

        # 加载测试对
        pairs = self._load_test_pairs(pairs_file)

        # 计算相似度分数和标签
        scores = []
        labels = []

        for label, utt1, utt2 in tqdm(pairs, desc="Evaluating pairs"):
            if utt1 in embeddings and utt2 in embeddings:
                # 计算余弦相似度
                embed1 = embeddings[utt1]
                embed2 = embeddings[utt2]
                score = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

                scores.append(score)
                labels.append(label)

        # 计算EER和minDCF
        eer, min_dcf, _, _, _ = self.calculate_eer(np.array(scores), np.array(labels))
        return eer, min_dcf

    def _load_test_pairs(self, pairs_file):
        """
        加载VoxCeleb1测试对
        :param pairs_file: 测试对文件路径
        :return: 测试对列表 [(label, utt1_id, utt2_id)]
        """
        pairs = []

        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) == 3:  # 匹配对
                    label = 1
                    utt1 = f"id{parts[0]}/{parts[0]}_{parts[1].zfill(4)}.wav"
                    utt2 = f"id{parts[0]}/{parts[0]}_{parts[2].zfill(4)}.wav"
                elif len(parts) == 4:  # 不匹配对
                    label = 0
                    utt1 = f"id{parts[0]}/{parts[0]}_{parts[1].zfill(4)}.wav"
                    utt2 = f"id{parts[2]}/{parts[2]}_{parts[3].zfill(4)}.wav"
                else:
                    continue

                pairs.append((label, utt1, utt2))

        return pairs

    def evaluate_noise_robustness(self, base_dataset, noise_types=None, snr_levels=None):
        """
        评估模型在不同噪声条件下的鲁棒性
        :param base_dataset: 基础数据集
        :param noise_types: 噪声类型列表
        :param snr_levels: 信噪比列表
        :return: 结果字典 {noise_type: {snr: (eer, min_dcf)}}
        """
        noise_types = noise_types or self.config.NOISE_TYPES
        snr_levels = snr_levels or self.config.SNR_LEVELS

        results = {}

        for noise_type in noise_types:
            results[noise_type] = {}

            for snr in snr_levels:
                # 创建带噪声的数据集
                noisy_dataset = SpeakerRecognitionDataset(
                    base_dataset.dataset_name,
                    split=base_dataset.split,
                    add_noise=True,
                    noise_type=noise_type,
                    snr_db=snr
                )

                # 创建数据加载器
                dataloader = torch.utils.data.DataLoader(
                    noisy_dataset,
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.config.NUM_WORKERS
                )

                # 计算嵌入向量
                embeddings = self.compute_embeddings(dataloader)

                # 评估
                eer, min_dcf = self.evaluate_voxceleb1(embeddings=embeddings)
                results[noise_type][snr] = (eer, min_dcf)

                print(f"Noise: {noise_type}, SNR: {snr}dB, EER: {eer:.4f}, minDCF: {min_dcf:.4f}")

        return results

    def evaluate_duration_sensitivity(self, base_dataset, duration_levels=None):
        """
        评估模型对不同音频时长的敏感性
        :param base_dataset: 基础数据集
        :param duration_levels: 时长列表(秒)
        :return: 结果字典 {duration: (eer, min_dcf)}
        """
        duration_levels = duration_levels or self.config.DURATION_LEVELS
        results = {}

        for duration in duration_levels:
            # 创建指定时长的数据集
            duration_dataset = SpeakerRecognitionDataset(
                base_dataset.dataset_name,
                split=base_dataset.split,
                duration=duration
            )

            # 创建数据加载器
            dataloader = torch.utils.data.DataLoader(
                duration_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS
            )

            # 计算嵌入向量
            embeddings = self.compute_embeddings(dataloader)

            # 评估
            eer, min_dcf = self.evaluate_voxceleb1(embeddings=embeddings)
            results[duration] = (eer, min_dcf)

            print(f"Duration: {duration}s, EER: {eer:.4f}, minDCF: {min_dcf:.4f}")

        return results

    def evaluate_adversarial_robustness(self, base_dataset, attack_types=None, epsilons=None):
        """
        评估模型对抗攻击的鲁棒性
        :param base_dataset: 基础数据集
        :param attack_types: 攻击类型列表
        :param epsilons: 扰动强度列表
        :return: 结果字典 {attack_type: {epsilon: (eer, min_dcf)}}
        """
        attack_types = attack_types or self.config.ATTACK_TYPES
        epsilons = epsilons or self.config.EPSILONS

        results = {}

        for attack_type in attack_types:
            results[attack_type] = {}

            for epsilon in epsilons:
                # 创建对抗攻击数据集
                adversarial_dataset = AdversarialDataset(
                    base_dataset,
                    self.model,
                    attack_type=attack_type,
                    epsilon=epsilon,
                    device=self.device
                )

                # 创建数据加载器
                dataloader = torch.utils.data.DataLoader(
                    adversarial_dataset,
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.config.NUM_WORKERS
                )

                # 计算嵌入向量
                embeddings = self.compute_embeddings(dataloader)

                # 评估
                eer, min_dcf = self.evaluate_voxceleb1(embeddings=embeddings)
                results[attack_type][epsilon] = (eer, min_dcf)

                print(f"Attack: {attack_type}, Epsilon: {epsilon}, EER: {eer:.4f}, minDCF: {min_dcf:.4f}")

        return results

    def visualize_embeddings(self, dataloader, num_samples=1000, method='tsne'):
        """
        可视化嵌入向量
        :param dataloader: 数据加载器
        :param num_samples: 采样数量
        :param method: 可视化方法('tsne'或'umap')
        :return: 可视化数据(坐标和标签)
        """
        from sklearn.manifold import TSNE
        import umap

        # 计算嵌入向量
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                embeddings, _ = self.model(inputs)

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # 限制样本数量
                if len(all_embeddings) * self.config.BATCH_SIZE >= num_samples:
                    break

        all_embeddings = np.vstack(all_embeddings)[:num_samples]
        all_labels = np.hstack(all_labels)[:num_samples]

        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported visualization method: {method}")

        # 执行降维
        embedding_2d = reducer.fit_transform(all_embeddings)

        return embedding_2d, all_labels


# 对抗攻击数据集
class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, model, attack_type='fgsm', epsilon=0.01, device=None):
        """
        对抗攻击数据集
        :param base_dataset: 基础数据集
        :param model: 目标模型
        :param attack_type: 攻击类型
        :param epsilon: 扰动强度
        :param device: 计算设备
        """
        self.base_dataset = base_dataset
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将模型设置为评估模式
        self.model.eval()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 获取原始样本
        inputs, label = self.base_dataset[idx]
        inputs = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)

        # 应用对抗攻击
        if self.attack_type == 'fgsm':
            perturbed_inputs = self._fgsm_attack(inputs, label)
        elif self.attack_type == 'pgd':
            perturbed_inputs = self._pgd_attack(inputs, label)
        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")

        # 返回扰动后的样本
        return perturbed_inputs.squeeze(0).cpu().numpy(), label

    def _fgsm_attack(self, inputs, label):
        """
        快速梯度符号法(FGSM)攻击
        :param inputs: 输入样本
        :param label: 真实标签
        :return: 扰动后的样本
        """
        inputs.requires_grad = True

        # 前向传播
        self.model.zero_grad()
        embeddings, logits = self.model(inputs)
        loss = F.cross_entropy(logits, torch.tensor([label]).to(self.device))

        # 计算梯度
        loss.backward()

        # 生成扰动
        perturbation = self.epsilon * torch.sign(inputs.grad.data)

        # 应用扰动
        perturbed_inputs = inputs + perturbation
        perturbed_inputs = torch.clamp(perturbed_inputs, -1, 1)

        return perturbed_inputs

    def _pgd_attack(self, inputs, label, steps=10, alpha=None):
        """
        投影梯度下降(PGD)攻击
        :param inputs: 输入样本
        :param label: 真实标签
        :param steps: 迭代步数
        :param alpha: 步长(如果为None，则设置为epsilon/steps)
        :return: 扰动后的样本
        """
        alpha = alpha or self.epsilon / steps

        # 初始化扰动
        delta = torch.zeros_like(inputs, requires_grad=True)

        for _ in range(steps):
            # 前向传播
            self.model.zero_grad()
            embeddings, logits = self.model(inputs + delta)
            loss = F.cross_entropy(logits, torch.tensor([label]).to(self.device))

            # 计算梯度
            loss.backward()

            # 更新扰动
            delta.data = delta.data + alpha * torch.sign(delta.grad.data)
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.data = torch.clamp(inputs + delta.data, -1, 1) - inputs

            # 重置梯度
            delta.grad.zero_()

        # 应用扰动
        perturbed_inputs = inputs + delta.data
        perturbed_inputs = torch.clamp(perturbed_inputs, -1, 1)

        return perturbed_inputs


# 主函数示例
if __name__ == "__main__":
    # 加载模型
    model = CHiLAPModel()
    model.load_state_dict(torch.load("path/to/checkpoint.pth")["model_state_dict"])

    # 创建评估器
    evaluator = Evaluator(model)

    # 加载测试数据
    dataloaders = get_dataloaders("voxceleb1")

    # 评估基础性能
    eer, min_dcf = evaluator.evaluate_voxceleb1(dataloader=dataloaders["test"])
    print(f"Base EER: {eer:.4f}, minDCF: {min_dcf:.4f}")

    # 评估噪声鲁棒性
    noise_results = evaluator.evaluate_noise_robustness(dataloaders["test"])

    # 评估时长敏感性
    duration_results = evaluator.evaluate_duration_sensitivity(dataloaders["test"])

    # 评估对抗攻击鲁棒性
    adversarial_results = evaluator.evaluate_adversarial_robustness(dataloaders["test"])

    # 可视化嵌入向量
    embedding_2d, labels = evaluator.visualize_embeddings(dataloaders["test"])

    # 保存结果
    results = {
        "base_performance": {"eer": eer, "min_dcf": min_dcf},
        "noise_robustness": noise_results,
        "duration_sensitivity": duration_results,
        "adversarial_robustness": adversarial_results
    }

    import json

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)    