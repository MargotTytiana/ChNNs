import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json

# 导入项目模块
from librispeech_utils import LibriSpeechDataset, LibriSpeechFeatureExtractor, create_librispeech_dataloader
from librispeech_utils import LibriSpeechTrialGenerator, evaluate_speaker_verification
from c_hilap import CHiLAP
from variant_a import ChaosEnhancedModel_VariantA
from variant_b import ChaosEnhancedModel_VariantB
from ecapa_tdnn import ECAPA_TDNN
from xvector import XVector
from metrics import compute_eer, compute_mindcf, compute_csi, plot_det_curve, plot_embedding_space
from metrics import compute_recurrence_plot, compute_rqa_metrics, compute_lyapunov_exponent
from augmentation import AudioAugmentor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"从 {config_path} 加载配置")
    return config


def load_model(
        model_type: str,
        model_path: str,
        config: Dict[str, Any],
        device: torch.device
) -> nn.Module:
    """
    加载预训练模型。

    Args:
        model_type: 模型类型，'c_hilap', 'variant_a', 'variant_b', 'ecapa_tdnn' 或 'xvector'
        model_path: 模型检查点路径
        config: 模型配置
        device: 计算设备

    Returns:
        加载的模型
    """
    # 创建模型
    if model_type == 'c_hilap':
        model = CHiLAP(
            input_dim=config['model']['input_dim'],
            embed_dim=config['model']['embed_dim'],
            num_classes=0,  # 仅提取嵌入向量
            num_layers=config['model']['num_layers'],
            chaos_type=config['model']['chaos_type'],
            ks_entropy=config['model']['ks_entropy'],
            bifurcation_factor=config['model']['bifurcation_factor']
        ).to(device)
    elif model_type == 'variant_a':
        model = ChaosEnhancedModel_VariantA(
            acoustic_dim=config['model']['input_dim'],
            chaotic_dim=config['model']['chaotic_dim'],
            backbone_type=config['model']['backbone_type'],
            fusion_type=config['model']['fusion_type'],
            channels=config['model'].get('channels', 512),
            emb_dim=config['model']['embed_dim'],
            num_classes=0  # 仅提取嵌入向量
        ).to(device)
    elif model_type == 'variant_b':
        model = ChaosEnhancedModel_VariantB(
            input_dim=config['model']['input_dim'],
            backbone_type=config['model']['backbone_type'],
            chaos_type=config['model']['chaos_type'],
            emb_dim=config['model']['embed_dim'],
            num_classes=0  # 仅提取嵌入向量
        ).to(device)
    elif model_type == 'ecapa_tdnn':
        model = ECAPA_TDNN(
            input_dim=config['model']['input_dim'],
            channels=config['model'].get('channels', 512),
            emb_dim=config['model']['embed_dim']
        ).to(device)
    elif model_type == 'xvector':
        model = XVector(
            input_dim=config['model']['input_dim'],
            num_classes=0,  # 仅提取嵌入向量
            emb_dim=config['model']['embed_dim']
        ).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)

    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        # 训练脚本保存的格式
        if model_type == 'c_hilap' and 'c_hilap.state_dict' in checkpoint['model_state_dict']:
            # 从CHiLAPWithLoss中提取C-HiLAP模型权重
            model_state_dict = {
                k.replace('c_hilap.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('c_hilap.')
            }
            model.load_state_dict(model_state_dict)
        else:
            # 尝试直接加载
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                # 尝试移除模块前缀
                model_state_dict = {
                    k.replace('module.', ''): v
                    for k, v in checkpoint['model_state_dict'].items()
                }
                model.load_state_dict(model_state_dict)
    else:
        # 可能是直接保存的模型权重
        try:
            model.load_state_dict(checkpoint)
        except:
            # 尝试移除模块前缀
            model_state_dict = {
                k.replace('module.', ''): v
                for k, v in checkpoint.items()
            }
            model.load_state_dict(model_state_dict)

    logger.info(f"从 {model_path} 加载 {model_type} 模型")

    # 设置为评估模式
    model.eval()

    return model


def extract_embeddings(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从数据集中提取嵌入向量。

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        嵌入向量和对应的标签
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取嵌入向量"):
            # 将数据移动到设备
            features = batch['features'].to(device)
            labels = batch['speaker_labels'].cpu().numpy()

            # 提取嵌入向量
            embeddings = model(features, extract_embedding=True)

            # 收集嵌入向量和标签
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels)

    # 合并所有批次的结果
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    return all_embeddings, all_labels


def evaluate_trials(
        model: nn.Module,
        trials: List[Dict],
        feature_extractor: Any,
        device: torch.device,
        distance_metric: str = 'cosine',
        output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    评估说话人验证试验对。

    Args:
        model: 模型
        trials: 试验对列表
        feature_extractor: 特征提取器
        device: 计算设备
        distance_metric: 距离度量方式，'cosine' 或 'euclidean'
        output_path: 结果输出路径（可选）

    Returns:
        包含评估结果的字典
    """
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for trial in tqdm(trials, desc="评估试验对"):
            # 加载音频
            enroll_audio, _ = torchaudio.load(trial['enrollment'])
            test_audio, _ = torchaudio.load(trial['test'])

            # 提取特征
            enroll_feat = feature_extractor(enroll_audio).to(device)
            test_feat = feature_extractor(test_audio).to(device)

            # 添加批次维度
            enroll_feat = enroll_feat.unsqueeze(0)
            test_feat = test_feat.unsqueeze(0)

            # 提取嵌入向量
            enroll_emb = model(enroll_feat, extract_embedding=True)
            test_emb = model(test_feat, extract_embedding=True)

            # 计算相似度分数
            if distance_metric == 'cosine':
                # 余弦相似度
                score = F.cosine_similarity(enroll_emb, test_emb).item()
            elif distance_metric == 'euclidean':
                # 欧几里得距离（转换为相似度）
                distance = torch.norm(enroll_emb - test_emb).item()
                score = 1.0 / (1.0 + distance)  # 转换为[0,1]范围的相似度
            else:
                raise ValueError(f"不支持的距离度量: {distance_metric}")

            scores.append(score)
            labels.append(trial['label'])

    # 转换为numpy数组
    scores = np.array(scores)
    labels = np.array(labels)

    # 分离合法和冒充分数
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    # 计算EER和minDCF
    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    mindcf, _ = compute_mindcf(genuine_scores, impostor_scores)

    # 计算准确率
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == labels)

    # 创建结果字典
    results = {
        'eer': eer,
        'mindcf': mindcf,
        'threshold': threshold,
        'accuracy': accuracy
    }

    # 输出结果
    logger.info(f"评估结果:")
    logger.info(f"  EER: {eer:.4f}")
    logger.info(f"  minDCF: {mindcf:.4f}")
    logger.info(f"  阈值: {threshold:.4f}")
    logger.info(f"  准确率: {accuracy:.4f}")

    # 绘制DET曲线
    fig = plot_det_curve(genuine_scores, impostor_scores)

    # 保存结果
    if output_path:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存结果字典
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # 保存DET曲线
        fig.savefig(output_path.replace('.json', '_det.png'))

        logger.info(f"结果保存到 {output_path}")

    return results


def evaluate_noise_robustness(
        model: nn.Module,
        trials: List[Dict],
        feature_extractor: Any,
        device: torch.device,
        snr_levels: List[float] = [0, 5, 10, 15, 20],
        noise_types: List[str] = ['gaussian', 'babble'],
        output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    评估模型在不同噪声条件下的鲁棒性。

    Args:
        model: 模型
        trials: 试验对列表
        feature_extractor: 特征提取器
        device: 计算设备
        snr_levels: 信噪比水平列表（dB）
        noise_types: 噪声类型列表
        output_path: 结果输出路径（可选）

    Returns:
        包含不同噪声条件下评估结果的嵌套字典
    """
    model.eval()

    # 创建音频增强器
    augmentor = AudioAugmentor(sample_rate=16000)

    # 存储不同条件下的结果
    results = {}

    # 首先评估干净条件
    logger.info("评估干净条件...")
    clean_results = evaluate_trials(
        model=model,
        trials=trials,
        feature_extractor=feature_extractor,
        device=device,
        output_path=output_path.replace('.json', '_clean.json') if output_path else None
    )
    results['clean'] = clean_results

    # 评估不同噪声类型和SNR水平
    for noise_type in noise_types:
        for snr in snr_levels:
            logger.info(f"评估噪声类型: {noise_type}, SNR: {snr}dB...")

            # 创建带噪声的特征提取器
            noisy_extractor = lambda audio: feature_extractor(
                torch.tensor(augmentor.add_noise(
                    audio.numpy().squeeze(0),
                    snr_db=snr,
                    noise_type=noise_type
                )).unsqueeze(0)
            )

            # 评估带噪声的试验对
            noise_results = evaluate_trials(
                model=model,
                trials=trials,
                feature_extractor=noisy_extractor,
                device=device,
                output_path=output_path.replace('.json', f'_{noise_type}_snr{snr}.json') if output_path else None
            )

            # 存储结果
            results[f'{noise_type}_snr{snr}'] = noise_results

    # 计算混沌敏感性指数（CSI）
    if 'gaussian_snr5' in results:
        # 生成一些示例信号用于CSI计算
        clean_signal = np.random.randn(16000)  # 1秒的信号
        noisy_signal = clean_signal + 0.1 * np.random.randn(16000)  # 添加噪声

        csi = compute_csi(
            results['clean']['eer'],
            results['gaussian_snr5']['eer'],
            clean_signal,
            noisy_signal
        )

        logger.info(f"混沌敏感性指数 (CSI): {csi:.4f}")
        results['csi'] = csi

    # 保存汇总结果
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"噪声鲁棒性结果保存到 {output_path}")

    return results


def evaluate_duration_robustness(
        model: nn.Module,
        trials: List[Dict],
        feature_extractor: Any,
        device: torch.device,
        durations: List[float] = [1.0, 2.0, 3.0, 5.0],
        output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    评估模型在不同语音时长下的鲁棒性。

    Args:
        model: 模型
        trials: 试验对列表
        feature_extractor: 特征提取器
        device: 计算设备
        durations: 语音时长列表（秒）
        output_path: 结果输出路径（可选）

    Returns:
        包含不同时长条件下评估结果的嵌套字典
    """
    model.eval()

    # 存储不同条件下的结果
    results = {}

    # 评估不同时长
    for duration in durations:
        logger.info(f"评估时长: {duration}秒...")

        # 创建截断的特征提取器
        truncated_extractor = lambda audio: feature_extractor(
            audio[:, :int(16000 * duration)]  # 假设采样率为16kHz
        )

        # 评估截断的试验对
        duration_results = evaluate_trials(
            model=model,
            trials=trials,
            feature_extractor=truncated_extractor,
            device=device,
            output_path=output_path.replace('.json', f'_duration{duration}.json') if output_path else None
        )

        # 存储结果
        results[f'duration{duration}'] = duration_results

    # 保存汇总结果
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"时长鲁棒性结果保存到 {output_path}")

    return results


def evaluate_adversarial_robustness(
        model: nn.Module,
        trials: List[Dict],
        feature_extractor: Any,
        device: torch.device,
        epsilon_values: List[float] = [0.01, 0.05, 0.1],
        output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    评估模型在对抗攻击下的鲁棒性。

    Args:
        model: 模型
        trials: 试验对列表
        feature_extractor: 特征提取器
        device: 计算设备
        epsilon_values: 扰动大小列表
        output_path: 结果输出路径（可选）

    Returns:
        包含不同对抗攻击条件下评估结果的嵌套字典
    """
    model.eval()

    # 创建音频增强器
    augmentor = AudioAugmentor(sample_rate=16000)

    # 存储不同条件下的结果
    results = {}

    # 首先评估干净条件
    logger.info("评估干净条件...")
    clean_results = evaluate_trials(
        model=model,
        trials=trials,
        feature_extractor=feature_extractor,
        device=device,
        output_path=output_path.replace('.json', '_clean.json') if output_path else None
    )
    results['clean'] = clean_results

    # 评估不同扰动大小的FGSM攻击
    for epsilon in epsilon_values:
        logger.info(f"评估FGSM攻击, epsilon: {epsilon}...")

        # 创建对抗样本特征提取器
        def adversarial_extractor(audio):
            # 将音频转换为需要梯度的张量
            audio_tensor = audio.clone().detach().requires_grad_(True)

            # 提取特征
            features = feature_extractor(audio_tensor)

            # 添加批次维度
            features = features.unsqueeze(0)

            # 前向传播
            embeddings = model(features, extract_embedding=True)

            # 计算损失（最大化与原始嵌入的距离）
            loss = -torch.norm(embeddings)

            # 反向传播
            loss.backward()

            # 生成对抗样本
            audio_grad = audio_tensor.grad.sign()
            adversarial_audio = audio_tensor + epsilon * audio_grad

            # 裁剪到有效范围
            adversarial_audio = torch.clamp(adversarial_audio, -1.0, 1.0)

            # 提取对抗样本的特征
            return feature_extractor(adversarial_audio.detach())

        # 评估对抗样本
        adv_results = evaluate_trials(
            model=model,
            trials=trials[:100],  # 限制数量，因为生成对抗样本很耗时
            feature_extractor=adversarial_extractor,
            device=device,
            output_path=output_path.replace('.json', f'_fgsm_eps{epsilon}.json') if output_path else None
        )

        # 存储结果
        results[f'fgsm_eps{epsilon}'] = adv_results

    # 保存汇总结果
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"对抗鲁棒性结果保存到 {output_path}")

    return results


def visualize_embeddings(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        output_path: Optional[str] = None,
        max_samples: int = 1000
) -> None:
    """
    可视化说话人嵌入向量。

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备
        output_path: 结果输出路径（可选）
        max_samples: 最大样本数
    """
    # 提取嵌入向量
    embeddings, labels = extract_embeddings(model, dataloader, device)

    # 限制样本数量
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # 使用t-SNE可视化
    fig = plot_embedding_space(embeddings, labels, method='tsne')

    # 保存结果
    if output_path:
        fig.savefig(output_path)
        logger.info(f"嵌入向量可视化保存到 {output_path}")


def analyze_chaotic_properties(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    分析模型的混沌特性。

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备
        output_dir: 结果输出目录（可选）

    Returns:
        包含混沌特性指标的字典
    """
    model.eval()

    # 存储混沌特性指标
    chaotic_metrics = {
        'mle': [],
        'rqa_recurrence_rate': [],
        'rqa_determinism': [],
        'rqa_laminarity': []
    }

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 处理一个批次的数据
    for batch in tqdm(dataloader, desc="分析混沌特性"):
        # 将数据移动到设备
        features = batch['features'].to(device)

        # 获取中间轨迹（如果模型支持）
        try:
            # 尝试获取中间轨迹
            if hasattr(model, 'get_trajectory'):
                trajectory = model.get_trajectory(features).cpu().numpy()
            elif hasattr(model, 'c_hilap') and hasattr(model.c_hilap, 'get_trajectory'):
                trajectory = model.c_hilap.get_trajectory(features).cpu().numpy()
            else:
                # 如果模型不支持获取轨迹，使用特征作为信号
                trajectory = features.cpu().numpy()

            # 分析每个样本的混沌特性
            for i in range(trajectory.shape[0]):
                # 提取单个样本的轨迹或信号
                sample = trajectory[i]

                # 计算最大李雅普诺夫指数
                mle = compute_lyapunov_exponent(sample)
                chaotic_metrics['mle'].append(mle)

                # 计算递归图
                rp = compute_recurrence_plot(sample, embedding_dim=3, delay=1, threshold=0.1)

                # 计算RQA指标
                rqa = compute_rqa_metrics(rp)
                chaotic_metrics['rqa_recurrence_rate'].append(rqa['recurrence_rate'])
                chaotic_metrics['rqa_determinism'].append(rqa['determinism'])
                chaotic_metrics['rqa_laminarity'].append(rqa['laminarity'])

                # 可视化递归图（仅第一个样本）
                if i == 0 and output_dir:
                    plt.figure(figsize=(8, 8))
                    plt.imshow(rp, cmap='binary', origin='lower')
                    plt.title("Recurrence Plot")
                    plt.xlabel("Time Index")
                    plt.ylabel("Time Index")
                    plt.savefig(os.path.join(output_dir, "recurrence_plot.png"))
                    plt.close()
        except Exception as e:
            logger.warning(f"无法分析混沌特性: {e}")
            break

        # 只处理一个批次
        break

    # 计算平均指标
    results = {}
    for key, values in chaotic_metrics.items():
        if values:
            results[key] = float(np.mean(values))

    # 输出结果
    logger.info("混沌特性分析结果:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")

    # 保存结果
    if output_dir:
        with open(os.path.join(output_dir, "chaotic_metrics.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def compare_models(
        models: Dict[str, nn.Module],
        trials: List[Dict],
        feature_extractors: Dict[str, Any],
        device: torch.device,
        output_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    比较多个模型的性能。

    Args:
        models: 模型字典，键为模型名称，值为模型
        trials: 试验对列表
        feature_extractors: 特征提取器字典，键为模型名称，值为特征提取器
        device: 计算设备
        output_dir: 结果输出目录（可选）

    Returns:
        包含各模型评估结果的嵌套字典
    """
    results = {}

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 评估每个模型
    for model_name, model in models.items():
        logger.info(f"评估模型: {model_name}...")

        # 获取对应的特征提取器
        feature_extractor = feature_extractors.get(model_name, feature_extractors.get('default'))

        # 评估模型
        model_results = evaluate_trials(
            model=model,
            trials=trials,
            feature_extractor=feature_extractor,
            device=device,
            output_path=os.path.join(output_dir, f"{model_name}_results.json") if output_dir else None
        )

        # 存储结果
        results[model_name] = model_results

    # 创建比较表格
    comparison = {
        'model': [],
        'eer': [],
        'mindcf': [],
        'accuracy': []
    }

    for model_name, model_results in results.items():
        comparison['model'].append(model_name)
        comparison['eer'].append(model_results['eer'])
        comparison['mindcf'].append(model_results['mindcf'])
        comparison['accuracy'].append(model_results['accuracy'])

    # 创建DataFrame
    df = pd.DataFrame(comparison)

    # 输出比较表格
    logger.info("\n模型比较:")
    logger.info(df.to_string(index=False))

    # 保存比较表格
    if output_dir:
        df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    return results


def main(config_path: str):
    """
    主评估函数。

    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    output_dir = config['evaluation']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- 数据准备 ---
    # 创建特征提取器
    feature_extractor = LibriSpeechFeatureExtractor(
        sample_rate=config['data']['sample_rate'],
        feature_type=config['data']['feature_type'],
        feature_dim=config['model']['input_dim']
    )

    # 加载测试集
    test_dataset = LibriSpeechDataset(
        root_dir=config['data']['root_dir'],
        split=config['data']['test_split'],
        sample_rate=config['data']['sample_rate'],
        transform=feature_extractor,
        max_duration=config['data'].get('max_duration'),
        cache_dir=config['data']['cache_dir'],
        use_cache=True
    )

    # 创建测试数据加载器
    test_loader = create_librispeech_dataloader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers']
    )

    # 生成或加载试验对
    trials_path = config['evaluation'].get('trials_path')
    if trials_path and os.path.exists(trials_path):
        # 加载现有试验对
        with open(trials_path, 'r') as f:
            trials = json.load(f)
        logger.info(f"从 {trials_path} 加载了 {len(trials)} 个试验对")
    else:
        # 生成新的试验对
        trial_generator = LibriSpeechTrialGenerator(
            dataset=test_dataset,
            num_trials=config['evaluation'].get('num_trials', 1000),
            genuine_ratio=config['evaluation'].get('genuine_ratio', 0.5)
        )
        trials = trial_generator.generate_trials()

        # 保存试验对
        if trials_path:
            os.makedirs(os.path.dirname(trials_path), exist_ok=True)
            with open(trials_path, 'w') as f:
                json.dump(trials, f, indent=2)
            logger.info(f"生成并保存了 {len(trials)} 个试验对到 {trials_path}")

    # --- 模型加载 ---
    # 加载主模型
    model_path = config['model']['checkpoint']
    model_type = config['model']['type']
    model = load_model(model_type, model_path, config, device)

    # --- 评估 ---
    # 执行基本评估
    basic_results = evaluate_trials(
        model=model,
        trials=trials,
        feature_extractor=feature_extractor,
        device=device,
        output_path=os.path.join(output_dir, "basic_results.json")
    )

    # 执行噪声鲁棒性评估
    if config['evaluation'].get('evaluate_noise', False):
        noise_results = evaluate_noise_robustness(
            model=model,
            trials=trials,
            feature_extractor=feature_extractor,
            device=device,
            snr_levels=config['evaluation'].get('snr_levels', [0, 5, 10, 15, 20]),
            noise_types=config['evaluation'].get('noise_types', ['gaussian', 'babble']),
            output_path=os.path.join(output_dir, "noise_robustness_results.json")
        )

    # 执行时长鲁棒性评估
    if config['evaluation'].get('evaluate_duration', False):
        duration_results = evaluate_duration_robustness(
            model=model,
            trials=trials,
            feature_extractor=feature_extractor,
            device=device,
            durations=config['evaluation'].get('durations', [1.0, 2.0, 3.0, 5.0]),
            output_path=os.path.join(output_dir, "duration_robustness_results.json")
        )

    # 执行对抗鲁棒性评估
    if config['evaluation'].get('evaluate_adversarial', False):
        adversarial_results = evaluate_adversarial_robustness(
            model=model,
            trials=trials,
            feature_extractor=feature_extractor,
            device=device,
            epsilon_values=config['evaluation'].get('epsilon_values', [0.01, 0.05, 0.1]),
            output_path=os.path.join(output_dir, "adversarial_robustness_results.json")
        )

    # 可视化嵌入向量
    if config['evaluation'].get('visualize_embeddings', False):
        visualize_embeddings(
            model=model,
            dataloader=test_loader,
            device=device,
            output_path=os.path.join(output_dir, "embeddings_visualization.png"),
            max_samples=config['evaluation'].get('max_samples', 1000)
        )

    # 分析混沌特性
    if config['evaluation'].get('analyze_chaotic', False):
        chaotic_metrics = analyze_chaotic_properties(
            model=model,
            dataloader=test_loader,
            device=device,
            output_dir=os.path.join(output_dir, "chaotic_analysis")
        )

    # 比较多个模型（如果指定）
    if 'comparison_models' in config['evaluation']:
        models = {'main': model}
        feature_extractors = {'main': feature_extractor, 'default': feature_extractor}

        # 加载比较模型
        for comp_name, comp_config in config['evaluation']['comparison_models'].items():
            comp_model = load_model(
                comp_config['type'],
                comp_config['checkpoint'],
                comp_config,
                device
            )
            models[comp_name] = comp_model

            # 如果有特定的特征提取器
            if 'feature_type' in comp_config:
                comp_feature_extractor = LibriSpeechFeatureExtractor(
                    sample_rate=config['data']['sample_rate'],
                    feature_type=comp_config['feature_type'],
                    feature_dim=comp_config['input_dim']
                )
                feature_extractors[comp_name] = comp_feature_extractor

        # 比较模型
        comparison_results = compare_models(
            models=models,
            trials=trials,
            feature_extractors=feature_extractors,
            device=device,
            output_dir=os.path.join(output_dir, "model_comparison")
        )

    logger.info("评估完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估说话人识别模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    main(args.config)