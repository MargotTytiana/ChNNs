"""增强版说话人识别评估脚本 - 支持x-vector模型全面分析"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas as pd
import joblib
import seaborn as sns
from xvector_model import XVector  # 确保xvector_model.py在同一目录
from LS_DataLoader import LibriSpeechDataset, collate_fn  # 使用重写的数据加载器


# 配置日志和绘图
def setup_environment(result_dir):
    """创建结果目录并设置绘图风格"""
    os.makedirs(result_dir, exist_ok=True)
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })


def load_model(model_path, num_classes, device):
    """加载预训练的x-vector模型"""
    model = XVector(input_dim=39, emb_dim=512, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_eer(fpr, tpr, thresholds):
    """计算等错误率(EER)"""
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thr = interp1d(fpr, thresholds)(eer)
    return eer, thr


def compute_min_dcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    """计算最小检测代价函数(minDCF)"""
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    for i in range(len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    return min_c_det, min_c_det_threshold


def extract_embeddings(model, data_loader, device, feature_stats=None):
    """从数据集中提取所有嵌入向量和标签"""
    embeddings = []
    labels = []
    predictions = []
    speaker_ids = []
    file_paths = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取嵌入向量"):
            features = batch['features'].to(device)
            lengths = batch['lengths'].to(device)

            # 特征归一化（如果提供了统计信息）
            if feature_stats:
                mean = torch.tensor(feature_stats['mean'], device=device, dtype=torch.float32)
                std = torch.tensor(feature_stats['std'], device=device, dtype=torch.float32)
                features = (features - mean) / (std + 1e-5)

            # 前向传播（修正：不再进行转置）
            _, emb = model(features)  # 输入应为(batch, seq_len, feat_dim)

            # 收集结果
            embeddings.append(emb.cpu().numpy())
            labels.append(batch['labels'].numpy())
            predictions.append(_.argmax(dim=1).cpu().numpy())
            speaker_ids.extend(batch['speaker_ids'])
            file_paths.extend(batch.get('file_paths', [str(i) for i in range(len(batch['labels']))]))

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)

    return embeddings, labels, predictions, speaker_ids, file_paths


def plot_det_curve(fpr, fnr, result_dir):
    """绘制检测错误权衡(DET)曲线"""
    plt.figure()
    plt.semilogx(fpr, fnr, 'b-', linewidth=2)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.grid(True, which='both', linestyle='--')

    # 标记EER点
    eer_index = np.argmin(np.abs(fpr - fnr))
    plt.plot(fpr[eer_index], fnr[eer_index], 'ro', label=f'EER={fpr[eer_index]:.3f}')
    plt.legend()

    plt.savefig(os.path.join(result_dir, 'det_curve.png'))
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, result_dir):
    """绘制ROC曲线"""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, result_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names[:20],
                yticklabels=class_names[:20])
    plt.title('Normalized Confusion Matrix (Top 20 Speakers)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()


def plot_embedding_tsne(embeddings, labels, speaker_mapping, result_dir, n_speakers=20):
    """使用t-SNE可视化嵌入空间"""
    # 选择前n_speakers个说话人
    unique_labels = np.unique(labels)
    if len(unique_labels) > n_speakers:
        selected_labels = unique_labels[:n_speakers]
        mask = np.isin(labels, selected_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]

    # 反转speaker_mapping用于显示
    idx_to_speaker = {v: k for k, v in speaker_mapping.items()}
    speaker_names = [idx_to_speaker.get(lbl, f"Speaker_{lbl}") for lbl in labels]

    # 运行t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制结果
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels,
                          cmap='tab20', alpha=0.6, s=10)

    # 添加图例（只显示部分）
    handles, labels_scatter = scatter.legend_elements(num=min(20, len(np.unique(labels))))
    legend_labels = [speaker_names[i] for i in range(len(handles))]
    plt.legend(handles, legend_labels, title='Speakers', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('t-SNE Visualization of Speaker Embeddings')
    plt.savefig(os.path.join(result_dir, 'tsne_embeddings.png'), bbox_inches='tight')
    plt.close()


def compute_speaker_verification_scores(embeddings, labels, speaker_mapping, result_dir):
    """计算说话人验证性能指标"""
    # 为每个说话人创建平均嵌入
    speaker_embeddings = {}
    for speaker_id, idx in speaker_mapping.items():
        mask = np.array(labels) == idx
        if np.sum(mask) > 0:
            speaker_embeddings[speaker_id] = np.mean(embeddings[mask], axis=0)

    # 生成正负样本对
    positive_scores = []
    negative_scores = []

    # 正样本对：相同说话人的不同样本
    for speaker_id, emb in speaker_embeddings.items():
        mask = np.array(labels) == speaker_mapping[speaker_id]
        speaker_embs = embeddings[mask]

        # 计算所有样本对（避免自比较）
        for i in range(len(speaker_embs)):
            for j in range(i + 1, len(speaker_embs)):
                score = np.dot(speaker_embs[i], speaker_embs[j]) / (
                        np.linalg.norm(speaker_embs[i]) * np.linalg.norm(speaker_embs[j]))
                positive_scores.append(score)

    # 负样本对：不同说话人的样本
    speaker_ids = list(speaker_embeddings.keys())
    for i in range(len(speaker_ids)):
        for j in range(i + 1, len(speaker_ids)):
            score = np.dot(speaker_embeddings[speaker_ids[i]], speaker_embeddings[speaker_ids[j]]) / (
                    np.linalg.norm(speaker_embeddings[speaker_ids[i]]) *
                    np.linalg.norm(speaker_embeddings[speaker_ids[j]]))
            negative_scores.append(score)

    # 合并所有分数
    y_true = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])
    y_scores = np.concatenate([positive_scores, negative_scores])

    # 计算ROC曲线和EER
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr  # 计算False Negative Rate

    # 计算EER和minDCF
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    min_dcf, min_dcf_threshold = compute_min_dcf(fnr, fpr, thresholds)

    # 绘制曲线
    plot_det_curve(fpr, fnr, result_dir)
    plot_roc_curve(fpr, tpr, roc_auc, result_dir)

    # 保存分数分布图
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=100, alpha=0.5, label='Positive Pairs (Same Speaker)')
    plt.hist(negative_scores, bins=100, alpha=0.5, label='Negative Pairs (Different Speakers)')
    plt.axvline(x=eer_threshold, color='r', linestyle='--', label=f'EER Threshold={eer_threshold:.2f}')
    plt.axvline(x=min_dcf_threshold, color='g', linestyle='--', label=f'minDCF Threshold={min_dcf_threshold:.2f}')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Count')
    plt.title('Speaker Verification Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'score_distribution.png'))
    plt.close()

    return eer, min_dcf, roc_auc, eer_threshold


def analyze_top_errors(y_true, y_pred, speaker_mapping, file_paths, result_dir, top_n=20):
    """分析分类错误最多的说话人"""
    # 创建反向映射：索引 -> 说话人ID
    idx_to_speaker = {idx: spk for spk, idx in speaker_mapping.items()}

    # 计算每个说话人的错误率
    speaker_errors = {}
    speaker_counts = {}

    for i in range(len(y_true)):
        # 使用反向映射获取说话人ID
        speaker_id = idx_to_speaker[y_true[i]]

        if speaker_id not in speaker_counts:
            speaker_counts[speaker_id] = 0
            speaker_errors[speaker_id] = 0

        speaker_counts[speaker_id] += 1
        if y_true[i] != y_pred[i]:
            speaker_errors[speaker_id] += 1

    # 计算错误率
    error_rates = {}
    for speaker_id in speaker_counts:
        error_rates[speaker_id] = speaker_errors[speaker_id] / speaker_counts[speaker_id]

    # 排序并取top_n
    sorted_errors = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # 创建结果表格
    error_df = pd.DataFrame(sorted_errors, columns=['Speaker ID', 'Error Rate'])
    error_df['Total Samples'] = [speaker_counts[sp] for sp in error_df['Speaker ID']]
    error_df['Error Count'] = [speaker_errors[sp] for sp in error_df['Speaker ID']]

    # 保存到CSV
    error_df.to_csv(os.path.join(result_dir, 'top_error_speakers.csv'), index=False)

    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_errors)), [x[1] for x in sorted_errors])
    plt.xticks(range(len(sorted_errors)), [x[0] for x in sorted_errors], rotation=45)
    plt.xlabel('Speaker ID')
    plt.ylabel('Error Rate')
    plt.title(f'Top {top_n} Speakers with Highest Error Rates')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'top_error_speakers.png'))
    plt.close()

    return error_df


def evaluate_speaker_identification(y_true, y_pred, speaker_mapping, result_dir):
    """评估说话人辨认性能"""
    # 计算准确率
    accuracy = np.mean(y_true == y_pred)

    # 创建反向映射用于分类报告
    idx_to_speaker = {idx: spk for spk, idx in speaker_mapping.items()}
    speaker_names = [idx_to_speaker[i] for i in range(len(speaker_mapping))]

    # 生成分类报告
    class_report = classification_report(y_true, y_pred, target_names=speaker_names[:50], output_dict=True)

    # 保存报告
    report_df = pd.DataFrame(class_report).transpose()
    report_df.to_csv(os.path.join(result_dir, 'classification_report.csv'))

    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, speaker_names, result_dir)

    # 分析错误最多的说话人
    top_errors_df = analyze_top_errors(y_true, y_pred, speaker_mapping, [], result_dir)

    return accuracy, report_df, top_errors_df


def save_results(results, result_dir):
    """保存评估结果到文本文件"""
    with open(os.path.join(result_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("===== Speaker Recognition Evaluation Results =====\n\n")

        # 说话人辨认结果
        f.write("--- Speaker Identification ---\n")
        f.write(f"Accuracy: {results['id_accuracy']:.4f}\n")
        f.write(f"Top-1 Error Rate: {1 - results['id_accuracy']:.4f}\n\n")

        # 说话人验证结果
        f.write("--- Speaker Verification ---\n")
        f.write(f"Equal Error Rate (EER): {results['eer']:.4f}\n")
        f.write(f"Minimum Detection Cost (minDCF): {results['min_dcf']:.4f}\n")
        f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
        f.write(f"EER Threshold: {results['eer_threshold']:.4f}\n\n")

        # 附加信息
        f.write("--- Additional Information ---\n")
        f.write(f"Number of Speakers: {results['num_speakers']}\n")
        f.write(f"Number of Samples: {results['num_samples']}\n")
        f.write(f"Average Embedding Norm: {results['avg_emb_norm']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='X-Vector Speaker Recognition Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory to features')
    parser.add_argument('--feature_stats', type=str, required=True, help='Path to feature statistics')  # 改为必需参数
    parser.add_argument('--speaker_mapping', type=str, required=True, help='Path to speaker mapping file')
    parser.add_argument('--result_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--subset', type=str, default='test-clean', help='Subset to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # 设置环境和设备
    setup_environment(args.result_dir)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 加载说话人映射
    speaker_mapping = joblib.load(args.speaker_mapping)
    num_classes = len(speaker_mapping)
    print(f"Loaded speaker mapping with {num_classes} speakers")

    # 加载特征统计信息
    feature_stats = joblib.load(args.feature_stats)
    print("Loaded feature statistics")

    # 创建数据集
    test_dataset = LibriSpeechDataset(
        metadata=args.metadata,
        feature_dir=args.feature_dir,
        feature_stats=feature_stats,  # 传入特征统计信息
        augment=False,
        subset=args.subset
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    print(f"Created test loader with {len(test_loader.dataset)} samples")

    # 加载模型
    model = load_model(args.model_path, num_classes, device)
    print("Model loaded successfully")

    # 提取嵌入向量和预测结果（传入特征统计信息）
    embeddings, labels, predictions, speaker_ids, file_paths = extract_embeddings(
        model, test_loader, device, feature_stats
    )
    print(f"Extracted embeddings: {embeddings.shape}")

    # 评估结果存储
    results = {
        'num_speakers': num_classes,
        'num_samples': len(test_loader.dataset),
        'avg_emb_norm': np.mean(np.linalg.norm(embeddings, axis=1))
    }

    # 评估说话人辨认性能
    id_accuracy, id_report, top_errors = evaluate_speaker_identification(
        labels, predictions, speaker_mapping, args.result_dir
    )
    results['id_accuracy'] = id_accuracy
    print(f"Speaker Identification Accuracy: {id_accuracy:.4f}")

    # 评估说话人验证性能
    eer, min_dcf, roc_auc, eer_threshold = compute_speaker_verification_scores(
        embeddings, labels, speaker_mapping, args.result_dir
    )
    results.update({
        'eer': eer,
        'min_dcf': min_dcf,
        'roc_auc': roc_auc,
        'eer_threshold': eer_threshold
    })
    print(f"Speaker Verification EER: {eer:.4f}, minDCF: {min_dcf:.4f}, AUC: {roc_auc:.4f}")

    # 可视化嵌入空间
    plot_embedding_tsne(embeddings, labels, speaker_mapping, args.result_dir)
    print("Embedding visualization completed")

    # 保存完整结果
    save_results(results, args.result_dir)

    # 保存嵌入向量和标签（用于进一步分析）
    np.save(os.path.join(args.result_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.result_dir, 'labels.npy'), labels)
    np.save(os.path.join(args.result_dir, 'predictions.npy'), predictions)

    print(f"All results saved to: {args.result_dir}")


if __name__ == '__main__':
    main()