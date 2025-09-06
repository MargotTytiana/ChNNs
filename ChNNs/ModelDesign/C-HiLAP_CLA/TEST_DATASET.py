# 数据流程调试脚本
import glob
import os

import matplotlib.pyplot as plt
from data_loader import create_dataloaders


def debug_features():
    # 创建小型测试数据集
    train_loader, _, _, _ = create_dataloaders(
        train_dir="P:/PycharmProjects/pythonProject1/dataset",
        dev_dir=None,
        test_dir=None,
        batch_size=4,
        cache_dir=None
    )

    # 获取第一个batch
    batch = next(iter(train_loader))
    features = batch['audio']
    labels = batch['label']

    # 打印特征信息
    print(f"特征形状: {features.shape} (应为 [batch, feature_dim])")
    print(f"特征示例统计:")
    print(f"  最小值: {features.min().item():.4f}")
    print(f"  最大值: {features.max().item():.4f}")
    print(f"  平均值: {features.mean().item():.4f}")
    print(f"  标准差: {features.std().item():.4f}")

    # 可视化第一个样本的特征
    plt.figure(figsize=(12, 6))
    plt.title(f"Speaker {labels[0].item()} 的混沌特征")
    plt.plot(features[0].numpy().T)  # 假设特征维度为 [timesteps, features]
    plt.xlabel("Time steps")
    plt.ylabel("Feature value")
    plt.show()


def validate_labels():
    """验证标签一致性"""
    _, _, _, speaker_mapping = create_dataloaders(
        train_dir="P:/PycharmProjects/pythonProject1/dataset",
        dev_dir="P:/PycharmProjects/pythonProject1/devDataset",
        test_dir="P:/PycharmProjects/pythonProject1/testDataset",
        cache_dir=None
    )

    # 验证所有数据集的speaker都在全局映射中
    for dataset_type in ['train', 'dev', 'test']:
        dir_path = f"P:/PycharmProjects/pythonProject1/{'dataset' if dataset_type == 'train' else dataset_type + 'Dataset'}"
        files = glob.glob(os.path.join(dir_path, "**", "*.flac"), recursive=True)
        speakers = set(os.path.basename(os.path.dirname(os.path.dirname(f))) for f in files)

        missing = [s for s in speakers if s not in speaker_mapping]
        if missing:
            print(f"警告！{dataset_type}集中存在未映射的说话人: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        else:
            print(f"{dataset_type}集标签映射正常")


if __name__ == "__main__":
    validate_labels()
    debug_features()
