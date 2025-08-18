# train.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Any

# 导入项目模块
from librispeech_utils import LibriSpeechDataset, LibriSpeechFeatureExtractor, create_librispeech_dataloader
from c_hilap import CHiLAP, CHiLAPWithLoss
from classification import AAMSoftmax, LyapunovRegularizationLoss, PhaseSynchronizationLoss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_environment(seed: int):
    """设置随机种子和GPU环境"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"环境设置完成，随机种子: {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"从 {config_path} 加载配置")
    return config


def save_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, loss: float, path: str):
    """保存模型检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logger.info(f"在 epoch {epoch} 保存检查点到 {path}")


def train_one_epoch(
        model: CHiLAPWithLoss,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        epoch: int,
        num_epochs: int
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        # 将数据移动到设备
        features = batch['features'].to(device)
        labels = batch['speaker_labels'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播和损失计算
        loss, loss_dict = model(features, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新权重
        optimizer.step()

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 累加损失
        total_loss += loss.item()

        # 更新进度条
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce_loss=f"{loss_dict['ce_loss'].item():.4f}",
            sync_loss=f"{loss_dict['sync_loss'].item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} 完成, 平均损失: {avg_loss:.4f}")

    return avg_loss


def main(config_path: str):
    """主训练函数"""
    # 加载配置
    config = load_config(config_path)

    # 设置环境
    setup_environment(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # --- 数据准备 ---
    feature_extractor = LibriSpeechFeatureExtractor(
        sample_rate=config['data']['sample_rate'],
        feature_type=config['data']['feature_type'],
        feature_dim=config['model']['input_dim']
    )

    train_dataset = LibriSpeechDataset(
        root_dir=config['data']['root_dir'],
        split=config['data']['train_split'],
        sample_rate=config['data']['sample_rate'],
        transform=feature_extractor,
        cache_dir=config['data']['cache_dir']
    )

    train_loader = create_librispeech_dataloader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )

    num_classes = len(train_dataset.get_speakers())
    logger.info(f"训练集包含 {num_classes} 个说话人")

    # --- 模型初始化 ---
    c_hilap_model = CHiLAP(
        input_dim=config['model']['input_dim'],
        embed_dim=config['model']['embed_dim'],
        num_classes=0,  # C-HiLAP本身不进行分类
        num_layers=config['model']['num_layers'],
        chaos_type=config['model']['chaos_type'],
        ks_entropy=config['model']['ks_entropy'],
        bifurcation_factor=config['model']['bifurcation_factor']
    ).to(device)

    # 创建带有损失函数的模型
    model_with_loss = CHiLAPWithLoss(
        c_hilap=c_hilap_model,
        num_classes=num_classes,
        sync_weight=config['loss']['sync_weight']
    ).to(device)

    logger.info(f"模型初始化完成，参数量: {sum(p.numel() for p in model_with_loss.parameters() if p.requires_grad):,}")

    # --- 优化器和学习率调度器 ---
    optimizer = AdamW(
        model_with_loss.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    scheduler_type = config['scheduler']['type']
    if scheduler_type == 'cyclic':
        scheduler = CyclicLR(
            optimizer,
            base_lr=config['scheduler']['base_lr'],
            max_lr=config['scheduler']['max_lr'],
            step_size_up=len(train_loader) * config['scheduler']['step_size_up_epochs'],
            mode='triangular2',
            cycle_momentum=False
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * config['train']['num_epochs'],
            eta_min=config['scheduler']['eta_min']
        )
    else:
        scheduler = None

    # --- 训练循环 ---
    best_loss = float('inf')

    for epoch in range(config['train']['num_epochs']):
        avg_loss = train_one_epoch(
            model=model_with_loss,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            num_epochs=config['train']['num_epochs']
        )

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                epoch=epoch,
                model=model_with_loss,
                optimizer=optimizer,
                loss=avg_loss,
                path=os.path.join(config['train']['checkpoint_dir'], "best_model.pth")
            )

        # 定期保存检查点
        if (epoch + 1) % config['train']['save_interval'] == 0:
            save_checkpoint(
                epoch=epoch,
                model=model_with_loss,
                optimizer=optimizer,
                loss=avg_loss,
                path=os.path.join(config['train']['checkpoint_dir'], f"epoch_{epoch + 1}.pth")
            )

    logger.info("训练完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练C-HiLAP模型")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    main(args.config)