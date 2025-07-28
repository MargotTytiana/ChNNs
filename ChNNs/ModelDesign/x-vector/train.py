import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from LibriSpeechDataLoader import LibriSpeechDataset
from xvector_model import XVector
from aam_loss import AAMSoftmax
import numpy as np
import os
from tqdm import tqdm


def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # 获取数据和标签
        data = batch['audio'].to(device)  # 原始音频
        labels = batch['label'].to(device)

        # 特征提取（这里需要添加MFCC特征提取步骤）
        # 实际应用中，最好在数据加载时预提取特征
        mfcc_features = extract_mfcc(data)  # 伪代码，需要实现

        # 前向传播
        optimizer.zero_grad()
        outputs, embeddings = model(mfcc_features)
        loss = loss_fn(embeddings, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total
    print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%')
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='X-Vector Training')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of speakers')
    parser.add_argument('--save_model', type=str, default='model', help='Path to save the model')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_dataset = LibriSpeechDataset(
        metadata=args.metadata,
        subset='train-clean-100'
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    model = XVector(input_dim=39, emb_dim=512, num_classes=args.num_classes).to(device)
    loss_fn = AAMSoftmax(emb_dim=512, num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(args.save_model, 'xvector_model.pth'))
    torch.save(loss_fn.state_dict(), os.path.join(args.save_model, 'aam_softmax.pth'))


if __name__ == '__main__':
    main()