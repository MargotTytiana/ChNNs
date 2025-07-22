import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import glob
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 数据预处理 - 适配您的数据集结构
class SpeechEmotionDataset(Dataset):
    def __init__(self, base_path, max_length=500, n_mfcc=40, actor_ids=None):
        """
        :param base_path: 数据集根目录 (e.g., 'F:/F/LifeLongLearning/TUNI/Thesis/Code/archive')
        :param max_length: MFCC序列最大长度
        :param n_mfcc: MFCC特征维度
        :param actor_ids: 使用的演员ID列表 (None表示使用全部)
        """
        self.data = []
        self.labels = []
        self.max_length = max_length
        self.n_mfcc = n_mfcc

        # 情感标签映射
        emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

        # 获取所有演员目录
        if actor_ids is None:
            actor_dirs = glob.glob(os.path.join(base_path, 'Actor_*'))
        else:
            actor_dirs = [os.path.join(base_path, f'Actor_{id:02d}') for id in actor_ids]

        print(f"处理{len(actor_dirs)}个演员目录...")

        # 遍历所有演员目录
        for actor_dir in actor_dirs:
            if not os.path.isdir(actor_dir):
                continue

            # 获取该演员的所有wav文件
            wav_files = glob.glob(os.path.join(actor_dir, '*.wav'))

            for wav_file in wav_files:
                # 提取文件名并解析情感
                filename = os.path.basename(wav_file)
                parts = filename.split('-')

                # 确保文件名格式正确
                if len(parts) < 7:
                    continue

                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, None)

                if emotion is None:
                    continue

                # 加载音频并提取MFCC特征
                try:
                    y, sr = librosa.load(wav_file, sr=16000)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

                    # 标准化并填充/截断到固定长度
                    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                    if mfcc.shape[1] > max_length:
                        mfcc = mfcc[:, :max_length]
                    else:
                        pad_width = max_length - mfcc.shape[1]
                        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

                    # 添加到数据集
                    self.data.append(mfcc)
                    self.labels.append(emotion)
                except Exception as e:
                    print(f"处理文件 {wav_file} 时出错: {str(e)}")

        # 将标签转换为数字
        self.emotion_to_idx = {e: i for i, e in enumerate(set(self.labels))}
        self.labels_idx = [self.emotion_to_idx[l] for l in self.labels]
        self.num_classes = len(self.emotion_to_idx)

        print(f"数据集加载完成: {len(self.data)}个样本, {self.num_classes}种情感")
        print("情感分布:")
        for emotion, idx in self.emotion_to_idx.items():
            count = self.labels.count(emotion)
            print(f"  {emotion}: {count}个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels_idx[idx], dtype=torch.long)


# 2. Rossler混沌神经元实现
class RosslerCell(nn.Module):
    def __init__(self, input_size, hidden_size, a=0.2, b=0.2, c=5.7, dt=0.1):
        super(RosslerCell, self).__init__()
        self.hidden_size = hidden_size
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        self.dt = dt

        # 输入到隐藏状态的权重
        self.w_ih = nn.Linear(input_size, 3 * hidden_size)

        # 隐藏状态到隐藏状态的权重
        self.w_hh = nn.Linear(3 * hidden_size, 3 * hidden_size)

        # Lyapunov指数监控
        self.le_reg = LyapunovRegularizer(target_le=0.05)

        # 初始化隐藏状态
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重"""
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hx=None):
        batch_size = x.size(0)

        # Initialize hidden state if None
        if hx is None:
            hx = torch.zeros(batch_size, 3, self.hidden_size, device=x.device)
            hx += 0.01 * torch.randn_like(hx)

        # Split input and hidden state
        x_state = hx[:, 0, :]  # [batch_size, hidden_size]
        y_state = hx[:, 1, :]
        z_state = hx[:, 2, :]

        # Process input
        i2h = self.w_ih(x)  # [batch_size, 3*hidden_size]
        i2h = i2h.view(batch_size, 3, self.hidden_size)
        i_x = i2h[:, 0, :]
        i_y = i2h[:, 1, :]
        i_z = i2h[:, 2, :]

        # Rossler equations
        dx = (-y_state - z_state + i_x) * self.dt
        dy = (x_state + self.a * y_state + i_y) * self.dt
        dz = (self.b + z_state * (x_state - self.c) + i_z) * self.dt

        # Update states
        new_x = x_state + dx
        new_y = y_state + dy
        new_z = z_state + dz

        # Combine new states
        new_hx = torch.stack([new_x, new_y, new_z], dim=1)  # [batch_size, 3, hidden_size]

        # Compute Lyapunov loss
        le_loss = self.le_reg(new_hx)

        return new_hx, le_loss


# Lyapunov正则化器
class LyapunovRegularizer(nn.Module):
    def __init__(self, target_le=0.05, reg_strength=0.1):
        super().__init__()
        self.target_le = target_le
        self.reg_strength = reg_strength
        self.le_history = []

    def forward(self, state):
        # state shape: [batch_size, 3, hidden_size]
        batch_size, _, hidden_size = state.shape

        with torch.enable_grad():
            state = state.detach().requires_grad_(True)

            # Create perturbation with same shape as state
            perturbation = 1e-6 * torch.randn_like(state)

            # Split state into components
            x = state[:, 0, :]  # [batch_size, hidden_size]
            y = state[:, 1, :]
            z = state[:, 2, :]

            # Compute derivatives
            dx = -y - z
            dy = x + 0.2 * y
            dz = 0.2 + z * (x - 5.7)

            # Compute Jacobian-vector product approximation
            # We'll compute the effect of perturbation on each component

            # Compute how perturbation affects dx
            jvp_dx = torch.autograd.grad(dx.sum(), state, grad_outputs=torch.ones_like(dx),
                                         retain_graph=True, create_graph=False)[0]

            # Compute how perturbation affects dy
            jvp_dy = torch.autograd.grad(dy.sum(), state, grad_outputs=torch.ones_like(dy),
                                         retain_graph=True, create_graph=False)[0]

            # Compute how perturbation affects dz
            jvp_dz = torch.autograd.grad(dz.sum(), state, grad_outputs=torch.ones_like(dz),
                                         retain_graph=True, create_graph=False)[0]

            # Combine the effects
            jvp = torch.stack([jvp_dx, jvp_dy, jvp_dz], dim=1)  # [batch_size, 3, 3, hidden_size]

            # Compute norm (simplified approximation)
            # For each batch, compute the Frobenius norm of the Jacobian
            norm = torch.norm(jvp, p='fro', dim=(1, 2, 3))  # [batch_size]

            # Approximate Lyapunov exponent
            le_approx = torch.log(norm) / 1.0

        current_le = le_approx.mean().item()
        self.le_history.append(current_le)

        reg_loss = self.reg_strength * torch.abs(le_approx.mean() - self.target_le)

        return reg_loss


# 3. 混沌神经网络架构
class ChaoticSpeechNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(ChaoticSpeechNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Rossler混沌层
        self.chaos_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.chaos_cells.append(RosslerCell(hidden_size, hidden_size))

        # 输出层
        self.fc = nn.Linear(3 * hidden_size, num_classes)

        # 层归一化
        self.ln = nn.LayerNorm(hidden_size)

        # 注意力机制
        self.attention = nn.Linear(3 * hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        le_loss_total = 0.0

        # 初始化隐藏状态
        hx = None
        all_states = []

        # 处理序列
        for t in range(seq_len):
            # 输入投影
            x_t = self.input_proj(x[:, t, :])

            # 通过所有混沌层
            for i, cell in enumerate(self.chaos_cells):
                if i == 0:
                    input_t = x_t
                else:
                    input_t = h_state

                hx, le_loss = cell(input_t, hx)
                le_loss_total += le_loss
                h_state = hx[:, -1, :]
                h_state = self.ln(h_state)

            all_states.append(hx)

        all_states = torch.stack(all_states, dim=0)
        flat_states = all_states.view(seq_len, batch_size, -1)

        # 注意力加权
        attn_weights = torch.softmax(self.attention(flat_states).squeeze(-1), dim=0)
        context = torch.sum(attn_weights.unsqueeze(-1) * flat_states, dim=0)

        # 分类
        output = self.fc(context)

        return output, le_loss_total / seq_len


# 2. 传统LSTM模型
class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # 注意力机制
        self.attention = nn.Linear(hidden_size, 1)

        # 输出层
        self.fc = nn.Linear(hidden_size, num_classes)

        # 层归一化
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.ln(out)

        # 注意力机制
        attn_weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * out, dim=1)

        # 分类
        output = self.fc(context)

        return output, torch.tensor(0.0)  # 返回0作为LE损失占位符


# 训练函数 (适配两种模型)
def train_model(model, model_name, dataloader, test_dataloader, num_epochs=30, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 修改：移除verbose参数或根据PyTorch版本处理
    try:
        # 尝试新版本PyTorch的verbose参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        # 回退到不支持verbose的版本
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        print(f"警告: 当前PyTorch版本不支持ReduceLROnPlateau的verbose参数")

    history = {
        'train_loss': [], 'train_acc': [], 'test_acc': [],
        'time_per_epoch': [], 'le_history': []
    }

    best_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs, le_loss = model(inputs)
            ce_loss = criterion(outputs, labels)

            # 混沌模型有额外的LE损失
            if 'Chaotic' in model_name:
                loss = ce_loss + le_loss
            else:
                loss = ce_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录LE历史（仅混沌模型）
            if 'Chaotic' in model_name:
                history['le_history'].append(le_loss.item())

            if (i + 1) % 10 == 0:
                print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}')

        # 计算训练精度
        epoch_time = time.time() - start_time
        train_loss = running_loss / len(dataloader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['time_per_epoch'].append(epoch_time)

        # 在测试集上评估
        test_acc = evaluate_model(model, test_dataloader)
        history['test_acc'].append(test_acc)

        # 更新学习率
        scheduler.step(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}_model.pth')
            print(f"保存最佳模型，测试精度: {test_acc:.2f}%")

        print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], '
              f'Time: {epoch_time:.1f}s, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    return history


# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# 结果可视化
def plot_results(history, model_name, save_prefix):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    if model_name == 'Chaotic RNN':
        plt.subplot(2, 2, 3)
        plt.plot(history['le_history'])
        plt.title('Lyapunov Exponent History')
        plt.xlabel('Step')
        plt.ylabel('LE Loss')

        plt.subplot(2, 2, 4)
        window_size = 100
        le_ma = np.convolve(history['le_history'], np.ones(window_size) / window_size, mode='valid')
        plt.plot(le_ma)
        plt.title(f'Lyapunov Exponent (Moving Avg, window={window_size})')
        plt.xlabel('Step')
        plt.ylabel('Smoothed LE')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_training_results.png')
    print(f"{model_name} 训练结果图已保存至: {save_prefix}_training_results.png")
    plt.show()


# 混淆矩阵可视化
def plot_confusion_matrix(model, dataloader, class_names, model_name, save_path):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"混淆矩阵已保存至: {save_path}")
    plt.show()

    # 打印分类报告
    print(f"\n{model_name} 分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


# 主函数
def main():
    # 参数配置
    config = {
        'base_path': 'P://PycharmProjects//pythonProject1//ChNNs//archive',
        'batch_size': 32,
        'max_length': 500,
        'n_mfcc': 40,
        'hidden_size': 128,
        'num_layers': 2,
        'lr': 0.001,
        'num_epochs': 30,
        'test_size': 0.2,
        'seed': 42,
        'actor_ids': list(range(1, 25))
    }

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 加载数据集
    print("加载数据集...")
    dataset = SpeechEmotionDataset(
        base_path=config['base_path'],
        max_length=config['max_length'],
        n_mfcc=config['n_mfcc'],
        actor_ids=config['actor_ids']
    )

    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=config['test_size'],
        random_state=config['seed'],
        stratify=dataset.labels_idx
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    class_names = list(dataset.emotion_to_idx.keys())

    # 创建并训练混沌模型
    print("\n" + "=" * 50)
    print("训练混沌神经网络")
    print("=" * 50)
    chaotic_model = ChaoticSpeechNet(
        input_size=config['n_mfcc'],
        hidden_size=config['hidden_size'],
        num_classes=dataset.num_classes,
        num_layers=config['num_layers']
    ).to(device)

    chaotic_history = train_model(
        chaotic_model,
        "Chaotic RNN",
        train_dataloader,
        test_dataloader,
        num_epochs=config['num_epochs'],
        lr=config['lr']
    )

    # 可视化混沌模型结果
    plot_results(chaotic_history, "Chaotic RNN", "chaotic")
    plot_confusion_matrix(chaotic_model, test_dataloader, class_names,
                          "Chaotic RNN", "chaotic_confusion_matrix.png")

    # 创建并训练LSTM模型
    print("\n" + "=" * 50)
    print("训练标准LSTM模型")
    print("=" * 50)
    lstm_model = StandardLSTM(
        input_size=config['n_mfcc'],
        hidden_size=config['hidden_size'],
        num_classes=dataset.num_classes,
        num_layers=config['num_layers']
    ).to(device)

    lstm_history = train_model(
        lstm_model,
        "Standard LSTM",
        train_dataloader,
        test_dataloader,
        num_epochs=config['num_epochs'],
        lr=config['lr']
    )

    # 可视化LSTM模型结果
    plot_results(lstm_history, "Standard LSTM", "lstm")
    plot_confusion_matrix(lstm_model, test_dataloader, class_names,
                          "Standard LSTM", "lstm_confusion_matrix.png")

    # 横向对比结果
    plt.figure(figsize=(14, 8))

    # 精度对比
    plt.subplot(2, 2, 1)
    plt.plot(chaotic_history['test_acc'], 'r-', label='Chaotic RNN Test Acc')
    plt.plot(lstm_history['test_acc'], 'b-', label='LSTM Test Acc')
    plt.plot(chaotic_history['train_acc'], 'r--', label='Chaotic RNN Train Acc')
    plt.plot(lstm_history['train_acc'], 'b--', label='LSTM Train Acc')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 损失对比
    plt.subplot(2, 2, 2)
    plt.plot(chaotic_history['train_loss'], 'r-', label='Chaotic RNN')
    plt.plot(lstm_history['train_loss'], 'b-', label='LSTM')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 训练时间对比
    plt.subplot(2, 2, 3)
    plt.bar(['Chaotic RNN', 'LSTM'],
            [np.mean(chaotic_history['time_per_epoch']),
             np.mean(lstm_history['time_per_epoch'])])
    plt.title('Average Training Time per Epoch')
    plt.ylabel('Time (seconds)')

    # 最终性能对比
    chaotic_final_acc = chaotic_history['test_acc'][-1]
    lstm_final_acc = lstm_history['test_acc'][-1]
    plt.subplot(2, 2, 4)
    plt.bar(['Chaotic RNN', 'LSTM'], [chaotic_final_acc, lstm_final_acc])
    plt.title('Final Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(min(chaotic_final_acc, lstm_final_acc) - 5, max(chaotic_final_acc, lstm_final_acc) + 5)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("模型对比图已保存至: model_comparison.png")
    plt.show()

    # 打印最终对比结果
    print("\n" + "=" * 50)
    print("模型性能对比总结")
    print("=" * 50)
    print(f"混沌神经网络 最终测试精度: {chaotic_final_acc:.2f}%")
    print(f"标准LSTM模型 最终测试精度: {lstm_final_acc:.2f}%")
    print(
        f"平均每轮训练时间 - 混沌模型: {np.mean(chaotic_history['time_per_epoch']):.1f}s, LSTM: {np.mean(lstm_history['time_per_epoch']):.1f}s")

    # 情感类别性能对比
    chaotic_preds = []
    lstm_preds = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)

            # 混沌模型预测
            chaotic_outputs, _ = chaotic_model(inputs)
            _, chaotic_pred = torch.max(chaotic_outputs, 1)
            chaotic_preds.extend(chaotic_pred.cpu().numpy())

            # LSTM模型预测
            lstm_outputs, _ = lstm_model(inputs)
            _, lstm_pred = torch.max(lstm_outputs, 1)
            lstm_preds.extend(lstm_pred.cpu().numpy())

            true_labels.extend(labels.numpy())

    # 计算每个类别的准确率
    chaotic_class_acc = []
    lstm_class_acc = []

    for i, emotion in enumerate(class_names):
        indices = [j for j, label in enumerate(true_labels) if label == i]

        chaotic_correct = sum(1 for j in indices if chaotic_preds[j] == i)
        lstm_correct = sum(1 for j in indices if lstm_preds[j] == i)

        chaotic_acc = chaotic_correct / len(indices) * 100 if len(indices) > 0 else 0
        lstm_acc = lstm_correct / len(indices) * 100 if len(indices) > 0 else 0

        chaotic_class_acc.append(chaotic_acc)
        lstm_class_acc.append(lstm_acc)

    # 绘制类别性能对比
    plt.figure(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width / 2, chaotic_class_acc, width, label='Chaotic RNN')
    plt.bar(x + width / 2, lstm_class_acc, width, label='Standard LSTM')

    plt.xlabel('Emotion Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Emotion Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_performance_comparison.png')
    print("类别性能对比图已保存至: class_performance_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
