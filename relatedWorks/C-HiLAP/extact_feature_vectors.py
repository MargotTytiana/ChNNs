import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.font_manager as fm

# 尝试查找系统中可用的中文字体
chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']
font_path = None

for font in chinese_fonts:
    try:
        font_path = fm.findfont(font, fallback_to_default=False)
        if font_path:
            plt.rcParams['font.family'] = font
            break
    except:
        continue

# 如果没有找到中文字体，则使用默认字体
if font_path is None:
    plt.rcParams['font.family'] = ['sans-serif']
    # 添加字体回退选项
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'] + plt.rcParams[
        'font.sans-serif']

# 1. 生成模拟相空间数据
np.random.seed(42)
num_samples = 3  # 样本数量
d_e = 5  # 嵌入维度
T = 100  # 时间步长
phase_space = np.random.randn(num_samples, d_e, T)  # 形状: [3,5,100]


# 2. 模拟混沌特征提取 (对应C-HiLAP中的MLSA和RQA)
def extract_chaotic_features(phase_data):
    """从单个样本的相空间数据中提取4个关键特征"""
    d_e, T = phase_data.shape
    # 1. 最大Lyapunov指数 (MLSA提取)
    lyap_max = np.mean(phase_data[:, 1:] - phase_data[:, :-1]) * 5  # 模拟计算

    # 2. 递归率RR (RQA特征)
    dist_matrix = np.linalg.norm(phase_data[:, :, None] - phase_data[:, None, :], axis=0)
    eps = 0.5 * np.mean(dist_matrix)  # 阈值计算
    RR = np.mean(dist_matrix < eps)

    # 3. 确定性DET (RQA特征)
    diagonal_counts = np.sum([np.sum(dist_matrix.diagonal(k) < eps) for k in range(1, 10)])
    DET = diagonal_counts / np.sum(dist_matrix < eps) if np.sum(dist_matrix < eps) > 0 else 0

    # 4. 层流性LAM (RQA特征)
    vertical_counts = np.sum([np.sum(dist_matrix[:, k:] < eps, axis=0).sum() for k in range(1, 10)])
    LAM = vertical_counts / np.sum(dist_matrix < eps) if np.sum(dist_matrix < eps) > 0 else 0

    return np.array([lyap_max, RR, DET, LAM]), eps  # 返回阈值


# 提取所有样本的特征和阈值
features = []
eps_values = []
for sample in phase_space:
    feat, eps = extract_chaotic_features(sample)
    features.append(feat)
    eps_values.append(eps)
features = np.array(features)

# 3. 可视化设置
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(num_samples, 4, width_ratios=[3, 2, 2, 1])
custom_cmap = LinearSegmentedColormap.from_list("chaos", ["#f0f9e8", "#006d2c"])

# 为每个样本绘制转换过程
for i in range(num_samples):
    # 子图1: 原始相空间数据 [dₑ×T]
    ax1 = fig.add_subplot(gs[i, 0])
    im1 = ax1.imshow(phase_space[i], aspect='auto', cmap=custom_cmap,
                     extent=[0, T - 1, d_e - 1, 0])
    ax1.set_title(f'样本{i + 1} 相空间数据\n[dₑ×T] = [{d_e}×{T}]')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('嵌入维度')
    if i == 0:
        fig.colorbar(im1, ax=ax1, label='状态值')

    # 子图2: 最大Lyapunov指数计算示意
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.plot(phase_space[i, 0, :20], label='状态轨迹')
    ax2.plot(phase_space[i, 1, :20], label='邻近轨迹')
    ax2.set_title('Lyapunov指数计算\n(轨迹分离率)')
    ax2.set_xlabel('时间步')
    ax2.legend(fontsize=8)
    ax2.text(0.5, -0.3, f'λ_max = {features[i, 0]:.2f}',
             ha='center', transform=ax2.transAxes)

    # 子图3: 递归图与RQA特征 (使用当前样本的eps)
    ax3 = fig.add_subplot(gs[i, 2])
    dist_matrix = np.linalg.norm(phase_space[i][:, :, None] - phase_space[i][:, None, :], axis=0)
    im3 = ax3.imshow(dist_matrix < eps_values[i], cmap='binary', aspect='auto')  # 修正此处
    ax3.set_title('递归图 (RQA基础)')
    ax3.set_xlabel('时间i')
    ax3.set_ylabel('时间j')
    ax3.text(1.1, 0.5, f'RR={features[i, 1]:.2f}\nDET={features[i, 2]:.2f}\nLAM={features[i, 3]:.2f}',
             va='center', transform=ax3.transAxes, bbox=dict(facecolor='white'))

# 子图4: 最终特征矩阵 [样本数×4]
ax4 = fig.add_subplot(gs[:, 3])
im4 = ax4.imshow(features, aspect='auto', cmap='coolwarm',
                 extent=[0, 3, num_samples - 1, 0])
ax4.set_yticks(range(num_samples))
ax4.set_yticklabels([f'样本{i + 1}' for i in range(num_samples)])
ax4.set_xticks(range(4))
ax4.set_xticklabels(['λ_max', 'RR', 'DET', 'LAM'])
ax4.set_title('最终特征矩阵\n[样本数×4]')
fig.colorbar(im4, ax=ax4, label='特征值')

# 添加特征说明
plt.figtext(0.5, 0.01,
            "特征含义: 1)最大Lyapunov指数(混沌性) 2)递归率(状态重复性) 3)确定性(轨迹规律性) 4)层流性(连续递归长度)",
            ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()