import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

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


# 1. 生成模拟相空间数据 (d_e=5, T=100)
np.random.seed(42)
d_e = 5  # 嵌入维度
T = 100  # 时间步长
phase_space_data = np.random.randn(d_e, T)  # 形状: (d_e, T) = (5, 100)

# 2. 滑动窗口参数
window_size = 10  # 窗口大小
step = 10  # 步长 (无重叠)
n_windows = (T - window_size) // step + 1  # 窗口数量: (100-10)/10 +1 = 10
feature_dim = 4  # 每个窗口提取的特征维度


# 3. 特征提取函数 (模拟从窗口数据中提取4个特征)
def extract_features(window_data):
    # 实际应用中可能是: 最大Lyapunov指数、递归率、关联维数等
    return np.array([
        window_data.mean(),  # 特征1: 均值
        window_data.var(),  # 特征2: 方差
        np.max(window_data) - np.min(window_data),  # 特征3: 极差
        np.sqrt(np.mean(window_data ** 2))  # 特征4: 均方根
    ])


# 4. 预处理所有窗口的特征 (最终形状: 10×4)
all_features = []
for i in range(n_windows):
    start = i * step
    end = start + window_size
    window = phase_space_data[:, start:end]  # 窗口数据: (5,10)
    features = extract_features(window)
    all_features.append(features)
all_features = np.array(all_features)  # 形状: (10,4)

# 5. 可视化设置
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

# 子图1: 原始相空间数据 (d_e × T)
ax1 = fig.add_subplot(gs[0, :])
cax1 = ax1.imshow(phase_space_data, aspect='auto', cmap='viridis',
                  extent=[0, T, d_e - 1, 0])  # y轴为嵌入维度
ax1.set_title(f'Original Phase Space Data (d_e={d_e} × T={T})')
ax1.set_xlabel('Time Step (T)')
ax1.set_ylabel('Embedding Dimension (d_e)')
fig.colorbar(cax1, ax=ax1, label='Signal')

# 滑动窗口标记 (初始状态)
window_rect = plt.Rectangle((0, 0), window_size, d_e - 1,
                            edgecolor='red', facecolor='none', linewidth=2)
ax1.add_patch(window_rect)

# 子图2: 当前窗口数据 (局部放大)
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Current Slide Windows Data (Zoomed)')
ax2.set_xlabel('Time Step within Window')
ax2.set_ylabel('Embedding Dimension (d_e)')
window_im = ax2.imshow(np.zeros((d_e, window_size)), aspect='auto', cmap='viridis')

# 子图3: 提取的特征向量 (10×4)
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Compressed Feature Vectors (10×4)')
ax3.set_xlabel('Feature Dimension (4 chaotic features)')
ax3.set_ylabel('Window Index (10 windows)')
feature_mat = ax3.imshow(np.zeros((n_windows, feature_dim)), aspect='auto',
                        cmap='plasma', vmin=all_features.min(), vmax=all_features.max())
fig.colorbar(feature_mat, ax=ax3, label='Feature Value')


# 6. 动画更新函数
def update(frame):
    # 更新窗口位置
    start = frame * step
    end = start + window_size
    window_rect.set_x(start)

    # 更新窗口数据显示
    current_window = phase_space_data[:, start:end]
    window_im.set_data(current_window)
    ax2.set_xlim(0, window_size - 1)
    ax2.set_ylim(d_e - 1, 0)  # 保持与原始图一致的y轴方向

    # 更新特征矩阵 (累积显示)
    temp_features = all_features.copy()
    temp_features[frame + 1:] = np.nan  # 未处理的窗口设为NaN (灰色)
    feature_mat.set_data(temp_features)

    return window_rect, window_im, feature_mat


# 7. 生成动画
ani = FuncAnimation(
    fig, update, frames=n_windows,
    interval=500, blit=True
)

plt.tight_layout()
plt.show()

# 可选: 保存动画
ani.save('sliding_window_compression.gif', writer='pillow', fps=2)
