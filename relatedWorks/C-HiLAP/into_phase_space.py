import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import soundfile as sf  # 用于读取真实音频（可选）

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. 生成模拟数据
np.random.seed(42)
num_samples = 2  # 样本数量
T = 50  # 每个样本的时间步长
d_e = 4  # 嵌入维度 (根据虚假最近邻法确定)
tau = 5  # 时延参数 (根据自相关函数过零点确定)

# 原始语音信号: 形状 [样本数×T]
raw_signals = np.array([
    np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, T)) + 0.2 * np.random.randn(T),  # 样本1
    np.sin(2 * np.pi * 1.8 * np.linspace(0, 10, T)) + 0.2 * np.random.randn(T)  # 样本2
])

# 2. 相空间重构: 转换为 [样本数×d_e×T]
# 对于每个样本，构建d_e行，每行是原始信号的时延版本
phase_space = []
for s in range(num_samples):
    sample_phase = []
    for i in range(d_e):
        # 第i行: s(t + i*tau)，超出部分用最后值填充
        shifted = np.roll(raw_signals[s], -i * tau)
        shifted[-i * tau:] = raw_signals[s][-1]  # 填充边界
        sample_phase.append(shifted)
    phase_space.append(np.array(sample_phase))
phase_space = np.array(phase_space)  # 最终形状: [2,4,50]

# 3. 可视化设置
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(num_samples, 2, width_ratios=[1, 3])

for sample_idx in range(num_samples):
    # 左侧: 原始信号 [1×T]
    ax1 = fig.add_subplot(gs[sample_idx, 0])
    ax1.plot(raw_signals[sample_idx], color='blue')
    ax1.set_title(f'样本{sample_idx + 1}原始信号 [1×T]')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('信号幅度')
    ax1.set_xlim(0, T - 1)

    # 标记时延采样点 (以第一个时间步为例)
    for i in range(d_e):
        t = i * tau
        if t < T:
            ax1.scatter(t, raw_signals[sample_idx, t], color='red', s=50, zorder=3)
            ax1.text(t, raw_signals[sample_idx, t] + 0.3, f't+{i}τ', fontsize=8)

    # 右侧: 重构后的相空间 [d_e×T]
    ax2 = fig.add_subplot(gs[sample_idx, 1])
    cax = ax2.imshow(phase_space[sample_idx], aspect='auto', cmap='viridis',
                     extent=[0, T - 1, d_e - 1, 0])  # y轴为嵌入维度
    ax2.set_title(f'Sample {sample_idx + 1} Phase Space Reconstruction [d_e×T]')
    ax2.set_xlabel('Time Step (T)')
    ax2.set_ylabel('Embedding Dimension (d_e=4)')
    fig.colorbar(cax, ax=ax2, label='Signal')

    # 标记时延对应关系
    for i in range(d_e):
        ax2.axhline(y=i, color='white', linestyle='--', alpha=0.3)

# 添加原理说明
plt.figtext(0.5, 0.01,
            f'转换原理: 基于Takens定理，通过时延嵌入 s(t)→[s(t), s(t+τ), s(t+2τ), s(t+3τ)]，τ={tau}',
            ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 1])  # 预留底部文本空间
plt.show()
