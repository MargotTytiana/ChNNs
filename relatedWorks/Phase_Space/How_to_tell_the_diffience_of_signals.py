import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA

# 设置中文字体（修改部分）
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


# 生成三种基础信号（代表不同声源）
def generate_periodic(t, freq=2):
    """周期信号（如元音）：二维环面嵌入三维"""
    return np.array([np.sin(2 * np.pi * freq * t),
                     np.sin(4 * np.pi * freq * t),
                     np.cos(2 * np.pi * freq * t)]).T


def generate_chaotic(t):
    """混沌信号（如摩擦音）：洛伦兹吸引子"""
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))
    dt = t[1] - t[0]
    # 洛伦兹方程参数
    sigma, rho, beta = 10., 28., 8. / 3.
    for i in range(1, len(t)):
        dx = sigma * (y[i - 1] - x[i - 1])
        dy = x[i - 1] * (rho - z[i - 1]) - y[i - 1]
        dz = x[i - 1] * y[i - 1] - beta * z[i - 1]
        x[i] = x[i - 1] + dx * dt
        y[i] = y[i - 1] + dy * dt
        z[i] = z[i - 1] + dz * dt
    return np.vstack([x, y, z]).T


def generate_noise(t):
    """噪声信号（如背景嘶嘶声）"""
    return np.random.randn(len(t), 3)


# 生成时间序列
t = np.linspace(0, 10, 5000)
signal_A = generate_periodic(t, freq=1.5)  # 周期信号
signal_B = generate_chaotic(t)  # 混沌信号
signal_C = generate_noise(t)  # 噪声信号

# 混合信号 (线性叠加)
mixed_signal = 0.4 * signal_A + 0.5 * signal_B + 0.1 * signal_C

# 可视化混合信号在三维相空间
fig = plt.figure(figsize=(18, 12))

# 1. 原始信号分离展示
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot(signal_A[:, 0], signal_A[:, 1], signal_A[:, 2], 'b', alpha=0.5, label='周期信号')
ax1.plot(signal_B[:, 0], signal_B[:, 1], signal_B[:, 2], 'r', alpha=0.5, label='混沌信号')
ax1.plot(signal_C[:, 0], signal_C[:, 1], signal_C[:, 2], 'g', alpha=0.1, label='噪声')
ax1.set_title("原始信号在三维相空间")
ax1.legend()

# 2. 混合信号展示
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot(mixed_signal[:, 0], mixed_signal[:, 1], mixed_signal[:, 2], 'm', alpha=0.3)
ax2.set_title("混合信号在三维相空间")

# 3. 相空间分离（几何分解）
ax3 = fig.add_subplot(233, projection='3d')


# 分离算法：基于曲率聚类
def curvature(x):
    """计算轨迹曲率（表征动力学特性）"""
    dx = np.gradient(x, axis=0)
    ddx = np.gradient(dx, axis=0)
    num = np.linalg.norm(np.cross(dx, ddx), axis=1)
    denom = np.linalg.norm(dx, axis=1) ** 3
    return np.where(denom > 1e-10, num / denom, 0)


# 计算混合信号曲率
curve = curvature(mixed_signal)

# 根据曲率分离信号
threshold = np.percentile(curve, 85)
periodic_mask = curve < threshold / 3
chaotic_mask = curve > threshold
noise_mask = (~periodic_mask) & (~chaotic_mask)

# 绘制分离结果
ax3.plot(mixed_signal[periodic_mask, 0], mixed_signal[periodic_mask, 1],
         mixed_signal[periodic_mask, 2], 'b', alpha=0.3, label='分离的周期信号')
ax3.plot(mixed_signal[chaotic_mask, 0], mixed_signal[chaotic_mask, 1],
         mixed_signal[chaotic_mask, 2], 'r', alpha=0.3, label='分离的混沌信号')
ax3.plot(mixed_signal[noise_mask, 0], mixed_signal[noise_mask, 1],
         mixed_signal[noise_mask, 2], 'g', alpha=0.1, label='分离的噪声')
ax3.set_title("相空间几何分离结果")
ax3.legend()


# 4. 高维相空间重构（5维）
def delay_embedding(signal, dim=5, tau=50):
    """时延嵌入构建高维相空间"""
    embedded = np.zeros((len(signal) - (dim - 1) * tau, dim))
    for i in range(dim):
        embedded[:, i] = signal[i * tau: len(signal) - (dim - 1 - i) * tau, 0]
    return embedded


high_dim_signal = delay_embedding(mixed_signal, dim=5, tau=50)

# 5. 高维空间可视化（PCA降维）
pca = PCA(n_components=3)
high_dim_3d = pca.fit_transform(high_dim_signal)

ax4 = fig.add_subplot(234, projection='3d')
ax4.scatter(high_dim_3d[:, 0], high_dim_3d[:, 1], high_dim_3d[:, 2],
            c=curve[:len(high_dim_3d)], cmap='viridis', alpha=0.3)
ax4.set_title("高维重构后的混合信号\n(PCA投影到三维)")


# 6. 吸引子拓扑不变量计算
def correlation_dimension(signal, r_range=np.logspace(-2, 0, 20)):
    """计算关联维数（吸引子分形特征）"""
    dists = np.sqrt(np.sum((signal[:, None] - signal[None, :]) ** 2, axis=-1))
    C_r = [np.mean(dists < r) for r in r_range]
    return np.polyfit(np.log(r_range), np.log(C_r), 1)[0]


dim_periodic = correlation_dimension(mixed_signal[periodic_mask])
dim_chaotic = correlation_dimension(mixed_signal[chaotic_mask])
dim_noise = correlation_dimension(mixed_signal[noise_mask])

ax5 = fig.add_subplot(235)
r_vals = np.logspace(-2, 0, 20)
ax5.loglog(r_vals, [np.mean(np.sqrt(np.sum((mixed_signal[periodic_mask][:, None] -
                                            mixed_signal[periodic_mask][None, :]) ** 2, axis=-1)) < r)
                    for r in r_vals], 'bo-', label=f'周期信号 D={dim_periodic:.2f}')
ax5.loglog(r_vals, [np.mean(np.sqrt(np.sum((mixed_signal[chaotic_mask][:, None] -
                                            mixed_signal[chaotic_mask][None, :]) ** 2, axis=-1)) < r)
                    for r in r_vals], 'ro-', label=f'混沌信号 D={dim_chaotic:.2f}')
ax5.loglog(r_vals, [np.mean(np.sqrt(np.sum((mixed_signal[noise_mask][:, None] -
                                            mixed_signal[noise_mask][None, :]) ** 2, axis=-1)) < r)
                    for r in r_vals], 'go-', label=f'噪声 D={dim_noise:.2f}')
ax5.set_title("吸引子关联维数")
ax5.legend()

plt.tight_layout()
plt.show()
