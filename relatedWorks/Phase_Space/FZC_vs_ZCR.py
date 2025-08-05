import numpy as np
import matplotlib.pyplot as plt

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

# 创建两个测试信号
t = np.linspace(0, 0.1, 1000)  # 100ms时间

# 低频周期信号 (类似浊音)
low_freq = 0.5 * np.sin(2 * np.pi * 50 * t)

# 高频噪声信号 (类似清音)
high_freq = 0.2 * np.random.randn(len(t))


# 计算自相关函数
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


# 计算过零率
def zcr(signal):
    return len(np.where(np.diff(np.sign(signal)))[0]) / (t[-1] - t[0])


# 绘图
plt.figure(figsize=(12, 8))

# 低频信号分析
plt.subplot(221)
plt.plot(t, low_freq, 'b')
plt.title(f"低频信号 (ZCR={zcr(low_freq):.1f}次/秒)")
plt.xlabel("时间 (秒)")
plt.ylabel("振幅")

plt.subplot(222)
acf_low = autocorr(low_freq)
zero_cross = np.where(acf_low < 0)[0]
tau0 = t[zero_cross[0]] if zero_cross.size > 0 else t[-1]
plt.plot(t[:len(acf_low)], acf_low, 'g')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(tau0, color='r', label=f'过零点={tau0 * 1000:.1f}ms')
plt.title("低频信号自相关")
plt.legend()

# 高频信号分析
plt.subplot(223)
plt.plot(t, high_freq, 'r')
plt.title(f"高频信号 (ZCR={zcr(high_freq):.1f}次/秒)")
plt.xlabel("时间 (秒)")
plt.ylabel("振幅")

plt.subplot(224)
acf_high = autocorr(high_freq)
zero_cross_high = np.where(acf_high < 0)[0]
tau0_high = t[zero_cross_high[0]] if zero_cross_high.size > 0 else t[-1]
plt.plot(t[:len(acf_high)], acf_high, 'm')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(tau0_high, color='r', label=f'过零点={tau0_high * 1000:.1f}ms')
plt.title("高频信号自相关")
plt.legend()

plt.tight_layout()
plt.show()
