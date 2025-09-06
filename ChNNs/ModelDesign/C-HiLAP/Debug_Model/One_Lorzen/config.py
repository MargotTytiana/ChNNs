# config.py - 全局配置参数，供所有模块导入使用
import numpy as np

# 训练参数
EPOCHS = 500  # 减少训练轮数
LR = 0.01  # 初始较大学习率
LR_DECAY = 0.95  # 学习率衰减因子
WEIGHT_DECAY = 1e-5
SAVE_INTERVAL = 10
VAL_INTERVAL = 1  # 每个epoch都验证
CHECKPOINT_DIR = "./checkpoints"

# 优化参数
WARMUP_EPOCHS = 5  # 学习率预热轮数
GRAD_CLIP = 1.0  # 梯度裁剪阈值

# 损失函数权重
CE_WEIGHT = 1.0  # 交叉熵损失权重
LYAPUNOV_WEIGHT = 0.0  # 李雅普诺夫稳定性损失权重（与原模型保持一致）

# 早停参数
PATIENCE = 10
MIN_DELTA = 0.001

# 内存优化参数
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
ENABLE_MIXED_PRECISION = False  # CPU上禁用混合精度

# -------------------------- 数据集路径配置 --------------------------
# LibriSpeech数据集路径（仅保留该数据集）
train_dir = "P:/PycharmProjects/pythonProject1/dataset/LibriSpeech/train-clean-100"
dev_dir = "P:/PycharmProjects/pythonProject1/devDataset/LibriSpeech/dev-clean"
test_dir = "P:/PycharmProjects/pythonProject1/testDataset/LibriSpeech/test-clean"

# -------------------------- 音频处理参数 --------------------------
SAMPLE_RATE = 16000  # 采样率（与模型输入匹配）
DURATION = 1.0  # 音频固定时长（秒），适配Lorenz系统时序处理
MAX_SAMPLES = int(SAMPLE_RATE * DURATION)  # 最大采样点数（1秒=16000点）
MIN_AUDIO_DURATION = 0.5  # 最小有效音频时长（秒），过滤过短音频

# -------------------------- 噪声增强参数 --------------------------
NOISE_TYPES = ["white", "babble"]  # 仅保留两种核心噪声类型
SNR_LEVELS = [0, 5, 10]  # 信噪比等级（dB），避免过多等级增加计算量
BABBLE_NUM_SPEAKERS = 3  # 混响噪声的说话人数量

# -------------------------- 数据加载器参数 --------------------------
BATCH_SIZE = 16  # 批次大小（适配GPU内存，可根据设备调整）
NUM_WORKERS = 2  # 数据加载线程数（避免线程过多导致内存溢出）
VALID_RATIO = 0.1  # 训练集划分验证集的比例
MAX_SEQ_LEN = 16000  # 最大序列长度（与MAX_SAMPLES一致，适配模型输入）
DEBUG_MODE = True  # 调试模式：使用小规模数据加速测试
DEBUG_SAMPLE_SIZE = 5000  # 调试模式下训练集最大样本数

# -------------------------- Lorenz混沌系统参数（RK4迭代） --------------------------
LORENZ_SIGMA = 10.0  # 洛伦兹系统标准参数
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0
RK4_STEP_SIZE = 0.01  # RK4数值解法的时间步长
RK4_ITER_STEPS = 10  # 每个音频样本的RK4迭代步数（提取混沌特征）