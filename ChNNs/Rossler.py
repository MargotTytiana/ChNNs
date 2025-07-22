import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 1. 定义Rossler混沌系统
def rossler_system(x, y, z, a=0.2, b=0.2, c=5.7):
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return dx, dy, dz


# 2. 生成Rossler吸引子数据（用于训练或初始化）
def generate_rossler_attractor(num_steps=10000, dt=0.01):
    xs = np.zeros(num_steps + 1)
    ys = np.zeros(num_steps + 1)
    zs = np.zeros(num_steps + 1)

    # 初始条件
    xs[0], ys[0], zs[0] = 0.1, 0.1, 0.1

    for i in range(num_steps):
        dx, dy, dz = rossler_system(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + dx * dt
        ys[i + 1] = ys[i] + dy * dt
        zs[i + 1] = zs[i] + dz * dt

    return xs, ys, zs


# 3. 定义混沌启发的神经网络层
class ChaoticLayer(nn.Module):
    def __init__(self, input_dim, output_dim, chaos_strength=0.1):
        super(ChaoticLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.chaos_strength = chaos_strength
        self.x, self.y, self.z = 0.1, 0.1, 0.1  # Rossler状态初始化

    def forward(self, x):
        # 常规线性变换
        output = self.linear(x)

        # 注入Rossler混沌动态
        dx, dy, dz = rossler_system(self.x, self.y, self.z)
        self.x += dx * 0.01 * self.chaos_strength
        self.y += dy * 0.01 * self.chaos_strength
        self.z += dz * 0.01 * self.chaos_strength

        # 将混沌状态作为非线性扰动添加到输出
        chaotic_noise = torch.tensor([self.x, self.y, self.z][:output.shape[-1]], dtype=torch.float32)
        return output + chaotic_noise * self.chaos_strength


# 4. 构建完整神经网络
class ChaoticNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChaoticNN, self).__init__()
        self.chaotic_layer1 = ChaoticLayer(input_dim, hidden_dim)
        self.chaotic_layer2 = ChaoticLayer(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.chaotic_layer1(x))
        x = torch.relu(self.chaotic_layer2(x))
        return self.linear_out(x)


# 5. 可视化Rossler吸引子（验证混沌系统）
def plot_rossler_attractor():
    xs, ys, zs = generate_rossler_attractor()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_title("Rossler Attractor")
    plt.show()


# 6. 测试网络
if __name__ == "__main__":
    # 可视化Rossler系统
    plot_rossler_attractor()

    # 初始化网络
    model = ChaoticNN(input_dim=3, hidden_dim=64, output_dim=1)
    input_data = torch.randn(10, 3)  # 随机输入数据
    output = model(input_data)
    print("Output shape:", output.shape)