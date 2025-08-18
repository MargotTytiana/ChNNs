import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def draw_neural_network(input_size, hidden_sizes, output_size, figsize=(12, 8)):
    """
    绘制全连接神经网络结构图

    参数:
    input_size: 输入层神经元数量
    hidden_sizes: 隐藏层神经元数量列表，每个元素代表一层的神经元数量
    output_size: 输出层神经元数量
    figsize: 图像大小
    """
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 创建图像
    fig, ax = plt.subplots(figsize=figsize)

    # 计算总层数
    total_layers = 2 + len(hidden_sizes)  # 输入层 + 隐藏层 + 输出层

    # 每层的神经元数量
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    # 计算每层的x坐标位置
    x_positions = np.linspace(0.1, 0.9, total_layers)

    # 存储所有神经元的位置
    neuron_positions = []

    # 绘制神经元
    for i, (x, size) in enumerate(zip(x_positions, layer_sizes)):
        # 计算当前层神经元的y坐标
        y_positions = np.linspace(0.1, 0.9, size)
        neuron_positions.append((x, y_positions))

        # 绘制神经元（圆形）
        for y in y_positions:
            circle = Circle((x, y), 0.02, fill=True, color='skyblue', edgecolor='black')
            ax.add_patch(circle)

            # 添加神经元编号
            ax.text(x, y, f"{i + 1}-{y_positions.tolist().index(y) + 1}",
                    ha='center', va='center', fontsize=8)

    # 绘制层标签
    layer_labels = ['输入层', '隐藏层', '输出层']
    ax.text(x_positions[0], 1.02, layer_labels[0], ha='center', va='bottom', fontsize=12, fontweight='bold')

    for i in range(1, total_layers - 1):
        ax.text(x_positions[i], 1.02, f'{layer_labels[1]} {i}', ha='center', va='bottom', fontsize=12,
                fontweight='bold')

    ax.text(x_positions[-1], 1.02, layer_labels[2], ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 绘制连接（线）
    for i in range(total_layers - 1):
        x1, y1s = neuron_positions[i]
        x2, y2s = neuron_positions[i + 1]

        # 全连接：当前层每个神经元与下一层每个神经元连接
        for y1 in y1s:
            for y2 in y2s:
                ax.plot([x1, x2], [y1, y2], 'gray', linestyle='-', linewidth=0.5, alpha=0.6)

    # 设置坐标轴范围和属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.axis('off')  # 隐藏坐标轴

    plt.title('全连接神经网络结构', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# 示例：绘制一个3-4-2结构的神经网络（输入层3个神经元，1个隐藏层4个神经元，输出层2个神经元）
if __name__ == "__main__":
    # 可以修改这些参数来绘制不同结构的神经网络
    input_neurons = 3
    hidden_layers = [4, 3]  # 两个隐藏层，分别有4和3个神经元
    output_neurons = 2

    fig = draw_neural_network(input_neurons, hidden_layers, output_neurons)
    plt.show()

    # 可选：保存图像
    # fig.savefig('neural_network.png', dpi=300, bbox_inches='tight')

