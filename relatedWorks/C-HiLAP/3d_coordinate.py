import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 创建图形和3D坐标轴
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 坐标轴范围 (从0到100厘米，原点为立方体一角)
limit = 100
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_zlim(0, limit)

# 设置坐标轴标签
ax.set_xlabel('X (cm)', color='#1f77b4')
ax.set_ylabel('Y (cm)', color='#2ca02c')
ax.set_zlabel('Z (cm)', color='#ff7f0e')

# 绘制实线坐标轴（使用与标签匹配的颜色）
ax.plot([0, limit], [0, 0], [0, 0], color='#1f77b4', linewidth=2.5)  # X轴
ax.plot([0, 0], [0, limit], [0, 0], color='#2ca02c', linewidth=2.5)  # Y轴
ax.plot([0, 0], [0, 0], [0, limit], color='#ff7f0e', linewidth=2.5)  # Z轴

# 标记原点
ax.scatter(0, 0, 0, color='black', s=150, marker='o', zorder=5)
ax.text(2, 2, 2, '(0,0,0)', color='black', fontweight='bold')

# 绘制每隔50cm的虚线网格线（浅灰色）
grid_interval = 50
grid_color = '#d6d6d6'  # 淡灰色网格，无平面颜色

# 绘制平行于X轴的网格线
for y in range(0, limit + 1, grid_interval):
    for z in range(0, limit + 1, grid_interval):
        if y != 0 or z != 0:  # 跳过已经绘制的坐标轴
            ax.plot([0, limit], [y, y], [z, z], color=grid_color, linestyle='--', linewidth=0.8)

# 绘制平行于Y轴的网格线
for x in range(0, limit + 1, grid_interval):
    for z in range(0, limit + 1, grid_interval):
        if x != 0 or z != 0:  # 跳过已经绘制的坐标轴
            ax.plot([x, x], [0, limit], [z, z], color=grid_color, linestyle='--', linewidth=0.8)

# 绘制平行于Z轴的网格线
for x in range(0, limit + 1, grid_interval):
    for y in range(0, limit + 1, grid_interval):
        if x != 0 or y != 0:  # 跳过已经绘制的坐标轴
            ax.plot([x, x], [y, y], [0, limit], color=grid_color, linestyle='--', linewidth=0.8)

# 在立方体内随机选择一个点
np.random.seed(42)  # 设置随机种子，确保结果可重现
x = np.random.uniform(0, limit)
y = np.random.uniform(0, limit)
z = np.random.uniform(0, limit)

# 标记该点并显示坐标（使用深紫色）
ax.scatter(x, y, z, color='#9467bd', s=120, marker='*', label=f'Point ({x:.1f}, {y:.1f}, {z:.1f})', zorder=4)
ax.text(x+2, y+2, z+2, f'({x:.1f}, {y:.1f}, {z:.1f})', color='#9467bd', fontsize=10, fontweight='bold')

# 绘制三个方向的向量（从原点到随机点，与对应坐标轴同色）
ax.quiver(0, 0, 0, x, 0, 0, color='#1f77b4', linewidth=2, label='X', alpha=0.7)
ax.quiver(0, 0, 0, 0, y, 0, color='#2ca02c', linewidth=2, label='Y', alpha=0.7)
ax.quiver(0, 0, 0, 0, 0, z, color='#ff7f0e', linewidth=2, label='Z', alpha=0.7)

# 在随机点上标记向量方向（短小箭头表示方向）
ax.quiver(x, y, z, 5, 0, 0, color='#1f77b4', linewidth=1.5, alpha=0.9)  # X方向
ax.quiver(x, y, z, 0, 5, 0, color='#2ca02c', linewidth=1.5, alpha=0.9)  # Y方向
ax.quiver(x, y, z, 0, 0, 5, color='#ff7f0e', linewidth=1.5, alpha=0.9)  # Z方向

# 添加图例
ax.legend(loc='upper left')

# 设置标题
# ax.set_title('立方体三维坐标系统 (单位: 厘米)', color='#444444')

# 显示图形
plt.tight_layout()
plt.show()
