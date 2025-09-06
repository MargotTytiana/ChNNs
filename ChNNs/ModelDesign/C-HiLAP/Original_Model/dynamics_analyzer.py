import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


class DynamicalSystemAnalyzer:
    """动力系统分析器，用于计算李雅普诺夫指数、重构相空间等"""

    def __init__(self, system_type='continuous', time_step=0.01, max_time=100):
        """
        初始化动力系统分析器
        :param system_type: 系统类型，'continuous'或'discrete'
        :param time_step: 时间步长
        :param max_time: 最大模拟时间
        """
        self.system_type = system_type
        self.time_step = time_step
        self.max_time = max_time
        self.time_points = np.arange(0, max_time, time_step)
        self.trajectory = None
        self.system_function = None

    def set_system_function(self, system_function):
        """
        设置动力系统方程
        :param system_function: 系统方程函数
        """
        self.system_function = system_function

    def simulate_trajectory(self, initial_conditions, params=None):
        """
        模拟系统轨迹
        :param initial_conditions: 初始条件
        :param params: 系统参数
        :return: 系统轨迹
        """
        if self.system_function is None:
            raise ValueError("请先设置系统方程")

        if self.system_type == 'continuous':
            # 连续系统使用scipy的solve_ivp求解
            sol = solve_ivp(
                lambda t, y: self.system_function(t, y, params),
                [0, self.max_time],
                initial_conditions,
                t_eval=self.time_points,
                method='RK45'
            )
            self.trajectory = sol.y.T  # [时间点, 状态变量]

        elif self.system_type == 'discrete':
            # 离散系统使用迭代求解
            n_points = len(self.time_points)
            n_vars = len(initial_conditions)
            self.trajectory = np.zeros((n_points, n_vars))
            self.trajectory[0] = initial_conditions

            for i in range(1, n_points):
                self.trajectory[i] = self.system_function(
                    self.trajectory[i - 1], params
                )

        return self.trajectory

    def calculate_max_lyapunov_exponent(self, trajectory=None, eps=1e-8, steps=None):
        """
        计算最大李雅普诺夫指数
        :param trajectory: 系统轨迹，若为None则使用已模拟的轨迹
        :param eps: 初始扰动大小
        :param steps: 计算步数，若为None则使用全部轨迹
        :return: 最大李雅普诺夫指数
        """
        if trajectory is None:
            if self.trajectory is None:
                raise ValueError("请先模拟系统轨迹")
            trajectory = self.trajectory

        if steps is None:
            steps = len(trajectory) - 1

        n_points, n_vars = trajectory.shape
        steps = min(steps, n_points - 1)

        # 选择一个随机初始点
        start_idx = np.random.randint(0, n_points - steps)

        # 初始化两个相邻点
        point1 = trajectory[start_idx].copy()
        point2 = point1 + np.random.normal(0, eps, n_vars)
        point2 /= np.linalg.norm(point2 - point1) * eps

        # 跟踪这两个点的演化并计算李雅普诺夫指数
        lyapunov_sum = 0

        for i in range(steps):
            # 演化第一个点
            if self.system_type == 'continuous':
                # 对于连续系统，使用小步长数值积分
                sol1 = solve_ivp(
                    lambda t, y: self.system_function(t, y, None),
                    [0, self.time_step],
                    point1,
                    t_eval=[self.time_step],
                    method='RK45'
                )
                new_point1 = sol1.y[:, 0]
            else:
                # 对于离散系统，直接应用系统函数
                new_point1 = self.system_function(point1, None)

            # 演化第二个点
            if self.system_type == 'continuous':
                sol2 = solve_ivp(
                    lambda t, y: self.system_function(t, y, None),
                    [0, self.time_step],
                    point2,
                    t_eval=[self.time_step],
                    method='RK45'
                )
                new_point2 = sol2.y[:, 0]
            else:
                new_point2 = self.system_function(point2, None)

            # 计算新距离
            distance = np.linalg.norm(new_point2 - new_point1)

            # 累加李雅普诺夫指数贡献
            lyapunov_sum += np.log(distance / eps)

            # 重新正交化
            point1 = new_point1
            point2 = point1 + (new_point2 - new_point1) * eps / distance

        # 计算平均李雅普诺夫指数
        max_lyapunov = lyapunov_sum / (steps * self.time_step)

        return max_lyapunov

    def reconstruct_phase_space(self, signal, embedding_dim=3, time_delay=10):
        """
        重构相空间
        :param signal: 单变量时间序列
        :param embedding_dim: 嵌入维度
        :param time_delay: 时间延迟
        :return: 重构的相空间
        """
        n_points = len(signal)
        n_vectors = n_points - (embedding_dim - 1) * time_delay

        if n_vectors <= 0:
            raise ValueError("信号长度不足以进行相空间重构")

        phase_space = np.zeros((n_vectors, embedding_dim))

        for i in range(embedding_dim):
            phase_space[:, i] = signal[i * time_delay: i * time_delay + n_vectors]

        return phase_space

    def calculate_correlation_dimension(self, phase_space, min_r=0.01, max_r=1.0, num_r=20):
        """
        计算关联维度
        :param phase_space: 相空间点集
        :param min_r: 最小半径
        :param max_r: 最大半径
        :param num_r: 半径采样点数
        :return: 关联维度估计值
        """
        # 计算所有点对之间的距离
        distances = pdist(phase_space, 'euclidean')

        # 计算不同半径下的关联积分
        r_values = np.logspace(np.log10(min_r), np.log10(max_r), num_r)
        C = np.zeros_like(r_values)

        for i, r in enumerate(r_values):
            C[i] = np.sum(distances < r) / len(distances)

        # 线性拟合log(C(r)) vs log(r)的斜率
        valid_indices = np.where((C > 0) & (C < 1))[0]
        if len(valid_indices) < 2:
            return 0

        log_r = np.log(r_values[valid_indices])
        log_C = np.log(C[valid_indices])

        # 线性回归
        slope, _ = np.polyfit(log_r, log_C, 1)
        return slope

    def plot_phase_portrait(self, phase_space=None, dimensions=None, title=None):
        """
        绘制相图
        :param phase_space: 相空间点集，若为None则使用已重构的相空间
        :param dimensions: 要绘制的维度索引列表
        :param title: 图标题
        """
        if phase_space is None:
            if self.trajectory is None:
                raise ValueError("请先模拟系统轨迹或提供相空间点集")
            phase_space = self.trajectory

        n_points, n_dims = phase_space.shape

        if dimensions is None:
            dimensions = list(range(min(3, n_dims)))

        if len(dimensions) == 1:
            # 绘制时间序列
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_points[:n_points], phase_space[:, dimensions[0]])
            plt.xlabel('Time')
            plt.ylabel(f'x{dimensions[0]}')

        elif len(dimensions) == 2:
            # 绘制2D相图
            plt.figure(figsize=(10, 10))
            plt.plot(phase_space[:, dimensions[0]], phase_space[:, dimensions[1]], 'b-', alpha=0.5)
            plt.xlabel(f'x{dimensions[0]}')
            plt.ylabel(f'x{dimensions[1]}')

        elif len(dimensions) == 3:
            # 绘制3D相图
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(
                phase_space[:, dimensions[0]],
                phase_space[:, dimensions[1]],
                phase_space[:, dimensions[2]],
                'b-', alpha=0.5
            )
            ax.set_xlabel(f'x{dimensions[0]}')
            ax.set_ylabel(f'x{dimensions[1]}')
            ax.set_zlabel(f'x{dimensions[2]}')

        else:
            raise ValueError("只能绘制1D、2D或3D相图")

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.show()

    def plot_lyapunov_spectrum(self, trajectory=None, max_exponents=5):
        """
        绘制李雅普诺夫指数谱
        :param trajectory: 系统轨迹
        :param max_exponents: 计算的最大指数数量
        """
        if trajectory is None:
            if self.trajectory is None:
                raise ValueError("请先模拟系统轨迹")
            trajectory = self.trajectory

        n_points, n_vars = trajectory.shape
        max_exponents = min(max_exponents, n_vars)

        # 计算李雅普诺夫指数谱
        lyapunov_exponents = []

        for i in range(max_exponents):
            # 这里简化处理，实际应该使用更复杂的算法计算完整的谱
            # 此处仅为示例，使用不同的扰动方向计算近似值
            np.random.seed(i)  # 设置不同的随机种子
            exponent = self.calculate_max_lyapunov_exponent(
                trajectory,
                eps=1e-8,
                steps=n_points // 2
            )
            lyapunov_exponents.append(exponent)

        # 排序指数
        lyapunov_exponents.sort(reverse=True)

        # 绘制李雅普诺夫指数谱
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(lyapunov_exponents) + 1), lyapunov_exponents)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('Lyapunov Exponent')
        plt.title('Lyapunov Exponent Spectrum')
        plt.xticks(range(1, len(lyapunov_exponents) + 1))
        plt.tight_layout()
        plt.show()

        return lyapunov_exponents


# 洛伦兹系统示例
def lorenz_system(t, state, params):
    """
    洛伦兹系统
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    sigma, rho, beta = params
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


# 逻辑斯蒂映射示例
def logistic_map(state, params):
    """
    逻辑斯蒂映射
    x_{n+1} = r * x_n * (1 - x_n)
    """
    r = params
    x = state[0]
    return [r * x * (1 - x)]


# 测试代码
if __name__ == "__main__":
    # 测试洛伦兹系统
    print("测试洛伦兹系统...")
    analyzer = DynamicalSystemAnalyzer(system_type='continuous', time_step=0.01, max_time=50)
    analyzer.set_system_function(lorenz_system)

    # 模拟轨迹
    trajectory = analyzer.simulate_trajectory([1.0, 1.0, 1.0], params=[10.0, 28.0, 8.0 / 3.0])

    # 计算最大李雅普诺夫指数
    max_lyapunov = analyzer.calculate_max_lyapunov_exponent()
    print(f"洛伦兹系统最大李雅普诺夫指数: {max_lyapunov:.6f}")

    # 绘制相图
    analyzer.plot_phase_portrait(title="Lorenz Attractor")

    # 绘制李雅普诺夫指数谱
    lyapunov_spectrum = analyzer.plot_lyapunov_spectrum()
    print(f"李雅普诺夫指数谱: {lyapunov_spectrum}")

    # 测试相空间重构
    print("\n测试相空间重构...")
    # 从洛伦兹系统中提取一个变量
    signal = trajectory[:, 0]

    # 重构相空间
    reconstructed = analyzer.reconstruct_phase_space(signal, embedding_dim=3, time_delay=10)

    # 计算关联维度
    corr_dim = analyzer.calculate_correlation_dimension(reconstructed)
    print(f"关联维度: {corr_dim:.6f}")

    # 绘制重构的相图
    analyzer.plot_phase_portrait(
        phase_space=reconstructed,
        title="Reconstructed Lorenz Attractor"
    )

    # 测试逻辑斯蒂映射
    print("\n测试逻辑斯蒂映射...")
    analyzer = DynamicalSystemAnalyzer(system_type='discrete', time_step=1, max_time=1000)
    analyzer.set_system_function(logistic_map)

    # 模拟轨迹 (r=3.9时系统处于混沌状态)
    trajectory = analyzer.simulate_trajectory([0.5], params=3.9)

    # 计算最大李雅普诺夫指数
    max_lyapunov = analyzer.calculate_max_lyapunov_exponent()
    print(f"逻辑斯蒂映射最大李雅普诺夫指数 (r=3.9): {max_lyapunov:.6f}")

    # 绘制相图
    analyzer.plot_phase_portrait(title="Logistic Map (r=3.9)")

    # 分岔图示例
    print("\n生成逻辑斯蒂映射分岔图...")
    r_values = np.linspace(2.5, 4.0, 1000)
    final_states = []

    for r in r_values:
        # 模拟轨迹
        trajectory = analyzer.simulate_trajectory([0.5], params=r)
        # 取最后100个状态点
        final_states.append(trajectory[-100:])

    # 绘制分岔图
    plt.figure(figsize=(10, 6))
    for i, r in enumerate(r_values):
        plt.plot([r] * len(final_states[i]), final_states[i], 'k.', markersize=0.5)

    plt.xlabel('r')
    plt.ylabel('x')
    plt.title('Bifurcation Diagram of Logistic Map')
    plt.tight_layout()
    plt.show()

