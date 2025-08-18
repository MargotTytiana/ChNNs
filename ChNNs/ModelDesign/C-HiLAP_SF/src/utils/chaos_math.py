import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable


class ChaoticSystem(nn.Module):
    """
    混沌系统的基类，定义了混沌系统的通用接口。

    所有具体的混沌系统都应该继承这个基类并实现forward方法。
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        计算混沌系统的状态导数。

        Args:
            t: 时间点（在自治系统中通常不使用，但ODE求解器需要）
            state: 当前状态向量

        Returns:
            状态导数向量
        """
        raise NotImplementedError("子类必须实现forward方法")

    def get_initial_state(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        获取系统的初始状态。

        Args:
            batch_size: 批次大小
            device: 计算设备

        Returns:
            初始状态张量
        """
        raise NotImplementedError("子类必须实现get_initial_state方法")


class LorenzSystem(ChaoticSystem):
    """
    洛伦兹混沌系统。

    洛伦兹系统由以下微分方程描述：
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Args:
        sigma: 系统参数 sigma
        rho: 系统参数 rho
        beta: 系统参数 beta
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            sigma: float = 10.0,
            rho: float = 28.0,
            beta: float = 8.0 / 3.0,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化系统参数
        if learnable:
            self.sigma = nn.Parameter(torch.tensor(float(sigma)))
            self.rho = nn.Parameter(torch.tensor(float(rho)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer('sigma', torch.tensor(float(sigma)))
            self.register_buffer('rho', torch.tensor(float(rho)))
            self.register_buffer('beta', torch.tensor(float(beta)))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        计算洛伦兹系统的状态导数。

        Args:
            t: 时间点
            state: 当前状态 [x, y, z] 或批次状态 [batch_size, 3]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt] 或批次状态导数 [batch_size, 3]
        """
        if state.dim() == 1:
            # 单个状态
            x, y, z = state

            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z

            return torch.stack([dx, dy, dz])
        else:
            # 批次状态
            x, y, z = state[:, 0], state[:, 1], state[:, 2]

            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z

            return torch.stack([dx, dy, dz], dim=1)

    def get_initial_state(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        获取洛伦兹系统的初始状态。

        Args:
            batch_size: 批次大小
            device: 计算设备

        Returns:
            初始状态张量，形状为 [batch_size, 3]
        """
        if device is None:
            device = next(self.parameters()).device

        # 典型的初始状态
        initial_state = torch.tensor([1.0, 1.0, 1.0], device=device)

        if batch_size == 1:
            return initial_state
        else:
            return initial_state.unsqueeze(0).expand(batch_size, -1)


class ChenLeeSystem(ChaoticSystem):
    """
    陈-李混沌系统。

    陈-李系统由以下微分方程描述：
        dx/dt = a * y
        dy/dt = b * x + c * y * z
        dz/dt = -2 * y^2

    Args:
        a: 系统参数 a
        b: 系统参数 b
        c: 系统参数 c
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            a: float = 5.0,
            b: float = -10.0,
            c: float = -0.38,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化系统参数
        if learnable:
            self.a = nn.Parameter(torch.tensor(float(a)))
            self.b = nn.Parameter(torch.tensor(float(b)))
            self.c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer('a', torch.tensor(float(a)))
            self.register_buffer('b', torch.tensor(float(b)))
            self.register_buffer('c', torch.tensor(float(c)))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        计算陈-李系统的状态导数。

        Args:
            t: 时间点
            state: 当前状态 [x, y, z] 或批次状态 [batch_size, 3]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt] 或批次状态导数 [batch_size, 3]
        """
        if state.dim() == 1:
            # 单个状态
            x, y, z = state

            dx = self.a * y
            dy = self.b * x + self.c * y * z
            dz = -2 * y * y

            return torch.stack([dx, dy, dz])
        else:
            # 批次状态
            x, y, z = state[:, 0], state[:, 1], state[:, 2]

            dx = self.a * y
            dy = self.b * x + self.c * y * z
            dz = -2 * y * y

            return torch.stack([dx, dy, dz], dim=1)

    def get_initial_state(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        获取陈-李系统的初始状态。

        Args:
            batch_size: 批次大小
            device: 计算设备

        Returns:
            初始状态张量，形状为 [batch_size, 3]
        """
        if device is None:
            device = next(self.parameters()).device

        # 典型的初始状态
        initial_state = torch.tensor([1.0, 0.0, 0.0], device=device)

        if batch_size == 1:
            return initial_state
        else:
            return initial_state.unsqueeze(0).expand(batch_size, -1)


class RosslerSystem(ChaoticSystem):
    """
    罗斯勒混沌系统。

    罗斯勒系统由以下微分方程描述：
        dx/dt = -y - z
        dy/dt = x + a * y
        dz/dt = b + z * (x - c)

    Args:
        a: 系统参数 a
        b: 系统参数 b
        c: 系统参数 c
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            a: float = 0.2,
            b: float = 0.2,
            c: float = 5.7,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化系统参数
        if learnable:
            self.a = nn.Parameter(torch.tensor(float(a)))
            self.b = nn.Parameter(torch.tensor(float(b)))
            self.c = nn.Parameter(torch.tensor(float(c)))
        else:
            self.register_buffer('a', torch.tensor(float(a)))
            self.register_buffer('b', torch.tensor(float(b)))
            self.register_buffer('c', torch.tensor(float(c)))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        计算罗斯勒系统的状态导数。

        Args:
            t: 时间点
            state: 当前状态 [x, y, z] 或批次状态 [batch_size, 3]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt] 或批次状态导数 [batch_size, 3]
        """
        if state.dim() == 1:
            # 单个状态
            x, y, z = state

            dx = -y - z
            dy = x + self.a * y
            dz = self.b + z * (x - self.c)

            return torch.stack([dx, dy, dz])
        else:
            # 批次状态
            x, y, z = state[:, 0], state[:, 1], state[:, 2]

            dx = -y - z
            dy = x + self.a * y
            dz = self.b + z * (x - self.c)

            return torch.stack([dx, dy, dz], dim=1)

    def get_initial_state(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        获取罗斯勒系统的初始状态。

        Args:
            batch_size: 批次大小
            device: 计算设备

        Returns:
            初始状态张量，形状为 [batch_size, 3]
        """
        if device is None:
            device = next(self.parameters()).device

        # 典型的初始状态
        initial_state = torch.tensor([0.0, 0.0, 0.0], device=device)

        if batch_size == 1:
            return initial_state
        else:
            return initial_state.unsqueeze(0).expand(batch_size, -1)


class ChuaSystem(ChaoticSystem):
    """
    蔡氏电路混沌系统。

    蔡氏电路系统由以下微分方程描述：
        dx/dt = alpha * (y - h(x))
        dy/dt = x - y + z
        dz/dt = -beta * y

    其中 h(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)

    Args:
        alpha: 系统参数 alpha
        beta: 系统参数 beta
        m0: 系统参数 m0
        m1: 系统参数 m1
        learnable: 是否将系统参数设为可学习
    """

    def __init__(
            self,
            alpha: float = 15.6,
            beta: float = 28.0,
            m0: float = -1.143,
            m1: float = -0.714,
            learnable: bool = True
    ):
        super().__init__()

        # 初始化系统参数
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
            self.m0 = nn.Parameter(torch.tensor(float(m0)))
            self.m1 = nn.Parameter(torch.tensor(float(m1)))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha)))
            self.register_buffer('beta', torch.tensor(float(beta)))
            self.register_buffer('m0', torch.tensor(float(m0)))
            self.register_buffer('m1', torch.tensor(float(m1)))

    def _h_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        蔡氏电路的非线性函数 h(x)。

        Args:
            x: 输入值

        Returns:
            h(x) 的值
        """
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (torch.abs(x + 1) - torch.abs(x - 1))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        计算蔡氏电路系统的状态导数。

        Args:
            t: 时间点
            state: 当前状态 [x, y, z] 或批次状态 [batch_size, 3]

        Returns:
            状态导数 [dx/dt, dy/dt, dz/dt] 或批次状态导数 [batch_size, 3]
        """
        if state.dim() == 1:
            # 单个状态
            x, y, z = state

            h_x = self._h_function(x)
            dx = self.alpha * (y - h_x)
            dy = x - y + z
            dz = -self.beta * y

            return torch.stack([dx, dy, dz])
        else:
            # 批次状态
            x, y, z = state[:, 0], state[:, 1], state[:, 2]

            h_x = self._h_function(x)
            dx = self.alpha * (y - h_x)
            dy = x - y + z
            dz = -self.beta * y

            return torch.stack([dx, dy, dz], dim=1)

    def get_initial_state(self, batch_size: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        获取蔡氏电路系统的初始状态。

        Args:
            batch_size: 批次大小
            device: 计算设备

        Returns:
            初始状态张量，形状为 [batch_size, 3]
        """
        if device is None:
            device = next(self.parameters()).device

        # 典型的初始状态
        initial_state = torch.tensor([0.7, 0.0, 0.0], device=device)

        if batch_size == 1:
            return initial_state
        else:
            return initial_state.unsqueeze(0).expand(batch_size, -1)


class ChaoticMap(nn.Module):
    """
    混沌映射的基类，定义了混沌映射的通用接口。

    所有具体的混沌映射都应该继承这个基类并实现forward方法。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用混沌映射。

        Args:
            x: 输入值，范围通常在[0,1]之间

        Returns:
            映射后的值
        """
        raise NotImplementedError("子类必须实现forward方法")

    def iterate(self, x0: torch.Tensor, n_iterations: int) -> torch.Tensor:
        """
        迭代应用混沌映射。

        Args:
            x0: 初始值
            n_iterations: 迭代次数

        Returns:
            迭代后的值
        """
        x = x0
        for _ in range(n_iterations):
            x = self.forward(x)
        return x


class LogisticMap(ChaoticMap):
    """
    Logistic映射: x_{n+1} = r * x_n * (1 - x_n)

    Args:
        r: 控制参数，当r > 3.57时系统表现出混沌行为
        learnable: 是否将参数设为可学习
    """

    def __init__(self, r: float = 3.9, learnable: bool = True):
        super().__init__()

        if learnable:
            self.r = nn.Parameter(torch.tensor(float(r)))
        else:
            self.register_buffer('r', torch.tensor(float(r)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用Logistic映射。

        Args:
            x: 输入值，范围应在[0,1]之间

        Returns:
            映射后的值
        """
        return self.r * x * (1 - x)


class TentMap(ChaoticMap):
    """
    帐篷映射: x_{n+1} = mu * min(x_n, 1-x_n)

    Args:
        mu: 控制参数，当mu=2时系统表现出混沌行为
        learnable: 是否将参数设为可学习
    """

    def __init__(self, mu: float = 2.0, learnable: bool = True):
        super().__init__()

        if learnable:
            self.mu = nn.Parameter(torch.tensor(float(mu)))
        else:
            self.register_buffer('mu', torch.tensor(float(mu)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用帐篷映射。

        Args:
            x: 输入值，范围应在[0,1]之间

        Returns:
            映射后的值
        """
        return self.mu * torch.min(x, 1 - x)


class HenonMap(ChaoticMap):
    """
    Henon映射:
        x_{n+1} = 1 - a * x_n^2 + b * y_n
        y_{n+1} = x_n

    Args:
        a: 控制参数a
        b: 控制参数b
        learnable: 是否将参数设为可学习
    """

    def __init__(self, a: float = 1.4, b: float = 0.3, learnable: bool = True):
        super().__init__()

        if learnable:
            self.a = nn.Parameter(torch.tensor(float(a)))
            self.b = nn.Parameter(torch.tensor(float(b)))
        else:
            self.register_buffer('a', torch.tensor(float(a)))
            self.register_buffer('b', torch.tensor(float(b)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        应用Henon映射。

        Args:
            state: 输入状态 [x, y] 或批次状态 [batch_size, 2]

        Returns:
            映射后的状态
        """
        if state.dim() == 1:
            # 单个状态
            x, y = state

            new_x = 1 - self.a * x * x + self.b * y
            new_y = x

            return torch.stack([new_x, new_y])
        else:
            # 批次状态
            x, y = state[:, 0], state[:, 1]

            new_x = 1 - self.a * x * x + self.b * y
            new_y = x

            return torch.stack([new_x, new_y], dim=1)

    def iterate(self, x0: torch.Tensor, n_iterations: int) -> torch.Tensor:
        """
        迭代应用Henon映射。

        Args:
            x0: 初始状态 [x, y] 或批次状态 [batch_size, 2]
            n_iterations: 迭代次数

        Returns:
            迭代后的状态
        """
        state = x0
        for _ in range(n_iterations):
            state = self.forward(state)
        return state


def reconstruct_phase_space(
        signal: torch.Tensor,
        embedding_dim: int,
        delay: int
) -> torch.Tensor:
    """
    使用时间延迟嵌入方法重构相空间（Takens定理）。

    Args:
        signal: 输入信号，形状为 [batch_size, seq_len] 或 [seq_len]
        embedding_dim: 嵌入维度
        delay: 时间延迟

    Returns:
        重构的相空间轨迹，形状为 [batch_size, seq_len-(embedding_dim-1)*delay, embedding_dim]
        或 [seq_len-(embedding_dim-1)*delay, embedding_dim]
    """
    # 检查输入维度
    is_batched = signal.dim() > 1

    if not is_batched:
        # 添加批次维度
        signal = signal.unsqueeze(0)

    batch_size, seq_len = signal.shape

    # 计算输出长度
    output_len = seq_len - (embedding_dim - 1) * delay

    if output_len <= 0:
        raise ValueError(f"Signal too short for the given embedding_dim ({embedding_dim}) and delay ({delay})")

    # 初始化输出张量
    output = torch.zeros(batch_size, output_len, embedding_dim, device=signal.device)

    # 填充输出张量
    for i in range(embedding_dim):
        output[:, :, i] = signal[:, i * delay:i * delay + output_len]

    if not is_batched:
        # 移除批次维度
        output = output.squeeze(0)

    return output


def compute_recurrence_plot(
        trajectory: torch.Tensor,
        threshold: float,
        norm_type: str = 'euclidean'
) -> torch.Tensor:
    """
    计算相空间轨迹的递归图。

    Args:
        trajectory: 相空间轨迹，形状为 [n_points, embedding_dim] 或 [batch_size, n_points, embedding_dim]
        threshold: 递归阈值
        norm_type: 距离范数类型，'euclidean'、'manhattan' 或 'max'

    Returns:
        递归图矩阵，形状为 [n_points, n_points] 或 [batch_size, n_points, n_points]
    """
    # 检查输入维度
    is_batched = trajectory.dim() > 2

    if not is_batched:
        # 添加批次维度
        trajectory = trajectory.unsqueeze(0)

    batch_size, n_points, _ = trajectory.shape

    # 计算成对距离
    if norm_type == 'euclidean':
        # 使用广播计算欧几里得距离
        # 展开为 [batch_size, n_points, 1, embedding_dim] 和 [batch_size, 1, n_points, embedding_dim]
        dist = torch.sqrt(torch.sum(
            (trajectory.unsqueeze(2) - trajectory.unsqueeze(1)) ** 2,
            dim=3
        ))
    elif norm_type == 'manhattan':
        # 使用广播计算曼哈顿距离
        dist = torch.sum(
            torch.abs(trajectory.unsqueeze(2) - trajectory.unsqueeze(1)),
            dim=3
        )
    elif norm_type == 'max':
        # 使用广播计算最大范数距离
        dist = torch.max(
            torch.abs(trajectory.unsqueeze(2) - trajectory.unsqueeze(1)),
            dim=3
        )[0]
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

    # 应用阈值
    rp = (dist <= threshold).float()

    if not is_batched:
        # 移除批次维度
        rp = rp.squeeze(0)

    return rp


def compute_rqa_metrics(rp: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    计算递归图的递归量化分析(RQA)指标。

    Args:
        rp: 递归图矩阵，形状为 [n_points, n_points] 或 [batch_size, n_points, n_points]

    Returns:
        包含RQA指标的字典
    """
    # 检查输入维度
    is_batched = rp.dim() > 2

    if not is_batched:
        # 添加批次维度
        rp = rp.unsqueeze(0)

    batch_size, n_points, _ = rp.shape

    # 递归率(RR)
    rr = torch.mean(rp, dim=(1, 2))

    # 对角线和垂直线结构的检测需要复杂的算法
    # 这里提供一个简化版本

    # 确定性(DET)：对角线结构的比例
    # 通过检查对角线上的点来近似
    det = torch.zeros(batch_size, device=rp.device)

    for b in range(batch_size):
        diag_sum = 0
        for k in range(1, n_points):
            diag = torch.diagonal(rp[b], offset=k)
            # 寻找连续的1
            diag_diff = torch.diff(
                torch.cat([torch.tensor([0.0], device=rp.device), diag, torch.tensor([0.0], device=rp.device)]))
            # 连续1的起始索引
            starts = torch.where(diag_diff == 1)[0]
            # 连续1的结束索引
            ends = torch.where(diag_diff == -1)[0]

            if len(starts) > 0 and len(ends) > 0:
                # 计算对角线长度
                lengths = ends - starts
                # 只考虑长度大于1的对角线
                diag_sum += torch.sum((lengths > 1).float() * lengths.float())

        # 计算确定性
        if torch.sum(rp[b]) > 0:
            det[b] = diag_sum / torch.sum(rp[b])

    # 层流性(LAM)：垂直线结构的比例
    # 通过检查列上的点来近似
    lam = torch.zeros(batch_size, device=rp.device)

    for b in range(batch_size):
        vert_sum = 0
        for j in range(n_points):
            col = rp[b, :, j]
            # 寻找连续的1
            col_diff = torch.diff(
                torch.cat([torch.tensor([0.0], device=rp.device), col, torch.tensor([0.0], device=rp.device)]))
            # 连续1的起始索引
            starts = torch.where(col_diff == 1)[0]
            # 连续1的结束索引
            ends = torch.where(col_diff == -1)[0]

            if len(starts) > 0 and len(ends) > 0:
                # 计算垂直线长度
                lengths = ends - starts
                # 只考虑长度大于1的垂直线
                vert_sum += torch.sum((lengths > 1).float() * lengths.float())

        # 计算层流性
        if torch.sum(rp[b]) > 0:
            lam[b] = vert_sum / torch.sum(rp[b])

    # 熵(ENTR)：对角线长度分布的香农熵
    # 这需要完整的对角线长度分布，简化版本中省略

    # 平均对角线长度(L)和最大对角线长度(Lmax)也省略

    # 返回计算的指标
    metrics = {
        'recurrence_rate': rr,
        'determinism': det,
        'laminarity': lam
    }

    if not is_batched:
        # 移除批次维度
        metrics = {k: v.squeeze(0) for k, v in metrics.items()}

    return metrics


def estimate_lyapunov_exponent(
        trajectory: torch.Tensor,
        k_neighbors: int = 5,
        max_steps: int = 20
) -> torch.Tensor:
    """
    使用Rosenstein算法估计最大李雅普诺夫指数。

    Args:
        trajectory: 相空间轨迹，形状为 [n_points, embedding_dim] 或 [batch_size, n_points, embedding_dim]
        k_neighbors: 用于估计的近邻点数量
        max_steps: 最大时间步数

    Returns:
        最大李雅普诺夫指数
    """
    # 检查输入维度
    is_batched = trajectory.dim() > 2

    if not is_batched:
        # 添加批次维度
        trajectory = trajectory.unsqueeze(0)

    batch_size, n_points, embedding_dim = trajectory.shape

    # 初始化结果
    mle = torch.zeros(batch_size, device=trajectory.device)

    for b in range(batch_size):
        # 计算所有点对之间的距离
        distances = torch.cdist(trajectory[b], trajectory[b])

        # 设置对角线元素为无穷大（避免选择自身作为近邻）
        distances.fill_diagonal_(float('inf'))

        # 设置时间相近的点的距离为无穷大（避免时间相关性）
        for i in range(n_points):
            for j in range(max(0, i - 10), min(n_points, i + 11)):
                distances[i, j] = float('inf')

        # 寻找每个点的k个最近邻
        _, indices = torch.topk(distances, k=k_neighbors, dim=1, largest=False)

        # 计算平均发散率
        divergence = torch.zeros(max_steps, device=trajectory.device)
        valid_count = 0

        for i in range(n_points - max_steps):
            for k in range(k_neighbors):
                j = indices[i, k].item()

                if j < n_points - max_steps:
                    # 计算初始距离
                    d0 = torch.norm(trajectory[b, i] - trajectory[b, j])

                    if d0 > 0:
                        # 跟踪轨迹发散
                        for step in range(max_steps):
                            # 计算step时间后的距离
                            d = torch.norm(trajectory[b, i + step] - trajectory[b, j + step])

                            # 更新发散率
                            if d > 0:
                                divergence[step] += torch.log(d / d0)

                        valid_count += 1

        if valid_count > 0:
            # 计算平均发散率
            divergence = divergence / valid_count

            # 使用线性回归估计MLE
            x = torch.arange(max_steps, device=trajectory.device).float()
            y = divergence

            # 计算斜率（最小二乘法）
            mean_x = torch.mean(x)
            mean_y = torch.mean(y)

            numerator = torch.sum((x - mean_x) * (y - mean_y))
            denominator = torch.sum((x - mean_x) ** 2)

            if denominator > 0:
                slope = numerator / denominator
                mle[b] = slope

    if not is_batched:
        # 移除批次维度
        mle = mle.squeeze(0)

    return mle


def compute_correlation_dimension(
        trajectory: torch.Tensor,
        max_radius: float = 10.0,
        n_radii: int = 20
) -> torch.Tensor:
    """
    使用Grassberger-Procaccia算法计算相关维数。

    Args:
        trajectory: 相空间轨迹，形状为 [n_points, embedding_dim] 或 [batch_size, n_points, embedding_dim]
        max_radius: 最大搜索半径
        n_radii: 半径数量

    Returns:
        相关维数
    """
    # 检查输入维度
    is_batched = trajectory.dim() > 2

    if not is_batched:
        # 添加批次维度
        trajectory = trajectory.unsqueeze(0)

    batch_size, n_points, _ = trajectory.shape

    # 初始化结果
    corr_dim = torch.zeros(batch_size, device=trajectory.device)

    # 生成对数均匀分布的半径
    radii = torch.logspace(-2, torch.log10(torch.tensor(max_radius)), n_radii, device=trajectory.device)

    for b in range(batch_size):
        # 计算所有点对之间的距离
        distances = torch.cdist(trajectory[b], trajectory[b])

        # 设置对角线元素为无穷大（避免自相关）
        distances.fill_diagonal_(float('inf'))

        # 计算相关积分
        correlation_sum = torch.zeros(n_radii, device=trajectory.device)

        for i, r in enumerate(radii):
            # 计算距离小于r的点对数量
            correlation_sum[i] = torch.sum(distances < r).float()

        # 归一化
        total_pairs = n_points * (n_points - 1)
        correlation_sum = correlation_sum / total_pairs

        # 在对数空间中进行线性回归
        x = torch.log(radii)
        y = torch.log(correlation_sum + 1e-10)  # 添加小常数避免log(0)

        # 计算斜率（最小二乘法）
        valid_idx = correlation_sum > 0
        if torch.sum(valid_idx) > 1:
            x_valid = x[valid_idx]
            y_valid = y[valid_idx]

            mean_x = torch.mean(x_valid)
            mean_y = torch.mean(y_valid)

            numerator = torch.sum((x_valid - mean_x) * (y_valid - mean_y))
            denominator = torch.sum((x_valid - mean_x) ** 2)

            if denominator > 0:
                slope = numerator / denominator
                corr_dim[b] = slope

    if not is_batched:
        # 移除批次维度
        corr_dim = corr_dim.squeeze(0)

    return corr_dim


def compute_sample_entropy(
        signal: torch.Tensor,
        m: int = 2,
        r: float = 0.2
) -> torch.Tensor:
    """
    计算信号的样本熵。

    Args:
        signal: 输入信号，形状为 [batch_size, seq_len] 或 [seq_len]
        m: 模板长度
        r: 容差（通常为信号标准差的0.1-0.25倍）

    Returns:
        样本熵
    """
    # 检查输入维度
    is_batched = signal.dim() > 1

    if not is_batched:
        # 添加批次维度
        signal = signal.unsqueeze(0)

    batch_size, seq_len = signal.shape

    # 初始化结果
    sample_entropy = torch.zeros(batch_size, device=signal.device)

    for b in range(batch_size):
        # 归一化信号
        sig = signal[b]
        sig = (sig - torch.mean(sig)) / (torch.std(sig) + 1e-10)

        # 计算容差
        tolerance = r * torch.std(sig)

        # 计算m和m+1长度模板的匹配数
        count_m = 0
        count_m_plus_1 = 0

        for i in range(seq_len - m):
            template_m = sig[i:i + m]
            template_m_plus_1 = sig[i:i + m + 1] if i + m + 1 <= seq_len else None

            for j in range(i + 1, seq_len - m + 1):
                # 检查m长度模板是否匹配
                if torch.max(torch.abs(template_m - sig[j:j + m])) < tolerance:
                    count_m += 1

                    # 检查m+1长度模板是否匹配
                    if template_m_plus_1 is not None and j + m + 1 <= seq_len:
                        if torch.max(torch.abs(template_m_plus_1 - sig[j:j + m + 1])) < tolerance:
                            count_m_plus_1 += 1

        # 计算样本熵
        if count_m > 0 and count_m_plus_1 > 0:
            sample_entropy[b] = -torch.log(count_m_plus_1 / count_m)
        else:
            sample_entropy[b] = torch.tensor(float('inf'), device=signal.device)

    if not is_batched:
        # 移除批次维度
        sample_entropy = sample_entropy.squeeze(0)

    return sample_entropy


def compute_kolmogorov_sinai_entropy(
        system: ChaoticSystem,
        n_trajectories: int = 10,
        n_steps: int = 1000,
        dt: float = 0.01
) -> torch.Tensor:
    """
    估计混沌系统的Kolmogorov-Sinai熵。

    Args:
        system: 混沌系统
        n_trajectories: 轨迹数量
        n_steps: 每个轨迹的步数
        dt: 时间步长

    Returns:
        估计的K-S熵
    """
    device = next(system.parameters()).device

    # 生成初始条件
    initial_states = system.get_initial_state(n_trajectories, device)

    # 添加小的随机扰动
    initial_states = initial_states + 0.01 * torch.randn_like(initial_states)

    # 跟踪轨迹
    trajectories = []
    current_states = initial_states

    for _ in range(n_steps):
        # 计算导数
        derivatives = system(torch.tensor(0.0, device=device), current_states)

        # 更新状态（简单的欧拉方法）
        current_states = current_states + dt * derivatives

        # 存储状态
        trajectories.append(current_states.clone())

    # 将轨迹堆叠为张量
    trajectories = torch.stack(trajectories, dim=1)  # [n_trajectories, n_steps, state_dim]

    # 计算轨迹之间的平均发散率
    divergence_rates = []

    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            # 计算初始距离
            d0 = torch.norm(trajectories[i, 0] - trajectories[j, 0])

            if d0 > 0:
                # 计算最终距离
                d_final = torch.norm(trajectories[i, -1] - trajectories[j, -1])

                # 计算发散率
                if d_final > d0:
                    rate = torch.log(d_final / d0) / (n_steps * dt)
                    divergence_rates.append(rate.item())

    # K-S熵近似为正的李雅普诺夫指数之和
    if divergence_rates:
        ks_entropy = torch.tensor(sum(max(0, rate) for rate in divergence_rates) / len(divergence_rates))
    else:
        ks_entropy = torch.tensor(0.0)

    return ks_entropy


def compute_bifurcation_diagram(
        chaotic_map: ChaoticMap,
        param_name: str,
        param_range: Tuple[float, float],
        param_steps: int = 1000,
        x0: float = 0.5,
        n_iterations: int = 1000,
        n_discard: int = 500
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算混沌映射的分岔图。

    Args:
        chaotic_map: 混沌映射
        param_name: 参数名称
        param_range: 参数范围
        param_steps: 参数步数
        x0: 初始值
        n_iterations: 迭代次数
        n_discard: 丢弃的初始迭代次数

    Returns:
        参数值和对应的轨迹点
    """
    device = next(chaotic_map.parameters()).device

    # 生成参数值
    param_values = torch.linspace(param_range[0], param_range[1], param_steps, device=device)

    # 存储结果
    bifurcation_points = []

    # 对每个参数值进行迭代
    for param in param_values:
        # 设置参数值
        setattr(chaotic_map, param_name, param)

        # 初始化状态
        if isinstance(chaotic_map, HenonMap):
            x = torch.tensor([x0, x0], device=device)
        else:
            x = torch.tensor(x0, device=device)

        # 丢弃初始迭代
        for _ in range(n_discard):
            x = chaotic_map(x)

        # 收集稳态轨迹点
        points = []
        for _ in range(n_iterations - n_discard):
            x = chaotic_map(x)
            if isinstance(chaotic_map, HenonMap):
                points.append(x[0].item())
            else:
                points.append(x.item())

        bifurcation_points.append(points)

    # 转换为张量
    bifurcation_points = torch.tensor(bifurcation_points, device=device)

    return param_values, bifurcation_points


# 示例用法
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建洛伦兹系统
    lorenz = LorenzSystem().to(device)

    # 生成初始状态
    state = lorenz.get_initial_state(batch_size=1, device=device)

    # 模拟轨迹
    from torchdiffeq import odeint

    t = torch.linspace(0, 50, 5000, device=device)
    trajectory = odeint(lorenz, state, t)

    print(f"洛伦兹轨迹形状: {trajectory.shape}")

    # 计算相空间重构
    signal = trajectory[:, 0]  # 使用x分量
    reconstructed = reconstruct_phase_space(signal, embedding_dim=3, delay=10)

    print(f"重构的相空间形状: {reconstructed.shape}")

    # 计算递归图
    rp = compute_recurrence_plot(reconstructed, threshold=2.0)

    print(f"递归图形状: {rp.shape}")

    # 计算RQA指标
    rqa_metrics = compute_rqa_metrics(rp)

    print("RQA指标:")
    for key, value in rqa_metrics.items():
        print(f"  {key}: {value.item():.4f}")

    # 估计最大李雅普诺夫指数
    mle = estimate_lyapunov_exponent(reconstructed)

    print(f"最大李雅普诺夫指数: {mle.item():.4f}")

    # 计算相关维数
    corr_dim = compute_correlation_dimension(reconstructed)

    print(f"相关维数: {corr_dim.item():.4f}")

    # 计算样本熵
    sample_entropy = compute_sample_entropy(signal)

    print(f"样本熵: {sample_entropy.item():.4f}")

    # 创建Logistic映射
    logistic_map = LogisticMap().to(device)

    # 计算分岔图
    param_values, bifurcation_points = compute_bifurcation_diagram(
        logistic_map,
        param_name='r',
        param_range=(3.5, 4.0),
        param_steps=100
    )

    print(f"分岔图参数值形状: {param_values.shape}")
    print(f"分岔图点形状: {bifurcation_points.shape}")