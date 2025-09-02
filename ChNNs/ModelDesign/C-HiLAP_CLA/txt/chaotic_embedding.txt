import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class ChaoticSystem:
    """
    Base class for chaotic dynamical systems

    混沌动力系统的基类
    """

    def __init__(self):
        """
        Initialize the chaotic system

        初始化混沌系统
        """
        pass

    def equations(self, t: float, state: np.ndarray, *args) -> np.ndarray:
        """
        Differential equations of the chaotic system

        混沌系统的微分方程

        Args:
            t (float): Time
                      时间
            state (np.ndarray): Current state
                              当前状态
            *args: Additional parameters
                  附加参数

        Returns:
            np.ndarray: State derivatives
                       状态导数
        """
        raise NotImplementedError("Subclasses must implement this method")

    def solve(
            self,
            initial_state: np.ndarray,
            t_span: Tuple[float, float],
            t_points: int = 1000,
            params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the system of differential equations

        求解微分方程组

        Args:
            initial_state (np.ndarray): Initial state
                                      初始状态
            t_span (Tuple[float, float]): Time span [t_start, t_end]
                                         时间跨度 [起始时间, 结束时间]
            t_points (int): Number of time points
                           时间点数量
            params (Dict, optional): Parameters for the system
                                    系统参数

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time points and trajectory
                                          时间点和轨迹
        """
        t = np.linspace(t_span[0], t_span[1], t_points)

        if params is None:
            params = {}

        # Solve the ODE system
        # 求解ODE系统
        solution = solve_ivp(
            lambda t, y: self.equations(t, y, **params),
            t_span,
            initial_state,
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        return solution.t, solution.y.T

    def plot_trajectory(self, trajectory: np.ndarray, title: str = "Chaotic Trajectory") -> None:
        """
        Plot the trajectory in phase space

        在相空间中绘制轨迹

        Args:
            trajectory (np.ndarray): System trajectory
                                    系统轨迹
            title (str): Plot title
                        图表标题
        """
        dim = trajectory.shape[1]

        if dim >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=0.5)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()
        elif dim == 2:
            plt.figure(figsize=(10, 8))
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.5)
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(trajectory[:, 0], 'b-')
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.grid(True)
            plt.show()


class LorenzSystem(ChaoticSystem):
    """
    Lorenz chaotic system

    洛伦兹混沌系统
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        """
        Initialize the Lorenz system

        初始化洛伦兹系统

        Args:
            sigma (float): Sigma parameter
                          Sigma参数
            rho (float): Rho parameter
                        Rho参数
            beta (float): Beta parameter
                         Beta参数
        """
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def equations(self, t: float, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Lorenz system equations

        洛伦兹系统方程

        Args:
            t (float): Time
                      时间
            state (np.ndarray): Current state [x, y, z]
                              当前状态 [x, y, z]
            **kwargs: Additional parameters (can override default parameters)
                     附加参数（可覆盖默认参数）

        Returns:
            np.ndarray: State derivatives [dx/dt, dy/dt, dz/dt]
                       状态导数 [dx/dt, dy/dt, dz/dt]
        """
        # Get parameters (use defaults if not provided)
        # 获取参数（如果未提供，则使用默认值）
        sigma = kwargs.get('sigma', self.sigma)
        rho = kwargs.get('rho', self.rho)
        beta = kwargs.get('beta', self.beta)

        x, y, z = state

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        return np.array([dx_dt, dy_dt, dz_dt])


class RosslerSystem(ChaoticSystem):
    """
    Rössler chaotic system

    Rössler混沌系统
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        """
        Initialize the Rössler system

        初始化Rössler系统

        Args:
            a (float): Parameter a
                      参数a
            b (float): Parameter b
                      参数b
            c (float): Parameter c
                      参数c
        """
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def equations(self, t: float, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Rössler system equations

        Rössler系统方程

        Args:
            t (float): Time
                      时间
            state (np.ndarray): Current state [x, y, z]
                              当前状态 [x, y, z]
            **kwargs: Additional parameters (can override default parameters)
                     附加参数（可覆盖默认参数）

        Returns:
            np.ndarray: State derivatives [dx/dt, dy/dt, dz/dt]
                       状态导数 [dx/dt, dy/dt, dz/dt]
        """
        # Get parameters (use defaults if not provided)
        # 获取参数（如果未提供，则使用默认值）
        a = kwargs.get('a', self.a)
        b = kwargs.get('b', self.b)
        c = kwargs.get('c', self.c)

        x, y, z = state

        dx_dt = -y - z
        dy_dt = x + a * y
        dz_dt = b + z * (x - c)

        return np.array([dx_dt, dy_dt, dz_dt])


class ChuaSystem(ChaoticSystem):
    """
    Chua's circuit chaotic system

    蔡氏电路混沌系统
    """

    def __init__(self, alpha: float = 15.6, beta: float = 28.0, mu0: float = -1.143, mu1: float = -0.714):
        """
        Initialize the Chua's circuit system

        初始化蔡氏电路系统

        Args:
            alpha (float): Alpha parameter
                          Alpha参数
            beta (float): Beta parameter
                         Beta参数
            mu0 (float): Mu0 parameter
                        Mu0参数
            mu1 (float): Mu1 parameter
                        Mu1参数
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mu0 = mu0
        self.mu1 = mu1

    def equations(self, t: float, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Chua's circuit equations

        蔡氏电路方程

        Args:
            t (float): Time
                      时间
            state (np.ndarray): Current state [x, y, z]
                              当前状态 [x, y, z]
            **kwargs: Additional parameters (can override default parameters)
                     附加参数（可覆盖默认参数）

        Returns:
            np.ndarray: State derivatives [dx/dt, dy/dt, dz/dt]
                       状态导数 [dx/dt, dy/dt, dz/dt]
        """
        # Get parameters (use defaults if not provided)
        # 获取参数（如果未提供，则使用默认值）
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)
        mu0 = kwargs.get('mu0', self.mu0)
        mu1 = kwargs.get('mu1', self.mu1)

        x, y, z = state

        # Chua's diode nonlinearity
        # 蔡氏二极管非线性
        h_x = mu1 * x + 0.5 * (mu0 - mu1) * (abs(x + 1) - abs(x - 1))

        dx_dt = alpha * (y - x - h_x)
        dy_dt = x - y + z
        dz_dt = -beta * y

        return np.array([dx_dt, dy_dt, dz_dt])


class ChaoticEmbeddingLayer(nn.Module):
    """
    Chaotic embedding layer for neural networks

    神经网络的混沌嵌入层
    """

    def __init__(
            self,
            input_dim: int,
            chaotic_dim: int = 3,
            trajectory_points: int = 100,
            system_type: str = 'lorenz',
            t_span: Tuple[float, float] = (0.0, 10.0),
            trainable_params: bool = True
    ):
        """
        Initialize the chaotic embedding layer

        初始化混沌嵌入层

        Args:
            input_dim (int): Input feature dimension
                            输入特征维度
            chaotic_dim (int): Dimension of the chaotic system
                              混沌系统的维度
            trajectory_points (int): Number of points in the trajectory
                                    轨迹中的点数
            system_type (str): Type of chaotic system ('lorenz', 'rossler', 'chua')
                              混沌系统类型（'lorenz', 'rossler', 'chua'）
            t_span (Tuple[float, float]): Time span for trajectory generation
                                         轨迹生成的时间跨度
            trainable_params (bool): Whether system parameters are trainable
                                   系统参数是否可训练
        """
        super().__init__()
        self.input_dim = input_dim
        self.chaotic_dim = chaotic_dim
        self.trajectory_points = trajectory_points
        self.system_type = system_type
        self.t_span = t_span

        # Create mapping from input features to initial state
        # 创建从输入特征到初始状态的映射
        self.initial_state_mapper = nn.Linear(input_dim, chaotic_dim)

        # Initialize chaotic system parameters
        # 初始化混沌系统参数
        if system_type == 'lorenz':
            self.system = LorenzSystem()
            # Create trainable parameters for Lorenz system
            # 为洛伦兹系统创建可训练参数
            self.sigma = nn.Parameter(torch.tensor(10.0), requires_grad=trainable_params)
            self.rho = nn.Parameter(torch.tensor(28.0), requires_grad=trainable_params)
            self.beta = nn.Parameter(torch.tensor(8.0 / 3.0), requires_grad=trainable_params)

        elif system_type == 'rossler':
            self.system = RosslerSystem()
            # Create trainable parameters for Rössler system
            # 为Rössler系统创建可训练参数
            self.a = nn.Parameter(torch.tensor(0.2), requires_grad=trainable_params)
            self.b = nn.Parameter(torch.tensor(0.2), requires_grad=trainable_params)
            self.c = nn.Parameter(torch.tensor(5.7), requires_grad=trainable_params)

        elif system_type == 'chua':
            self.system = ChuaSystem()
            # Create trainable parameters for Chua's circuit
            # 为蔡氏电路创建可训练参数
            self.alpha = nn.Parameter(torch.tensor(15.6), requires_grad=trainable_params)
            self.beta = nn.Parameter(torch.tensor(28.0), requires_grad=trainable_params)
            self.mu0 = nn.Parameter(torch.tensor(-1.143), requires_grad=trainable_params)
            self.mu1 = nn.Parameter(torch.tensor(-0.714), requires_grad=trainable_params)

        else:
            raise ValueError(f"Unknown system type: {system_type}")

        # Create time points tensor
        # 创建时间点张量
        self.register_buffer('t_points', torch.linspace(t_span[0], t_span[1], trajectory_points))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the chaotic embedding layer

        混沌嵌入层的前向传播

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
                             形状为[batch_size, input_dim]的输入张量

        Returns:
            torch.Tensor: Chaotic trajectories of shape [batch_size, trajectory_points, chaotic_dim]
                         形状为[batch_size, trajectory_points, chaotic_dim]的混沌轨迹
        """
        batch_size = x.shape[0]

        # Map input features to initial states
        # 将输入特征映射到初始状态
        initial_states = self.initial_state_mapper(x)  # [batch_size, chaotic_dim]

        # Get system parameters
        # 获取系统参数
        params = self._get_system_params()

        # Generate trajectories for each sample in the batch
        # 为批次中的每个样本生成轨迹
        trajectories = []

        for i in range(batch_size):
            # Get initial state for this sample
            # 获取此样本的初始状态
            initial_state = initial_states[i].detach().cpu().numpy()

            # Solve the chaotic system
            # 求解混沌系统
            _, trajectory = self.system.solve(
                initial_state=initial_state,
                t_span=self.t_span,
                t_points=self.trajectory_points,
                params=params
            )

            trajectories.append(torch.tensor(trajectory, device=x.device, dtype=x.dtype))

        # Stack trajectories into a batch
        # 将轨迹堆叠成一个批次
        return torch.stack(trajectories)

    def _get_system_params(self) -> Dict:
        """
        Get the current system parameters

        获取当前系统参数

        Returns:
            Dict: Dictionary of system parameters
                 系统参数字典
        """
        if self.system_type == 'lorenz':
            return {
                'sigma': self.sigma.item(),
                'rho': self.rho.item(),
                'beta': self.beta.item()
            }
        elif self.system_type == 'rossler':
            return {
                'a': self.a.item(),
                'b': self.b.item(),
                'c': self.c.item()
            }
        elif self.system_type == 'chua':
            return {
                'alpha': self.alpha.item(),
                'beta': self.beta.item(),
                'mu0': self.mu0.item(),
                'mu1': self.mu1.item()
            }
        else:
            return {}


class DifferentiableChaoticEmbedding(nn.Module):
    """
    Fully differentiable chaotic embedding layer

    完全可微分的混沌嵌入层
    """

    def __init__(
            self,
            input_dim: int,
            chaotic_dim: int = 3,
            trajectory_points: int = 100,
            system_type: str = 'lorenz',
            t_span: Tuple[float, float] = (0.0, 10.0),
            integration_steps: int = 1000
    ):
        """
        Initialize the differentiable chaotic embedding layer

        初始化可微分混沌嵌入层

        Args:
            input_dim (int): Input feature dimension
                            输入特征维度
            chaotic_dim (int): Dimension of the chaotic system
                              混沌系统的维度
            trajectory_points (int): Number of points in the trajectory
                                    轨迹中的点数
            system_type (str): Type of chaotic system ('lorenz', 'rossler', 'chua')
                              混沌系统类型（'lorenz', 'rossler', 'chua'）
            t_span (Tuple[float, float]): Time span for trajectory generation
                                         轨迹生成的时间跨度
            integration_steps (int): Number of steps for numerical integration
                                   数值积分的步数
        """
        super().__init__()
        self.input_dim = input_dim
        self.chaotic_dim = chaotic_dim
        self.trajectory_points = trajectory_points
        self.system_type = system_type
        self.t_span = t_span
        self.integration_steps = integration_steps

        # Create mapping from input features to initial state
        # 创建从输入特征到初始状态的映射
        self.initial_state_mapper = nn.Linear(input_dim, chaotic_dim)

        # Create mapping from input features to system parameters
        # 创建从输入特征到系统参数的映射
        if system_type == 'lorenz':
            self.param_mapper = nn.Linear(input_dim, 3)  # sigma, rho, beta
            self.param_base = nn.Parameter(torch.tensor([10.0, 28.0, 8.0 / 3.0]))
            self.param_scale = nn.Parameter(torch.tensor([2.0, 5.0, 1.0]))

        elif system_type == 'rossler':
            self.param_mapper = nn.Linear(input_dim, 3)  # a, b, c
            self.param_base = nn.Parameter(torch.tensor([0.2, 0.2, 5.7]))
            self.param_scale = nn.Parameter(torch.tensor([0.1, 0.1, 1.0]))

        elif system_type == 'chua':
            self.param_mapper = nn.Linear(input_dim, 4)  # alpha, beta, mu0, mu1
            self.param_base = nn.Parameter(torch.tensor([15.6, 28.0, -1.143, -0.714]))
            self.param_scale = nn.Parameter(torch.tensor([3.0, 5.0, 0.2, 0.2]))

        else:
            raise ValueError(f"Unknown system type: {system_type}")

        # Create time points tensor
        # 创建时间点张量
        dt = (t_span[1] - t_span[0]) / (integration_steps - 1)
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('t_points', torch.linspace(t_span[0], t_span[1], trajectory_points))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the differentiable chaotic embedding layer

        可微分混沌嵌入层的前向传播

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
                             形状为[batch_size, input_dim]的输入张量

        Returns:
            torch.Tensor: Chaotic trajectories of shape [batch_size, trajectory_points, chaotic_dim]
                         形状为[batch_size, trajectory_points, chaotic_dim]的混沌轨迹
        """
        batch_size = x.shape[0]

        # Map input features to initial states
        # 将输入特征映射到初始状态
        initial_states = self.initial_state_mapper(x)  # [batch_size, chaotic_dim]

        # Map input features to system parameters
        # 将输入特征映射到系统参数
        raw_params = self.param_mapper(x)  # [batch_size, num_params]

        # Apply sigmoid and scale to get actual parameters
        # 应用sigmoid并缩放以获得实际参数
        params = self.param_base + self.param_scale * torch.tanh(raw_params * 0.1)

        # Integrate the chaotic system using differentiable operations
        # 使用可微分操作积分混沌系统
        trajectories = self._integrate_system(initial_states, params)

        # Sample trajectories at the specified time points
        # 在指定的时间点采样轨迹
        sampled_trajectories = self._sample_trajectories(trajectories)

        return sampled_trajectories

    def _integrate_system(self, initial_states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Integrate the chaotic system using differentiable operations

        使用可微分操作积分混沌系统

        Args:
            initial_states (torch.Tensor): Initial states [batch_size, chaotic_dim]
                                         初始状态 [batch_size, chaotic_dim]
            params (torch.Tensor): System parameters [batch_size, num_params]
                                  系统参数 [batch_size, num_params]

        Returns:
            torch.Tensor: Integrated trajectories [batch_size, integration_steps, chaotic_dim]
                         积分轨迹 [batch_size, integration_steps, chaotic_dim]
        """
        from torch.utils.checkpoint import checkpoint
        batch_size = initial_states.shape[0]

        # Initialize trajectories with initial states
        # 用初始状态初始化轨迹
        trajectories = [initial_states]

        # Integrate using 4th order Runge-Kutta method
        # 使用四阶龙格-库塔方法积分
        current_state = initial_states
        for i in range(1, self.integration_steps):
            # Use gradient checkpointing to reduce memory usage
            current_state = checkpoint(
                self._runge_kutta_step,
                current_state,
                params,
                self.dt,
                use_reentrant=False
            )
            trajectories.append(current_state)

        # Stack all states
        return torch.stack(trajectories, dim=1)

    def _runge_kutta_step(self, state: torch.Tensor, params: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Perform one step of the 4th order Runge-Kutta integration

        执行一步四阶龙格-库塔积分

        Args:
            state (torch.Tensor): Current state [batch_size, chaotic_dim]
                                当前状态 [batch_size, chaotic_dim]
            params (torch.Tensor): System parameters [batch_size, num_params]
                                  系统参数 [batch_size, num_params]
            dt (torch.Tensor): Time step
                              时间步长

        Returns:
            torch.Tensor: Next state [batch_size, chaotic_dim]
                         下一个状态 [batch_size, chaotic_dim]
        """
        k1 = self._system_equations(state, params)
        k2 = self._system_equations(state + dt * k1 / 2, params)
        k3 = self._system_equations(state + dt * k2 / 2, params)
        k4 = self._system_equations(state + dt * k3, params)

        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _system_equations(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Chaotic system equations

        混沌系统方程

        Args:
            state (torch.Tensor): Current state [batch_size, chaotic_dim]
                                当前状态 [batch_size, chaotic_dim]
            params (torch.Tensor): System parameters [batch_size, num_params]
                                  系统参数 [batch_size, num_params]

        Returns:
            torch.Tensor: State derivatives [batch_size, chaotic_dim]
                         状态导数 [batch_size, chaotic_dim]
        """
        if self.system_type == 'lorenz':
            x, y, z = state[:, 0], state[:, 1], state[:, 2]
            sigma, rho, beta = params[:, 0], params[:, 1], params[:, 2]

            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z

            return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)

        elif self.system_type == 'rossler':
            x, y, z = state[:, 0], state[:, 1], state[:, 2]
            a, b, c = params[:, 0], params[:, 1], params[:, 2]

            dx_dt = -y - z
            dy_dt = x + a * y
            dz_dt = b + z * (x - c)

            return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)

        elif self.system_type == 'chua':
            x, y, z = state[:, 0], state[:, 1], state[:, 2]
            alpha, beta, mu0, mu1 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

            # Chua's diode nonlinearity
            # 蔡氏二极管非线性
            h_x = mu1 * x + 0.5 * (mu0 - mu1) * (torch.abs(x + 1) - torch.abs(x - 1))

            dx_dt = alpha * (y - x - h_x)
            dy_dt = x - y + z
            dz_dt = -beta * y

            return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)

        else:
            raise ValueError(f"Unknown system type: {self.system_type}")

    def _sample_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Sample trajectories at specified time points

        在指定的时间点采样轨迹

        Args:
            trajectories (torch.Tensor): Full trajectories [batch_size, integration_steps, chaotic_dim]
                                        完整轨迹 [batch_size, integration_steps, chaotic_dim]

        Returns:
            torch.Tensor: Sampled trajectories [batch_size, trajectory_points, chaotic_dim]
                         采样轨迹 [batch_size, trajectory_points, chaotic_dim]
        """
        batch_size = trajectories.shape[0]

        # If trajectory_points equals integration_steps, return the full trajectory
        # 如果trajectory_points等于integration_steps，返回完整轨迹
        if self.trajectory_points == self.integration_steps:
            return trajectories

        # Otherwise, sample at regular intervals
        # 否则，按固定间隔采样
        indices = torch.linspace(0, self.integration_steps - 1, self.trajectory_points).long()
        sampled_trajectories = trajectories[:, indices, :]

        return sampled_trajectories


if __name__ == "__main__":
    # Create a Lorenz system
    # 创建一个洛伦兹系统
    lorenz = LorenzSystem()

    # Solve the system
    # 求解系统
    t, trajectory = lorenz.solve(
        initial_state=np.array([1.0, 1.0, 1.0]),
        t_span=(0, 50),
        t_points=5000
    )

    # Plot the trajectory
    # 绘制轨迹
    lorenz.plot_trajectory(trajectory, "Lorenz Attractor")

    # Create a chaotic embedding layer
    # 创建一个混沌嵌入层
    embedding_layer = ChaoticEmbeddingLayer(
        input_dim=20,
        chaotic_dim=3,
        trajectory_points=100,
        system_type='lorenz'
    )

    # Create a differentiable chaotic embedding layer
    # 创建一个可微分混沌嵌入层
    diff_embedding_layer = DifferentiableChaoticEmbedding(
        input_dim=20,
        chaotic_dim=3,
        trajectory_points=100,
        system_type='lorenz'
    )

    # Create a random input tensor
    # 创建一个随机输入张量
    x = torch.randn(16, 20)  # batch_size=16, input_dim=20

    # Forward pass through the embedding layer
    # 通过嵌入层进行前向传播
    trajectories = embedding_layer(x)
    print(f"Trajectories shape: {trajectories.shape}")

    # Forward pass through the differentiable embedding layer
    # 通过可微分嵌入层进行前向传播
    diff_trajectories = diff_embedding_layer(x)
    print(f"Differentiable trajectories shape: {diff_trajectories.shape}")

    # Compute gradients (only works with differentiable layer)
    # 计算梯度（仅适用于可微分层）
    loss = diff_trajectories.mean()
    loss.backward()
    print("Gradients computed successfully")