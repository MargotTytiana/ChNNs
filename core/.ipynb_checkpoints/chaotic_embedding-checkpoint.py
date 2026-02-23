import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union

import os
import sys
from pathlib import Path

def setup_module_imports(current_file: str = __file__):
    try:
        from setup_imports import setup_project_imports
        return setup_project_imports(current_file), True
    except ImportError:
        current_dir = Path(current_file).resolve().parent
        project_root = current_dir.parent  # for most case
        
        paths = [str(project_root), str(project_root/'core'), 
                str(project_root/'models'), str(project_root/'features'),
                str(project_root/'data'), str(project_root/'utils')]
        
        for path in paths:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
        return project_root, False

PROJECT_ROOT, _ = setup_module_imports()


class ChaoticEmbedding(nn.Module):
    """
    Chaotic Embedding Layer - Core module for embedding speech features into chaotic dynamics.
    
    This layer maps low-dimensional static features into high-dimensional chaotic system 
    trajectories, leveraging topological properties of chaotic attractors to enhance 
    feature discriminability.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        system_type: str = 'lorenz',
        evolution_time: float = 0.5,
        time_step: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8/3,
        coupling_strength: float = 1.0,
        noise_level: float = 0.0,
        device: str = 'cpu',
        use_bifurcation_control: bool = False
    ):
        """
        Initialize the Chaotic Embedding Layer.
        
        Args:
            input_dim: Dimension of input feature vector (default: 4)
            system_type: Type of chaotic system ('lorenz', 'rossler', 'mackey_glass')
            evolution_time: Duration of chaotic evolution in seconds
            time_step: Numerical integration time step
            sigma, rho, beta: Lorenz system parameters
            coupling_strength: Strength of feature coupling to system
            noise_level: Small noise to prevent identical trajectories
            device: Computation device ('cpu' or 'cuda')
        """
        super(ChaoticEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.system_type = system_type
        self.evolution_time = evolution_time
        self.time_step = time_step
        self.device = device
        self.noise_level = noise_level
        
        # Lorenz system parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.coupling_strength = coupling_strength
        
        # Mackey-Glass system parameters
        # dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
        self.mg_beta = 0.2      # Production rate
        self.mg_gamma = 0.1     # Decay rate
        self.mg_tau = 17.0      # Delay time (in time units)
        self.mg_n = 10.0        # Nonlinearity exponent
        
        # Calculate trajectory dimensions
        self.num_steps = int(evolution_time / time_step)
        self.state_dim = 3  # For all systems: use 3D embedding
        
        # Mackey-Glass delay buffer size (number of steps for delay)
        self.mg_delay_steps = int(self.mg_tau / time_step)
        
        # For Mackey-Glass: we use time-delay embedding to create 3D trajectory
        # The delays for embedding: [0, tau/3, 2*tau/3]
        self.mg_embedding_delays = [0, self.mg_delay_steps // 3, 2 * self.mg_delay_steps // 3]

        # Add after self.param_adapter definition
        if use_bifurcation_control:
            self.bifurcation_net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
            self.use_bifurcation_control = True
        else:
            self.bifurcation_net = None
            self.use_bifurcation_control = False
        
        # Feature mapping networks
        self.initial_state_mapper = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, self.state_dim),
            nn.Tanh()
        )
        
        self.coupling_mapper = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, self.state_dim),
            nn.Tanh()
        )
        
        # Parameter adaptation network
        self.param_adapter = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3),  # Adapt sigma, rho, beta
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize network parameters with careful scaling."""
        for module in [self.initial_state_mapper, self.coupling_mapper, self.param_adapter]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def _lorenz_dynamics(
        self, 
        state: torch.Tensor, 
        coupling: torch.Tensor, 
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lorenz system dynamics: dx/dt = F(x, coupling, params)
        
        Args:
            state: Current state [batch_size, 3] - (x, y, z)
            coupling: Coupling terms [batch_size, 3]
            params: Adapted parameters [batch_size, 3] - (sigma, rho, beta)
            
        Returns:
            State derivatives [batch_size, 3]
        """
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        sigma, rho, beta = params[:, 0], params[:, 1], params[:, 2]
        coupling_x, coupling_y, coupling_z = coupling[:, 0], coupling[:, 1], coupling[:, 2]
        
        # Lorenz equations with coupling
        dx_dt = sigma * (y - x) + coupling_x
        dy_dt = x * (rho - z) - y + coupling_y  
        dz_dt = x * y - beta * z + coupling_z
        
        return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
    
    def _rossler_dynamics(
        self, 
        state: torch.Tensor, 
        coupling: torch.Tensor, 
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Rossler system dynamics (alternative chaotic system).
        
        Args:
            state: Current state [batch_size, 3] - (x, y, z)
            coupling: Coupling terms [batch_size, 3]  
            params: System parameters [batch_size, 3] - (a, b, c)
            
        Returns:
            State derivatives [batch_size, 3]
        """
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        a, b, c = params[:, 0], params[:, 1], params[:, 2]
        coupling_x, coupling_y, coupling_z = coupling[:, 0], coupling[:, 1], coupling[:, 2]
        
        # Rossler equations with coupling
        dx_dt = -y - z + coupling_x
        dy_dt = x + a * y + coupling_y
        dz_dt = b + z * (x - c) + coupling_z
        
        return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)

    def _chua_dynamics(
        self,
        state: torch.Tensor,
        coupling: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Chua's circuit dynamics.

        Chua's equations:
            dx/dt = alpha * (y - x - f(x))
            dy/dt = x - y + z
            dz/dt = -beta * y
        where f(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|) is the piecewise-linear nonlinearity.

        Args:
            state: Current state [batch_size, 3] - (x, y, z)
            coupling: Coupling terms [batch_size, 3]
            params: System parameters [batch_size, 3] - (alpha, beta, m0)
                    m1 is fixed at a standard value relative to m0.

        Returns:
            State derivatives [batch_size, 3]
        """
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        alpha, beta, m0 = params[:, 0], params[:, 1], params[:, 2]
        coupling_x, coupling_y, coupling_z = coupling[:, 0], coupling[:, 1], coupling[:, 2]

        # Standard Chua diode slope values: m1 = -0.5, m0 varies
        m1 = torch.full_like(m0, -0.5)

        # Piecewise-linear Chua diode nonlinearity
        h_x = m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1.0) - torch.abs(x - 1.0))

        dx_dt = alpha * (y - x - h_x) + coupling_x
        dy_dt = x - y + z + coupling_y
        dz_dt = -beta * y + coupling_z

        return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)

    def _mackey_glass_dynamics(
        self,
        x_current: torch.Tensor,
        x_delayed: torch.Tensor,
        coupling: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Mackey-Glass delay differential equation dynamics.
        
        The Mackey-Glass equation:
            dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t) + coupling
        
        Args:
            x_current: Current state value [batch_size, 1]
            x_delayed: Delayed state value x(t-tau) [batch_size, 1]
            coupling: Coupling term [batch_size, 1]
            params: System parameters [batch_size, 3] - (beta, gamma, n)
            
        Returns:
            State derivative [batch_size, 1]
        """
        beta_mg = params[:, 0]   # Production rate (~0.2)
        gamma = params[:, 1]     # Decay rate (~0.1)
        n = params[:, 2]         # Nonlinearity exponent (~10)
        
        # Ensure x_delayed is positive for numerical stability
        x_delayed_safe = torch.clamp(x_delayed.squeeze(-1), min=1e-8)
        x_current_safe = x_current.squeeze(-1)
        coupling_val = coupling[:, 0] if coupling.dim() > 1 else coupling
        
        # Mackey-Glass equation
        x_delayed_pow_n = torch.pow(x_delayed_safe, n)
        dx_dt = beta_mg * x_delayed_safe / (1.0 + x_delayed_pow_n) - gamma * x_current_safe + coupling_val
        
        return dx_dt.unsqueeze(-1)
    
    def _runge_kutta_4(
        self, 
        state: torch.Tensor, 
        coupling: torch.Tensor, 
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        4th-order Runge-Kutta numerical integration for ODEs.
        
        Note: For Mackey-Glass (DDE), use _integrate_mackey_glass instead.
        
        Args:
            state: Current state [batch_size, 3]
            coupling: Coupling terms [batch_size, 3]
            params: System parameters [batch_size, 3]
            
        Returns:
            Next state [batch_size, 3]
        """
        h = self.time_step
        
        # Choose dynamics function based on system type
        if self.system_type == 'lorenz':
            dynamics_fn = self._lorenz_dynamics
        elif self.system_type == 'rossler':
            dynamics_fn = self._rossler_dynamics
        elif self.system_type == 'chua':
            dynamics_fn = self._chua_dynamics
        else:
            raise ValueError(f"Unsupported system type for RK4: {self.system_type}. "
                           f"Use _integrate_mackey_glass for mackey_glass system.")
        
        # RK4 integration
        k1 = dynamics_fn(state, coupling, params)
        k2 = dynamics_fn(state + 0.5 * h * k1, coupling, params)
        k3 = dynamics_fn(state + 0.5 * h * k2, coupling, params)
        k4 = dynamics_fn(state + h * k3, coupling, params)
        
        return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _integrate_mackey_glass(
        self,
        initial_state: torch.Tensor,
        coupling: torch.Tensor,
        params: torch.Tensor,
        history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate Mackey-Glass DDE using RK4 with delay history.
        
        Args:
            initial_state: Current state [batch_size, 1]
            coupling: Coupling terms [batch_size, 1]
            params: System parameters [batch_size, 3]
            history: History buffer [batch_size, delay_steps]
            
        Returns:
            Tuple of (next_state, updated_history)
        """
        h = self.time_step
        batch_size = initial_state.shape[0]
        
        # Get delayed value from history buffer
        x_delayed = history[:, 0:1]  # Oldest value in buffer
        
        # RK4 for delay differential equation
        # Note: For DDE, we use the same delayed value throughout one RK4 step
        # This is a common approximation for small time steps
        
        k1 = self._mackey_glass_dynamics(initial_state, x_delayed, coupling, params)
        k2 = self._mackey_glass_dynamics(initial_state + 0.5 * h * k1, x_delayed, coupling, params)
        k3 = self._mackey_glass_dynamics(initial_state + 0.5 * h * k2, x_delayed, coupling, params)
        k4 = self._mackey_glass_dynamics(initial_state + h * k3, x_delayed, coupling, params)
        
        next_state = initial_state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update history buffer (shift and add new value)
        updated_history = torch.cat([history[:, 1:], next_state], dim=1)
        
        return next_state, updated_history
    
    def _generate_mackey_glass_trajectory(
        self,
        initial_states: torch.Tensor,
        couplings: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate Mackey-Glass trajectory with time-delay embedding for 3D representation.
        
        The 1D Mackey-Glass signal is embedded into 3D using time delays:
        [x(t), x(t-d1), x(t-d2)] where d1, d2 are chosen delay values.
        
        Args:
            initial_states: Initial conditions [batch_size, 3] (we use first component)
            couplings: Coupling terms [batch_size, 3] (we use first component)
            params: System parameters [batch_size, 3] - (beta, gamma, n)
            
        Returns:
            3D embedded trajectory [batch_size, num_steps, 3]
        """
        batch_size = initial_states.shape[0]
        
        # Total steps needed including warmup for delay buffer
        warmup_steps = self.mg_delay_steps + max(self.mg_embedding_delays)
        total_steps = warmup_steps + self.num_steps
        
        # Initialize 1D signal storage
        signal_1d = torch.zeros(
            batch_size, total_steps,
            device=self.device, dtype=torch.float32
        )
        
        # Initialize history buffer with small random values around 0.5
        # This is a typical initialization for Mackey-Glass
        history = torch.ones(
            batch_size, self.mg_delay_steps,
            device=self.device, dtype=torch.float32
        ) * 0.5
        
        # Add variation from initial states
        init_val = initial_states[:, 0:1].abs() * 0.5 + 0.3  # Range roughly [0.3, 0.8]
        history = history + (init_val - 0.5) * 0.2  # Add some variation
        
        # Current state
        state = init_val.clone()
        
        # Use only first coupling component for 1D system
        coupling_1d = couplings[:, 0:1] * 0.01  # Scale down coupling for MG
        
        # Integrate the system
        for t in range(total_steps):
            signal_1d[:, t] = state.squeeze(-1)
            
            # Add small noise for numerical stability
            if self.noise_level > 0:
                noise = torch.randn_like(state) * self.noise_level
                state = state + noise
            
            # Integrate one step
            state, history = self._integrate_mackey_glass(
                state, coupling_1d, params, history
            )
            
            # Clamp to prevent explosion
            state = torch.clamp(state, min=0.0, max=3.0)
        
        # Extract the main trajectory (after warmup)
        main_signal = signal_1d[:, warmup_steps:]
        
        # Create 3D embedding using time delays
        trajectory_3d = torch.zeros(
            batch_size, self.num_steps, 3,
            device=self.device, dtype=torch.float32
        )
        
        # Time-delay embedding: [x(t), x(t-d1), x(t-d2)]
        d1 = self.mg_embedding_delays[1]
        d2 = self.mg_embedding_delays[2]
        
        # We need to access signal with delays, so we use the full signal buffer
        full_signal = signal_1d[:, (warmup_steps - d2):]  # Start from earliest needed point
        
        for t in range(self.num_steps):
            trajectory_3d[:, t, 0] = full_signal[:, t + d2]      # x(t)
            trajectory_3d[:, t, 1] = full_signal[:, t + d2 - d1] # x(t - d1)
            trajectory_3d[:, t, 2] = full_signal[:, t]           # x(t - d2)
        
        return trajectory_3d
    
    def _generate_trajectory(
        self, 
        initial_states: torch.Tensor, 
        couplings: torch.Tensor, 
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate chaotic trajectory through numerical integration.
        
        Args:
            initial_states: Initial conditions [batch_size, 3]
            couplings: Coupling terms [batch_size, 3]
            params: System parameters [batch_size, 3]
            
        Returns:
            Trajectory tensor [batch_size, num_steps, 3]
        """
        # Dispatch to specialized method for Mackey-Glass
        if self.system_type == 'mackey_glass':
            return self._generate_mackey_glass_trajectory(initial_states, couplings, params)
        
        # Standard ODE systems (Lorenz, Rossler)
        batch_size = initial_states.shape[0]
        trajectory = torch.zeros(
            batch_size, self.num_steps, self.state_dim,
            device=self.device, dtype=torch.float32
        )
        
        # Initialize trajectory
        state = initial_states.clone()
        trajectory[:, 0, :] = state
        
        # Evolve system
        for t in range(1, self.num_steps):
            # Add small noise for numerical stability
            if self.noise_level > 0:
                noise = torch.randn_like(state) * self.noise_level
                state = state + noise
                
            # Integrate one step
            state = self._runge_kutta_4(state, couplings, params)
            trajectory[:, t, :] = state
            
        return trajectory
    
    def _adapt_parameters(self, features: torch.Tensor) -> torch.Tensor:
        """
        Adapt system parameters based on input features.
        
        Args:
            features: Input feature vector [batch_size, input_dim]
            
        Returns:
            Adapted parameters [batch_size, 3]
        """
        param_scales = self.param_adapter(features)
        
        if self.system_type == 'lorenz':
            # Scale parameters around typical Lorenz values
            sigma = self.sigma * (0.5 + param_scales[:, 0])  # 5-15
            rho = self.rho * (0.5 + param_scales[:, 1])      # 14-42
            beta = self.beta * (0.5 + param_scales[:, 2])    # 1.3-4.0
            
            # FIX v2: Bifurcation control as MODULATION, not replacement
            # This preserves the learned param_adapter mapping while adding
            # bifurcation-aware adjustment
            if self.use_bifurcation_control and self.bifurcation_net is not None:
                regime_signal = self.bifurcation_net(features).squeeze(-1)  # [batch], in [0, 1]
                
                # Option A: Multiplicative modulation (Â±20%)
                # modulation_factor in [0.8, 1.2]
                modulation_factor = 0.8 + 0.4 * regime_signal
                rho = rho * modulation_factor
                
                # Clamp to safe chaotic range
                rho = torch.clamp(rho, min=20.0, max=50.0)
            
            return torch.stack([sigma, rho, beta], dim=1)
            
        elif self.system_type == 'rossler':
            # Typical Rossler parameters
            a = 0.2 * (0.5 + param_scales[:, 0])   # 0.1-0.3
            b = 0.2 * (0.5 + param_scales[:, 1])   # 0.1-0.3  
            c = 5.7 * (0.5 + param_scales[:, 2])   # 2.85-8.55
            return torch.stack([a, b, c], dim=1)
            
        elif self.system_type == 'mackey_glass':
            # Mackey-Glass system parameters
            # Typical parameters: beta=0.2, gamma=0.1, n=10
            # param_scales is already computed from param_adapter(features)
            beta_mg = self.mg_beta * (0.8 + 0.4 * param_scales[:, 0])   # 0.16-0.24 (around 0.2)
            gamma = self.mg_gamma * (0.8 + 0.4 * param_scales[:, 1])    # 0.08-0.12 (around 0.1)
            n = self.mg_n * (0.8 + 0.4 * param_scales[:, 2])            # 8-12 (around 10)
            return torch.stack([beta_mg, gamma, n], dim=1)
        
        elif self.system_type == 'chua':
            # Chua circuit parameters
            alpha = self.param_adapter(features)[:, 0] * 5 + 15
            beta_chua = self.param_adapter(features)[:, 1] * 10 + 28
            m0 = self.param_adapter(features)[:, 2] * 0.5 - 1.14
            return torch.stack([alpha, beta_chua, m0], dim=1)
        
        else:
            # Default: return generic parameters
            param1 = self.param_adapter(features)[:, 0]
            param2 = self.param_adapter(features)[:, 1]
            param3 = self.param_adapter(features)[:, 2]
            return torch.stack([param1, param2, param3], dim=1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through chaotic embedding layer.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Chaotic trajectories [batch_size, num_steps, state_dim]
        """
        batch_size = features.shape[0]
        
        # Map features to initial conditions (scaled to reasonable range)
        initial_states = self.initial_state_mapper(features) * 2.0  # [-2, 2]
        
        # Map features to coupling terms (smaller scale)
        couplings = self.coupling_mapper(features) * self.coupling_strength
        
        # Adapt system parameters based on features
        params = self._adapt_parameters(features)
        
        # Generate chaotic trajectory
        trajectory = self._generate_trajectory(initial_states, couplings, params)
        
        return trajectory
    
    def get_attractor_statistics(self, trajectory: torch.Tensor) -> dict:
        """
        Extract basic statistics from chaotic trajectory.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, state_dim]
            
        Returns:
            Dictionary of attractor statistics
        """
        stats = {}
        
        # Basic statistics
        stats['mean'] = torch.mean(trajectory, dim=1)  # [batch_size, state_dim]
        stats['std'] = torch.std(trajectory, dim=1)    # [batch_size, state_dim]
        stats['range'] = torch.max(trajectory, dim=1)[0] - torch.min(trajectory, dim=1)[0]
        
        # Trajectory length (cumulative distance)
        diff = torch.diff(trajectory, dim=1)
        distances = torch.norm(diff, dim=2)
        stats['total_length'] = torch.sum(distances, dim=1)  # [batch_size]
        
        return stats
    
    def visualize_attractor(self, trajectory: torch.Tensor, sample_idx: int = 0):
        """
        Create 3D visualization of chaotic attractor (for debugging/analysis).
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, state_dim]
            sample_idx: Which sample to visualize
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Extract single trajectory
            traj = trajectory[sample_idx].detach().cpu().numpy()
            x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectory with color gradient
            colors = np.linspace(0, 1, len(x))
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(f'{self.system_type.capitalize()} Attractor')
            
            plt.colorbar(scatter, label='Time')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


class AdaptiveChaoticEmbedding(ChaoticEmbedding):
    """
    Adaptive version of ChaoticEmbedding with learnable system parameters.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make system parameters learnable
        self.sigma_base = nn.Parameter(torch.tensor(self.sigma, dtype=torch.float32))
        self.rho_base = nn.Parameter(torch.tensor(self.rho, dtype=torch.float32))  
        self.beta_base = nn.Parameter(torch.tensor(self.beta, dtype=torch.float32))
        
        # Learnable coupling strength
        self.coupling_scale = nn.Parameter(
            torch.tensor(self.coupling_strength, dtype=torch.float32)
        )
    
    def _adapt_parameters(self, features: torch.Tensor) -> torch.Tensor:
        """Override to use learnable base parameters."""
        param_scales = self.param_adapter(features)
        
        if self.system_type == 'lorenz':
            sigma = self.sigma_base * (0.5 + param_scales[:, 0])
            rho = self.rho_base * (0.5 + param_scales[:, 1])
            beta = self.beta_base * (0.5 + param_scales[:, 2])
            
            # FIX v2: Bifurcation control as MODULATION, not replacement
            if self.use_bifurcation_control and self.bifurcation_net is not None:
                regime_signal = self.bifurcation_net(features).squeeze(-1)  # [batch], in [0, 1]
                
                # Multiplicative modulation (Â±20%)
                modulation_factor = 0.8 + 0.4 * regime_signal
                rho = rho * modulation_factor
                
                # Clamp to safe chaotic range
                rho = torch.clamp(rho, min=20.0, max=50.0)
            
            return torch.stack([sigma, rho, beta], dim=1)
            
        else:
            sigma = param_scales[:, 0]
            rho = param_scales[:, 1] 
            beta = param_scales[:, 2]
            
        return torch.stack([sigma, rho, beta], dim=1)


# Factory function for easy instantiation
def create_chaotic_embedding(config: dict) -> ChaoticEmbedding:
    """
    Factory function to create chaotic embedding layer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ChaoticEmbedding instance
    """
    adaptive = config.get('adaptive', False)

    config_copy = config.copy()
    config_copy.pop('adaptive', None)
    
    if adaptive:
        return AdaptiveChaoticEmbedding(**config_copy)
    else:
        return ChaoticEmbedding(**config_copy)


if __name__ == "__main__":
    # Test the chaotic embedding layer
    print("Testing Chaotic Embedding Layer...")
    print("=" * 60)
    
    batch_size = 8
    test_features = torch.randn(batch_size, 4)
    print(f"Input features shape: {test_features.shape}")
    
    # Test all three system types
    system_types = ['lorenz', 'rossler', 'mackey_glass']
    
    for system_type in system_types:
        print(f"\n{'='*60}")
        print(f"Testing {system_type.upper()} system...")
        print(f"{'='*60}")
        
        # Create test configuration
        config = {
            'input_dim': 4,
            'system_type': system_type,
            'evolution_time': 0.5,
            'time_step': 0.01,
            'device': 'cpu',
            'noise_level': 0.001
        }
        
        # Create embedding layer
        embedding = create_chaotic_embedding(config)
        
        print(f"Number of steps: {embedding.num_steps}")
        print(f"State dimension: {embedding.state_dim}")
        if system_type == 'mackey_glass':
            print(f"Delay steps: {embedding.mg_delay_steps}")
            print(f"Embedding delays: {embedding.mg_embedding_delays}")
        
        with torch.no_grad():
            trajectories = embedding(test_features)
            stats = embedding.get_attractor_statistics(trajectories)
        
        print(f"Output trajectories shape: {trajectories.shape}")
        print(f"Trajectory statistics:")
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}, "
                      f"mean={value.mean().item():.4f}, std={value.std().item():.4f}")
        
        # Check for NaN or Inf
        if torch.isnan(trajectories).any():
            print("  WARNING: NaN values detected!")
        if torch.isinf(trajectories).any():
            print("  WARNING: Inf values detected!")
        
        # Test trajectory range
        print(f"  Trajectory range: [{trajectories.min().item():.4f}, {trajectories.max().item():.4f}]")
    
    # Test adaptive version with Lorenz
    print(f"\n{'='*60}")
    print("Testing Adaptive Chaotic Embedding (Lorenz)...")
    print(f"{'='*60}")
    
    config = {
        'input_dim': 4,
        'system_type': 'lorenz',
        'evolution_time': 0.2,
        'time_step': 0.01,
        'device': 'cpu',
        'adaptive': True
    }
    adaptive_embedding = create_chaotic_embedding(config)
    
    with torch.no_grad():
        adaptive_trajectories = adaptive_embedding(test_features)
    
    print(f"Adaptive trajectories shape: {adaptive_trajectories.shape}")
    print(f"Trajectory range: [{adaptive_trajectories.min().item():.4f}, {adaptive_trajectories.max().item():.4f}]")
    
    print("\n" + "=" * 60)
    print("Chaotic Embedding Layer test completed!")
    print("=" * 60)