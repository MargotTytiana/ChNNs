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
        device: str = 'cpu'
    ):
        """
        Initialize the Chaotic Embedding Layer.
        
        Args:
            input_dim: Dimension of input feature vector (default: 4)
            system_type: Type of chaotic system ('lorenz', 'rossler', 'mackey_glass', 'chua')
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
        
        # System parameters (used for Lorenz by default)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.coupling_strength = coupling_strength
        
        # Calculate trajectory dimensions
        self.num_steps = int(evolution_time / time_step)
        self.state_dim = 3  # All systems use 3D state: (x, y, z)
        
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
            nn.Linear(8, 3),  # Adapt 3 system parameters
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
        Compute Rössler system dynamics (alternative chaotic system).
        
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
        
        # Rössler equations with coupling
        dx_dt = -y - z + coupling_x
        dy_dt = x + a * y + coupling_y
        dz_dt = b + z * (x - c) + coupling_z
        
        return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
    
    def _runge_kutta_4(
            self, 
            state: torch.Tensor, 
            coupling: torch.Tensor, 
            params: torch.Tensor
        ) -> torch.Tensor:
        """
        4th-order Runge-Kutta numerical integration.
        
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
        elif self.system_type == 'mackey_glass':
            dynamics_fn = self._mackey_glass_dynamics
        elif self.system_type == 'chua':
            dynamics_fn = self._chua_dynamics
        else:
            raise ValueError(f"Unsupported system type: {self.system_type}")
        
        # RK4 integration
        k1 = dynamics_fn(state, coupling, params)
        k2 = dynamics_fn(state + 0.5 * h * k1, coupling, params)
        k3 = dynamics_fn(state + 0.5 * h * k2, coupling, params)
        k4 = dynamics_fn(state + h * k3, coupling, params)
        
        return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


    def _mackey_glass_dynamics(
            self, 
            state: torch.Tensor, 
            coupling: torch.Tensor, 
            params: torch.Tensor
        ) -> torch.Tensor:
        """
        Mackey-Glass system dynamics (simplified version without delay).
        
        Since we can't easily handle delays in this framework, we use a simplified
        coupled oscillator approximation that produces chaotic behavior.
        
        Args:
            state: Current state [batch_size, 3] as (x, y, z)
            coupling: Coupling terms [batch_size, 3]
            params: System parameters [batch_size, 3] as (beta, gamma, n)
            
        Returns:
            Derivatives [batch_size, 3]
        """
        batch_size = state.shape[0]
        
        # Extract parameters with defaults
        beta = params[:, 0] if params is not None else 0.2
        gamma = params[:, 1] if params is not None else 0.1  
        n = params[:, 2] if params is not None else 10.0
        
        # Ensure parameters have correct shape
        if beta.dim() == 0:
            beta = beta.expand(batch_size)
        if gamma.dim() == 0:
            gamma = gamma.expand(batch_size)
        if n.dim() == 0:
            n = n.expand(batch_size)
        
        # Extract state variables
        x = state[:, 0]
        y = state[:, 1] 
        z = state[:, 2]
        
        # Extract coupling terms
        cx, cy, cz = coupling[:, 0], coupling[:, 1], coupling[:, 2]
        
        # Simplified Mackey-Glass dynamics (coupled oscillator approximation)
        # This creates a 3D system that approximates chaotic behavior
        dx_dt = beta * x / (1 + torch.pow(torch.abs(y), n)) - gamma * x + cx * z
        dy_dt = x - gamma * y + cy * x * z
        dz_dt = -gamma * z + cz * x * y
        
        # Alternative: More complex coupling for richer dynamics
        # dx_dt = beta * y / (1 + torch.pow(torch.abs(z), n)) - gamma * x + cx * (y - x)
        # dy_dt = beta * z / (1 + torch.pow(torch.abs(x), n)) - gamma * y + cy * (z - y) 
        # dz_dt = beta * x / (1 + torch.pow(torch.abs(y), n)) - gamma * z + cz * (x - z)
        
        derivatives = torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
        
        return derivatives

    def _chua_dynamics(
            self, 
            state: torch.Tensor, 
            coupling: torch.Tensor, 
            params: torch.Tensor
        ) -> torch.Tensor:
        """
        Chua circuit system dynamics.
        
        Chua's circuit is a simple electronic circuit that exhibits chaotic behavior.
        Equations:
            dx/dt = alpha * (y - x - f(x))
            dy/dt = x - y + z  
            dz/dt = -beta * y
            
        where f(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)
        
        Args:
            state: Current state [batch_size, 3] as (x, y, z)
            coupling: Coupling terms [batch_size, 3]
            params: System parameters [batch_size, 3] as (alpha, beta, m0_m1_ratio)
            
        Returns:
            Derivatives [batch_size, 3]
        """
        batch_size = state.shape[0]
        
        # Extract parameters with defaults
        alpha = params[:, 0] if params is not None else 15.6
        beta = params[:, 1] if params is not None else 28.0
        m0_m1_ratio = params[:, 2] if params is not None else -1.143  # m0/m1 ratio
        
        # Ensure parameters have correct shape
        if alpha.dim() == 0:
            alpha = alpha.expand(batch_size)
        if beta.dim() == 0:
            beta = beta.expand(batch_size)
        if m0_m1_ratio.dim() == 0:
            m0_m1_ratio = m0_m1_ratio.expand(batch_size)
        
        # Extract state variables
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        
        # Extract coupling terms
        cx, cy, cz = coupling[:, 0], coupling[:, 1], coupling[:, 2]
        
        # Chua's diode nonlinear function parameters
        # Typical values: m0 = -1/7, m1 = 2/7
        m0 = -0.714 * m0_m1_ratio  # ~ -1/7 when ratio = 1
        m1 = 0.285 + 0.1 * m0_m1_ratio  # ~ 2/7 when ratio = 1
        
        # Chua's diode piecewise-linear function
        f_x = m1 * x + 0.5 * (m0 - m1) * (
            torch.abs(x + 1.0) - torch.abs(x - 1.0)
        )
        
        # Chua circuit equations with coupling
        dx_dt = alpha * (y - x - f_x) + cx * y
        dy_dt = x - y + z + cy * z
        dz_dt = -beta * y + cz * x
        
        derivatives = torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
        
        return derivatives        
    
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
            return torch.stack([sigma, rho, beta], dim=1)
            
        elif self.system_type == 'rossler':
            # Typical Rössler parameters
            a = 0.2 * (0.5 + param_scales[:, 0])   # 0.1-0.3
            b = 0.2 * (0.5 + param_scales[:, 1])   # 0.1-0.3  
            c = 5.7 * (0.5 + param_scales[:, 2])   # 2.85-8.55
            return torch.stack([a, b, c], dim=1)
            
        elif self.system_type == 'mackey_glass':
            # Mackey-Glass system parameters
            # Typical parameters: beta=0.2, gamma=0.1, n=10
            beta_mg = 0.1 * param_scales[:, 0] + 0.2    # ~0.2
            gamma = 0.05 * param_scales[:, 1] + 0.1     # ~0.1
            n = 5.0 * param_scales[:, 2] + 8.0          # ~10
            return torch.stack([beta_mg, gamma, n], dim=1)
        
        elif self.system_type == 'chua':
            # Chua circuit parameters
            # Typical parameters: alpha=15.6, beta=28.0, m0/m1 ratio
            alpha = 10.0 * param_scales[:, 0] + 10.0    # ~15.6
            beta_chua = 20.0 * param_scales[:, 1] + 20.0  # ~28.0
            m0_m1_ratio = 2.0 * param_scales[:, 2] - 1.5  # ~-1.143
            return torch.stack([alpha, beta_chua, m0_m1_ratio], dim=1)
        
        else:
            # Default generic parameters
            return param_scales
    
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
    # Test the chaotic embedding layer with all systems
    print("Testing Chaotic Embedding Layer with All Systems...")
    
    systems_to_test = ['lorenz', 'rossler', 'mackey_glass', 'chua']
    
    for system in systems_to_test:
        print(f"\n=== Testing {system.upper()} System ===")
        
        # Create test configuration
        config = {
            'input_dim': 4,
            'system_type': system,
            'evolution_time': 0.1,  # Shorter for faster testing
            'time_step': 0.01,
            'device': 'cpu'
        }
        
        # Create embedding layer
        embedding = create_chaotic_embedding(config)
        
        # Test forward pass
        batch_size = 4
        test_features = torch.randn(batch_size, 4)
        
        print(f"Input features shape: {test_features.shape}")
        
        with torch.no_grad():
            trajectories = embedding(test_features)
            stats = embedding.get_attractor_statistics(trajectories)
        
        print(f"Output trajectories shape: {trajectories.shape}")
        print(f"Mean trajectory range: {stats['range'].mean().item():.4f}")
        print(f"Total trajectory length: {stats['total_length'].mean().item():.4f}")
    
    print("\nAll chaotic systems test completed!")