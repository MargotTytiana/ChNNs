import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union


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
        
        # Calculate trajectory dimensions
        self.num_steps = int(evolution_time / time_step)
        self.state_dim = 3  # For Lorenz system: (x, y, z)
        
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
        else:
            raise ValueError(f"Unsupported system type: {self.system_type}")
        
        # RK4 integration
        k1 = dynamics_fn(state, coupling, params)
        k2 = dynamics_fn(state + 0.5 * h * k1, coupling, params)
        k3 = dynamics_fn(state + 0.5 * h * k2, coupling, params)
        k4 = dynamics_fn(state + h * k3, coupling, params)
        
        return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
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
        elif self.system_type == 'rossler':
            # Typical Rössler parameters
            a = 0.2 * (0.5 + param_scales[:, 0])   # 0.1-0.3
            b = 0.2 * (0.5 + param_scales[:, 1])   # 0.1-0.3  
            c = 5.7 * (0.5 + param_scales[:, 2])   # 2.85-8.55
            sigma, rho, beta = a, b, c
        
        return torch.stack([sigma, rho, beta], dim=1)
    
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
    # Test the chaotic embedding layer
    print("Testing Chaotic Embedding Layer...")
    
    # Create test configuration
    config = {
        'input_dim': 4,
        'system_type': 'lorenz',
        'evolution_time': 0.2,
        'time_step': 0.01,
        'device': 'cpu'
    }
    
    # Create embedding layer
    embedding = create_chaotic_embedding(config)
    
    # Test forward pass
    batch_size = 8
    test_features = torch.randn(batch_size, 4)
    
    print(f"Input features shape: {test_features.shape}")
    
    with torch.no_grad():
        trajectories = embedding(test_features)
        stats = embedding.get_attractor_statistics(trajectories)
    
    print(f"Output trajectories shape: {trajectories.shape}")
    print(f"Trajectory statistics:")
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
    
    # Test adaptive version
    print("\nTesting Adaptive Chaotic Embedding...")
    config['adaptive'] = True
    adaptive_embedding = create_chaotic_embedding(config)
    
    with torch.no_grad():
        adaptive_trajectories = adaptive_embedding(test_features)
    
    print(f"Adaptive trajectories shape: {adaptive_trajectories.shape}")
    
    print("\nChaotic Embedding Layer test completed!")