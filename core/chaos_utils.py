"""
Robust Chaos Analysis Utilities for Speaker Recognition.

This module provides numerically stable implementations of chaotic systems,
Lyapunov exponent calculations, and chaos analysis tools with comprehensive
error handling and precision management.

Author: C-HiLAP Project
Date: 2025
"""
import numpy as np
import warnings
from scipy import integrate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from scipy.signal import detrend
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict, Optional, Union, Any
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def setup_module_imports(current_file: str = __file__):
    """Setup imports for current module.""" 
    try:
        from setup_imports import setup_project_imports
        return setup_project_imports(current_file), True
    except ImportError:
        current_dir = Path(current_file).resolve().parent  # core
        project_root = current_dir.parent  # core -> Model
        
        paths_to_add = [
            str(project_root),
            str(project_root / 'core'),
            str(project_root / 'utils'),
        ]
        
        for path in paths_to_add:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
        
        return project_root, False

# Setup imports
PROJECT_ROOT, USING_IMPORT_MANAGER = setup_module_imports()

# =============================================================================
# project model import
# =============================================================================
try:
    from core.phase_space_reconstruction import PhaseSpaceReconstructor, EmbeddingConfig
    HAS_PHASE_SPACE = True
except ImportError as e:
    HAS_PHASE_SPACE = False
    warnings.warn(f"phase_space_reconstruction not available: {e}")

try:
    from utils.numerical_stability import NumericalConfig, safe_divide
    HAS_NUMERICAL_UTILS = True
except ImportError as e:
    HAS_NUMERICAL_UTILS = False
    warnings.warn(f"numerical_stability not available: {e}")
    # Simple fallback
    safe_divide = lambda x, y: x / (y + 1e-12)
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaoticSystem:
    """Base class for chaotic dynamical systems."""
    
    def __init__(self):
        self.dimension = None
        self.parameters = {}
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """Define the differential equations of the system."""
        raise NotImplementedError("Subclasses must implement equations method")
    
    def get_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Get Jacobian matrix at given state (optional)."""
        return None


class LorenzSystem(ChaoticSystem):
    """Lorenz chaotic system implementation with numerical stability."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        super().__init__()
        self.dimension = 3
        self.parameters = {
            'sigma': float(sigma),
            'rho': float(rho), 
            'beta': float(beta)
        }
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """Lorenz system differential equations."""
        x, y, z = state
        sigma, rho, beta = self.parameters['sigma'], self.parameters['rho'], self.parameters['beta']
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def get_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix of Lorenz system."""
        x, y, z = state
        sigma, rho, beta = self.parameters['sigma'], self.parameters['rho'], self.parameters['beta']
        
        jacobian = np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])
        
        return jacobian


class RosslerSystem(ChaoticSystem):
    """Rossler chaotic system implementation."""
    
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        super().__init__()
        self.dimension = 3
        self.parameters = {
            'a': float(a),
            'b': float(b),
            'c': float(c)
        }
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """Rossler system differential equations."""
        x, y, z = state
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        
        dx_dt = -y - z
        dy_dt = x + a * y
        dz_dt = b + z * (x - c)
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def get_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix of Rossler system."""
        x, y, z = state
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        
        jacobian = np.array([
            [0, -1, -1],
            [1, a, 0],
            [z, 0, x - c]
        ])
        
        return jacobian


class ChenSystem(ChaoticSystem):
    """Chen chaotic system implementation."""
    
    def __init__(self, a: float = 35.0, b: float = 3.0, c: float = 28.0):
        super().__init__()
        self.dimension = 3
        self.parameters = {
            'a': float(a),
            'b': float(b),
            'c': float(c)
        }
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """Chen system differential equations."""
        x, y, z = state
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        
        dx_dt = a * (y - x)
        dy_dt = (c - a) * x - x * z + c * y
        dz_dt = x * y - b * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def get_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix of Chen system."""
        x, y, z = state
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        
        jacobian = np.array([
            [-a, a, 0],
            [c - a - z, c, -x],
            [y, x, -b]
        ])
        
        return jacobian


def validate_solve_ivp_params(t_span: Tuple[float, float], t_eval: Optional[np.ndarray] = None) -> bool:
    """
    Robust validation for scipy.integrate.solve_ivp parameters.
    
    Args:
        t_span: Integration time span (t_start, t_end)
        t_eval: Time evaluation points
        
    Returns:
        True if parameters are valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise ValueError("t_span must be a tuple or list with 2 elements: (t_start, t_end)")
    
    t_start, t_end = float(t_span[0]), float(t_span[1])
    
    if t_start >= t_end:
        raise ValueError(f"t_start ({t_start}) must be less than t_end ({t_end})")
    
    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        
        if t_eval.size == 0:
            raise ValueError("t_eval array is empty")
        
        # Use machine precision tolerance for boundary checks
        eps = np.finfo(float).eps
        tol = max(abs(t_start), abs(t_end)) * eps * 100
        
        if t_eval[0] < t_start - tol:
            raise ValueError(f"First t_eval value ({t_eval[0]}) is before t_start ({t_start})")
        
        if t_eval[-1] > t_end + tol:
            raise ValueError(f"Last t_eval value ({t_eval[-1]}) is after t_end ({t_end})")
    
    return True


def create_safe_time_array(t_start: float, t_end: float, num_points: int) -> np.ndarray:
    """
    Generate time evaluation array guaranteed to respect boundaries.
    
    Args:
        t_start: Start time
        t_end: End time  
        num_points: Number of points
        
    Returns:
        Safe time array within boundaries
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    
    t_eval = np.linspace(t_start, t_end, num_points)
    
    # Force exact boundary matching to prevent floating-point drift
    t_eval[0] = t_start
    t_eval[-1] = t_end
    
    return t_eval


class ChaoticSystemSolver:
    """Numerically robust solver for chaotic dynamical systems."""
    
    def __init__(self, system: ChaoticSystem):
        self.system = system
        
    def solve(self, initial_state: np.ndarray, time_span: Tuple[float, float], 
              dt: float = 0.01, method: str = 'rk45', 
              transient_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve chaotic system with robust error handling.
        
        Args:
            initial_state: Initial conditions
            time_span: Integration time span (t_start, t_end)
            dt: Time step size
            method: Integration method ('rk45', 'rk4', 'euler')
            transient_time: Transient time to remove (if any)
            
        Returns:
            Tuple of (time_array, trajectory_array)
        """
        initial_state = np.asarray(initial_state, dtype=float)
        t_start, t_end = float(time_span[0]), float(time_span[1])
        
        if initial_state.size != self.system.dimension:
            raise ValueError(f"Initial state size ({initial_state.size}) doesn't match system dimension ({self.system.dimension})")
        
        # Handle transient time properly
        if transient_time is not None and transient_time > 0:
            # Extend integration time to include transient
            t_start_extended = t_start
            t_end_extended = t_end + transient_time
            integration_span = (t_start_extended, t_end_extended)
            
            # Calculate number of points for extended integration
            total_time = t_end_extended - t_start_extended
            num_points = max(10, int((t_end - t_start) / dt) + 1)  # min 10 is reasonable
            t_eval = create_safe_time_array(t_start_extended, t_end_extended, num_points)
        else:
            integration_span = (t_start, t_end)
            total_time = t_end - t_start
            num_points = max(10, int((t_end - t_start) / dt) + 1)  # min 10 is reasonable
            t_eval = create_safe_time_array(t_start, t_end, num_points)
        
        # Validate parameters before integration
        validate_solve_ivp_params(integration_span, t_eval)
        
        if method.lower() == 'rk45':
            return self._solve_rk45(initial_state, integration_span, t_eval, transient_time, time_span)
        elif method.lower() == 'rk4':
            return self._solve_rk4(initial_state, integration_span, t_eval, transient_time, time_span)
        elif method.lower() == 'euler':
            return self._solve_euler(initial_state, integration_span, t_eval, transient_time, time_span)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def _solve_rk45(self, initial_state: np.ndarray, integration_span: Tuple[float, float], 
                    t_eval: np.ndarray, transient_time: Optional[float], 
                    target_span: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using scipy's RK45 method with fallback strategies."""
        
        # Enhanced integration parameters for chaotic systems
        integration_params = {
            'rtol': 1e-8,      # Tighter relative tolerance
            'atol': 1e-11,     # Tighter absolute tolerance
            'max_step': 0.1,   # Prevent excessive steps
            't_eval': t_eval
        }
        
        # Try different solvers if RK45 fails
        solvers = ['RK45', 'DOP853', 'Radau']
        
        for solver in solvers:
            try:
                integration_params['method'] = solver
                sol = integrate.solve_ivp(
                    self.system.equations, integration_span, initial_state, **integration_params
                )
                
                if sol.success:
                    return self._process_solution(sol, transient_time, target_span)
                else:
                    logger.warning(f"Solver {solver} finished but not successful: {sol.message}")
                    
            except ValueError as e:
                if "not within" in str(e):
                    # Apply automatic correction for boundary issues
                    logger.warning(f"Boundary issue with {solver}, correcting t_eval")
                    t_eval_corrected = np.clip(t_eval, integration_span[0], integration_span[1])
                    t_eval_corrected[0] = integration_span[0]
                    t_eval_corrected[-1] = integration_span[1]
                    integration_params['t_eval'] = t_eval_corrected
                    continue
                else:
                    logger.warning(f"Solver {solver} failed: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Solver {solver} failed with unexpected error: {e}")
                continue
        
        raise RuntimeError("All integration methods failed")
    
    def _solve_rk4(self, initial_state: np.ndarray, integration_span: Tuple[float, float],
                   t_eval: np.ndarray, transient_time: Optional[float],
                   target_span: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using custom RK4 implementation."""
        
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.01
        t_current = integration_span[0]
        state_current = initial_state.copy()
        
        t_result = []
        trajectory_result = []
        
        for t_target in t_eval:
            while t_current < t_target:
                step_size = min(dt, t_target - t_current)
                
                # RK4 integration step
                k1 = self.system.equations(t_current, state_current)
                k2 = self.system.equations(t_current + step_size/2, state_current + step_size*k1/2)
                k3 = self.system.equations(t_current + step_size/2, state_current + step_size*k2/2)
                k4 = self.system.equations(t_current + step_size, state_current + step_size*k3)
                
                state_current = state_current + step_size * (k1 + 2*k2 + 2*k3 + k4) / 6
                t_current += step_size
            
            t_result.append(t_current)
            trajectory_result.append(state_current.copy())
        
        t_result = np.array(t_result)
        trajectory_result = np.array(trajectory_result).T
        
        # Create mock solution object for processing
        class MockSolution:
            def __init__(self, t, y):
                self.t = t
                self.y = y
                self.success = True
        
        sol = MockSolution(t_result, trajectory_result)
        return self._process_solution(sol, transient_time, target_span)
    
    def _solve_euler(self, initial_state: np.ndarray, integration_span: Tuple[float, float],
                     t_eval: np.ndarray, transient_time: Optional[float],
                     target_span: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using Euler method (for testing/fallback)."""
        
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.001  # Small step for stability
        t_current = integration_span[0]
        state_current = initial_state.copy()
        
        t_result = []
        trajectory_result = []
        
        for t_target in t_eval:
            while t_current < t_target:
                step_size = min(dt, t_target - t_current)
                
                # Euler integration step
                derivative = self.system.equations(t_current, state_current)
                state_current = state_current + step_size * derivative
                t_current += step_size
            
            t_result.append(t_current)
            trajectory_result.append(state_current.copy())
        
        t_result = np.array(t_result)
        trajectory_result = np.array(trajectory_result).T
        
        # Create mock solution object for processing
        class MockSolution:
            def __init__(self, t, y):
                self.t = t
                self.y = y
                self.success = True
        
        sol = MockSolution(t_result, trajectory_result)
        return self._process_solution(sol, transient_time, target_span)
    
    def _process_solution(self, sol, transient_time: Optional[float], 
                         target_span: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Process integration solution and handle transient removal."""
        
        if transient_time is not None and transient_time > 0:
            # Remove transient part
            transient_cutoff = target_span[0] + transient_time
            valid_indices = sol.t >= transient_cutoff
            
            if not np.any(valid_indices):
                raise ValueError("Transient time is too long, no valid data remains")
            
            t_result = sol.t[valid_indices]
            trajectory_result = sol.y[:, valid_indices]
        else:
            # Keep original time span
            valid_indices = (sol.t >= target_span[0]) & (sol.t <= target_span[1])
            
            if not np.any(valid_indices):
                # Fallback: use all data if no points in target span
                t_result = sol.t
                trajectory_result = sol.y
            else:
                t_result = sol.t[valid_indices]
                trajectory_result = sol.y[:, valid_indices]
        
        return t_result, trajectory_result


class LyapunovExponentCalculator:
    """Calculate Lyapunov exponents for chaotic systems."""
    
    def __init__(self, system: ChaoticSystem):
        self.system = system
        
        if not hasattr(system, 'get_jacobian') or system.get_jacobian(np.zeros(system.dimension)) is None:
            warnings.warn(f"System {type(system).__name__} doesn't provide Jacobian matrix. "
                         "Lyapunov calculation may be less accurate.")
    
    def calculate_largest_lyapunov(self, initial_state: np.ndarray, time_span: Tuple[float, float],
                                  dt: float = 0.01, n_iterations: int = 100) -> float:
        """
        Calculate largest Lyapunov exponent using the standard algorithm.
        
        Args:
            initial_state: Initial conditions
            time_span: Integration time span
            dt: Time step
            n_iterations: Number of renormalization iterations
            
        Returns:
            Largest Lyapunov exponent
        """
        if n_iterations < 10:
            warnings.warn("Too few iterations for reliable Lyapunov calculation")
        
        solver = ChaoticSystemSolver(self.system)
        
        # Small perturbation
        perturbation = 1e-6
        initial_perturbed = initial_state + perturbation * np.ones_like(initial_state)
        
        total_separation_log = 0.0
        current_state = initial_state.copy()
        current_perturbed = initial_perturbed.copy()
        
        renorm_time = (time_span[1] - time_span[0]) / n_iterations
        
        for i in range(n_iterations):
            t_start = time_span[0] + i * renorm_time
            t_end = t_start + renorm_time
            
            # Integrate main trajectory
            try:
                t1, traj1 = solver.solve(current_state, (t_start, t_end), dt=dt)
                t2, traj2 = solver.solve(current_perturbed, (t_start, t_end), dt=dt)
                
                # Get final states
                final_state = traj1[:, -1]
                final_perturbed = traj2[:, -1]
                
                # Calculate separation
                separation = np.linalg.norm(final_perturbed - final_state)
                
                if separation <= 0:
                    logger.warning("Zero separation detected, using fallback value")
                    separation = perturbation * 1e-6
                
                # Accumulate logarithmic separation
                total_separation_log += np.log(separation / perturbation)
                
                # Renormalize perturbation
                direction = (final_perturbed - final_state) / separation
                current_state = final_state
                current_perturbed = final_state + perturbation * direction
                
            except Exception as e:
                logger.warning(f"Integration failed at iteration {i}: {e}")
                break
        
        # Calculate average exponential growth rate
        total_time = min(i + 1, n_iterations) * renorm_time
        largest_lyapunov = total_separation_log / total_time
        
        return largest_lyapunov
    
    def calculate_spectrum(self, initial_state: np.ndarray, time_span: Tuple[float, float],
                          dt: float = 0.01, n_steps: int = 1000) -> np.ndarray:
        """
        Calculate full Lyapunov spectrum using QR decomposition.
        
        Args:
            initial_state: Initial conditions
            time_span: Integration time span
            dt: Time step
            n_steps: Number of integration steps
            
        Returns:
            Array of Lyapunov exponents (sorted descending)
        """
        if not hasattr(self.system, 'get_jacobian'):
            raise NotImplementedError("Full spectrum calculation requires Jacobian matrix")
        
        solver = ChaoticSystemSolver(self.system)
        dimension = self.system.dimension
        
        # Initialize orthonormal basis with better conditioning
        w = np.eye(dimension) + 1e-8 * np.random.randn(dimension, dimension)  # small disturbance
        lyap_sum = np.zeros(dimension)
        
        current_state = initial_state.copy()
        step_time = (time_span[1] - time_span[0]) / n_steps
        
        reorthogonalize_interval = max(1, n_steps // 100)  # reorthogonalize in every 1% step
        
        successful_steps = 0
        
        for step in range(n_steps):
            t_start = time_span[0] + step * step_time
            t_end = t_start + step_time
            
            try:
                # Integrate trajectory
                t_traj, trajectory = solver.solve(current_state, (t_start, t_end), dt=dt)
                current_state = trajectory[:, -1]
                
                # Get Jacobian at current state
                jacobian = self.system.get_jacobian(current_state)
                
                condition_number = np.linalg.cond(jacobian)
                if condition_number > 1e10:
                    logger.warning(f"High condition number ({condition_number:.2e}) at step {step}")
                    jacobian += 1e-12 * np.eye(dimension)
                
                # Evolve tangent vectors
                w = jacobian @ w
                
                if step % reorthogonalize_interval == 0 or condition_number > 1e10:
                    try:
                        q, r = np.linalg.qr(w)
                        
                        # stability
                        diagonal_elements = np.diag(r)
                        
                        min_threshold = 1e-12
                        fixed_count = 0
                        
                        for i in range(dimension):
                            if abs(diagonal_elements[i]) <= min_threshold:
                                r[i, i] = min_threshold if diagonal_elements[i] >= 0 else -min_threshold
                                fixed_count += 1
                        
                        if fixed_count > 0 and step % 100 == 0:
                            logger.debug(f"Fixed {fixed_count} small diagonal elements at step {step}")
                        
                        w = q
                        
                        for i in range(dimension):
                            log_value = np.log(max(abs(r[i, i]), min_threshold))
                            lyap_sum[i] += log_value
                            
                        successful_steps += 1
                        
                    except np.linalg.LinAlgError as e:
                        logger.warning(f"QR decomposition failed at step {step}: {e}")
                        w = np.eye(dimension) + 1e-6 * np.random.randn(dimension, dimension)
                        continue
                else:
                    successful_steps += 1
                    
            except Exception as e:
                logger.warning(f"Spectrum calculation failed at step {step}: {e}")
                continue
        
        if successful_steps == 0:
            logger.error("No successful integration steps")
            return np.zeros(dimension)
        
        # Calculate average growth rates
        total_time = successful_steps * step_time
        lyapunov_spectrum = lyap_sum / total_time
        
        # Sort in descending order
        lyapunov_spectrum = np.sort(lyapunov_spectrum)[::-1]
        
        if np.any(np.abs(lyapunov_spectrum) > 100):
            logger.warning(f"Unreasonably large Lyapunov exponents detected: {lyapunov_spectrum}")
        
        return lyapunov_spectrum


def correlation_dimension(data: np.ndarray, embedding_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate correlation dimension using Grassberger-Procaccia algorithm.
    
    Args:
        data: Time series data (1D) or trajectory data (2D)
        embedding_dim: Embedding dimension (auto-calculated if None)
        
    Returns:
        Tuple of (radii, correlations, correlation_dimension)
    """
    data = np.asarray(data)
    
    # Input validation
    if data.size == 0:
        raise ValueError("Empty data array")
    
    # Handle different input formats
    if data.ndim == 1:
        # Time series data - use embedding
        if len(data) < 100:
            raise ValueError("Insufficient data points for correlation dimension analysis")
        
        # Simple embedding for 1D data
        if embedding_dim is None:
            embedding_dim = min(5, len(data) // 20)
        
        # Create embedded vectors
        n_points = len(data) - embedding_dim + 1
        embedded_data = np.zeros((n_points, embedding_dim))
        for i in range(embedding_dim):
            embedded_data[:, i] = data[i:i + n_points]
        
        trajectory = embedded_data
    elif data.ndim == 2:
        # Trajectory data
        if data.shape[0] < 50:
            raise ValueError("Insufficient trajectory points for correlation dimension analysis")
        trajectory = data
    else:
        raise ValueError("Data must be 1D or 2D array")
    
    # Remove invalid points
    valid_mask = np.all(np.isfinite(trajectory), axis=1)
    trajectory = trajectory[valid_mask]
    
    if len(trajectory) < 10:
        raise ValueError("Too few valid data points after filtering")
    
    # Calculate pairwise distances
    try:
        distances = pdist(trajectory)
    except Exception as e:
        raise ValueError(f"Failed to calculate distances: {e}")
    
    if len(distances) == 0:
        raise ValueError("No valid distances calculated")
    
    # Define radius range
    max_dist = np.max(distances)
    min_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else max_dist * 1e-6
    
    radii = np.logspace(np.log10(min_dist), np.log10(max_dist), 50)
    
    # Calculate correlation integral
    correlations = []
    for r in radii:
        correlation_sum = np.mean(distances <= r)
        correlations.append(max(correlation_sum, 1e-10))  # Avoid log(0)
    
    correlations = np.array(correlations)
    
    # Estimate correlation dimension using linear regression
    valid_indices = correlations > 0
    if not np.any(valid_indices):
        logger.warning("No valid correlation values found")
        return radii, correlations, 0.0
    
    log_radii = np.log(radii[valid_indices])
    log_correlations = np.log(correlations[valid_indices])
    
    # Find linear region (middle portion of the curve)
    n_valid = len(log_radii)
    start_idx = n_valid // 4
    end_idx = 3 * n_valid // 4
    
    if end_idx <= start_idx:
        # Too few points, use all
        start_idx, end_idx = 0, n_valid
    
    try:
        slope, _, r_value, _, _ = linregress(
            log_radii[start_idx:end_idx], 
            log_correlations[start_idx:end_idx]
        )
        
        if r_value**2 < 0.5:
            logger.warning("Poor linear fit in correlation dimension estimation")
        
        correlation_dim = slope
    except Exception as e:
        logger.warning(f"Linear regression failed: {e}")
        correlation_dim = 0.0
    
    return radii, correlations, correlation_dim


def largest_lyapunov_from_data(data: np.ndarray, dt: float = 1.0, tau: int = 1, 
                              min_neighbors: int = 10) -> float:
    """
    Estimate largest Lyapunov exponent from time series data.
    
    Args:
        data: Time series data
        dt: Sampling time step
        tau: Time delay for embedding
        min_neighbors: Minimum number of neighbors
        
    Returns:
        Estimated largest Lyapunov exponent
    """
    data = np.asarray(data)
    
    if len(data) < 100:
        logger.warning("Insufficient data for Lyapunov estimation")
        return np.nan
    
    # Remove invalid values
    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    
    if len(data) < 50:
        return np.nan
    
    # Normalize data
    data = (data - np.mean(data)) / np.std(data)
    
    try:
        # Simple estimation using local divergence
        embedding_dim = min(5, len(data) // 20)
        
        # Create phase space reconstruction
        n_points = len(data) - (embedding_dim - 1) * tau
        embedded = np.zeros((n_points, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = data[i * tau:i * tau + n_points]
        
        # Find nearest neighbors and track divergence
        nbrs = NearestNeighbors(n_neighbors=min_neighbors + 1, metric='euclidean')
        nbrs.fit(embedded[:-1])  # Exclude last point to have evolution
        
        divergences = []
        evolution_times = []
        
        for i in range(min(len(embedded) - 10, 500)):  # Limit for performance
            distances, indices = nbrs.kneighbors([embedded[i]], return_distance=True)
            
            # Skip self-match
            neighbor_indices = indices[0][1:]
            neighbor_distances = distances[0][1:]
            
            # Track evolution of nearest neighbors
            for j, neighbor_idx in enumerate(neighbor_indices):
                if neighbor_idx < len(embedded) - 1:
                    # Calculate initial and evolved distances
                    initial_dist = neighbor_distances[j]
                    
                    if initial_dist > 0:
                        evolved_dist = np.linalg.norm(embedded[i + 1] - embedded[neighbor_idx + 1])
                        
                        if evolved_dist > 0:
                            divergence = np.log(evolved_dist / initial_dist)
                            divergences.append(divergence)
                            evolution_times.append(dt)
        
        if len(divergences) < 10:
            logger.warning("Too few valid divergences calculated")
            return np.nan
        
        # Estimate Lyapunov exponent
        lyapunov_estimate = np.mean(divergences) / np.mean(evolution_times)
        
        return lyapunov_estimate
        
    except Exception as e:
        logger.warning(f"Lyapunov estimation failed: {e}")
        return np.nan


# def hurst_exponent(data: np.ndarray, max_lag: Optional[int] = None) -> float:
#     """
#     Calculate Hurst exponent using Detrended Fluctuation Analysis.
    
#     Args:
#         data: Time series data
#         max_lag: Maximum lag for analysis
        
#     Returns:
#         Hurst exponent (0.5 for random, >0.5 for persistent, <0.5 for anti-persistent)
#     """
#     data = np.asarray(data)
    
#     if len(data) < 20:
#         logger.warning("Insufficient data for Hurst exponent calculation")
#         return 0.5  # Default value
    
#     # Remove invalid values
#     valid_mask = np.isfinite(data)
#     data = data[valid_mask]
    
#     if len(data) < 10:
#         return 0.5
    
#     # Remove linear trend
#     try:
#         data = detrend(data)
#     except:
#         data = data - np.mean(data)
    
#     n = len(data)
#     if max_lag is None:
#         max_lag = min(n // 4, 100)
    
#     # Create profile (cumulative sum)
#     profile = np.cumsum(data - np.mean(data))
    
#     # Calculate fluctuations at different scales
#     lags = np.unique(np.logspace(1, np.log10(max_lag), 15).astype(int))
#     lags = lags[lags < len(profile)]
    
#     if len(lags) < 3:
#         return 0.5
    
#     fluctuations = []
    
#     for lag in lags:
#         # Number of complete windows
#         n_windows = len(profile) // lag
        
#         if n_windows < 2:
#             continue
        
#         # Divide into non-overlapping windows
#         windowed_profile = profile[:n_windows * lag].reshape(n_windows, lag)
        
#         # Calculate fluctuation for each window
#         window_fluctuations = []
#         for window in windowed_profile:
#             # Linear detrending within window
#             try:
#                 coeffs = np.polyfit(range(lag), window, 1)
#                 trend = np.polyval(coeffs, range(lag))
#                 detrended = window - trend
#                 fluctuation = np.sqrt(np.mean(detrended**2))
                
#                 if np.isfinite(fluctuation) and fluctuation > 0:
#                     window_fluctuations.append(fluctuation)
#             except:
#                 continue
        
#         if window_fluctuations:
#             fluctuations.append(np.mean(window_fluctuations))
    
#     if len(fluctuations) < 3:
#         return 0.5
    
#     # Linear regression in log-log space
#     try:
#         valid_lags = lags[:len(fluctuations)]
#         log_lags = np.log(valid_lags)
#         log_fluctuations = np.log(fluctuations)
        
#         slope, _, r_value, _, _ = linregress(log_lags, log_fluctuations)
        
#         # Hurst exponent should be between 0 and 1
#         hurst = np.clip(slope, 0.0, 1.0)
        
#         if r_value**2 < 0.5:
#             logger.warning("Poor linear fit in Hurst exponent calculation")
        
#         return hurst
        
#     except Exception as e:
#         logger.warning(f"Hurst exponent calculation failed: {e}")
#         return 0.5


def is_chaotic(lyapunov_spectrum: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Determine if system is chaotic based on Lyapunov spectrum.
    
    Args:
        lyapunov_spectrum: Array of Lyapunov exponents
        tolerance: Tolerance for considering exponent as positive
        
    Returns:
        True if system is chaotic (at least one positive Lyapunov exponent)
    """
    if len(lyapunov_spectrum) == 0:
        return False
    
    # System is chaotic if it has at least one positive Lyapunov exponent
    return np.any(lyapunov_spectrum > tolerance)


def attractor_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """
    Calculate Kaplan-Yorke dimension from Lyapunov spectrum.
    
    Args:
        lyapunov_spectrum: Array of Lyapunov exponents (sorted descending)
        
    Returns:
        Kaplan-Yorke dimension
    """
    spectrum = np.sort(lyapunov_spectrum)[::-1]  # Ensure descending order
    
    # Find the largest k such that sum of first k exponents is positive
    cumsum = np.cumsum(spectrum)
    positive_indices = np.where(cumsum > 0)[0]
    
    if len(positive_indices) == 0:
        return 1.0  # No positive sum, dimension is 1
    
    k = positive_indices[-1]  # Largest index with positive cumsum
    
    if k + 1 >= len(spectrum):
        return float(len(spectrum))  # All exponents sum to positive
    
    # Kaplan-Yorke dimension
    dimension = k + 1 + cumsum[k] / abs(spectrum[k + 1])
    
    return dimension


def kolmogorov_entropy(lyapunov_spectrum: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Sinai entropy from Lyapunov spectrum.
    
    Args:
        lyapunov_spectrum: Array of Lyapunov exponents
        
    Returns:
        Kolmogorov-Sinai entropy (sum of positive Lyapunov exponents)
    """
    positive_exponents = lyapunov_spectrum[lyapunov_spectrum > 0]
    return np.sum(positive_exponents)


def create_chaotic_system(system_type: str, **kwargs) -> ChaoticSystem:
    """
    Factory function to create chaotic systems.
    
    Args:
        system_type: Type of system ('lorenz', 'rossler', 'chen')
        **kwargs: System-specific parameters
        
    Returns:
        Chaotic system instance
    """
    system_type = system_type.lower()
    
    if system_type == 'lorenz':
        return LorenzSystem(**kwargs)
    elif system_type == 'rossler':
        return RosslerSystem(**kwargs)
    elif system_type == 'chen':
        return ChenSystem(**kwargs)
    else:
        raise ValueError(f"Unknown chaotic system type: {system_type}")

def validate_chaotic_parameters(
    system_type: str,
    evolution_time: Optional[float] = None,
    time_step: Optional[float] = None,
    coupling_strength: Optional[float] = None,
    noise_level: Optional[float] = None,
    **kwargs
) -> Tuple[bool, str]:
    """
    Validate parameters for chaotic system integration.
    
    Args:
        system_type: Type of chaotic system ('lorenz', 'rossler', 'chen', etc.)
        evolution_time: Total evolution time for integration
        time_step: Integration time step
        coupling_strength: Coupling strength parameter
        noise_level: Noise level parameter
        **kwargs: Additional system-specific parameters
        
    Returns:
        Tuple of (is_valid, message)
    """
    system_type = system_type.lower()
    
    # Validate system type
    valid_systems = ['lorenz', 'rossler', 'chen', 'mackey_glass', 'chua']
    if system_type not in valid_systems:
        return False, f"Invalid system type '{system_type}'. Must be one of: {valid_systems}"
    
    # Validate evolution time
    if evolution_time is not None:
        if evolution_time <= 0:
            return False, f"evolution_time must be positive, got {evolution_time}"
        if evolution_time > 100:
            return False, f"evolution_time too large ({evolution_time}), may cause numerical issues"
    
    # Validate time step
    if time_step is not None:
        if time_step <= 0:
            return False, f"time_step must be positive, got {time_step}"
        if time_step > 1.0:
            return False, f"time_step too large ({time_step}), may cause integration errors"
        if evolution_time is not None and time_step > evolution_time / 10:
            return False, f"time_step ({time_step}) should be much smaller than evolution_time ({evolution_time})"
    
    # Validate coupling strength
    if coupling_strength is not None:
        if coupling_strength < 0:
            return False, f"coupling_strength must be non-negative, got {coupling_strength}"
        if coupling_strength > 10:
            return False, f"coupling_strength too large ({coupling_strength}), may cause instability"
    
    # Validate noise level
    if noise_level is not None:
        if noise_level < 0:
            return False, f"noise_level must be non-negative, got {noise_level}"
        if noise_level > 0.1:
            return False, f"noise_level too large ({noise_level}), may dominate chaotic dynamics"
    
    # System-specific validation
    if system_type == 'lorenz':
        # Lorenz system is generally stable with default parameters
        pass
    
    elif system_type == 'rossler':
        # Rossler system may need smaller time steps
        if time_step is not None and time_step > 0.01:
            return False, "Rossler system requires smaller time_step (≤0.01) for numerical stability"
    
    elif system_type == 'mackey_glass':
        # Mackey-Glass needs specific delay parameter
        delay_tau = kwargs.get('delay_tau', 17)
        if delay_tau <= 0:
            return False, f"Mackey-Glass delay_tau must be positive, got {delay_tau}"
    
    elif system_type == 'chua':
        # Chua circuit can be sensitive to parameters
        if noise_level is not None and noise_level > 0.001:
            return False, "Chua system requires very small noise_level (≤0.001)"
    
    return True, "Parameters valid"


def optimize_chaotic_parameters(
    system_type: str,
    evolution_time: Optional[float] = None,
    time_step: Optional[float] = None,
    coupling_strength: Optional[float] = None,
    noise_level: Optional[float] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Optimize parameters for chaotic system to ensure numerical stability.
    
    Args:
        system_type: Type of chaotic system
        evolution_time: Evolution time (will be adjusted if needed)
        time_step: Time step (will be adjusted if needed)
        coupling_strength: Coupling strength (will be adjusted if needed)
        noise_level: Noise level (will be adjusted if needed)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of optimized parameters
    """
    system_type = system_type.lower()
    optimized = {}
    
    # System-specific optimal ranges
    optimal_ranges = {
        'lorenz': {
            'evolution_time': (0.1, 2.0),
            'time_step': (0.001, 0.05),
            'coupling_strength': (0.5, 2.0),
            'noise_level': (0.0001, 0.01)
        },
        'rossler': {
            'evolution_time': (0.5, 3.0),
            'time_step': (0.001, 0.01),
            'coupling_strength': (0.3, 1.5),
            'noise_level': (0.0001, 0.005)
        },
        'chen': {
            'evolution_time': (0.1, 2.0),
            'time_step': (0.001, 0.05),
            'coupling_strength': (0.5, 2.0),
            'noise_level': (0.0001, 0.01)
        },
        'mackey_glass': {
            'evolution_time': (0.5, 5.0),
            'time_step': (0.01, 0.1),
            'coupling_strength': (0.1, 1.0),
            'noise_level': (0.0001, 0.005)
        },
        'chua': {
            'evolution_time': (0.1, 1.0),
            'time_step': (0.001, 0.01),
            'coupling_strength': (0.5, 2.0),
            'noise_level': (0.00001, 0.001)
        }
    }
    
    # Get optimal ranges for this system
    if system_type not in optimal_ranges:
        logger.warning(f"Unknown system type '{system_type}', using default ranges")
        ranges = optimal_ranges['lorenz']
    else:
        ranges = optimal_ranges[system_type]
    
    # Optimize evolution_time
    if evolution_time is not None:
        min_time, max_time = ranges['evolution_time']
        if evolution_time < min_time:
            optimized['evolution_time'] = min_time
            logger.info(f"Adjusted evolution_time from {evolution_time} to {min_time}")
        elif evolution_time > max_time:
            optimized['evolution_time'] = max_time
            logger.info(f"Adjusted evolution_time from {evolution_time} to {max_time}")
        else:
            optimized['evolution_time'] = evolution_time
    
    # Optimize time_step
    if time_step is not None:
        min_step, max_step = ranges['time_step']
        if time_step < min_step:
            optimized['time_step'] = min_step
            logger.info(f"Adjusted time_step from {time_step} to {min_step}")
        elif time_step > max_step:
            optimized['time_step'] = max_step
            logger.info(f"Adjusted time_step from {time_step} to {max_step}")
        else:
            optimized['time_step'] = time_step
        
        # Ensure time_step is reasonable relative to evolution_time
        if evolution_time is not None:
            max_allowed_step = evolution_time / 10
            if optimized['time_step'] > max_allowed_step:
                optimized['time_step'] = max_allowed_step
                logger.info(f"Adjusted time_step to {max_allowed_step} (evolution_time/10)")
    
    # Optimize coupling_strength
    if coupling_strength is not None:
        min_coupling, max_coupling = ranges['coupling_strength']
        if coupling_strength < min_coupling:
            optimized['coupling_strength'] = min_coupling
            logger.info(f"Adjusted coupling_strength from {coupling_strength} to {min_coupling}")
        elif coupling_strength > max_coupling:
            optimized['coupling_strength'] = max_coupling
            logger.info(f"Adjusted coupling_strength from {coupling_strength} to {max_coupling}")
        else:
            optimized['coupling_strength'] = coupling_strength
    
    # Optimize noise_level
    if noise_level is not None:
        min_noise, max_noise = ranges['noise_level']
        if noise_level < min_noise:
            optimized['noise_level'] = min_noise
            logger.info(f"Adjusted noise_level from {noise_level} to {min_noise}")
        elif noise_level > max_noise:
            optimized['noise_level'] = max_noise
            logger.info(f"Adjusted noise_level from {noise_level} to {max_noise}")
        else:
            optimized['noise_level'] = noise_level
    
    # Include any additional parameters
    for key, value in kwargs.items():
        if key not in optimized:
            optimized[key] = value
    
    return optimized


def improved_hurst_exponent(data: np.ndarray, max_lag: Optional[int] = None, 
                          min_lag: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Improved Hurst exponent calculation with better linear fitting.
    
    Args:
        data: Time series data
        max_lag: Maximum lag for analysis
        min_lag: Minimum lag for analysis
        
    Returns:
        Tuple of (hurst_exponent, diagnostics_dict)
    """
    data = np.asarray(data)
    
    if len(data) < 50:
        logger.warning("Insufficient data for Hurst exponent calculation")
        return 0.5, {"error": "insufficient_data", "n_points": len(data)}
    
    # Remove invalid values and detrend
    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    
    if len(data) < 30:
        return 0.5, {"error": "too_few_valid_points", "n_points": len(data)}
    
    # Remove linear and quadratic trends for better stability
    try:
        # Try quadratic detrending first
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 2)
        trend = np.polyval(coeffs, x)
        data_detrended = data - trend
    except:
        try:
            # Fallback to linear detrending
            data_detrended = detrend(data)
        except:
            # Final fallback: remove mean only
            data_detrended = data - np.mean(data)
    
    # Create profile (cumulative sum of detrended data)
    profile = np.cumsum(data_detrended)
    n = len(profile)
    
    # Set lag ranges adaptively
    if min_lag is None:
        min_lag = max(4, n // 50)  # At least 4 points, but reasonable minimum
    if max_lag is None:
        max_lag = min(n // 4, 100)  # At most 1/4 of data length
    
    if max_lag <= min_lag:
        max_lag = min_lag + 10
    
    # Generate logarithmic spaced lags for better coverage
    lags = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), 20).astype(int)
    )
    lags = lags[(lags >= min_lag) & (lags <= max_lag)]
    
    if len(lags) < 4:
        return 0.5, {"error": "insufficient_lags", "n_lags": len(lags)}
    
    fluctuations = []
    valid_lags = []
    
    for lag in lags:
        n_windows = len(profile) // lag
        
        if n_windows < 2:
            continue
        
        window_fluctuations = []
        windowed_profile = profile[:n_windows * lag].reshape(n_windows, lag)
        
        for window in windowed_profile:
            try:
                # Use linear detrending within each window
                window_x = np.arange(lag)
                coeffs = np.polyfit(window_x, window, 1)
                trend = np.polyval(coeffs, window_x)
                detrended_window = window - trend
                
                fluctuation = np.sqrt(np.mean(detrended_window**2))
                
                if np.isfinite(fluctuation) and fluctuation > 1e-12:
                    window_fluctuations.append(fluctuation)
            except Exception as e:
                continue
        
        if len(window_fluctuations) >= 2:  # Require at least 2 valid windows
            avg_fluctuation = np.mean(window_fluctuations)
            fluctuations.append(avg_fluctuation)
            valid_lags.append(lag)
    
    if len(fluctuations) < 4:
        return 0.5, {"error": "insufficient_fluctuations", "n_fluctuations": len(fluctuations)}
    
    # Convert to log space for linear regression
    log_lags = np.log(valid_lags)
    log_fluctuations = np.log(fluctuations)
    
    # Find the most linear region using R² optimization
    best_r2 = -np.inf
    best_slope = 0.5
    best_intercept = 0
    best_indices = (0, len(log_lags))
    
    n_points = len(log_lags)
    
    # Try different segments to find the best linear region
    for start in range(0, n_points - 3):
        for end in range(start + 4, n_points + 1):
            segment_lags = log_lags[start:end]
            segment_flucts = log_fluctuations[start:end]
            
            try:
                slope, intercept, r_value, p_value, std_err = linregress(
                    segment_lags, segment_flucts
                )
                
                r2 = r_value ** 2
                
                if r2 > best_r2 and abs(slope) <= 2.0:  # Reasonable slope constraint
                    best_r2 = r2
                    best_slope = slope
                    best_intercept = intercept
                    best_indices = (start, end)
            except:
                continue
    
    hurst = np.clip(best_slope, 0.0, 1.0)
    
    diagnostics = {
        "r_squared": best_r2,
        "n_points_used": best_indices[1] - best_indices[0],
        "total_lags": len(valid_lags),
        "fit_quality": "good" if best_r2 > 0.9 else "fair" if best_r2 > 0.7 else "poor",
        "valid_data_points": len(data)
    }
    
    if best_r2 < 0.5:
        logger.warning(f"Poor Hurst exponent fit (R²={best_r2:.3f}). Data may not exhibit long-range correlation.")
    
    return hurst, diagnostics


def rs_hurst_estimate(data: np.ndarray) -> float:
    """
    Alternative Hurst exponent estimation using R/S method.
    """
    data = np.asarray(data)
    n = len(data)
    
    # Calculate rescaled range for different segment sizes
    segment_sizes = np.unique(np.logspace(np.log10(10), np.log10(n//4), 15).astype(int))
    segment_sizes = segment_sizes[segment_sizes <= n//4]
    
    rs_ratios = []
    
    for size in segment_sizes:
        n_segments = n // size
        if n_segments < 2:
            continue
            
        segment_rs = []
        
        for i in range(n_segments):
            segment = data[i*size:(i+1)*size]
            if len(segment) < 2:
                continue
                
            # Calculate mean and cumulative deviations
            mean_val = np.mean(segment)
            deviations = segment - mean_val
            cumulative_deviations = np.cumsum(deviations)
            
            # Range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            # Standard deviation
            S = np.std(segment)
            
            if S > 1e-12:
                rs_ratio = R / S
                if rs_ratio > 0:
                    segment_rs.append(rs_ratio)
        
        if segment_rs:
            rs_ratios.append(np.mean(segment_rs))
        else:
            rs_ratios.append(np.nan)
    
    # Remove NaN values
    valid_indices = ~np.isnan(rs_ratios)
    if np.sum(valid_indices) < 3:
        return 0.5
    
    valid_sizes = segment_sizes[valid_indices]
    valid_rs = np.array(rs_ratios)[valid_indices]
    
    # Linear regression in log-log space
    try:
        log_sizes = np.log(valid_sizes)
        log_rs = np.log(valid_rs)
        
        slope, _, r_value, _, _ = linregress(log_sizes, log_rs)
        hurst = slope
        
        if r_value**2 < 0.5:
            logger.warning("Poor R/S method linear fit")
            
        return np.clip(hurst, 0.0, 1.0)
    except:
        return 0.5


def robust_hurst_exponent(data: np.ndarray, method: str = 'auto') -> float:
    """
    Robust Hurst exponent calculation with multiple fallback methods.
    
    Args:
        data: Time series data
        method: Calculation method ('auto', 'dfa', 'rs')
        
    Returns:
        Hurst exponent estimate
    """
    data = np.asarray(data)
    
    if len(data) < 40:
        return 0.5
    
    # Try the improved DFA method first
    try:
        hurst, diagnostics = improved_hurst_exponent(data)
        
        if diagnostics.get("r_squared", 0) > 0.6:  # Acceptable fit
            return hurst
        else:
            logger.info(f"Poor DFA fit (R²={diagnostics.get('r_squared', 0):.3f}), trying alternative methods")
    except Exception as e:
        logger.warning(f"Improved Hurst calculation failed: {e}")
    
    # Fallback: Simple R/S method
    try:
        return rs_hurst_estimate(data)
    except Exception as e:
        logger.warning(f"R/S method also failed: {e}")
    
    # Final fallback
    return 0.5


# 替换原来的 hurst_exponent 函数
def hurst_exponent(data: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis.
    
    Args:
        data: Time series data
        max_lag: Maximum lag for analysis
        
    Returns:
        Hurst exponent (0.5 for random, >0.5 for persistent, <0.5 for anti-persistent)
    """
    return robust_hurst_exponent(data, method='auto')


if __name__ == "__main__":
    print(f"✓ Project Root: {PROJECT_ROOT}")
    print(f"✓ Import Manager: {USING_IMPORT_MANAGER}")
    print(f"✓ Module imports successful")
    
    print("Testing Chaos Utils...")
    
    lorenz = LorenzSystem()
    solver = ChaoticSystemSolver(lorenz)
    initial_state = np.array([1.0, 1.0, 1.0])
    
    time_span = (0.0, 10.0)
    dt = 0.01
    
    print(f"Integration setup: time_span={time_span}, dt={dt}")
    print(f"Expected points: ~{int((time_span[1] - time_span[0]) / dt)}")
    
    try:
        t, trajectory = solver.solve(initial_state, time_span, dt=dt, method='rk45')
        
        print(f"Generated trajectory with {trajectory.shape[1]} points")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Time range: [{t[0]:.3f}, {t[-1]:.3f}]")
        print(f"Final state: {trajectory[:, -1]}")
        
        if trajectory.shape[1] < 100:
            print(f"Warning: Only {trajectory.shape[1]} points generated, need at least 100 for reliable analysis")
            
            print("Retrying with longer integration time...")
            time_span_long = (0.0, 20.0)
            dt_small = 0.005
            t, trajectory = solver.solve(initial_state, time_span_long, dt=dt_small, method='rk45')
            print(f"Retry: Generated trajectory with {trajectory.shape[1]} points")
        
        if trajectory.shape[1] >= 100:
            lyap_calc = LyapunovExponentCalculator(lorenz)
            largest_lyap = lyap_calc.calculate_largest_lyapunov(
                initial_state, time_span, dt=0.01, n_iterations=20
            )
            print(f"Largest Lyapunov exponent: {largest_lyap:.6f}")
            
            try:
                trajectory_for_corr = trajectory.T
                print(f"Trajectory for correlation dim: {trajectory_for_corr.shape}")
                
                if trajectory_for_corr.shape[0] >= 50:
                    radii, corrs, dim = correlation_dimension(trajectory_for_corr)
                    print(f"Correlation dimension: {dim:.3f}")
                else:
                    print(f"Skipping correlation dimension: need ≥50 points, have {trajectory_for_corr.shape[0]}")
                    
            except Exception as e:
                print(f"Correlation dimension calculation failed: {e}")
                
            try:
                # 使用改进的 Hurst 指数计算
                hurst, diagnostics = improved_hurst_exponent(trajectory[0, :])
                print(f"Hurst exponent: {hurst:.3f} (R²={diagnostics.get('r_squared', 0):.3f})")
                if diagnostics.get('r_squared', 0) < 0.7:
                    print(f"  Note: Fit quality is {diagnostics.get('fit_quality', 'unknown')}")
            except Exception as e:
                print(f"Hurst exponent calculation failed: {e}")
                # 备选方法
                hurst = robust_hurst_exponent(trajectory[0, :])
                print(f"Robust Hurst exponent: {hurst:.3f}")
        
        else:
            print("Insufficient data for chaos analysis")
            
    except Exception as e:
        print(f"Integration failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        print("\nDebugging integration issue...")
        try:
            simple_span = (0.0, 1.0)
            simple_dt = 0.1
            t_simple, traj_simple = solver.solve(initial_state, simple_span, dt=simple_dt)
            print(f"Simple integration successful: {traj_simple.shape[1]} points")
        except Exception as e2:
            print(f"Even simple integration failed: {e2}")
    
    print("Chaos utils testing completed.")


def debug_integration_parameters():
    """debug integration parameters setting"""
    print("\n=== Integration Parameter Debug ===")
    
    time_spans = [(0.0, 1.0), (0.0, 5.0), (0.0, 10.0)]
    dts = [0.1, 0.05, 0.01]
    
    lorenz = LorenzSystem()
    solver = ChaoticSystemSolver(lorenz)
    initial_state = np.array([1.0, 1.0, 1.0])
    
    for span in time_spans:
        for dt in dts:
            try:
                expected_points = int((span[1] - span[0]) / dt) + 1
                t, trajectory = solver.solve(initial_state, span, dt=dt)
                actual_points = trajectory.shape[1]
                
                print(f"span={span}, dt={dt}: expected {expected_points}, got {actual_points}")
                
                if actual_points < expected_points * 0.9:
                    print(f"  WARNING: Significant point loss!")
                    
            except Exception as e:
                print(f"span={span}, dt={dt}: FAILED - {e}")
    
    print("=== Debug Complete ===\n")


# delete the annotate if wanna debug
debug_integration_parameters()