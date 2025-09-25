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

# AFTER (修复后的统一导入方式):
import os
import sys
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# 统一导入设置
# =============================================================================
def setup_module_imports(current_file: str = __file__):
    """Setup imports for current module.""" 
    try:
        from setup_imports import setup_project_imports
        return setup_project_imports(current_file), True
    except ImportError:
        current_dir = Path(current_file).resolve().parent  # core目录
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
# 项目模块导入 (带安全检查)
# =============================================================================
try:
    from chaos_utils import (
        largest_lyapunov_from_data, correlation_dimension, 
        hurst_exponent, kolmogorov_entropy
    )
    HAS_CHAOS_UTILS = True
except ImportError as e:
    HAS_CHAOS_UTILS = False
    warnings.warn(f"chaos_utils not available: {e}")
    # 简单fallback，不需要大量代码
    largest_lyapunov_from_data = lambda x: 0.0
    correlation_dimension = lambda x: 2.0

try:
    from phase_space_reconstruction import PhaseSpaceReconstructor, EmbeddingConfig
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
            num_points = max(int(total_time / dt) + 1, 100)
            t_eval = create_safe_time_array(t_start_extended, t_end_extended, num_points)
        else:
            integration_span = (t_start, t_end)
            total_time = t_end - t_start
            num_points = max(int(total_time / dt) + 1, 100)
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
        w = np.eye(dimension) + 1e-8 * np.random.randn(dimension, dimension)  # 添加小扰动
        lyap_sum = np.zeros(dimension)
        
        current_state = initial_state.copy()
        step_time = (time_span[1] - time_span[0]) / n_steps
        
        # 更频繁的重正交化
        reorthogonalize_interval = max(1, n_steps // 100)  # 每1%的步数重正交化一次
        
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
                
                # 检查雅可比矩阵的条件数
                condition_number = np.linalg.cond(jacobian)
                if condition_number > 1e10:
                    logger.warning(f"High condition number ({condition_number:.2e}) at step {step}")
                    # 添加数值正则化
                    jacobian += 1e-12 * np.eye(dimension)
                
                # Evolve tangent vectors
                w = jacobian @ w
                
                # 定期重正交化或当矩阵条件数过大时
                if step % reorthogonalize_interval == 0 or condition_number > 1e10:
                    try:
                        q, r = np.linalg.qr(w)
                        
                        # 数值稳定性处理
                        diagonal_elements = np.diag(r)
                        
                        # 检查并修复小的或负的对角元素
                        min_threshold = 1e-12
                        fixed_count = 0
                        
                        for i in range(dimension):
                            if abs(diagonal_elements[i]) <= min_threshold:
                                r[i, i] = min_threshold if diagonal_elements[i] >= 0 else -min_threshold
                                fixed_count += 1
                        
                        if fixed_count > 0 and step % 100 == 0:  # 只在每100步报告一次
                            logger.debug(f"Fixed {fixed_count} small diagonal elements at step {step}")
                        
                        w = q
                        
                        # 累加对数增长率
                        for i in range(dimension):
                            # 使用绝对值来处理可能的负值
                            log_value = np.log(max(abs(r[i, i]), min_threshold))
                            lyap_sum[i] += log_value
                            
                        successful_steps += 1
                        
                    except np.linalg.LinAlgError as e:
                        logger.warning(f"QR decomposition failed at step {step}: {e}")
                        # 重新初始化正交基
                        w = np.eye(dimension) + 1e-6 * np.random.randn(dimension, dimension)
                        continue
                else:
                    # 不进行QR分解的步骤，仍然计数
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
        
        # 基本合理性检查
        if np.any(np.abs(lyapunov_spectrum) > 100):
            logger.warning(f"Unreasonably large Lyapunov exponents detected: {lyapunov_spectrum}")
            # 可以选择返回更保守的估计或重新计算
        
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


def hurst_exponent(data: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis.
    
    Args:
        data: Time series data
        max_lag: Maximum lag for analysis
        
    Returns:
        Hurst exponent (0.5 for random, >0.5 for persistent, <0.5 for anti-persistent)
    """
    data = np.asarray(data)
    
    if len(data) < 20:
        logger.warning("Insufficient data for Hurst exponent calculation")
        return 0.5  # Default value
    
    # Remove invalid values
    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    
    if len(data) < 10:
        return 0.5
    
    # Remove linear trend
    try:
        data = detrend(data)
    except:
        data = data - np.mean(data)
    
    n = len(data)
    if max_lag is None:
        max_lag = min(n // 4, 100)
    
    # Create profile (cumulative sum)
    profile = np.cumsum(data - np.mean(data))
    
    # Calculate fluctuations at different scales
    lags = np.unique(np.logspace(1, np.log10(max_lag), 15).astype(int))
    lags = lags[lags < len(profile)]
    
    if len(lags) < 3:
        return 0.5
    
    fluctuations = []
    
    for lag in lags:
        # Number of complete windows
        n_windows = len(profile) // lag
        
        if n_windows < 2:
            continue
        
        # Divide into non-overlapping windows
        windowed_profile = profile[:n_windows * lag].reshape(n_windows, lag)
        
        # Calculate fluctuation for each window
        window_fluctuations = []
        for window in windowed_profile:
            # Linear detrending within window
            try:
                coeffs = np.polyfit(range(lag), window, 1)
                trend = np.polyval(coeffs, range(lag))
                detrended = window - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                
                if np.isfinite(fluctuation) and fluctuation > 0:
                    window_fluctuations.append(fluctuation)
            except:
                continue
        
        if window_fluctuations:
            fluctuations.append(np.mean(window_fluctuations))
    
    if len(fluctuations) < 3:
        return 0.5
    
    # Linear regression in log-log space
    try:
        valid_lags = lags[:len(fluctuations)]
        log_lags = np.log(valid_lags)
        log_fluctuations = np.log(fluctuations)
        
        slope, _, r_value, _, _ = linregress(log_lags, log_fluctuations)
        
        # Hurst exponent should be between 0 and 1
        hurst = np.clip(slope, 0.0, 1.0)
        
        if r_value**2 < 0.5:
            logger.warning("Poor linear fit in Hurst exponent calculation")
        
        return hurst
        
    except Exception as e:
        logger.warning(f"Hurst exponent calculation failed: {e}")
        return 0.5


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


if __name__ == "__main__":
    print(f"✓ Project Root: {PROJECT_ROOT}")
    print(f"✓ Import Manager: {USING_IMPORT_MANAGER}")
    print(f"✓ Module imports successful")
    
    print("Testing Chaos Utils...")
    
    # 创建Lorenz系统并求解
    lorenz = LorenzSystem()
    solver = ChaoticSystemSolver(lorenz)
    initial_state = np.array([1.0, 1.0, 1.0])
    
    # 修复：使用合适的参数确保生成足够的点
    time_span = (0.0, 10.0)  # 10秒积分时间
    dt = 0.01  # 0.01秒时间步长
    
    print(f"Integration setup: time_span={time_span}, dt={dt}")
    print(f"Expected points: ~{int((time_span[1] - time_span[0]) / dt)}")
    
    try:
        t, trajectory = solver.solve(initial_state, time_span, dt=dt, method='rk45')
        
        print(f"Generated trajectory with {trajectory.shape[1]} points")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Time range: [{t[0]:.3f}, {t[-1]:.3f}]")
        print(f"Final state: {trajectory[:, -1]}")
        
        # 检查轨迹是否足够长
        if trajectory.shape[1] < 100:
            print(f"Warning: Only {trajectory.shape[1]} points generated, need at least 100 for reliable analysis")
            
            # 尝试用更长的积分时间
            print("Retrying with longer integration time...")
            time_span_long = (0.0, 20.0)
            dt_small = 0.005
            t, trajectory = solver.solve(initial_state, time_span_long, dt=dt_small, method='rk45')
            print(f"Retry: Generated trajectory with {trajectory.shape[1]} points")
        
        # 计算最大李雅普诺夫指数
        if trajectory.shape[1] >= 100:
            lyap_calc = LyapunovExponentCalculator(lorenz)
            largest_lyap = lyap_calc.calculate_largest_lyapunov(
                initial_state, time_span, dt=0.01, n_iterations=20
            )
            print(f"Largest Lyapunov exponent: {largest_lyap:.6f}")
            
            # 相关维数计算 - 使用转置的轨迹数据
            try:
                trajectory_for_corr = trajectory.T  # 转置：时间×维度
                print(f"Trajectory for correlation dim: {trajectory_for_corr.shape}")
                
                if trajectory_for_corr.shape[0] >= 50:
                    radii, corrs, dim = correlation_dimension(trajectory_for_corr)
                    print(f"Correlation dimension: {dim:.3f}")
                else:
                    print(f"Skipping correlation dimension: need ≥50 points, have {trajectory_for_corr.shape[0]}")
                    
            except Exception as e:
                print(f"Correlation dimension calculation failed: {e}")
                
            # 测试Hurst指数
            try:
                hurst = hurst_exponent(trajectory[0, :])  # 使用x坐标时间序列
                print(f"Hurst exponent: {hurst:.3f}")
            except Exception as e:
                print(f"Hurst exponent calculation failed: {e}")
        
        else:
            print("Insufficient data for chaos analysis")
            
    except Exception as e:
        print(f"Integration failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # 调试信息
        print("\nDebugging integration issue...")
        try:
            # 尝试更简单的参数
            simple_span = (0.0, 1.0)
            simple_dt = 0.1
            t_simple, traj_simple = solver.solve(initial_state, simple_span, dt=simple_dt)
            print(f"Simple integration successful: {traj_simple.shape[1]} points")
        except Exception as e2:
            print(f"Even simple integration failed: {e2}")
    
    print("Chaos utils testing completed.")


# 额外的调试函数
def debug_integration_parameters():
    """调试积分参数设置"""
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


# 如果你想运行调试，取消下面的注释
debug_integration_parameters()