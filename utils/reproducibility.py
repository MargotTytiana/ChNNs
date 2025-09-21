"""
Reproducibility Management System for Chaotic Speaker Recognition Project
Ensures deterministic behavior across different runs and environments
by managing random seeds and computational settings.
"""

import os
import sys
import json
import random
import hashlib
import platform
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Import ML/Scientific libraries with error handling
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Audio processing libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False



import os
import sys
import random
import platform
import subprocess
from typing import Dict, Any, Optional
import numpy as np

def set_seed(seed: int = 42):
    """
    设置所有随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 如果有 PyTorch，设置 PyTorch 种子
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # 如果有 TensorFlow，设置 TensorFlow 种子
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # 设置 Python 的哈希种子（需要在程序启动时设置）
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息用于实验记录
    
    Returns:
        包含系统信息的字典
    """
    system_info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_executable': sys.executable,
    }
    
    # CPU 信息
    try:
        system_info['cpu_count'] = os.cpu_count()
    except:
        system_info['cpu_count'] = 'unknown'
    
    # 内存信息
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info['total_memory'] = memory.total
        system_info['available_memory'] = memory.available
    except ImportError:
        system_info['total_memory'] = 'unknown'
        system_info['available_memory'] = 'unknown'
    
    # GPU 信息
    system_info['gpu_info'] = get_gpu_info()
    
    # Python 包版本
    system_info['package_versions'] = get_package_versions()
    
    # Git 信息（如果在 git 仓库中）
    system_info['git_info'] = get_git_info()
    
    return system_info


def get_gpu_info() -> Dict[str, Any]:
    """
    获取 GPU 信息
    
    Returns:
        GPU 信息字典
    """
    gpu_info = {
        'cuda_available': False,
        'cuda_version': 'unknown',
        'gpu_count': 0,
        'gpu_names': []
    }
    
    # PyTorch CUDA 信息
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    
    # 尝试使用 nvidia-smi 获取信息
    if not gpu_info['cuda_available']:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_names = result.stdout.strip().split('\n')
                gpu_info['gpu_count'] = len(gpu_names)
                gpu_info['gpu_names'] = gpu_names
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return gpu_info


def get_package_versions() -> Dict[str, str]:
    """
    获取重要包的版本信息
    
    Returns:
        包版本字典
    """
    packages = [
        'numpy', 'scipy', 'scikit-learn', 'librosa', 'soundfile',
        'torch', 'tensorflow', 'matplotlib', 'pandas'
    ]
    
    versions = {}
    for package in packages:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                versions[package] = module.__version__
            else:
                versions[package] = 'unknown'
        except ImportError:
            versions[package] = 'not installed'
    
    return versions


def get_git_info() -> Dict[str, str]:
    """
    获取 Git 仓库信息
    
    Returns:
        Git 信息字典
    """
    git_info = {
        'commit_hash': 'unknown',
        'branch': 'unknown',
        'is_dirty': 'unknown'
    }
    
    try:
        # 获取当前提交哈希
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
        
        # 获取当前分支
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # 检查是否有未提交的更改
        result = subprocess.run(['git', 'diff', '--quiet'], 
                              capture_output=True, timeout=5)
        git_info['is_dirty'] = result.returncode != 0
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return git_info


def create_reproducible_environment(seed: int = 42, 
                                  disable_cuda_benchmark: bool = True) -> Dict[str, Any]:
    """
    创建可重复的实验环境
    
    Args:
        seed: 随机种子
        disable_cuda_benchmark: 是否禁用 CUDA benchmark 以确保确定性
        
    Returns:
        环境配置信息
    """
    # 设置种子
    set_seed(seed)
    
    # 获取系统信息
    system_info = get_system_info()
    
    # PyTorch 特定设置
    try:
        import torch
        if disable_cuda_benchmark:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        # 设置线程数以确保确定性
        torch.set_num_threads(1)
        
    except ImportError:
        pass
    
    # TensorFlow 特定设置
    try:
        import tensorflow as tf
        # 设置确定性操作
        if hasattr(tf.config.experimental, 'enable_op_determinism'):
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass
    
    environment_config = {
        'seed': seed,
        'system_info': system_info,
        'reproducibility_settings': {
            'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not_set'),
            'cuda_deterministic': disable_cuda_benchmark
        }
    }
    
    return environment_config


def save_environment_info(output_path: str, seed: int = 42):
    """
    保存环境信息到文件
    
    Args:
        output_path: 输出文件路径
        seed: 随机种子
    """
    import json
    from pathlib import Path
    
    env_info = create_reproducible_environment(seed)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(env_info, f, indent=2, default=str)


def verify_reproducibility(reference_file: str) -> Dict[str, bool]:
    """
    验证当前环境与参考环境的一致性
    
    Args:
        reference_file: 参考环境信息文件路径
        
    Returns:
        验证结果字典
    """
    import json
    from pathlib import Path
    
    verification_results = {
        'python_version_match': False,
        'package_versions_match': False,
        'platform_match': False,
        'overall_compatible': False
    }
    
    try:
        with open(reference_file, 'r') as f:
            reference_info = json.load(f)
        
        current_info = get_system_info()
        
        # 检查 Python 版本
        ref_python = reference_info['system_info']['python_version'].split()[0]
        cur_python = current_info['python_version'].split()[0]
        verification_results['python_version_match'] = ref_python == cur_python
        
        # 检查关键包版本
        ref_packages = reference_info['system_info']['package_versions']
        cur_packages = current_info['package_versions']
        
        critical_packages = ['numpy', 'scipy', 'torch', 'librosa']
        package_matches = []
        
        for package in critical_packages:
            if package in ref_packages and package in cur_packages:
                package_matches.append(ref_packages[package] == cur_packages[package])
        
        verification_results['package_versions_match'] = all(package_matches) if package_matches else False
        
        # 检查平台
        verification_results['platform_match'] = (
            reference_info['system_info']['system'] == current_info['system']
        )
        
        # 整体兼容性评估
        verification_results['overall_compatible'] = (
            verification_results['python_version_match'] and
            verification_results['package_versions_match']
        )
        
    except Exception as e:
        print(f"验证过程中出错: {e}")
    
    return verification_results




class ReproducibilityManager:
    """
    Comprehensive reproducibility management for machine learning experiments.
    Handles random seed setting, environment configuration, and reproducibility verification.
    """
    
    def __init__(self, seed: Optional[int] = None, 
                 strict_mode: bool = True,
                 log_file: Optional[str] = None):
        """
        Initialize reproducibility manager
        
        Args:
            seed: Random seed to use. If None, generates one based on current time
            strict_mode: Whether to enforce strict reproducibility settings
            log_file: Optional file to log reproducibility information
        """
        self.seed = seed if seed is not None else self._generate_seed()
        self.strict_mode = strict_mode
        self.log_file = log_file
        self.environment_info = {}
        self.reproducibility_log = []
        
        # Set seeds immediately upon initialization
        self.set_global_seed(self.seed)
        
        # Collect environment information
        self._collect_environment_info()
    
    def _generate_seed(self) -> int:
        """Generate a deterministic seed based on current timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        seed = int(hashlib.md5(timestamp.encode()).hexdigest()[:8], 16) % (2**31)
        return seed
    
    def _log(self, message: str, level: str = "INFO"):
        """Internal logging function"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        self.reproducibility_log.append(log_entry)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
    
    def set_global_seed(self, seed: int):
        """Set random seed for all available libraries"""
        self.seed = seed
        self._log(f"Setting global random seed to: {seed}")
        
        # Python built-in random
        random.seed(seed)
        self._log("Python random seed set")
        
        # Operating system random
        os.environ['PYTHONHASHSEED'] = str(seed)
        self._log(f"PYTHONHASHSEED set to: {seed}")
        
        # NumPy
        if HAS_NUMPY:
            np.random.seed(seed)
            self._log("NumPy random seed set")
        else:
            self._log("NumPy not available - skipping NumPy seed", "WARNING")
        
        # PyTorch
        if HAS_TORCH:
            torch.manual_seed(seed)
            torch.random.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                self._log("PyTorch CUDA seeds set")
            
            # Set additional PyTorch reproducibility settings
            if self.strict_mode:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
                self._log("PyTorch strict reproducibility mode enabled")
            
            self._log("PyTorch seeds set")
        else:
            self._log("PyTorch not available - skipping PyTorch seed", "WARNING")
        
        # TensorFlow
        if HAS_TF:
            tf.random.set_seed(seed)
            if self.strict_mode:
                os.environ['TF_DETERMINISTIC_OPS'] = '1'
                os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
                self._log("TensorFlow strict reproducibility mode enabled")
            self._log("TensorFlow seed set")
        else:
            self._log("TensorFlow not available - skipping TensorFlow seed", "WARNING")
    
    def _collect_environment_info(self):
        """Collect comprehensive environment information"""
        self.environment_info = {
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation()
            },
            "environment_variables": {
                "PYTHONHASHSEED": os.environ.get('PYTHONHASHSEED', 'not_set'),
                "CUDA_VISIBLE_DEVICES": os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set'),
                "OMP_NUM_THREADS": os.environ.get('OMP_NUM_THREADS', 'not_set'),
                "MKL_NUM_THREADS": os.environ.get('MKL_NUM_THREADS', 'not_set')
            },
            "libraries": {}
        }
        
        # Collect library versions
        if HAS_NUMPY:
            self.environment_info["libraries"]["numpy"] = np.__version__
        
        if HAS_TORCH:
            self.environment_info["libraries"]["torch"] = torch.__version__
            if torch.cuda.is_available():
                self.environment_info["libraries"]["cuda_version"] = torch.version.cuda
                self.environment_info["libraries"]["cudnn_version"] = torch.backends.cudnn.version()
        
        if HAS_TF:
            self.environment_info["libraries"]["tensorflow"] = tf.__version__
        
        if HAS_LIBROSA:
            self.environment_info["libraries"]["librosa"] = librosa.__version__
        
        if HAS_SOUNDFILE:
            self.environment_info["libraries"]["soundfile"] = sf.__version__
        
        self._log("Environment information collected")
    
    def configure_deterministic_operations(self):
        """Configure additional settings for deterministic operations"""
        self._log("Configuring deterministic operations")
        
        # Set thread counts for reproducible parallel processing
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"
        if "MKL_NUM_THREADS" not in os.environ:
            os.environ["MKL_NUM_THREADS"] = "1"
        
        # PyTorch specific settings
        if HAS_TORCH and self.strict_mode:
            # Disable multithreading for deterministic behavior
            torch.set_num_threads(1)
            
            # Additional deterministic settings
            if hasattr(torch, 'set_deterministic_debug_mode'):
                torch.set_deterministic_debug_mode(1)
            
            self._log("PyTorch deterministic operations configured")
        
        # TensorFlow specific settings
        if HAS_TF and self.strict_mode:
            # Configure TensorFlow for deterministic behavior
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            # Set additional TF environment variables
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            
            self._log("TensorFlow deterministic operations configured")
    
    def verify_reproducibility(self, test_operations: Optional[List[callable]] = None) -> bool:
        """
        Verify that operations produce consistent results
        
        Args:
            test_operations: List of functions to test for reproducibility
            
        Returns:
            bool: True if all operations are reproducible
        """
        self._log("Starting reproducibility verification")
        
        if test_operations is None:
            test_operations = self._get_default_test_operations()
        
        all_reproducible = True
        
        for i, operation in enumerate(test_operations):
            try:
                # Run operation twice with same seed
                self.set_global_seed(self.seed)
                result1 = operation()
                
                self.set_global_seed(self.seed)
                result2 = operation()
                
                # Check if results are identical
                if self._compare_results(result1, result2):
                    self._log(f"Operation {i+1} is reproducible")
                else:
                    self._log(f"Operation {i+1} is NOT reproducible", "ERROR")
                    all_reproducible = False
                    
            except Exception as e:
                self._log(f"Error testing operation {i+1}: {str(e)}", "ERROR")
                all_reproducible = False
        
        if all_reproducible:
            self._log("All operations are reproducible", "SUCCESS")
        else:
            self._log("Some operations are not reproducible", "WARNING")
        
        return all_reproducible
    
    def _get_default_test_operations(self) -> List[callable]:
        """Get default operations to test for reproducibility"""
        operations = []
        
        # Python random test
        operations.append(lambda: [random.random() for _ in range(10)])
        
        # NumPy test
        if HAS_NUMPY:
            operations.append(lambda: np.random.rand(10))
            operations.append(lambda: np.random.normal(0, 1, 10))
        
        # PyTorch test
        if HAS_TORCH:
            operations.append(lambda: torch.rand(10).numpy())
            operations.append(lambda: torch.randn(10).numpy())
        
        return operations
    
    def _compare_results(self, result1, result2) -> bool:
        """Compare two results for equality"""
        try:
            if HAS_NUMPY and isinstance(result1, np.ndarray):
                return np.array_equal(result1, result2)
            elif isinstance(result1, (list, tuple)):
                return all(abs(a - b) < 1e-10 for a, b in zip(result1, result2))
            else:
                return result1 == result2
        except:
            return str(result1) == str(result2)
    
    def save_reproducibility_state(self, filepath: str):
        """Save current reproducibility state to file"""
        state = {
            "environment_info": self.environment_info,
            "reproducibility_log": self.reproducibility_log,
            "strict_mode": self.strict_mode
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)
        
        self._log(f"Reproducibility state saved to: {filepath}")
    
    def load_reproducibility_state(self, filepath: str):
        """Load and apply reproducibility state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Apply the seed from loaded state
        loaded_seed = state["environment_info"]["seed"]
        self.set_global_seed(loaded_seed)
        
        self._log(f"Reproducibility state loaded from: {filepath}")
        self._log(f"Applied seed: {loaded_seed}")
        
        return state
    
    def create_reproducible_split(self, data_size: int, 
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15) -> Dict[str, List[int]]:
        """
        Create reproducible train/val/test splits
        
        Args:
            data_size: Total size of dataset
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            
        Returns:
            Dict containing train, val, test indices
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Create indices and shuffle with current seed
        indices = list(range(data_size))
        random.shuffle(indices)
        
        # Calculate split points
        train_end = int(data_size * train_ratio)
        val_end = train_end + int(data_size * val_ratio)
        
        splits = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:]
        }
        
        self._log(f"Created reproducible data splits:")
        self._log(f"  Train: {len(splits['train'])} samples")
        self._log(f"  Val: {len(splits['val'])} samples") 
        self._log(f"  Test: {len(splits['test'])} samples")
        
        return splits
    
    def get_environment_hash(self) -> str:
        """Generate a hash of the current environment for reproducibility tracking"""
        env_str = json.dumps(self.environment_info, sort_keys=True, default=str)
        return hashlib.md5(env_str.encode()).hexdigest()[:16]
    
    def print_summary(self):
        """Print a summary of reproducibility settings"""
        print("\n" + "="*60)
        print("REPRODUCIBILITY SUMMARY")
        print("="*60)
        print(f"Seed: {self.seed}")
        print(f"Strict Mode: {self.strict_mode}")
        print(f"Environment Hash: {self.get_environment_hash()}")
        print(f"Python Version: {self.environment_info['platform']['python_version']}")
        
        print("\nAvailable Libraries:")
        for lib, version in self.environment_info["libraries"].items():
            print(f"  {lib}: {version}")
        
        print("\nEnvironment Variables:")
        for var, value in self.environment_info["environment_variables"].items():
            print(f"  {var}: {value}")
        
        print("="*60)


# Global reproducibility manager instance
_global_reproducibility_manager: Optional[ReproducibilityManager] = None


def setup_reproducibility(seed: Optional[int] = None,
                         strict_mode: bool = True,
                         verify: bool = True,
                         log_file: Optional[str] = None) -> ReproducibilityManager:
    """
    Setup global reproducibility settings
    
    Args:
        seed: Random seed to use
        strict_mode: Whether to enforce strict reproducibility
        verify: Whether to verify reproducibility after setup
        log_file: Optional log file for reproducibility information
        
    Returns:
        ReproducibilityManager instance
    """
    global _global_reproducibility_manager
    
    _global_reproducibility_manager = ReproducibilityManager(
        seed=seed,
        strict_mode=strict_mode,
        log_file=log_file
    )
    
    # Configure deterministic operations
    _global_reproducibility_manager.configure_deterministic_operations()
    
    # Verify reproducibility if requested
    if verify:
        _global_reproducibility_manager.verify_reproducibility()
    
    # Print summary
    _global_reproducibility_manager.print_summary()
    
    return _global_reproducibility_manager


def get_reproducibility_manager() -> Optional[ReproducibilityManager]:
    """Get the global reproducibility manager"""
    return _global_reproducibility_manager


def set_seed(seed: int):
    """Quick function to set global seed"""
    if _global_reproducibility_manager:
        _global_reproducibility_manager.set_global_seed(seed)
    else:
        setup_reproducibility(seed=seed)


def reproducible_function(seed_offset: int = 0):
    """Decorator to make functions reproducible with optional seed offset"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_reproducibility_manager()
            if manager:
                # Use base seed plus offset for this function
                func_seed = manager.seed + seed_offset
                manager.set_global_seed(func_seed)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Test the reproducibility system
    print("Testing Reproducibility Manager...")
    
    # Setup reproducibility
    manager = setup_reproducibility(seed=42, strict_mode=True, verify=True)
    
    # Test reproducible operations
    print("\nTesting reproducible operations...")
    
    @reproducible_function(seed_offset=0)
    def test_random_generation():
        results = []
        results.append(random.random())
        if HAS_NUMPY:
            results.append(np.random.rand())
        if HAS_TORCH:
            results.append(torch.rand(1).item())
        return results
    
    # Run twice to verify reproducibility
    result1 = test_random_generation()
    result2 = test_random_generation()
    
    print(f"First run: {result1}")
    print(f"Second run: {result2}")
    print(f"Results identical: {result1 == result2}")
    
    # Test data splitting
    print("\nTesting reproducible data splitting...")
    splits1 = manager.create_reproducible_split(1000, 0.7, 0.15, 0.15)
    splits2 = manager.create_reproducible_split(1000, 0.7, 0.15, 0.15)
    
    print(f"Splits identical: {splits1 == splits2}")
    
    # Save state
    manager.save_reproducibility_state("test_reproducibility_state.json")
    print("\nReproducibility state saved to test_reproducibility_state.json")
    
    print("\nReproducibility testing completed!")