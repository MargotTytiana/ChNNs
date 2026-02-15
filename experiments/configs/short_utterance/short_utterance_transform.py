"""
Short Utterance Transform for Speaker Recognition Robustness Testing

This module provides transformation functions to simulate short utterance conditions
by randomly or systematically cropping audio segments to specific durations.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
import random


class ShortUtteranceTransform:
    """
    Transform audio to simulate short utterance conditions.
    
    This transform randomly crops audio to a specified duration to test
    model robustness with limited temporal information.
    """
    
    def __init__(
        self,
        target_duration: float = 1.0,
        sample_rate: int = 16000,
        crop_mode: str = 'random',
        pad_if_shorter: bool = True,
        pad_value: float = 0.0
    ):
        """
        Initialize short utterance transform.
        
        Args:
            target_duration: Target duration in seconds
            sample_rate: Audio sample rate
            crop_mode: 'random', 'start', 'middle', or 'end'
            pad_if_shorter: Whether to pad audio shorter than target
            pad_value: Value to use for padding
        """
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration * sample_rate)
        self.crop_mode = crop_mode
        self.pad_if_shorter = pad_if_shorter
        self.pad_value = pad_value
        
        if crop_mode not in ['random', 'start', 'middle', 'end']:
            raise ValueError(f"Invalid crop_mode: {crop_mode}")
    
    def __call__(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply short utterance transformation.
        
        Args:
            audio: Input audio tensor/array of shape (samples,) or (channels, samples)
            
        Returns:
            Cropped/padded audio of target duration
        """
        is_tensor = isinstance(audio, torch.Tensor)
        
        # Convert to tensor if numpy
        if not is_tensor:
            audio_tensor = torch.from_numpy(audio)
        else:
            audio_tensor = audio
        
        # Handle multi-channel audio
        if audio_tensor.dim() > 1:
            # Assume shape is (channels, samples)
            audio_tensor = audio_tensor[0]  # Take first channel
        
        audio_length = audio_tensor.shape[0]
        
        # Case 1: Audio is longer than target - crop
        if audio_length > self.target_samples:
            cropped = self._crop_audio(audio_tensor, audio_length)
        
        # Case 2: Audio is shorter than target
        elif audio_length < self.target_samples:
            if self.pad_if_shorter:
                cropped = self._pad_audio(audio_tensor)
            else:
                # Return as is if padding disabled
                cropped = audio_tensor
        
        # Case 3: Audio is exactly target length
        else:
            cropped = audio_tensor
        
        # Convert back to numpy if input was numpy
        if not is_tensor:
            return cropped.numpy()
        return cropped
    
    def _crop_audio(self, audio: torch.Tensor, audio_length: int) -> torch.Tensor:
        """Crop audio based on crop_mode."""
        if self.crop_mode == 'random':
            # Random start position
            max_start = audio_length - self.target_samples
            start_idx = random.randint(0, max_start)
        
        elif self.crop_mode == 'start':
            start_idx = 0
        
        elif self.crop_mode == 'middle':
            start_idx = (audio_length - self.target_samples) // 2
        
        elif self.crop_mode == 'end':
            start_idx = audio_length - self.target_samples
        
        else:
            raise ValueError(f"Invalid crop_mode: {self.crop_mode}")
        
        end_idx = start_idx + self.target_samples
        return audio[start_idx:end_idx]
    
    def _pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad audio to target length."""
        pad_length = self.target_samples - audio.shape[0]
        
        if isinstance(audio, torch.Tensor):
            padded = torch.nn.functional.pad(
                audio, (0, pad_length), value=self.pad_value
            )
        else:
            # Should not reach here but keep for safety
            padded = np.pad(
                audio, (0, pad_length), constant_values=self.pad_value
            )
        
        return padded
    
    def __repr__(self) -> str:
        return (f"ShortUtteranceTransform("
                f"duration={self.target_duration}s, "
                f"mode={self.crop_mode})")


class MultiDurationTransform:
    """
    Transform that randomly selects from multiple duration options.
    
    Useful for training with varied short utterance lengths.
    """
    
    def __init__(
        self,
        durations: list = [0.5, 1.0, 2.0, 3.0],
        sample_rate: int = 16000,
        crop_mode: str = 'random',
        duration_weights: Optional[list] = None
    ):
        """
        Initialize multi-duration transform.
        
        Args:
            durations: List of possible durations in seconds
            sample_rate: Audio sample rate
            crop_mode: Crop mode for each duration
            duration_weights: Optional weights for sampling durations
        """
        self.durations = durations
        self.sample_rate = sample_rate
        self.crop_mode = crop_mode
        self.duration_weights = duration_weights
        
        # Create transform for each duration
        self.transforms = {
            dur: ShortUtteranceTransform(
                target_duration=dur,
                sample_rate=sample_rate,
                crop_mode=crop_mode
            )
            for dur in durations
        }
    
    def __call__(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Apply randomly selected duration transform."""
        # Randomly select duration
        if self.duration_weights:
            duration = random.choices(self.durations, weights=self.duration_weights)[0]
        else:
            duration = random.choice(self.durations)
        
        # Apply corresponding transform
        return self.transforms[duration](audio)
    
    def __repr__(self) -> str:
        return f"MultiDurationTransform(durations={self.durations})"


def create_short_utterance_dataset_wrapper(
    base_dataset,
    target_duration: float = 1.0,
    sample_rate: int = 16000,
    crop_mode: str = 'random'
):
    """
    Create a wrapped dataset that applies short utterance transform.
    
    Args:
        base_dataset: Original PyTorch dataset
        target_duration: Target duration in seconds
        sample_rate: Audio sample rate
        crop_mode: Crop mode
        
    Returns:
        Dataset with short utterance transform applied
    """
    transform = ShortUtteranceTransform(
        target_duration=target_duration,
        sample_rate=sample_rate,
        crop_mode=crop_mode
    )
    
    class ShortUtteranceDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            audio, label = self.dataset[idx]
            audio = self.transform(audio)
            return audio, label
        
        @property
        def num_classes(self):
            return getattr(self.dataset, 'num_classes', None)
    
    return ShortUtteranceDataset(base_dataset, transform)


# Utility functions for experimental analysis
def compute_duration_performance_curve(
    model,
    test_dataset,
    durations: list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    sample_rate: int = 16000,
    batch_size: int = 32,
    device: str = 'cuda'
):
    import torch
    from torch.utils.data import DataLoader

    # 保险 1: 显式将整个模型树递归移动到目标设备
    device = torch.device(device)
    model.to(device)
    model.eval()
    results = {}
    
    for duration in durations:
        print(f"Evaluating at {duration}s duration...")
        wrapped_dataset = create_short_utterance_dataset_wrapper(
            test_dataset, target_duration=duration, 
            sample_rate=sample_rate, crop_mode='random'
        )
        
        loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False)
        
        correct, total = 0, 0

        # 终极对齐：强制遍历所有子模块并移动设备
        device_obj = torch.device(device)
        model.to(device_obj)
        
        # 额外保险：有些动态属性需要手动移动
        for name, module in model.named_modules():
            module.to(device_obj)
                     
        with torch.no_grad():
            for audio, labels in loader:
                # 显式使用目标设备
                audio = audio.to(device_obj).float()
                labels = labels.to(device_obj)
                
                try:
                    outputs = model(audio)
                except RuntimeError as e:
                    if "device" in str(e):
                        # 如果发生设备不匹配，做最后一次全量同步
                        model.to(audio.device)
                        outputs = model(audio, labels=None, return_intermediates=False)
                    else:
                        raise e
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        results[duration] = accuracy
        print(f"  Accuracy: {accuracy:.2f}%")
    
    return results