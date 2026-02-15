#!/usr/bin/env python3
"""
Integration Example: Adding Short Utterance Testing to Existing Experiments

This script shows how to add short utterance evaluation to your existing
training pipeline without major code changes.
"""

import torch
from torch.utils.data import DataLoader
from short_utterance_transform import ShortUtteranceTransform, create_short_utterance_dataset_wrapper


# ============================================================================
# Example 1: Add short utterance test after training
# ============================================================================

def add_short_utterance_test_to_baseline_experiment(experiment):
    """
    Add short utterance testing to baseline experiment.
    
    Args:
        experiment: Your BaselineExperiment or ChaoticExperiment instance
    """
    print("\n" + "="*60)
    print("SHORT UTTERANCE ROBUSTNESS TEST")
    print("="*60)
    
    # Test configurations
    durations = [0.5, 1.0, 1.5, 2.0, 3.0]
    sample_rate = 16000
    device = experiment.device
    
    # Get test dataset
    test_dataset = experiment.test_loader.dataset
    
    results = {}
    
    for duration in durations:
        print(f"\nTesting at {duration}s...")
        
        # Create wrapped dataset
        short_dataset = create_short_utterance_dataset_wrapper(
            test_dataset,
            target_duration=duration,
            sample_rate=sample_rate,
            crop_mode='random'
        )
        
        # Create dataloader
        short_loader = DataLoader(
            short_dataset,
            batch_size=experiment.config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )
        
        # Evaluate
        experiment.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, labels in short_loader:
                audio = audio.to(device)
                labels = labels.to(device)
                
                outputs = experiment.model(audio)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        results[duration] = accuracy
        
        print(f"  Accuracy: {accuracy:.2f}%")
    
    # Save results
    import json
    output_path = experiment.output_dir / 'short_utterance_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return results


# ============================================================================
# Example 2: Use as data augmentation during training
# ============================================================================

def create_multi_duration_training_loader(dataset, batch_size=32):
    """
    Create a dataloader with random duration augmentation.
    
    This can improve model robustness by exposing it to various
    utterance lengths during training.
    """
    from short_utterance_transform import MultiDurationTransform
    
    # Create transform that randomly selects durations
    transform = MultiDurationTransform(
        durations=[1.0, 1.5, 2.0, 2.5, 3.0],
        sample_rate=16000,
        crop_mode='random',
        duration_weights=[0.1, 0.2, 0.3, 0.2, 0.2]  # Favor medium lengths
    )
    
    # Wrap dataset
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform):
            self.dataset = base_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            audio, label = self.dataset[idx]
            audio = self.transform(audio)
            return audio, label
    
    augmented_dataset = AugmentedDataset(dataset, transform)
    
    return DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


# ============================================================================
# Example 3: Standalone evaluation function
# ============================================================================

def evaluate_model_on_short_utterances(
    model,
    test_dataset,
    durations=[0.5, 1.0, 2.0, 3.0],
    device='cuda',
    batch_size=32
):
    """
    Standalone function to evaluate any model on short utterances.
    
    Args:
        model: Trained PyTorch model
        test_dataset: Test dataset
        durations: List of durations to test
        device: Device to use
        batch_size: Batch size
        
    Returns:
        Dictionary mapping durations to accuracies
    """
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for duration in durations:
        # Create short utterance dataset
        short_dataset = create_short_utterance_dataset_wrapper(
            test_dataset,
            target_duration=duration,
            sample_rate=16000,
            crop_mode='random'
        )
        
        # Create loader
        loader = DataLoader(
            short_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, labels in loader:
                audio = audio.to(device)
                labels = labels.to(device)
                
                outputs = model(audio)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        results[duration] = accuracy
        
        print(f"Duration {duration}s: {accuracy:.2f}%")
    
    return results


# ============================================================================
# Example 4: Modify existing training script
# ============================================================================

def example_modified_training_script():
    """
    Example showing how to modify your existing training script.
    """
    
    # Your existing training code
    # ... (model creation, dataset loading, etc.)
    
    # After training is complete:
    print("\nTraining completed. Running short utterance test...")
    
    # Quick short utterance test
    short_results = evaluate_model_on_short_utterances(
        model=trained_model,
        test_dataset=test_dataset,
        durations=[0.5, 1.0, 2.0, 3.0]
    )
    
    # Log results
    print("\nShort Utterance Results:")
    for dur, acc in short_results.items():
        print(f"  {dur}s: {acc:.2f}%")
    
    # Calculate robustness metric
    if 3.0 in short_results and 0.5 in short_results:
        robustness = (short_results[0.5] / short_results[3.0]) * 100
        print(f"\nRobustness (0.5s vs 3.0s): {robustness:.1f}%")


# ============================================================================
# Example 5: Using in Jupyter Notebook
# ============================================================================

"""
# In your Jupyter notebook:

from short_utterance_transform import ShortUtteranceTransform
import torch

# Load your trained model
model = torch.load('path/to/model.pth')
model.eval()

# Create transform
transform = ShortUtteranceTransform(
    target_duration=1.0,  # 1 second
    sample_rate=16000,
    crop_mode='random'
)

# Test on single audio
audio, label = test_dataset[0]
short_audio = transform(audio)

# Predict
with torch.no_grad():
    output = model(short_audio.unsqueeze(0))
    prediction = torch.argmax(output, dim=1)

print(f"Predicted speaker: {prediction.item()}")
print(f"True speaker: {label}")
"""


# ============================================================================
# Main example usage
# ============================================================================

if __name__ == '__main__':
    print("Integration Examples for Short Utterance Testing")
    print("="*60)
    print("\nThis script demonstrates how to integrate short utterance")
    print("testing into your existing experiments.")
    print("\nSee the example functions above for different use cases:")
    print("  1. add_short_utterance_test_to_baseline_experiment()")
    print("  2. create_multi_duration_training_loader()")
    print("  3. evaluate_model_on_short_utterances()")
    print("  4. example_modified_training_script()")
    print("\nFor actual experiments, use:")
    print("  - short_utterance_experiment.py (full experiment)")
    print("  - quick_short_utterance_test.py (quick test)")
