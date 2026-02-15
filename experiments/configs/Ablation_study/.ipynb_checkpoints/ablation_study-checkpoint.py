"""
Ablation Study - Based on 96.61% Baseline
==========================================

正确的消融实验设计：
- Baseline (full): 使用 experiment_H2c_0.05.yaml 的完整配置
- 预期: Val ~96.61%

只移除每个组件，其他保持不变
"""

import torch
import torch.nn as nn
import yaml
import json
import copy
from pathlib import Path
from typing import Dict, Any
import logging
import sys
import os
from datetime import datetime

# Setup imports
def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent.parent
    paths = [
        str(model_dir),
        str(model_dir/'experiments'), 
        str(model_dir/'models'),
        str(model_dir/'features'),
        str(model_dir/'data'),
        str(model_dir/'utils')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

from chaotic_experiment import ChaoticExperiment
from chaotic_network import ChaoticSpeakerRecognitionNetwork


# ============================================================================
# Simplified Replacement Components (same as before but confirmed working)
# ============================================================================

class SimpleStatisticalFeatures(nn.Module):
    def __init__(self, output_dim=7):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(-1)
            
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        
        features = torch.cat([mean, std, min_val, max_val], dim=-1)
        
        current_dim = features.shape[-1]
        if current_dim < self.output_dim:
            padding = torch.zeros(features.shape[0], self.output_dim - current_dim, device=features.device)
            features = torch.cat([features, padding], dim=-1)
        elif current_dim > self.output_dim:
            features = features[:, :self.output_dim]
            
        return features


class PhaseSpaceBypass(nn.Module):
    """Bypass with downsampling to prevent OOM."""
    def __init__(self, output_length=200, output_dim=10):
        super().__init__()
        self.output_length = output_length
        self.output_dim = output_dim
        
    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        x = x.transpose(1, 2)
        x = nn.functional.adaptive_avg_pool1d(x, self.output_length)
        x = x.transpose(1, 2)
        x = x.repeat(1, 1, self.output_dim)
        return x


class AveragePooling(nn.Module):
    def __init__(self, output_dim=None):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x):
        if len(x.shape) == 3:
            pooled = x.mean(dim=1)
        else:
            pooled = x
            
        if self.output_dim is not None and pooled.shape[-1] != self.output_dim:
            if pooled.shape[-1] < self.output_dim:
                padding = torch.zeros(pooled.shape[0], self.output_dim - pooled.shape[-1], device=pooled.device)
                pooled = torch.cat([pooled, padding], dim=-1)
            elif pooled.shape[-1] > self.output_dim:
                pooled = pooled[:, :self.output_dim]
        return pooled


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.net(x_flat)


class AblationModel(nn.Module):
    """Apply ablation to base model."""
    def __init__(self, base_model, ablation_type: str):
        super().__init__()
        self.ablation_type = ablation_type
        self.base_model = base_model
        
        try:
            self.device = next(base_model.parameters()).device
        except:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self._apply_ablation()
        
    def _apply_ablation(self):
        if self.ablation_type == 'full':
            self.logger.info("[ABLATION] Full model - no changes")
            
        elif self.ablation_type == 'no_phase_space':
            dim = getattr(self.base_model, 'embedding_dim', 10)
            self.base_model.phase_space = PhaseSpaceBypass(200, dim).to(self.device)
            self.logger.info("[ABLATION] Removed: Phase Space Reconstruction")
            
        elif self.ablation_type == 'no_chaotic_features':
            mlsa_dim = getattr(self.base_model, 'mlsa_scales', 5)
            self.base_model.mlsa_extractor = SimpleStatisticalFeatures(mlsa_dim).to(self.device)
            self.base_model.rqa_extractor = SimpleStatisticalFeatures(13).to(self.device)
            self.logger.info("[ABLATION] Removed: Chaotic Features (MLSA + RQA)")
            
        elif self.ablation_type == 'no_chaotic_embedding':
            input_dim = 230
            output_dim = 150
            mlp = SimpleMLP(input_dim, output_dim).to(self.device)
            
            original_forward = mlp.forward
            def reshaped_forward(x):
                out = original_forward(x)
                return out.reshape(out.shape[0], 50, 3)
            mlp.forward = reshaped_forward
            
            self.base_model.chaotic_embedding = mlp
            self.logger.info("[ABLATION] Removed: Chaotic Embedding")
            
        elif self.ablation_type == 'no_attractor_pooling':
            self.base_model.attractor_pooling = AveragePooling(117).to(self.device)
            self.logger.info("[ABLATION] Removed: Attractor Pooling")
            
        elif self.ablation_type == 'minimal':
            self.base_model.phase_space = PhaseSpaceBypass(200, 10).to(self.device)
            mlsa_dim = getattr(self.base_model, 'mlsa_scales', 5)
            self.base_model.mlsa_extractor = SimpleStatisticalFeatures(mlsa_dim).to(self.device)
            self.base_model.rqa_extractor = SimpleStatisticalFeatures(13).to(self.device)
            self.base_model.attractor_pooling = AveragePooling(117).to(self.device)
            self.logger.info("[ABLATION] Minimal baseline")

    def forward(self, x, *args, **kwargs):
        return self.base_model(x, return_intermediates=kwargs.get('return_intermediates', False))
    
    def compute_loss(self, *args, **kwargs):
        return self.base_model.compute_loss(*args, **kwargs)


# ============================================================================
# Ablation Experiment Manager - Based on 96.61% Baseline
# ============================================================================

class AblationExperiment:
    """
    消融实验管理器 - 基于96.61%的baseline配置
    
    保持以下完全不变：
    - 数据集: dev-clean-2
    - Sync weight: 0.05
    - Adversarial training: enabled
    - Stability loss: enabled
    - 所有超参数
    
    只修改：移除特定的模型组件
    """
    
    def __init__(self, baseline_config_path: str, output_dir: str = './ablation_results_correct'):
        self.baseline_config_path = baseline_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AblationExperiment")
        
        # Load baseline config
        with open(baseline_config_path, 'r') as f:
            self.baseline_config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded baseline config from: {baseline_config_path}")
        self.logger.info(f"Expected baseline performance: ~96.61% Val Acc")
        
        # Ablation configurations
        self.ablation_configs = [
            'full',                    # Baseline: 期望96.61%
            'no_phase_space',          # 移除相空间重构
            'no_chaotic_features',     # 移除混沌特征（MLSA+RQA）
            'no_chaotic_embedding',    # 移除混沌嵌入
            'no_attractor_pooling',    # 移除吸引子池化
            'minimal'                  # 最小模型（移除所有混沌组件）
        ]
        
        self.results = {}

    def create_ablation_config(self, ablation_type: str) -> Dict[str, Any]:
        """
        创建消融配置 - 保持baseline的所有设置，只标记移除组件
        """
        config = copy.deepcopy(self.baseline_config)
        
        # 修改实验名称
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['name'] = f'ablation_{ablation_type}'
        config['experiment']['description'] = f'Ablation: {ablation_type} removed from 96.61% baseline'
        
        # FIX: 提取关键参数到顶层（base_experiment.py和chaotic_experiment.py需要）
        if 'model' in config:
            # num_speakers
            if 'num_speakers' in config['model'] and 'num_speakers' not in config:
                config['num_speakers'] = config['model']['num_speakers']
            
            # chaotic_system (从 model.chaotic_embedding.system_type)
            if 'chaotic_embedding' in config['model']:
                system_type = config['model']['chaotic_embedding'].get('system_type')
                if system_type and 'chaotic_system' not in config:
                    config['chaotic_system'] = system_type
            
            # batch_size
            if 'training' in config and 'batch_size' in config['training']:
                if 'batch_size' not in config:
                    config['batch_size'] = config['training']['batch_size']
            
            # data_dir
            if 'data' in config and 'data_dir' in config['data']:
                if 'data_dir' not in config:
                    config['data_dir'] = config['data']['data_dir']
        
        # 添加ablation标记（用于后续处理）
        config['ablation'] = {
            'type': ablation_type,
            'baseline_config': str(self.baseline_config_path)
        }
        
        return config

    def run_single_ablation(
        self, 
        ablation_type: str, 
        epochs: int = 100,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """运行单个消融实验"""
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Running Ablation: {ablation_type}")
        self.logger.info(f"{'='*70}")
        
        # Create config
        config = self.create_ablation_config(ablation_type)
        
        # Save config
        config_path = self.output_dir / f'config_{ablation_type}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Create experiment
            experiment = ChaoticExperiment(
                config=config,
                experiment_name=f'ablation_{ablation_type}',
                output_dir=str(self.output_dir / ablation_type),
                device=device
            )
            
            # Initialize
            if not hasattr(experiment, 'model') or experiment.model is None:
                if hasattr(experiment, 'setup'):
                    experiment.setup()
                else:
                    self.logger.error("Cannot initialize experiment")
                    raise RuntimeError("Experiment initialization failed")
            
            # Apply ablation
            self.logger.info(f"Applying ablation: {ablation_type}")
            ablated_model = AblationModel(experiment.model, ablation_type)
            experiment.model = ablated_model
            
            # Count parameters
            total_params = sum(p.numel() for p in experiment.model.parameters())
            trainable_params = sum(p.numel() for p in experiment.model.parameters() if p.requires_grad)
            
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            # Train
            self.logger.info("Starting training...")
            start_time = datetime.now()
            experiment.train(num_epochs=epochs)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Load best checkpoint
            self.logger.info("="*70)
            self.logger.info("Loading best checkpoint for evaluation...")
            
            best_val_acc = 0.0
            best_epoch = 0
            
            if hasattr(experiment, 'checkpoint_manager'):
                checkpoint_dir = Path(experiment.checkpoint_manager.checkpoint_dir) / experiment.checkpoint_manager.experiment_name
                
                # Try to find best model
                best_path = checkpoint_dir / 'best_model.pt'
                if not best_path.exists():
                    # Try .pkl files
                    pkl_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pkl'))
                    if pkl_files:
                        pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        best_path = pkl_files[0]
                        self.logger.info(f"Using latest .pkl file: {best_path.name}")
                
                if best_path.exists():
                    try:
                        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
                        
                        # Load state dict
                        state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                        if state_key in checkpoint:
                            if hasattr(experiment.model, 'base_model'):
                                experiment.model.base_model.load_state_dict(checkpoint[state_key])
                            else:
                                experiment.model.load_state_dict(checkpoint[state_key])
                            
                            self.logger.info(f"✓ Loaded checkpoint: {best_path.name}")
                            
                            # Get metrics from checkpoint
                            best_epoch = checkpoint.get('epoch', 0)
                            best_val_acc = checkpoint.get('metric', checkpoint.get('best_metric', 0.0))
                            
                    except Exception as e:
                        self.logger.error(f"Failed to load checkpoint: {e}")
            
            # Get from state if not from checkpoint
            if hasattr(experiment, 'state'):
                if best_val_acc == 0.0 and hasattr(experiment.state, 'best_metric'):
                    best_val_acc = experiment.state.best_metric
                if best_epoch == 0 and hasattr(experiment.state, 'best_epoch'):
                    best_epoch = experiment.state.best_epoch
            
            # Final test
            self.logger.info("Running final test evaluation...")
            test_metrics = experiment.test()
            
            # Compile results
            results = {
                'ablation_type': ablation_type,
                'test_accuracy': test_metrics.get('accuracy', 0.0),
                'test_loss': test_metrics.get('loss', 0.0),
                'best_val_accuracy': best_val_acc,
                'best_epoch': best_epoch,
                'training_time_seconds': training_time,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'config_path': str(config_path),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Calculate performance drop vs full model
            if ablation_type != 'full' and 'full' in self.results:
                full_val = self.results['full']['best_val_accuracy']
                drop = full_val - best_val_acc
                results['val_acc_drop_vs_full'] = drop
                results['val_acc_drop_pct'] = (drop / full_val * 100) if full_val > 0 else 0.0
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"[RESULT] {ablation_type}")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"  Best Val Acc:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
            self.logger.info(f"  Best Epoch:    {best_epoch}")
            self.logger.info(f"  Test Acc:      {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
            self.logger.info(f"  Training Time: {training_time/60:.2f} min")
            self.logger.info(f"  Total Params:  {total_params:,}")
            
            if ablation_type != 'full' and 'val_acc_drop_vs_full' in results:
                self.logger.info(f"  Drop vs Full:  {results['val_acc_drop_vs_full']:.4f} ({results['val_acc_drop_pct']:.2f}%)")
            
            self.logger.info(f"{'='*70}\n")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ablation {ablation_type} failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'ablation_type': ablation_type,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_all_ablations(self, epochs: int = 100, device: str = 'cuda'):
        """运行所有消融实验"""
        
        self.logger.info("\n" + "="*70)
        self.logger.info("ABLATION STUDY - Based on 96.61% Baseline")
        self.logger.info("="*70)
        self.logger.info(f"Baseline config: {self.baseline_config_path}")
        self.logger.info(f"Number of ablations: {len(self.ablation_configs)}")
        self.logger.info(f"Epochs per run: {epochs}")
        self.logger.info("="*70 + "\n")
        
        for ablation_type in self.ablation_configs:
            self.results[ablation_type] = self.run_single_ablation(
                ablation_type, epochs, device
            )
            
            # Save intermediate results
            self.save_results()
        
        # Generate final summary
        self.generate_summary()

    def save_results(self):
        """保存结果"""
        results_path = self.output_dir / 'ablation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"Results saved to: {results_path}")

    def generate_summary(self):
        """生成总结报告"""
        self.logger.info("\n" + "="*70)
        self.logger.info("ABLATION STUDY SUMMARY")
        self.logger.info("="*70)
        
        # Get full model baseline
        full_result = self.results.get('full', {})
        full_val = full_result.get('best_val_accuracy', 0.0)
        
        self.logger.info(f"\nBaseline (full model): {full_val:.4f} ({full_val*100:.2f}%)")
        self.logger.info("-"*70)
        
        # Sort by performance
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items() if k != 'full' and v.get('success')],
            key=lambda x: x[1].get('best_val_accuracy', 0.0),
            reverse=True
        )
        
        self.logger.info("\nAblation Results (sorted by performance):")
        self.logger.info(f"{'Ablation':<25} {'Val Acc':<12} {'Test Acc':<12} {'Drop':<12} {'Status'}")
        self.logger.info("-"*70)
        
        for ablation_type, result in sorted_results:
            val_acc = result.get('best_val_accuracy', 0.0)
            test_acc = result.get('test_accuracy', 0.0)
            drop = full_val - val_acc if full_val > 0 else 0.0
            
            self.logger.info(
                f"{ablation_type:<25} "
                f"{val_acc:>6.2%}      "
                f"{test_acc:>6.2%}      "
                f"{drop:>+6.2%}      "
                f"{'✓' if result.get('success') else '✗'}"
            )
        
        # Failed experiments
        failed = [k for k, v in self.results.items() if not v.get('success')]
        if failed:
            self.logger.info(f"\nFailed experiments: {', '.join(failed)}")
        
        self.logger.info("="*70 + "\n")
        
        # Save summary report
        summary_path = self.output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Ablation Study Summary\n")
            f.write(f"Baseline: {full_val:.4f}\n\n")
            for ablation_type, result in sorted_results:
                val_acc = result.get('best_val_accuracy', 0.0)
                drop = full_val - val_acc
                f.write(f"{ablation_type}: {val_acc:.4f} (drop: {drop:.4f})\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ablation Study based on 96.61% baseline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='experiment_H2c_0.05.yaml',
        help='Path to baseline config (96.61% performance)'
    )
    parser.add_argument('--output-dir', type=str, default='./ablation_results_correct')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Run ablation study
    exp = AblationExperiment(args.config, args.output_dir)
    exp.run_all_ablations(args.epochs, args.device)


if __name__ == '__main__':
    main()