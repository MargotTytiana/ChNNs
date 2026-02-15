# ============================================================================
# sync_loss_extensions.py
# Extensions to PhaseSynchronizationLoss for Options B, C, D
# With warmup support for H4 experiments
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ScheduledSyncLoss(nn.Module):
    """
    Option B: Staged/Scheduled Sync Loss
    Supports different scheduling modes for sync and desync weights
    Now with warmup_epochs support for H4 experiments
    """
    
    def __init__(self, 
                 sync_weight: float = 0.1,
                 desync_weight: float = 0.1,
                 margin: float = 5.0,
                 sample_ratio: float = 0.3,
                 schedule_mode: str = "none",
                 phase1_epochs: int = 50,
                 warmup_epochs: int = 0):
        """
        Args:
            sync_weight: Weight for sync loss
            desync_weight: Weight for desync loss
            margin: Margin for hinge loss
            sample_ratio: Ratio of pairs to sample
            schedule_mode: "none" | "sync_first" | "desync_first" | "gradual_transition"
            phase1_epochs: Number of epochs for phase 1 (or transition period)
            warmup_epochs: Number of epochs to skip sync loss entirely (H4 experiment)
        """
        super().__init__()
        self.sync_weight = sync_weight
        self.desync_weight = desync_weight
        self.margin = margin
        self.sample_ratio = sample_ratio
        self.schedule_mode = schedule_mode
        self.phase1_epochs = phase1_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Called by trainer at the start of each epoch"""
        self.current_epoch = epoch
    
    def is_active(self) -> bool:
        """Check if sync loss should be applied (after warmup)"""
        return self.current_epoch >= self.warmup_epochs
    
    def get_current_weights(self) -> Tuple[float, float]:
        """Return (sync_weight, desync_weight) based on schedule and warmup"""
        
        # Check warmup first
        if not self.is_active():
            return 0.0, 0.0
        
        if self.schedule_mode == "none":
            return self.sync_weight, self.desync_weight
        
        elif self.schedule_mode == "sync_first":
            # Phase 1: only sync, Phase 2: only desync
            effective_epoch = self.current_epoch - self.warmup_epochs
            if effective_epoch < self.phase1_epochs:
                return self.sync_weight, 0.0
            else:
                return 0.0, self.desync_weight
        
        elif self.schedule_mode == "desync_first":
            # Phase 1: only desync, Phase 2: only sync
            effective_epoch = self.current_epoch - self.warmup_epochs
            if effective_epoch < self.phase1_epochs:
                return 0.0, self.desync_weight
            else:
                return self.sync_weight, 0.0
        
        elif self.schedule_mode == "gradual_transition":
            # Gradual transition from sync to desync
            effective_epoch = self.current_epoch - self.warmup_epochs
            progress = min(1.0, effective_epoch / max(1, self.phase1_epochs))
            current_sync = self.sync_weight * (1 - progress)
            current_desync = self.desync_weight * progress
            return current_sync, current_desync
        
        else:
            return self.sync_weight, self.desync_weight
    
    def _compute_pairwise_distances(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between all trajectories"""
        batch_size = trajectories.size(0)
        # Flatten: [batch, time, dim] -> [batch, time*dim]
        traj_flat = trajectories.reshape(batch_size, -1)
        
        # Compute pairwise L2 distances
        # distances[i,j] = ||traj_i - traj_j||
        distances = torch.cdist(traj_flat, traj_flat, p=2)
        return distances
    
    def _compute_sync_loss(self, trajectories: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Minimize distance between same-speaker trajectories"""
        distances = self._compute_pairwise_distances(trajectories)
        batch_size = labels.size(0)
        
        # Create same-speaker mask
        labels_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        same_speaker_mask = (labels_row == labels_col).float()
        
        # Exclude diagonal
        same_speaker_mask = same_speaker_mask - torch.eye(batch_size, device=labels.device)
        
        # Sample if needed
        if self.sample_ratio < 1.0:
            sample_mask = torch.rand_like(same_speaker_mask) < self.sample_ratio
            same_speaker_mask = same_speaker_mask * sample_mask.float()
        
        # Compute mean distance for same-speaker pairs
        num_pairs = same_speaker_mask.sum()
        if num_pairs > 0:
            sync_loss = (distances * same_speaker_mask).sum() / num_pairs
        else:
            sync_loss = torch.tensor(0.0, device=trajectories.device)
        
        return sync_loss
    
    def _compute_desync_loss(self, trajectories: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Maximize distance between different-speaker trajectories (hinge loss)"""
        distances = self._compute_pairwise_distances(trajectories)
        batch_size = labels.size(0)
        
        # Create different-speaker mask
        labels_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        diff_speaker_mask = (labels_row != labels_col).float()
        
        # Sample if needed
        if self.sample_ratio < 1.0:
            sample_mask = torch.rand_like(diff_speaker_mask) < self.sample_ratio
            diff_speaker_mask = diff_speaker_mask * sample_mask.float()
        
        # Hinge loss: max(0, margin - distance)
        hinge = torch.relu(self.margin - distances)
        
        # Compute mean hinge loss for different-speaker pairs
        num_pairs = diff_speaker_mask.sum()
        if num_pairs > 0:
            desync_loss = (hinge * diff_speaker_mask).sum() / num_pairs
        else:
            desync_loss = torch.tensor(0.0, device=trajectories.device)
        
        return desync_loss
    
    def forward(self, trajectories: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Check warmup
        if not self.is_active():
            return torch.tensor(0.0, device=trajectories.device, requires_grad=True)
        
        sync_w, desync_w = self.get_current_weights()
        
        loss = torch.tensor(0.0, device=trajectories.device)
        
        if sync_w > 0:
            sync_loss = self._compute_sync_loss(trajectories, labels)
            loss = loss + sync_w * sync_loss
        
        if desync_w > 0:
            desync_loss = self._compute_desync_loss(trajectories, labels)
            loss = loss + desync_w * desync_loss
        
        return loss
    
    def get_loss_components(self, trajectories: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """For logging: return individual loss components"""
        sync_w, desync_w = self.get_current_weights()
        
        components = {
            'sync_weight': sync_w,
            'desync_weight': desync_w,
            'sync_loss': 0.0,
            'desync_loss': 0.0,
            'warmup_active': not self.is_active(),
            'current_epoch': self.current_epoch,
        }
        
        if self.is_active():
            if sync_w > 0:
                components['sync_loss'] = self._compute_sync_loss(trajectories, labels).item()
            if desync_w > 0:
                components['desync_loss'] = self._compute_desync_loss(trajectories, labels).item()
        
        return components


class HierarchicalSyncLoss(nn.Module):
    """
    Option C: Hierarchical Sync Loss (FIXED)
    - Sync loss on TRAJECTORY space (pull same-speaker together)
    - Desync loss on EMBEDDING space (push different-speaker apart)
    """
    
    def __init__(self,
                 sync_weight: float = 0.1,
                 desync_weight: float = 0.1,
                 margin: float = 1.0,
                 sample_ratio: float = 0.3,
                 normalize: bool = True,
                 warmup_epochs: int = 0):
        super().__init__()
        self.sync_weight = sync_weight
        self.desync_weight = desync_weight
        self.margin = margin
        self.sample_ratio = sample_ratio
        self.normalize = normalize
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Called by trainer at the start of each epoch"""
        self.current_epoch = epoch
    
    def is_active(self) -> bool:
        """Check if sync loss should be applied (after warmup)"""
        return self.current_epoch >= self.warmup_epochs
    
    def forward(self, trajectories, embeddings, labels):
        # Check warmup
        if not self.is_active():
            return torch.tensor(0.0, device=trajectories.device, requires_grad=True)
        
        loss = torch.tensor(0.0, device=trajectories.device)
        
        if self.sync_weight > 0:
            sync_loss = self._trajectory_sync_loss(trajectories, labels)
            loss = loss + self.sync_weight * sync_loss
        
        if self.desync_weight > 0:
            desync_loss = self._embedding_desync_loss(embeddings, labels)
            loss = loss + self.desync_weight * desync_loss
        
        return loss
    
    def _trajectory_sync_loss(self, trajectories, labels):
        batch_size = trajectories.size(0)
        traj_flat = trajectories.reshape(batch_size, -1)
        
        if self.normalize:
            traj_flat = F.normalize(traj_flat, p=2, dim=1)
        
        distances = torch.cdist(traj_flat, traj_flat, p=2)
        
        # Same-speaker mask
        labels_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        same_mask = (labels_row == labels_col).float()
        same_mask = same_mask - torch.eye(batch_size, device=labels.device)
        
        if self.sample_ratio < 1.0:
            sample_mask = torch.rand_like(same_mask) < self.sample_ratio
            same_mask = same_mask * sample_mask.float()
        
        num_pairs = same_mask.sum()
        if num_pairs > 0:
            return (distances * same_mask).sum() / num_pairs
        return torch.tensor(0.0, device=trajectories.device)
    
    def _embedding_desync_loss(self, embeddings, labels):
        batch_size = embeddings.size(0)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Different-speaker mask
        labels_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        diff_mask = (labels_row != labels_col).float()
        
        if self.sample_ratio < 1.0:
            sample_mask = torch.rand_like(diff_mask) < self.sample_ratio
            diff_mask = diff_mask * sample_mask.float()
        
        hinge = torch.relu(self.margin - distances)
        
        num_pairs = diff_mask.sum()
        if num_pairs > 0:
            return (hinge * diff_mask).sum() / num_pairs
        return torch.tensor(0.0, device=embeddings.device)

        
class GradientSurgery:
    """
    Option D: Gradient Surgery
    Resolve gradient conflicts by projecting conflicting gradients
    
    Reference: "Gradient Surgery for Multi-Task Learning" (Yu et al., NeurIPS 2020)
    """
    
    @staticmethod
    def pcgrad(grad1: Dict[str, torch.Tensor], 
               grad2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        PCGrad: Project Conflicting Gradients
        If gradients conflict (negative cosine similarity), project grad2 onto 
        the plane perpendicular to grad1
        
        Args:
            grad1: Gradients from loss 1
            grad2: Gradients from loss 2
            
        Returns:
            grad2_modified: Modified gradients with conflicts resolved
        """
        grad2_modified = {}
        
        for name in grad2:
            if name not in grad1 or grad1[name] is None or grad2[name] is None:
                grad2_modified[name] = grad2[name]
                continue
            
            g1 = grad1[name].flatten()
            g2 = grad2[name].flatten()
            
            # Compute dot product
            dot = torch.dot(g1, g2)
            
            if dot < 0:
                # Conflict detected: project g2 away from g1
                # g2_proj = g2 - (g2·g1 / ||g1||²) * g1
                g1_norm_sq = torch.dot(g1, g1) + 1e-8
                projection = (dot / g1_norm_sq) * g1
                g2_proj = g2 - projection
                grad2_modified[name] = g2_proj.view_as(grad2[name])
            else:
                # No conflict: keep original
                grad2_modified[name] = grad2[name]
        
        return grad2_modified
    
    @staticmethod
    def compute_conflict_stats(grad1: Dict[str, torch.Tensor],
                               grad2: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute statistics about gradient conflicts"""
        num_params = 0
        num_conflicts = 0
        total_cos_sim = 0.0
        
        for name in grad1:
            if name not in grad2 or grad1[name] is None or grad2[name] is None:
                continue
            
            g1 = grad1[name].flatten()
            g2 = grad2[name].flatten()
            
            cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            
            num_params += 1
            total_cos_sim += cos_sim
            if cos_sim < 0:
                num_conflicts += 1
        
        return {
            'num_params': num_params,
            'num_conflicts': num_conflicts,
            'conflict_ratio': num_conflicts / max(1, num_params),
            'avg_cos_sim': total_cos_sim / max(1, num_params),
        }


class SyncLossWithGradientSurgery(nn.Module):
    """
    Option D: Sync/Desync loss with gradient surgery
    Now with warmup_epochs support for H4 experiments
    """
    
    def __init__(self,
                 sync_weight: float = 0.1,
                 desync_weight: float = 0.1,
                 margin: float = 5.0,
                 sample_ratio: float = 0.3,
                 use_gradient_surgery: bool = True,
                 warmup_epochs: int = 0):
        """
        Args:
            sync_weight: Weight for sync loss
            desync_weight: Weight for desync loss
            margin: Margin for hinge loss
            sample_ratio: Ratio of pairs to sample
            use_gradient_surgery: Whether to apply gradient surgery
            warmup_epochs: Number of epochs to skip sync loss entirely (H4 experiment)
        """
        super().__init__()
        self.sync_weight = sync_weight
        self.desync_weight = desync_weight
        self.margin = margin
        self.sample_ratio = sample_ratio
        self.use_gradient_surgery = use_gradient_surgery
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.surgery = GradientSurgery()
        
        # For logging
        self.last_conflict_stats = {}
    
    def set_epoch(self, epoch: int):
        """Called by trainer at the start of each epoch"""
        self.current_epoch = epoch
    
    def is_active(self) -> bool:
        """Check if sync loss should be applied (after warmup)"""
        return self.current_epoch >= self.warmup_epochs
    
    def _compute_sync_loss(self, trajectories: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Same as ScheduledSyncLoss"""
        batch_size = trajectories.size(0)
        traj_flat = trajectories.reshape(batch_size, -1)
        distances = torch.cdist(traj_flat, traj_flat, p=2)
        
        labels_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        same_mask = (labels_row == labels_col).float() - torch.eye(batch_size, device=labels.device)
        
        if self.sample_ratio < 1.0:
            sample_mask = torch.rand_like(same_mask) < self.sample_ratio
            same_mask = same_mask * sample_mask.float()
        
        num_pairs = same_mask.sum()
        if num_pairs > 0:
            return (distances * same_mask).sum() / num_pairs
        return torch.tensor(0.0, device=trajectories.device)
    
    def _compute_desync_loss(self, trajectories: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Same as ScheduledSyncLoss"""
        batch_size = trajectories.size(0)
        traj_flat = trajectories.reshape(batch_size, -1)
        distances = torch.cdist(traj_flat, traj_flat, p=2)
        
        labels_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        diff_mask = (labels_row != labels_col).float()
        
        if self.sample_ratio < 1.0:
            sample_mask = torch.rand_like(diff_mask) < self.sample_ratio
            diff_mask = diff_mask * sample_mask.float()
        
        hinge = torch.relu(self.margin - distances)
        
        num_pairs = diff_mask.sum()
        if num_pairs > 0:
            return (hinge * diff_mask).sum() / num_pairs
        return torch.tensor(0.0, device=trajectories.device)
    
    def forward_with_surgery(self, 
                             model: nn.Module,
                             trajectories: torch.Tensor, 
                             labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with gradient surgery applied
        
        This method handles backward passes internally and modifies model.grad
        """
        device = trajectories.device
        
        # Check warmup - return zero loss if still in warmup period
        if not self.is_active():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Step 1: Compute sync loss and gradients
        model.zero_grad()
        sync_loss = self._compute_sync_loss(trajectories, labels)
        
        if self.sync_weight > 0 and sync_loss.requires_grad:
            (self.sync_weight * sync_loss).backward(retain_graph=True)
            grad_sync = {n: p.grad.clone() if p.grad is not None else None 
                        for n, p in model.named_parameters()}
        else:
            grad_sync = {}
        
        # Step 2: Compute desync loss and gradients
        model.zero_grad()
        desync_loss = self._compute_desync_loss(trajectories, labels)
        
        if self.desync_weight > 0 and desync_loss.requires_grad:
            (self.desync_weight * desync_loss).backward(retain_graph=True)
            grad_desync = {n: p.grad.clone() if p.grad is not None else None 
                         for n, p in model.named_parameters()}
        else:
            grad_desync = {}
        
        # Step 3: Apply gradient surgery if enabled
        if self.use_gradient_surgery and grad_sync and grad_desync:
            # Compute conflict stats for logging
            self.last_conflict_stats = self.surgery.compute_conflict_stats(grad_sync, grad_desync)
            
            # Project conflicting gradients
            grad_desync = self.surgery.pcgrad(grad_sync, grad_desync)
        
        # Step 4: Apply combined gradients to model
        model.zero_grad()
        for name, param in model.named_parameters():
            combined_grad = torch.zeros_like(param)
            
            if name in grad_sync and grad_sync[name] is not None:
                combined_grad += grad_sync[name]
            if name in grad_desync and grad_desync[name] is not None:
                combined_grad += grad_desync[name]
            
            param.grad = combined_grad
        
        # Return total loss for logging
        total_loss = self.sync_weight * sync_loss + self.desync_weight * desync_loss
        return total_loss
    
    def forward(self, trajectories: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard forward without surgery (for evaluation)"""
        # Check warmup
        if not self.is_active():
            return torch.tensor(0.0, device=trajectories.device, requires_grad=True)
        
        loss = torch.tensor(0.0, device=trajectories.device)
        
        if self.sync_weight > 0:
            loss = loss + self.sync_weight * self._compute_sync_loss(trajectories, labels)
        if self.desync_weight > 0:
            loss = loss + self.desync_weight * self._compute_desync_loss(trajectories, labels)
        
        return loss


# ============================================================================
# Factory function to create the appropriate loss based on config
# ============================================================================

def create_sync_loss(config: dict) -> Optional[nn.Module]:
    """
    Create sync loss module based on configuration
    
    Args:
        config: loss.synchronization config dict
        
    Returns:
        Appropriate sync loss module or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    mode = config.get('mode', 'standard')
    warmup_epochs = config.get('warmup_epochs', 0)
    
    if mode == 'hierarchical':
        return HierarchicalSyncLoss(
            sync_weight=config.get('sync_weight', 0.1),
            desync_weight=config.get('desync_weight', 0.1),
            margin=config.get('margin', 5.0),
            sample_ratio=config.get('sample_ratio', 0.3),
            warmup_epochs=warmup_epochs,
        )
    
    elif config.get('use_gradient_surgery', False):
        return SyncLossWithGradientSurgery(
            sync_weight=config.get('sync_weight', 0.1),
            desync_weight=config.get('desync_weight', 0.1),
            margin=config.get('margin', 5.0),
            sample_ratio=config.get('sample_ratio', 0.3),
            use_gradient_surgery=True,
            warmup_epochs=warmup_epochs,
        )
    
    elif config.get('schedule_mode', 'none') != 'none':
        return ScheduledSyncLoss(
            sync_weight=config.get('sync_weight', 0.1),
            desync_weight=config.get('desync_weight', 0.1),
            margin=config.get('margin', 5.0),
            sample_ratio=config.get('sample_ratio', 0.3),
            schedule_mode=config.get('schedule_mode', 'none'),
            phase1_epochs=config.get('phase1_epochs', 50),
            warmup_epochs=warmup_epochs,
        )
    
    else:
        # Standard mode with warmup support
        return ScheduledSyncLoss(
            sync_weight=config.get('sync_weight', 0.1),
            desync_weight=config.get('desync_weight', 0.1),
            margin=config.get('margin', 5.0),
            sample_ratio=config.get('sample_ratio', 0.3),
            schedule_mode='none',
            warmup_epochs=warmup_epochs,
        )


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("Testing sync_loss_extensions with warmup support...")
    
    # Test ScheduledSyncLoss with warmup
    sync_loss = ScheduledSyncLoss(
        sync_weight=0.1,
        desync_weight=0.1,
        warmup_epochs=50
    )
    
    print(f"ScheduledSyncLoss created with warmup_epochs=50")
    print(f"  Current epoch: {sync_loss.current_epoch}")
    print(f"  Is active: {sync_loss.is_active()}")
    
    sync_loss.set_epoch(25)
    print(f"After set_epoch(25):")
    print(f"  Is active: {sync_loss.is_active()}")
    print(f"  Current weights: {sync_loss.get_current_weights()}")
    
    sync_loss.set_epoch(50)
    print(f"After set_epoch(50):")
    print(f"  Is active: {sync_loss.is_active()}")
    print(f"  Current weights: {sync_loss.get_current_weights()}")
    
    # Test SyncLossWithGradientSurgery with warmup
    surgery_loss = SyncLossWithGradientSurgery(
        sync_weight=0.1,
        desync_weight=0.1,
        warmup_epochs=50
    )
    
    print(f"\nSyncLossWithGradientSurgery created with warmup_epochs=50")
    print(f"  Current epoch: {surgery_loss.current_epoch}")
    print(f"  Is active: {surgery_loss.is_active()}")
    
    surgery_loss.set_epoch(50)
    print(f"After set_epoch(50):")
    print(f"  Is active: {surgery_loss.is_active()}")
    
    # Test factory function
    config = {
        'enabled': True,
        'sync_weight': 0.1,
        'desync_weight': 0.1,
        'warmup_epochs': 50
    }
    
    loss_module = create_sync_loss(config)
    print(f"\nFactory created: {type(loss_module).__name__}")
    print(f"  warmup_epochs: {loss_module.warmup_epochs}")
    
    print("\nAll tests passed!")