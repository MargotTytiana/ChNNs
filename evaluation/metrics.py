import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class MetricsCalculator:
    """
    Comprehensive metrics calculator for speaker recognition and chaotic neural networks.
    
    Provides both standard classification metrics and specialized metrics for
    speaker recognition and chaotic system evaluation.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of speaker classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Speaker_{i}" for i in range(num_classes)]
        
        # Store results for analysis
        self.prediction_history = []
        self.target_history = []
        self.confidence_history = []
    
    def reset(self):
        """Reset stored history."""
        self.prediction_history = []
        self.target_history = []
        self.confidence_history = []
    
    def update(
        self, 
        predictions: Union[torch.Tensor, np.ndarray], 
        targets: Union[torch.Tensor, np.ndarray],
        confidences: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Model predictions (class indices or logits)
            targets: True labels
            confidences: Prediction confidences (optional)
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()
        if confidences is not None and isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().detach().numpy()
        
        # Handle logits (convert to class predictions)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            if confidences is None:
                # Extract confidences from logits/probabilities
                if np.max(predictions) > 1:  # Logits
                    probs = F.softmax(torch.from_numpy(predictions), dim=1).numpy()
                else:  # Already probabilities
                    probs = predictions
                confidences = np.max(probs, axis=1)
            predictions = np.argmax(predictions, axis=1)
        
        # Store for later analysis
        self.prediction_history.extend(predictions.tolist())
        self.target_history.extend(targets.tolist())
        if confidences is not None:
            self.confidence_history.extend(confidences.tolist())
    
    def compute_basic_metrics(
        self,
        predictions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        targets: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            predictions: Model predictions (if None, use stored history)
            targets: True labels (if None, use stored history)
            
        Returns:
            Dictionary of basic metrics
        """
        if predictions is None:
            predictions = np.array(self.prediction_history)
        if targets is None:
            targets = np.array(self.target_history)
        
        if len(predictions) == 0 or len(targets) == 0:
            return {}
        
        # Convert tensors if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Handle logits
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Precision, Recall, F1 (macro and weighted averages)
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
        
        for i in range(min(len(per_class_precision), self.num_classes)):
            metrics[f'precision_class_{i}'] = per_class_precision[i]
            metrics[f'recall_class_{i}'] = per_class_recall[i]
            metrics[f'f1_class_{i}'] = per_class_f1[i]
        
        return metrics
    
    def compute_advanced_metrics(
        self,
        predictions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        targets: Optional[Union[torch.Tensor, np.ndarray]] = None,
        logits: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute advanced classification metrics.
        
        Args:
            predictions: Model predictions
            targets: True labels  
            logits: Raw model outputs for probability-based metrics
            
        Returns:
            Dictionary of advanced metrics
        """
        if predictions is None:
            predictions = np.array(self.prediction_history)
        if targets is None:
            targets = np.array(self.target_history)
        
        if len(predictions) == 0 or len(targets) == 0:
            return {}
        
        metrics = {}
        
        # Top-k accuracy
        if logits is not None:
            if isinstance(logits, torch.Tensor):
                logits = logits.cpu().numpy()
            
            for k in [3, 5]:
                if k <= self.num_classes:
                    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
                    top_k_acc = np.mean([targets[i] in top_k_preds[i] for i in range(len(targets))])
                    metrics[f'top_{k}_accuracy'] = top_k_acc
        
        # Balanced accuracy (handles class imbalance)
        metrics['balanced_accuracy'] = self._compute_balanced_accuracy(predictions, targets)
        
        # Matthews Correlation Coefficient (for binary or multiclass)
        if self.num_classes == 2:
            metrics['mcc'] = self._compute_mcc(predictions, targets)
        
        # Confidence-based metrics
        if len(self.confidence_history) > 0:
            confidences = np.array(self.confidence_history)
            metrics['mean_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
            
            # Confidence calibration
            metrics.update(self._compute_confidence_metrics(predictions, targets, confidences))
        
        return metrics
    
    def _compute_balanced_accuracy(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """Compute balanced accuracy (average of per-class accuracies)."""
        class_accuracies = []
        
        for class_id in range(self.num_classes):
            class_mask = (targets == class_id)
            if np.sum(class_mask) == 0:
                continue
            
            class_predictions = predictions[class_mask]
            class_accuracy = np.mean(class_predictions == class_id)
            class_accuracies.append(class_accuracy)
        
        return np.mean(class_accuracies) if class_accuracies else 0.0
    
    def _compute_mcc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient."""
        try:
            cm = confusion_matrix(targets, predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                return mcc if not np.isnan(mcc) else 0.0
            else:
                # Multiclass MCC
                n = len(targets)
                cov_ytyp = 0
                cov_ypyp = 0
                cov_ytyt = 0
                
                for k in range(self.num_classes):
                    cov_ytyp += np.sum((targets == k) & (predictions == k)) * n - np.sum(targets == k) * np.sum(predictions == k)
                    cov_ypyp += np.sum(predictions == k) * n - np.sum(predictions == k) ** 2
                    cov_ytyt += np.sum(targets == k) * n - np.sum(targets == k) ** 2
                
                if cov_ypyp * cov_ytyt == 0:
                    return 0.0
                
                return cov_ytyp / np.sqrt(cov_ypyp * cov_ytyt)
        except:
            return 0.0
    
    def _compute_confidence_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray, 
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """Compute confidence-based metrics."""
        metrics = {}
        
        # Reliability (accuracy vs confidence correlation)
        correct = (predictions == targets).astype(int)
        if len(np.unique(confidences)) > 1:
            reliability, p_value = stats.pearsonr(confidences, correct)
            metrics['reliability'] = reliability if not np.isnan(reliability) else 0.0
            metrics['reliability_p_value'] = p_value if not np.isnan(p_value) else 1.0
        
        # Calibration error (Expected Calibration Error)
        ece = self._compute_ece(predictions, targets, confidences)
        metrics['expected_calibration_error'] = ece
        
        # Brier score (for probabilistic predictions)
        try:
            # Convert to one-hot targets
            one_hot_targets = np.eye(self.num_classes)[targets]
            # Use confidences as proxy for probabilities
            prob_matrix = np.zeros((len(predictions), self.num_classes))
            prob_matrix[np.arange(len(predictions)), predictions] = confidences
            # Normalize to sum to 1
            prob_matrix = prob_matrix / (prob_matrix.sum(axis=1, keepdims=True) + 1e-8)
            
            brier_score = np.mean(np.sum((prob_matrix - one_hot_targets) ** 2, axis=1))
            metrics['brier_score'] = brier_score
        except:
            pass
        
        return metrics
    
    def _compute_ece(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class SpeakerRecognitionMetrics:
    """
    Specialized metrics for speaker recognition tasks.
    
    Includes speaker verification metrics like EER, DET curves, and 
    identification metrics.
    """
    
    def __init__(self):
        self.verification_scores = []
        self.verification_labels = []
        self.identification_predictions = []
        self.identification_targets = []
    
    def update_verification(
        self,
        scores: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
    ):
        """
        Update verification metrics.
        
        Args:
            scores: Similarity/distance scores
            labels: Binary labels (1 for same speaker, 0 for different)
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        self.verification_scores.extend(scores.flatten().tolist())
        self.verification_labels.extend(labels.flatten().tolist())
    
    def update_identification(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ):
        """
        Update identification metrics.
        
        Args:
            predictions: Predicted speaker IDs
            targets: True speaker IDs
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Handle logits
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        self.identification_predictions.extend(predictions.tolist())
        self.identification_targets.extend(targets.tolist())
    
    def compute_verification_metrics(self) -> Dict[str, float]:
        """Compute speaker verification metrics."""
        if len(self.verification_scores) == 0:
            return {}
        
        scores = np.array(self.verification_scores)
        labels = np.array(self.verification_labels)
        
        metrics = {}
        
        # ROC curve and AUC
        try:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc
            
            # Equal Error Rate (EER)
            eer, eer_threshold = self._compute_eer(fpr, tpr, thresholds)
            metrics['eer'] = eer
            metrics['eer_threshold'] = eer_threshold
            
            # Detection Cost Function (DCF)
            dcf = self._compute_dcf(scores, labels)
            metrics['min_dcf'] = dcf
            
        except Exception as e:
            print(f"Warning: Could not compute verification metrics: {e}")
        
        return metrics
    
    def compute_identification_metrics(self) -> Dict[str, float]:
        """Compute speaker identification metrics."""
        if len(self.identification_predictions) == 0:
            return {}
        
        predictions = np.array(self.identification_predictions)
        targets = np.array(self.identification_targets)
        
        metrics = {}
        
        # Identification accuracy
        metrics['identification_accuracy'] = accuracy_score(targets, predictions)
        
        # Rank-based metrics
        metrics.update(self._compute_rank_metrics(predictions, targets))
        
        return metrics
    
    def _compute_eer(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray, 
        thresholds: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Equal Error Rate."""
        try:
            fnr = 1 - tpr
            # Find where FPR and FNR intersect
            eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer = 1. - interp1d(fpr, tpr)(eer_threshold)
            
            # Find corresponding threshold
            threshold_idx = np.argmin(np.abs(fpr + fnr - 1))
            threshold = thresholds[threshold_idx]
            
            return eer, threshold
        except:
            # Fallback method
            fnr = 1 - tpr
            diff = np.abs(fpr - fnr)
            min_idx = np.argmin(diff)
            eer = (fpr[min_idx] + fnr[min_idx]) / 2
            threshold = thresholds[min_idx] if min_idx < len(thresholds) else 0.0
            return eer, threshold
    
    def _compute_dcf(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0
    ) -> float:
        """Compute minimum Detection Cost Function."""
        try:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            fnr = 1 - tpr
            
            # Compute DCF for each threshold
            dcf_values = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
            
            # Return minimum DCF
            return np.min(dcf_values)
        except:
            return 1.0  # Maximum possible DCF
    
    def _compute_rank_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute rank-based identification metrics."""
        metrics = {}
        
        # For rank metrics, we need the full score matrix
        # This is a simplified version using only top-1 predictions
        
        # Rank-1 accuracy (same as identification accuracy)
        metrics['rank_1_accuracy'] = accuracy_score(targets, predictions)
        
        return metrics


class ChaoticMetrics:
    """
    Specialized metrics for chaotic neural network evaluation.
    
    Includes metrics for assessing chaotic dynamics quality,
    attractor properties, and system stability.
    """
    
    def __init__(self):
        self.trajectories = []
        self.attractors = []
        self.lyapunov_exponents = []
        self.embedding_quality = []
    
    def update_trajectories(self, trajectories: Union[torch.Tensor, np.ndarray]):
        """Update with new chaotic trajectories."""
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.cpu().numpy()
        
        self.trajectories.append(trajectories)
    
    def update_attractors(self, attractor_features: Union[torch.Tensor, np.ndarray]):
        """Update with attractor pooling features."""
        if isinstance(attractor_features, torch.Tensor):
            attractor_features = attractor_features.cpu().numpy()
        
        self.attractors.append(attractor_features)
    
    def compute_trajectory_metrics(self) -> Dict[str, float]:
        """Compute metrics for chaotic trajectory quality."""
        if len(self.trajectories) == 0:
            return {}
        
        metrics = {}
        all_trajectories = np.concatenate(self.trajectories, axis=0)
        
        # Trajectory stability
        metrics['trajectory_stability'] = self._compute_trajectory_stability(all_trajectories)
        
        # Attractor dimension estimation
        metrics['estimated_dimension'] = self._estimate_attractor_dimension(all_trajectories)
        
        # Trajectory diversity
        metrics['trajectory_diversity'] = self._compute_trajectory_diversity(all_trajectories)
        
        # Phase space coverage
        metrics['phase_space_coverage'] = self._compute_phase_space_coverage(all_trajectories)
        
        return metrics
    
    def compute_attractor_metrics(self) -> Dict[str, float]:
        """Compute metrics for attractor quality."""
        if len(self.attractors) == 0:
            return {}
        
        metrics = {}
        all_attractors = np.concatenate(self.attractors, axis=0)
        
        # Feature diversity
        metrics['attractor_diversity'] = np.mean(np.std(all_attractors, axis=0))
        
        # Feature separability
        metrics['feature_separability'] = self._compute_feature_separability(all_attractors)
        
        # Topological consistency
        metrics['topological_consistency'] = self._compute_topological_consistency(all_attractors)
        
        return metrics
    
    def _compute_trajectory_stability(self, trajectories: np.ndarray) -> float:
        """Compute stability of chaotic trajectories."""
        try:
            # Compute variance of trajectory points
            if len(trajectories.shape) == 3:  # [batch, time, dim]
                stability = np.mean(np.var(trajectories, axis=1))  # Variance across time
            else:
                stability = np.var(trajectories)
            
            return float(stability)
        except:
            return 0.0
    
    def _estimate_attractor_dimension(self, trajectories: np.ndarray) -> float:
        """Estimate the fractal dimension of attractors."""
        try:
            if len(trajectories.shape) != 3:
                return 3.0  # Default dimension
            
            # Simple box-counting dimension estimate
            batch_size, time_steps, dim = trajectories.shape
            
            # Flatten spatial dimensions
            points = trajectories.reshape(-1, dim)
            
            # Compute correlation dimension estimate
            n_samples = min(1000, len(points))  # Limit for computation efficiency
            sample_points = points[np.random.choice(len(points), n_samples, replace=False)]
            
            # Compute pairwise distances
            distances = np.linalg.norm(
                sample_points[:, np.newaxis] - sample_points[np.newaxis, :], 
                axis=2
            )
            
            # Remove diagonal
            distances = distances[np.triu_indices_from(distances, k=1)]
            
            if len(distances) > 0:
                # Use median distance as characteristic scale
                char_scale = np.median(distances)
                # Simple dimension estimate
                estimated_dim = min(dim, np.log(len(distances)) / np.log(1 + 1/char_scale))
                return max(1.0, float(estimated_dim))
            
        except:
            pass
        
        return 3.0  # Default for 3D systems
    
    def _compute_trajectory_diversity(self, trajectories: np.ndarray) -> float:
        """Compute diversity of trajectory patterns."""
        try:
            if len(trajectories.shape) != 3:
                return 0.0
            
            batch_size = trajectories.shape[0]
            if batch_size < 2:
                return 0.0
            
            # Compute pairwise trajectory distances
            diversities = []
            for i in range(min(batch_size, 50)):  # Limit computation
                for j in range(i + 1, min(batch_size, 50)):
                    traj_i = trajectories[i]
                    traj_j = trajectories[j]
                    
                    # Dynamic Time Warping distance (simplified)
                    distance = np.mean(np.linalg.norm(traj_i - traj_j, axis=1))
                    diversities.append(distance)
            
            return float(np.mean(diversities)) if diversities else 0.0
            
        except:
            return 0.0
    
    def _compute_phase_space_coverage(self, trajectories: np.ndarray) -> float:
        """Compute how well trajectories cover the phase space."""
        try:
            if len(trajectories.shape) != 3:
                return 0.0
            
            # Flatten all points
            all_points = trajectories.reshape(-1, trajectories.shape[-1])
            
            # Compute convex hull volume (simplified)
            ranges = np.ptp(all_points, axis=0)  # Range in each dimension
            volume = np.prod(ranges)
            
            # Normalize by number of points
            normalized_coverage = volume / len(all_points)
            
            return float(normalized_coverage)
            
        except:
            return 0.0
    
    def _compute_feature_separability(self, features: np.ndarray) -> float:
        """Compute separability of attractor features."""
        try:
            if len(features) < 2:
                return 0.0
            
            # Compute within-class vs between-class distances
            # This is simplified without class labels
            distances = []
            n_samples = min(len(features), 100)
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(features[i] - features[j])
                    distances.append(dist)
            
            return float(np.mean(distances)) if distances else 0.0
            
        except:
            return 0.0
    
    def _compute_topological_consistency(self, features: np.ndarray) -> float:
        """Compute topological consistency of features."""
        try:
            if len(features) < 3:
                return 0.0
            
            # Measure consistency in local neighborhoods
            consistencies = []
            n_samples = min(len(features), 50)
            
            for i in range(n_samples):
                # Find k nearest neighbors
                distances = np.linalg.norm(features - features[i], axis=1)
                k_nearest = np.argsort(distances)[1:6]  # Exclude self
                
                # Measure variance in neighborhood
                neighborhood = features[k_nearest]
                variance = np.mean(np.var(neighborhood, axis=0))
                consistencies.append(1.0 / (1.0 + variance))  # Higher consistency = lower variance
            
            return float(np.mean(consistencies))
            
        except:
            return 0.0


class StatisticalAnalyzer:
    """
    Statistical analysis tools for comparing different methods.
    """
    
    @staticmethod
    def compute_confidence_interval(
        data: Union[List[float], np.ndarray],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence interval for data.
        
        Args:
            data: Data points
            confidence: Confidence level
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        data = np.array(data)
        mean = np.mean(data)
        std_error = stats.sem(data)
        
        alpha = 1 - confidence
        degrees_freedom = len(data) - 1
        t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
        
        margin_error = t_critical * std_error
        
        return mean, mean - margin_error, mean + margin_error
    
    @staticmethod
    def perform_t_test(
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray],
        paired: bool = False
    ) -> Dict[str, float]:
        """
        Perform t-test between two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            paired: Whether to perform paired t-test
            
        Returns:
            Dictionary with test results
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
        else:
            statistic, p_value = stats.ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float(cohens_d),
            'group1_mean': float(np.mean(group1)),
            'group2_mean': float(np.mean(group2))
        }
    
    @staticmethod
    def perform_anova(groups: List[Union[List[float], np.ndarray]]) -> Dict[str, float]:
        """
        Perform one-way ANOVA test.
        
        Args:
            groups: List of groups to compare
            
        Returns:
            Dictionary with ANOVA results
        """
        groups = [np.array(group) for group in groups]
        
        statistic, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'num_groups': len(groups)
        }
    
    @staticmethod
    def compute_effect_size(
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray]
    ) -> float:
        """Compute Cohen's d effect size."""
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std


class VisualizationHelper:
    """
    Helper class for creating evaluation visualizations.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = 'Confusion Matrix'
    ) -> Optional[str]:
        """Plot confusion matrix."""
        if not PLOTTING_AVAILABLE:
            return None
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.save_dir:
            save_path = f"{self.save_dir}/confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return None
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = 'ROC Curve'
    ) -> Optional[str]:
        """Plot ROC curve."""
        if not PLOTTING_AVAILABLE:
            return None
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if self.save_dir:
            save_path = f"{self.save_dir}/roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return None
    
    def plot_metrics_comparison(
        self,
        methods: List[str],
        metrics: Dict[str, List[float]],
        title: str = 'Methods Comparison'
    ) -> Optional[str]:
        """Plot comparison of different methods."""
        if not PLOTTING_AVAILABLE:
            return None
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            axes[i].bar(methods, values)
            axes[i].set_title(f'{metric_name}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = f"{self.save_dir}/methods_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return None


# Convenience function for comprehensive evaluation
def evaluate_model_comprehensive(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    logits: Optional[Union[torch.Tensor, np.ndarray]] = None,
    confidences: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    include_advanced: bool = True,
    include_speaker_metrics: bool = True,
    verification_scores: Optional[np.ndarray] = None,
    verification_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with all available metrics.
    
    Args:
        predictions: Model predictions
        targets: True labels
        logits: Raw model outputs (optional)
        confidences: Prediction confidences (optional)
        num_classes: Number of classes
        class_names: Class names (optional)
        include_advanced: Whether to include advanced metrics
        include_speaker_metrics: Whether to include speaker-specific metrics
        verification_scores: Speaker verification scores (optional)
        verification_labels: Speaker verification labels (optional)
        
    Returns:
        Dictionary of all computed metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Infer number of classes if not provided
    if num_classes is None:
        if len(predictions.shape) > 1:
            num_classes = predictions.shape[1]
        else:
            num_classes = len(np.unique(targets))
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(predictions, targets, confidences)
    
    # Compute all metrics
    all_metrics = {}
    
    # Basic metrics
    all_metrics.update(calculator.compute_basic_metrics())
    
    # Advanced metrics
    if include_advanced:
        all_metrics.update(calculator.compute_advanced_metrics(logits=logits))
    
    # Speaker recognition metrics
    if include_speaker_metrics:
        speaker_metrics = SpeakerRecognitionMetrics()
        
        # Identification metrics
        speaker_metrics.update_identification(predictions, targets)
        all_metrics.update(speaker_metrics.compute_identification_metrics())
        
        # Verification metrics (if available)
        if verification_scores is not None and verification_labels is not None:
            speaker_metrics.update_verification(verification_scores, verification_labels)
            all_metrics.update(speaker_metrics.compute_verification_metrics())
    
    return all_metrics


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Evaluation Metrics...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 10
    
    # Synthetic predictions and targets
    targets = np.random.randint(0, n_classes, n_samples)
    logits = np.random.randn(n_samples, n_classes)
    
    # Add some signal to make predictions somewhat realistic
    for i in range(n_samples):
        logits[i, targets[i]] += 2.0  # Boost correct class
    
    predictions = np.argmax(logits, axis=1)
    confidences = np.max(logits, axis=1)
    
    # Test basic metrics
    calculator = MetricsCalculator(n_classes)
    calculator.update(predictions, targets, confidences)
    
    basic_metrics = calculator.compute_basic_metrics()
    print("Basic Metrics:")
    for metric, value in basic_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test advanced metrics
    advanced_metrics = calculator.compute_advanced_metrics(logits=logits)
    print("\nAdvanced Metrics:")
    for metric, value in advanced_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test speaker recognition metrics
    speaker_metrics = SpeakerRecognitionMetrics()
    
    # Synthetic verification data
    verification_scores = np.random.randn(500)
    verification_labels = np.random.randint(0, 2, 500)
    
    speaker_metrics.update_verification(verification_scores, verification_labels)
    speaker_metrics.update_identification(predictions, targets)
    
    verif_metrics = speaker_metrics.compute_verification_metrics()
    ident_metrics = speaker_metrics.compute_identification_metrics()
    
    print("\nSpeaker Verification Metrics:")
    for metric, value in verif_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nSpeaker Identification Metrics:")
    for metric, value in ident_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test comprehensive evaluation
    print("\nComprehensive Evaluation:")
    all_metrics = evaluate_model_comprehensive(
        predictions=predictions,
        targets=targets,
        num_classes=self.config['num_speakers'],  # 确保传递这个参数
        logits=logits,
        confidences=confidences,
        verification_scores=verification_scores,
        verification_labels=verification_labels
    )
    
    print(f"Total metrics computed: {len(all_metrics)}")
    
    # Test statistical analysis
    print("\nStatistical Analysis:")
    group1 = np.random.normal(0.85, 0.05, 20)  # Baseline method
    group2 = np.random.normal(0.90, 0.04, 20)  # Chaotic method
    
    t_test_result = StatisticalAnalyzer.perform_t_test(group2, group1)
    print(f"Chaotic vs Baseline t-test: p-value = {t_test_result['p_value']:.6f}")
    print(f"Significant improvement: {t_test_result['significant']}")
    print(f"Effect size (Cohen's d): {t_test_result['effect_size']:.3f}")
    
    mean, lower, upper = StatisticalAnalyzer.compute_confidence_interval(group2)
    print(f"Chaotic method 95% CI: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
    
    print("\nMetrics testing completed successfully!")