import torch
import math
from typing import Dict, Any, Optional
import numpy as np


class DynamicLossScheduler:
    """
    Dynamic loss scheduler for LLM-guided MoE training.
    Implements adaptive weight scheduling for different loss components
    following the specifications in the implementation guide.
    """

    def __init__(self, total_epochs: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dynamic loss scheduler.

        Args:
            total_epochs: Total number of training epochs
            config: Configuration dictionary with scheduler parameters
        """
        self.total_epochs = total_epochs
        self.config = config or {}

        # Default configuration
        self.default_config = {
            'cls_weight_schedule': 'stable',  # 'stable', 'decay', 'adaptive'
            'consistency_warmup_ratio': 0.3,  # Warmup ratio for consistency loss
            'diversity_cycle_length': 20,     # Cycle length for diversity loss
            'diversity_base_weight': 0.01,    # Base weight for diversity loss
            'diversity_max_weight': 0.05,     # Maximum weight for diversity loss
            'gate_weight_schedule': 'constant',  # 'constant', 'adaptive'
            'select_weight_schedule': 'adaptive',  # 'constant', 'adaptive', 'decay'
        }

        # Merge with provided config
        self.scheduler_config = {**self.default_config, **self.config}

        # Compute derived parameters
        self.consistency_warmup_epochs = int(
            self.scheduler_config['consistency_warmup_ratio'] * total_epochs
        )

    def get_cls_weight(self, epoch: int) -> float:
        """
        Get classification loss weight for given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Classification loss weight
        """
        schedule = self.scheduler_config['cls_weight_schedule']

        if schedule == 'stable':
            return 1.0 if epoch < 50 else 0.8
        elif schedule == 'decay':
            # Linear decay after epoch 100
            if epoch < 100:
                return 1.0
            else:
                return 1.0 - 0.3 * (epoch - 100) / (self.total_epochs - 100)
        elif schedule == 'adaptive':
            # Adaptive based on training progress
            progress = epoch / self.total_epochs
            if progress < 0.3:
                return 1.0
            elif progress < 0.7:
                return 0.9
            else:
                return 0.8
        else:
            return 1.0

    def get_consistency_weight(self, epoch: int) -> float:
        """
        Get consistency loss weight for given epoch.
        Implements progressive increase with cosine scheduling.

        Args:
            epoch: Current epoch

        Returns:
            Consistency loss weight
        """
        if epoch < self.consistency_warmup_epochs:
            # Linear warmup
            progress = epoch / self.consistency_warmup_epochs
            return 0.1 + 0.9 * progress
        else:
            # Cosine scheduling after warmup
            remaining_epochs = self.total_epochs - self.consistency_warmup_epochs
            progress = (epoch - self.consistency_warmup_epochs) / remaining_epochs
            cosine_weight = 0.5 * (1 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cosine_weight

    def get_diversity_weight(self, epoch: int) -> float:
        """
        Get diversity loss weight for given epoch.
        Implements periodic enhancement as specified in the guide.

        Args:
            epoch: Current epoch

        Returns:
            Diversity loss weight
        """
        cycle_length = self.scheduler_config['diversity_cycle_length']
        base_weight = self.scheduler_config['diversity_base_weight']
        max_weight = self.scheduler_config['diversity_max_weight']

        cycle_progress = (epoch % cycle_length) / cycle_length

        if cycle_progress < 0.5:
            # Increasing phase
            return base_weight + (max_weight - base_weight) * (2 * cycle_progress)
        else:
            # Decreasing phase
            return base_weight + (max_weight - base_weight) * (2 * (1 - cycle_progress))

    def get_select_weight(self, epoch: int) -> float:
        """
        Get selection loss weight for given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Selection loss weight
        """
        schedule = self.scheduler_config['select_weight_schedule']

        if schedule == 'constant':
            return 1.0
        elif schedule == 'adaptive':
            # Adaptive based on training stage
            progress = epoch / self.total_epochs
            if progress < 0.2:
                return 0.5  # Lower weight early in training
            elif progress < 0.8:
                return 1.0  # Full weight during main training
            else:
                return 0.7  # Slightly reduced weight at the end
        elif schedule == 'decay':
            # Linear decay throughout training
            return 1.0 - 0.5 * (epoch / self.total_epochs)
        else:
            return 1.0

    def get_gate_weight(self, epoch: int) -> float:
        """
        Get gate loss weight for given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Gate loss weight
        """
        schedule = self.scheduler_config['gate_weight_schedule']

        if schedule == 'constant':
            return 1.0
        elif schedule == 'adaptive':
            # Higher weight when experts are likely to be unbalanced
            progress = epoch / self.total_epochs
            if progress < 0.3:
                return 1.2  # Higher weight early to encourage balanced expert usage
            elif progress < 0.7:
                return 1.0
            else:
                return 0.8  # Lower weight later when experts are specialized
        else:
            return 1.0

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get all loss weights for the given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Dictionary with loss weights
        """
        return {
            'cls_weight': self.get_cls_weight(epoch),
            'consistency_weight': self.get_consistency_weight(epoch),
            'diversity_weight': self.get_diversity_weight(epoch),
            'select_weight': self.get_select_weight(epoch),
            'gate_weight': self.get_gate_weight(epoch)
        }

    def compute_total_loss(self, epoch: int, loss_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute total weighted loss for the given epoch.

        Args:
            epoch: Current epoch
            loss_components: Dictionary with individual loss components

        Returns:
            Total weighted loss
        """
        weights = self.get_loss_weights(epoch)
        total_loss = torch.tensor(0.0, device=next(iter(loss_components.values())).device)

        for loss_name, loss_value in loss_components.items():
            weight_key = f"{loss_name}_weight"
            if weight_key in weights:
                total_loss += weights[weight_key] * loss_value
            else:
                # Default weight if not specified
                total_loss += loss_value

        return total_loss


class AdaptiveLossScheduler(DynamicLossScheduler):
    """
    Advanced adaptive loss scheduler that adjusts weights based on training dynamics.
    """

    def __init__(self, total_epochs: int, config: Optional[Dict[str, Any]] = None,
                 window_size: int = 10):
        """
        Initialize adaptive scheduler.

        Args:
            total_epochs: Total number of training epochs
            config: Configuration dictionary
            window_size: Window size for monitoring loss trends
        """
        super().__init__(total_epochs, config)
        self.window_size = window_size

        # Track loss history for adaptive adjustment
        self.loss_history = {
            'cls': [],
            'consistency': [],
            'diversity': [],
            'select': [],
            'gate': []
        }

        # Adaptive parameters
        self.adaptive_config = {
            'cls_stability_threshold': 0.1,
            'consistency_convergence_threshold': 0.05,
            'diversity_improvement_threshold': 0.01,
            'balance_tolerance': 0.3
        }

    def update_loss_history(self, epoch: int, loss_components: Dict[str, float]):
        """
        Update loss history with current epoch losses.

        Args:
            epoch: Current epoch
            loss_components: Dictionary with loss values
        """
        for loss_name, loss_value in loss_components.items():
            if loss_name in self.loss_history:
                self.loss_history[loss_name].append(loss_value)
                # Keep only recent history
                if len(self.loss_history[loss_name]) > self.window_size:
                    self.loss_history[loss_name].pop(0)

    def is_loss_stable(self, loss_name: str) -> bool:
        """
        Check if a loss component is stable based on recent history.

        Args:
            loss_name: Name of the loss component

        Returns:
            True if loss is stable, False otherwise
        """
        if loss_name not in self.loss_history:
            return False

        history = self.loss_history[loss_name]
        if len(history) < self.window_size // 2:
            return False

        # Compute variance of recent losses
        recent_losses = history[-self.window_size // 2:]
        variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)

        # Check stability based on relative variance
        if mean_loss > 0:
            relative_variance = variance / (mean_loss ** 2)
            threshold = self.adaptive_config['cls_stability_threshold']
            return relative_variance < threshold

        return True

    def get_adaptive_cls_weight(self, epoch: int) -> float:
        """
        Get adaptive classification loss weight.

        Args:
            epoch: Current epoch

        Returns:
            Adaptive classification loss weight
        """
        base_weight = self.get_cls_weight(epoch)

        if self.is_loss_stable('cls'):
            # Reduce weight if classification loss is stable
            return base_weight * 0.9
        else:
            # Increase weight if classification loss is unstable
            return base_weight * 1.1

    def get_adaptive_consistency_weight(self, epoch: int) -> float:
        """
        Get adaptive consistency loss weight.

        Args:
            epoch: Current epoch

        Returns:
            Adaptive consistency loss weight
        """
        base_weight = self.get_consistency_weight(epoch)

        # Check if consistency is converging
        if 'consistency' in self.loss_history and len(self.loss_history['consistency']) >= 3:
            recent_losses = self.loss_history['consistency'][-3:]
            if abs(recent_losses[-1] - recent_losses[0]) < self.adaptive_config['consistency_convergence_threshold']:
                # Reduce weight if consistency is converging
                return base_weight * 0.8

        return base_weight

    def get_adaptive_diversity_weight(self, epoch: int) -> float:
        """
        Get adaptive diversity loss weight.

        Args:
            epoch: Current epoch

        Returns:
            Adaptive diversity loss weight
        """
        base_weight = self.get_diversity_weight(epoch)

        # Check if diversity is improving
        if 'diversity' in self.loss_history and len(self.loss_history['diversity']) >= 2:
            recent_losses = self.loss_history['diversity'][-2:]
            improvement = recent_losses[-1] - recent_losses[0]

            if improvement < -self.adaptive_config['diversity_improvement_threshold']:
                # Increase weight if diversity is improving significantly
                return base_weight * 1.2
            elif improvement > self.adaptive_config['diversity_improvement_threshold']:
                # Decrease weight if diversity is getting worse
                return base_weight * 0.8

        return base_weight

    def check_loss_balance(self) -> Dict[str, bool]:
        """
        Check if different loss components are balanced.

        Returns:
            Dictionary indicating balance status for each loss pair
        """
        balance_status = {}

        # Check classification vs other losses
        if 'cls' in self.loss_history and len(self.loss_history['cls']) > 0:
            cls_mean = np.mean(self.loss_history['cls'])

            for other_loss in ['consistency', 'diversity', 'select']:
                if other_loss in self.loss_history and len(self.loss_history[other_loss]) > 0:
                    other_mean = np.mean(self.loss_history[other_loss])
                    ratio = abs(cls_mean - other_mean) / (cls_mean + 1e-8)
                    balance_status[f'cls_vs_{other_loss}'] = ratio < self.adaptive_config['balance_tolerance']

        return balance_status

    def get_adaptive_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get all adaptive loss weights for the given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Dictionary with adaptive loss weights
        """
        return {
            'cls_weight': self.get_adaptive_cls_weight(epoch),
            'consistency_weight': self.get_adaptive_consistency_weight(epoch),
            'diversity_weight': self.get_adaptive_diversity_weight(epoch),
            'select_weight': self.get_select_weight(epoch),
            'gate_weight': self.get_gate_weight(epoch)
        }