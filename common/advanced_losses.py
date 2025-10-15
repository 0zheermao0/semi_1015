import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) loss for expert selection.
    Implements pairwise preference learning from LLM rankings.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize DPO loss.

        Args:
            alpha: Temperature parameter for controlling preference strength
        """
        super(DPOLoss, self).__init__()
        self.alpha = alpha
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate_logits: torch.Tensor, preferences: List[Tuple[int, int]],
                node_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute DPO loss from expert preferences.

        Args:
            gate_logits: Gate logits for all nodes [num_nodes, num_experts]
            preferences: List of preference pairs (preferred_expert, less_preferred_expert)
            node_indices: Indices of nodes for which preferences are provided

        Returns:
            DPO loss value
        """
        if len(preferences) == 0 or len(node_indices) == 0:
            return torch.tensor(0.0, device=gate_logits.device, requires_grad=True)

        # Get gate logits for specific nodes
        node_gate_logits = gate_logits[node_indices]  # [len(node_indices), num_experts]

        total_loss = torch.tensor(0.0, device=gate_logits.device)
        num_pairs = len(preferences)

        for preferred_expert, less_preferred_expert in preferences:
            # Get logits for the preference pair
            preferred_logits = node_gate_logits[:, preferred_expert]  # [len(node_indices)]
            less_preferred_logits = node_gate_logits[:, less_preferred_expert]

            # Compute pairwise preference loss
            # We want preferred_expert to have higher logits than less_preferred_expert
            preference_logits = preferred_logits - less_preferred_logits  # [len(node_indices)]
            preference_probs = self.sigmoid(self.alpha * preference_logits)

            # DPO loss: -log(probability of correct preference)
            pair_loss = -torch.log(preference_probs + 1e-8).mean()
            total_loss += pair_loss

        return total_loss / max(num_pairs, 1)

    def compute_preference_accuracy(self, gate_logits: torch.Tensor, preferences: List[Tuple[int, int]],
                                   node_indices: torch.Tensor) -> float:
        """
        Compute preference accuracy for evaluation.

        Args:
            gate_logits: Gate logits for all nodes [num_nodes, num_experts]
            preferences: List of preference pairs
            node_indices: Indices of nodes for which preferences are provided

        Returns:
            Preference accuracy
        """
        if len(preferences) == 0 or len(node_indices) == 0:
            return 0.0

        node_gate_logits = gate_logits[node_indices]
        correct_predictions = 0
        total_predictions = 0

        for preferred_expert, less_preferred_expert in preferences:
            preferred_logits = node_gate_logits[:, preferred_expert]
            less_preferred_logits = node_gate_logits[:, less_preferred_expert]

            # Check if preferred expert has higher logits
            correct_predictions += (preferred_logits > less_preferred_logits).sum().item()
            total_predictions += len(node_indices)

        return correct_predictions / max(total_predictions, 1)


class EnhancedConsistencyLoss(nn.Module):
    """
    Enhanced consistency loss using KL divergence between expert predictions.
    Encourages consistency between experts that are ranked similarly by LLM.
    """

    def __init__(self, beta: float = 0.1, temperature: float = 1.0):
        """
        Initialize enhanced consistency loss.

        Args:
            beta: Weight for consistency loss
            temperature: Temperature for softmax computation
        """
        super(EnhancedConsistencyLoss, self).__init__()
        self.beta = beta
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, expert_outputs: torch.Tensor, rankings: Dict[int, List[int]],
                node_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced consistency loss based on LLM rankings.

        Args:
            expert_outputs: Expert outputs [num_nodes, num_experts, feature_dim]
            rankings: Dictionary mapping node_id to expert rankings
            node_indices: Indices of nodes with rankings

        Returns:
            Enhanced consistency loss
        """
        if len(rankings) == 0:
            return torch.tensor(0.0, device=expert_outputs.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=expert_outputs.device)
        num_nodes_considered = 0

        for i, node_idx in enumerate(node_indices):
            if node_idx.item() not in rankings:
                continue

            node_rankings = rankings[node_idx.item()]
            if len(node_rankings) < 2:
                continue

            # Get expert outputs for this node
            node_expert_outputs = expert_outputs[node_idx]  # [num_experts, feature_dim]

            # Create consistency pairs based on rankings
            consistency_pairs = []
            for j in range(len(node_rankings) - 1):
                for k in range(j + 1, len(node_rankings)):
                    expert_i = node_rankings[j]
                    expert_k = node_rankings[k]
                    if expert_i < expert_outputs.shape[1] and expert_k < expert_outputs.shape[1]:
                        consistency_pairs.append((expert_i, expert_k))

            # Compute consistency loss for each pair
            for expert_i, expert_k in consistency_pairs:
                output_i = node_expert_outputs[expert_i].unsqueeze(0)  # [1, feature_dim]
                output_k = node_expert_outputs[expert_k].unsqueeze(0)  # [1, feature_dim]

                # Convert to probability distributions using softmax
                prob_i = F.softmax(output_i / self.temperature, dim=-1)
                prob_k = F.softmax(output_k / self.temperature, dim=-1)

                # Compute KL divergence (symmetric)
                kl_loss = self.kl_div(F.log_softmax(output_i / self.temperature, dim=-1), prob_k)
                kl_loss += self.kl_div(F.log_softmax(output_k / self.temperature, dim=-1), prob_i)

                total_loss += kl_loss * 0.5  # Average of symmetric KL
                num_nodes_considered += 1

        if num_nodes_considered == 0:
            return torch.tensor(0.0, device=expert_outputs.device, requires_grad=True)

        return self.beta * total_loss / num_nodes_considered


class MMDDiversityLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) based diversity loss.
    Encourages diversity between expert representations by maximizing MMD distance.
    """

    def __init__(self, gamma: float = 1.0, device: str = 'cpu'):
        """
        Initialize MMD diversity loss.

        Args:
            gamma: RBF kernel bandwidth parameter
            device: Device for computation
        """
        super(MMDDiversityLoss, self).__init__()
        self.gamma = gamma
        self.device = device

    def rbf_kernel_torch(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute RBF kernel matrix in PyTorch.

        Args:
            X: Input tensor [n_samples, n_features]
            Y: Optional second input tensor [m_samples, n_features]

        Returns:
            Kernel matrix [n_samples, m_samples] or [n_samples, n_samples] if Y is None
        """
        if Y is None:
            Y = X

        # Compute pairwise squared Euclidean distances
        X_norm = torch.sum(X ** 2, dim=1).view(-1, 1)
        Y_norm = torch.sum(Y ** 2, dim=1).view(1, -1)
        squared_dist = X_norm + Y_norm - 2 * torch.mm(X, Y.t())

        # Apply RBF kernel
        K = torch.exp(-self.gamma * squared_dist)
        return K

    def compute_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between two sets of samples.

        Args:
            X: First set of samples [n_samples, n_features]
            Y: Second set of samples [m_samples, n_features]

        Returns:
            MMD value
        """
        K_XX = self.rbf_kernel_torch(X)
        K_YY = self.rbf_kernel_torch(Y)
        K_XY = self.rbf_kernel_torch(X, Y)

        m = X.shape[0]
        n = Y.shape[0]

        # Unbiased MMD estimator
        mmd_term_XX = (torch.sum(K_XX) - torch.trace(K_XX)) / (m * (m - 1))
        mmd_term_YY = (torch.sum(K_YY) - torch.trace(K_YY)) / (n * (n - 1))
        mmd_term_XY = torch.sum(K_XY) / (m * n)

        mmd_squared = mmd_term_XX + mmd_term_YY - 2 * mmd_term_XY
        return torch.sqrt(torch.clamp(mmd_squared, min=0.0))

    def forward(self, expert_outputs: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD diversity loss.

        Args:
            expert_outputs: Expert outputs [num_nodes, num_experts, feature_dim]
            gates: Gating weights [num_nodes, num_experts]

        Returns:
            MMD diversity loss
        """
        num_nodes, num_experts, feature_dim = expert_outputs.shape
        total_diversity_loss = torch.tensor(0.0, device=expert_outputs.device)

        # Compute MMD between all pairs of experts
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # Get activated nodes for each expert
                mask_i = gates[:, i] > 0
                mask_j = gates[:, j] > 0

                if not mask_i.any() or not mask_j.any():
                    continue

                # Get representations for activated nodes
                expert_i_repr = expert_outputs[mask_i, i]  # [n_i, feature_dim]
                expert_j_repr = expert_outputs[mask_j, j]  # [n_j, feature_dim]

                # Skip if either expert has too few activated nodes
                if expert_i_repr.shape[0] < 2 or expert_j_repr.shape[0] < 2:
                    continue

                # Compute MMD between expert representations
                mmd_distance = self.compute_mmd(expert_i_repr, expert_j_repr)

                # We want to maximize MMD (encourage diversity), so minimize negative MMD
                total_diversity_loss -= mmd_distance

        # Normalize by number of expert pairs
        num_pairs = num_experts * (num_experts - 1) // 2
        if num_pairs > 0:
            total_diversity_loss = total_diversity_loss / num_pairs

        return total_diversity_loss

    def compute_expert_diversity_metrics(self, expert_outputs: torch.Tensor,
                                        gates: torch.Tensor) -> Dict[str, float]:
        """
        Compute additional diversity metrics for analysis.

        Args:
            expert_outputs: Expert outputs [num_nodes, num_experts, feature_dim]
            gates: Gating weights [num_nodes, num_experts]

        Returns:
            Dictionary with diversity metrics
        """
        num_nodes, num_experts, feature_dim = expert_outputs.shape

        # Average pairwise MMD
        total_mmd = 0.0
        num_valid_pairs = 0

        mmd_distances = []

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                mask_i = gates[:, i] > 0
                mask_j = gates[:, j] > 0

                if not mask_i.any() or not mask_j.any():
                    continue

                expert_i_repr = expert_outputs[mask_i, i]
                expert_j_repr = expert_outputs[mask_j, j]

                if expert_i_repr.shape[0] < 2 or expert_j_repr.shape[0] < 2:
                    continue

                mmd_dist = self.compute_mmd(expert_i_repr, expert_j_repr)
                mmd_distances.append(mmd_dist.item())
                total_mmd += mmd_dist.item()
                num_valid_pairs += 1

        avg_mmd = total_mmd / max(num_valid_pairs, 1)
        std_mmd = np.std(mmd_distances) if mmd_distances else 0.0

        # Expert activation diversity
        expert_activations = (gates > 0).float().sum(dim=0)  # [num_experts]
        activation_entropy = -torch.sum(
            (expert_activations / expert_activations.sum()) *
            torch.log(expert_activations / expert_activations.sum() + 1e-8)
        ).item()

        return {
            'avg_pairwise_mmd': avg_mmd,
            'std_pairwise_mmd': std_mmd,
            'activation_entropy': activation_entropy,
            'num_valid_pairs': num_valid_pairs
        }


class GateEntropyRegularizer(nn.Module):
    """
    Entropy regularizer for gate distributions to prevent over-concentration.
    """

    def __init__(self, target_entropy: float = None, weight: float = 0.01):
        """
        Initialize gate entropy regularizer.

        Args:
            target_entropy: Target entropy for gate distributions (if None, maximize entropy)
            weight: Weight for entropy regularization
        """
        super(GateEntropyRegularizer, self).__init__()
        self.target_entropy = target_entropy
        self.weight = weight

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute gate entropy regularization loss.

        Args:
            gate_logits: Gate logits [num_nodes, num_experts]

        Returns:
            Entropy regularization loss
        """
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1)

        if self.target_entropy is not None:
            # Encourage entropy to be close to target
            entropy_loss = F.mse_loss(gate_entropy.mean(), torch.tensor(self.target_entropy))
        else:
            # Maximize entropy (minimize negative entropy)
            entropy_loss = -gate_entropy.mean()

        return self.weight * entropy_loss