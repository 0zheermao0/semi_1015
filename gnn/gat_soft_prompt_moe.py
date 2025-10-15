import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import numpy as np
from typing import Optional, Tuple, Dict, List


class SoftPrompt(nn.Module):
    """
    Soft-prompt module for expert specialization in GAT-based MoE.
    Each expert learns a specialized prompt vector to bias its attention patterns.
    """

    def __init__(self, prompt_length: int, hidden_dim: int, expert_id: int, dropout: float = 0.1):
        super(SoftPrompt, self).__init__()
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.expert_id = expert_id

        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_dim) * 0.1
        )

        # Expert-specific transformation for specialization
        self.specialization_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion weight for balancing original features and prompt influence
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for soft-prompt module.

        Args:
            x: Input features [num_nodes, hidden_dim]

        Returns:
            Prompt-enhanced features [num_nodes, hidden_dim]
        """
        # Compute prompt vector as mean of prompt embeddings
        prompt_vector = self.prompt_embeddings.mean(dim=0)  # [hidden_dim]

        # Apply expert-specific transformation
        specialized_features = self.specialization_transform(x)

        # Expand prompt vector to match batch size
        batch_size = x.size(0)
        expanded_prompt = prompt_vector.unsqueeze(0).expand(batch_size, -1)

        # Fusion with learnable weight (sigmoid to keep in [0,1])
        fusion_weight = torch.sigmoid(self.fusion_weight)
        fused_features = (1 - fusion_weight) * specialized_features + fusion_weight * expanded_prompt

        # Apply layer normalization
        output = self.layer_norm(fused_features)

        return output


class UnifiedGATExpert(nn.Module):
    """
    Unified GAT expert with soft-prompt specialization.
    All experts share the same architecture but learn different specializations via soft prompts.
    """

    def __init__(self, input_size: int, hidden_dim: int, output_size: int,
                 prompt_length: int = 8, expert_id: int = 0, num_heads: int = 4,
                 dropout: float = 0.1):
        super(UnifiedGATExpert, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.expert_id = expert_id
        self.num_heads = num_heads

        # Soft-prompt module for expert specialization
        self.soft_prompt = SoftPrompt(prompt_length, hidden_dim, expert_id, dropout)

        # Three-layer GAT architecture
        self.gat1 = GATConv(input_size, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        self.gat3 = GATConv(hidden_dim, output_size, heads=1, dropout=dropout, concat=False)

        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual connection projections if needed
        self.input_proj = nn.Linear(input_size, hidden_dim) if input_size != hidden_dim else None
        self.hidden_proj = nn.Linear(hidden_dim, output_size) if hidden_dim != output_size else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for unified GAT expert.

        Args:
            x: Node features [num_nodes, input_size]
            edge_index: Edge index [2, num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_dim]

        Returns:
            Expert output [num_nodes, output_size]
        """
        # Apply soft-prompt for expert specialization
        x_prompted = self.soft_prompt(x)

        # First GAT layer with residual connection
        h1 = self.gat1(x_prompted, edge_index, edge_attr)
        if self.input_proj is not None:
            h1 = h1 + self.input_proj(x_prompted)
        h1 = self.layer_norm1(h1)
        h1 = self.dropout(F.relu(h1))

        # Second GAT layer with residual connection
        h2 = self.gat2(h1, edge_index, edge_attr)
        h2 = h2 + h1
        h2 = self.layer_norm2(h2)
        h2 = self.dropout(F.relu(h2))

        # Third GAT layer
        out = self.gat3(h2, edge_index, edge_attr)
        if self.hidden_proj is not None:
            out = out + self.hidden_proj(h2)

        return out


class GATSoftPromptMoE(nn.Module):
    """
    GAT-based Mixture of Experts with Soft-Prompt specialization.
    Implements the unified architecture where all experts have the same GAT structure
    but learn different specializations through soft prompts.
    """

    def __init__(self, input_size: int, output_size: int, num_experts: int = 6,
                 prompt_length: int = 8, num_heads: int = 4, dropout: float = 0.1,
                 k: int = 1, coef: float = 1e-2, noisy_gating: bool = False):
        super(GATSoftPromptMoE, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        self.noisy_gating = noisy_gating

        # Create unified GAT experts with soft-prompt specialization
        hidden_dim = output_size  # Use output_size as hidden_dim for simplicity
        self.experts = nn.ModuleList([
            UnifiedGATExpert(input_size, hidden_dim, output_size, prompt_length, i, num_heads, dropout)
            for i in range(num_experts)
        ])

        # Gating network (can be either embedding-based or GAT-based)
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

        # Noise network for noisy gating (if enabled)
        if noisy_gating:
            self.noise_network = nn.Linear(input_size, num_experts)

        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

        # Register buffers for stability
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(k <= num_experts), f"k={k} must be <= num_experts={num_experts}"

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Compute squared coefficient of variation for load balancing."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute the true load per expert."""
        return (gates > 0).sum(0)

    def noisy_top_k_gating(self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2):
        """
        Noisy top-k gating for expert selection.

        Args:
            x: Input features [batch_size, input_size]
            train: Whether in training mode
            noise_epsilon: Small constant for numerical stability

        Returns:
            gates: Gating weights [batch_size, num_experts]
            load: Load per expert [num_experts]
            clean_logits: Clean logits without noise [batch_size, num_experts]
        """
        clean_logits = self.gate_network(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise_network(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Calculate top-k
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        # Create sparse gates
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Calculate load for load balancing loss
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load, clean_logits

    def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor,
                      noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor) -> torch.Tensor:
        """Helper for noisy top-k gating to compute probability of being in top k."""
        from torch.distributions import Normal

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def get_node_expert_assignment(self, x: torch.Tensor, edge_index: torch.Tensor, k: Optional[int] = None):
        """Get expert assignment for each node."""
        gates, load, clean_logits = self.noisy_top_k_gating(x, train=False)
        k = k if k is not None else self.k
        expert_probs, expert_indices = clean_logits.topk(k, dim=1)
        return expert_indices, expert_probs, gates

    def diversity_loss(self, expert_outputs: torch.Tensor, gates: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
        """
        Compute diversity-enhanced loss to encourage expert specialization.

        Args:
            expert_outputs: [num_nodes, num_experts, output_size]
            gates: [num_nodes, num_experts]
            tau: Temperature parameter

        Returns:
            Diversity loss scalar
        """
        num_nodes, num_experts, _ = expert_outputs.shape
        total_loss = 0.0

        for k in range(num_experts):
            mask = gates[:, k] > 0
            if not mask.any():
                continue

            h_k = expert_outputs[mask, k]
            n_k = h_k.size(0)

            if n_k <= 1:
                continue

            # Compute similarity matrix
            sim_matrix = torch.matmul(h_k, h_k.t()) / tau
            diag_mask = ~torch.eye(n_k, dtype=torch.bool, device=sim_matrix.device)
            denominator = torch.logsumexp(sim_matrix, dim=1)
            pos_sim = sim_matrix[diag_mask].view(n_k, -1)
            node_losses = -torch.mean(pos_sim / denominator.unsqueeze(1), dim=1)
            total_loss += node_losses.sum()

        if total_loss == 0:
            return torch.tensor(0.0, device=expert_outputs.device)
        return total_loss / (num_experts * num_nodes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for GAT+Soft-prompt MoE.

        Args:
            x: Node features [num_nodes, input_size]
            edge_index: Edge index [2, num_edges]
            edge_attr: Optional edge attributes

        Returns:
            output: Combined expert output [num_nodes, output_size]
            expert_outputs: Individual expert outputs [num_nodes, num_experts, output_size]
            diversity_loss_val: Diversity loss scalar
            clean_logits: Clean gating logits [num_nodes, num_experts]
        """
        # Get gating weights
        gates, load, clean_logits = self.noisy_top_k_gating(x, self.training)

        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x, edge_index, edge_attr)
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [num_nodes, num_experts, output_size]

        # Combine expert outputs using clean logits (similar to sparse mixture of experts)
        output = clean_logits.unsqueeze(dim=-1) * expert_outputs
        output = output.sum(dim=1)  # [num_nodes, output_size]

        # Compute diversity loss
        diversity_loss_val = self.diversity_loss(expert_outputs, clean_logits)

        # Compute load balancing loss
        importance = gates.sum(0)
        load_loss = self.cv_squared(importance) + self.cv_squared(load)
        load_loss *= self.loss_coef

        # Total auxiliary loss
        total_aux_loss = diversity_loss_val + load_loss

        return output, expert_outputs, total_aux_loss, clean_logits