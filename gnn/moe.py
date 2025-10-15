import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
from gnn.ppmi_conv import PPMIConv
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
    def dispatch(self, inp, edge_index, edge_attr):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # Note by Haotao:
        # self._batch_index: shape=(N_batch). The re-order indices from 0 to N_batch-1.
        # inp_exp: shape=inp.shape. The input Tensor re-ordered by self._batch_index along the batch dimension.
        # self._part_sizes: shape=(N_experts), sum=N_batch. self._part_sizes[i] is the number of samples routed towards expert[i].
        # return value: list [Tensor with shape[0]=self._part_sizes[i] for i in range(N_experts)]
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        edge_index_exp = edge_index[:,self._batch_index]
        edge_attr_exp = edge_attr[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(edge_index_exp, self._part_sizes, dim=1), torch.split(edge_attr_exp, self._part_sizes, dim=0)
    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()
    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with configurable GNN experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the output
    num_experts: an integer - number of experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    gnn_type: string - type of GNN layer ('gcn', 'sage', 'ppmi')
    num_hops: list of integers - number of hops for each expert (for multi-hop configurations)
    """
    def __init__(self, input_size, output_size, num_experts, noisy_gating=False, k=3, coef=1e-2, gnn_type='gcn', num_hops=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        self.num_hops = num_hops if num_hops is not None else [1] * num_experts
        if len(self.num_hops) != num_experts:
            raise ValueError(f"Length of num_hops ({len(self.num_hops)}) must match num_experts ({num_experts})")
        self.gnn_type = gnn_type
        # 所有专家都只用一层
        self.experts = nn.ModuleList([])
        for i in range(self.num_experts):
            if gnn_type == 'gcn':
                expert = GCNConv(input_size, output_size, normalize=False)
            elif gnn_type == 'sage':
                expert = SAGEConv(input_size, output_size, normalize=False)
            elif gnn_type == 'ppmi':
                expert = PPMIConv(input_size, output_size)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            self.experts.append(expert)
        # Small GNNs for gate and noise prediction
        self.gate_gnn = nn.Sequential(
            PPMIConv(input_size, num_experts),
            nn.ReLU(),
            # nn.BatchNorm1d(num_experts),
            # nn.Dropout(0.05),
            # PPMIConv(input_size, num_experts),
            # nn.ReLU()
        )
        self.noise_gnn = GCNConv(input_size, num_experts, normalize=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    def noisy_top_k_gating(self, x, edge_index, train, noise_epsilon=1e-2):
        """Noisy top-k gating using a small GNN.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            edge_index: edge index Tensor with shape [2, num_edges]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # clean_logits = self.gate_gnn(x, edge_index)
        out = x
        for layer in self.gate_gnn:
            if isinstance(layer, (GCNConv, SAGEConv, PPMIConv)):
                out = layer(out, edge_index)
            else:
                out = layer(out)
        clean_logits = out
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise_gnn(x, edge_index)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # Calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, clean_logits  # [batch_size, num_experts], [num_experts]

    def get_node_expert_assignment(self, x, edge_index, k=None):
        """获取每个节点被分配到的专家信息
        Args:
            x: 输入特征 [num_nodes, input_size]
            edge_index: 边索引 [2, num_edges]
            k: 整数，指定返回的top-k专家数量，若为None则使用self.k
        Returns:
            expert_indices: 每个节点对应的专家索引 [num_nodes, k]
            expert_probs: 每个专家对应的门控概率 [num_nodes, k]
            prob_distribution: 每个专家的完整概率分布 [num_nodes, num_experts]
        """
        # with torch.no_grad():
        # 获取gates和load
        gates, load, clean_logits = self.noisy_top_k_gating(x, edge_index, train=False)

        # 使用传入的k或默认的self.k
        k = k if k is not None else self.k
        
        # 获取每个节点的top-k专家索引和对应的概率
        expert_probs, expert_indices = clean_logits.topk(k, dim=1)

        return expert_indices, expert_probs, gates

    def get_k_hop_edge_index(self, edge_index, num_nodes, k):
        """返回k-hop邻接的edge_index"""
        if k == 1:
            return edge_index
        # 转为稀疏矩阵
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        adj_k = adj.copy()
        power = adj.copy()
        for _ in range(1, k):
            power = power @ adj
            adj_k = adj_k + power
        adj_k[adj_k > 0] = 1
        edge_index_k, _ = from_scipy_sparse_matrix(adj_k)
        return edge_index_k

    def get_expert_output(self, expert_idx, x, edge_index, edge_attr=None):
        """直接获取指定专家的原始编码输出
        Args:
            expert_idx (int): 专家索引 (0 ~ num_experts-1)
            x (Tensor): 节点特征矩阵 [num_nodes, input_size]
            edge_index (Tensor): 边索引 [2, num_edges]
            edge_attr (Tensor): 边属性 [num_edges, edge_dim]
        Returns:
            expert_output (Tensor): 专家原始输出 [num_nodes, output_size]
        """
        if not 0 <= expert_idx < self.num_experts:
            raise ValueError(f"Expert index {expert_idx} out of range [0, {self.num_experts-1}]")

        expert = self.experts[expert_idx]
        hop = self.num_hops[expert_idx]
        num_nodes = x.size(0)
        edge_index_k = self.get_k_hop_edge_index(edge_index, num_nodes, hop)
        return expert(x, edge_index_k, edge_attr)

    def diversity_loss(self, expert_outputs, gates, tau=0.2):
        """优化后的多样性增强损失计算
        
        Args:
            expert_outputs: [num_nodes, num_experts, d_feature] 每个专家对每个节点的编码
            gates: [num_nodes, num_experts] 门控矩阵 
            tau: float, 温度系数 τ
            
        Returns:
            loss: diversity-enhanced loss scalar
        """
        num_nodes, num_experts, _ = expert_outputs.shape
        total_loss = 0.0
        
        # 对每个专家并行计算loss
        for k in range(num_experts):
            # 获取当前专家的激活节点mask
            mask = gates[:, k] > 0
            if not mask.any():
                continue
                
            # 获取专家k的所有激活节点表示 [n_k, d_feature]
            h_k = expert_outputs[mask, k]
            n_k = h_k.size(0)
            
            if n_k <= 1:  # 至少需要2个节点才能计算相似度
                continue
                
            # 计算所有节点对的相似度矩阵 [n_k, n_k]
            sim_matrix = torch.matmul(h_k, h_k.t()) / tau
            
            # 创建对角线mask
            diag_mask = ~torch.eye(n_k, dtype=torch.bool, device=sim_matrix.device)
            
            # 计算分母(logsumexp)
            denominator = torch.logsumexp(sim_matrix, dim=1)  # [n_k]
            
            # 计算正样本的相似度(排除自身)
            pos_sim = sim_matrix[diag_mask].view(n_k, -1)  # [n_k, n_k-1]
            
            # 并行计算每个节点的loss
            node_losses = -torch.mean(pos_sim / denominator.unsqueeze(1), dim=1)  # [n_k]
            total_loss += node_losses.sum()

        # 按照公式(10)归一化    
        if total_loss == 0:
            return torch.tensor(0.0, device=expert_outputs.device)
        return total_loss / (num_experts * num_nodes)

    # def forward(self, x, edge_index, edge_attr=None):
    #     """Args:
    #     x: tensor shape [batch_size, input_size]
    #     edge_index: tensor shape [2, num_edges]
    #     edge_attr: tensor shape [num_edges, edge_dim] or None
    #     Returns:
    #     y: a tensor with shape [batch_size, output_size].
    #     extra_training_loss: a scalar.  This should be added into the overall
    #     training loss of the model.  The backpropagation of this loss
    #     encourages all experts to be approximately equally used across a batch.
    #     """
    #     gates, load, clean_logits = self.noisy_top_k_gating(x, edge_index, self.training)
    #     # calculate importance loss
    #     importance = gates.sum(0)
    #     #
    #     loss = self.cv_squared(importance) + self.cv_squared(load)
    #     loss *= self.loss_coef
    #     expert_outputs = []
    #     num_nodes = x.size(0)
    #     for i in range(self.num_experts):
    #         expert = self.experts[i]
    #         hop = self.num_hops[i]
    #         edge_index_k = self.get_k_hop_edge_index(edge_index, num_nodes, hop)
    #         expert_i_output = expert(x, edge_index_k, edge_attr)
    #         expert_outputs.append(expert_i_output)
    #     expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]
    #     # gates: shape=[num_nodes, num_experts]
    #     y = gates.unsqueeze(dim=-1) * expert_outputs
    #     y = y.sum(dim=1)
    #     return y, expert_outputs, loss, clean_logits
    
    def forward(self, x, edge_index, edge_attr=None):
        # ...existing code...
        gates, load, clean_logits = self.noisy_top_k_gating(x, edge_index, self.training)
        # calculate diversity-enhanced loss
        expert_outputs = []
        num_nodes = x.size(0)
        for i in range(self.num_experts):
            expert = self.experts[i]
            hop = self.num_hops[i]
            edge_index_k = self.get_k_hop_edge_index(edge_index, num_nodes, hop)
            expert_i_output = expert(x, edge_index_k, edge_attr)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]
        # gates: shape=[num_nodes, num_experts]
        y = clean_logits.unsqueeze(dim=-1) * expert_outputs
        y = y.sum(dim=1)
        # 多样性损失
        loss = self.diversity_loss(expert_outputs, clean_logits)
        return y, expert_outputs, loss, clean_logits