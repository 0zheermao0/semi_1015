# coding=utf-8
import os
import argparse
import sys
import random
import math
import time
import warnings
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Imports for CaNet and PyG utilities
from torch_geometric.utils import to_undirected, degree, remove_self_loops, add_self_loops, add_remaining_self_loops
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

from gnn.dataset.DomainData import DomainData # Assuming this is the correct path
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

# Helper function from CaNet script (gcn_conv)
def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N, dtype=x.dtype).float() # Ensure dtype matches x
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * d_norm_in * d_norm_out # Ensure dtype matches x
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x) # [N, D]

# --- CaNet Components Start ---
class GraphConvolutionBase(nn.Module):
    """ Basic GCN layer, optionally residual """
    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.residual:
            # Initialize weight_r carefully, maybe near zero or identity-like if in_features==out_features
            stdv_r = 1. / math.sqrt(self.in_features) # Or use self.out_features
            self.weight_r.data.uniform_(-stdv_r, stdv_r) # Or zeros initialization


    def forward(self, x, adj, x0): # x0 is the initial layer input, needed for graph env_enc
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
             # Use x (input to this layer) for residual, not x0 (initial input)
             # Ensure dimensions match if in_features != out_features, maybe add a linear projection for x
             # Assuming in_features == out_features for residual for simplicity here
             if self.in_features == self.out_features:
                output = output + torch.mm(x, self.weight_r) # Original had torch.mm(x, self.weight_r) which seems unusual for residual
             else:
                 # If dims don't match, project x first or omit this residual path
                 # output = output + torch.mm(x, self.weight_r) # Requires weight_r: in_features -> out_features
                 pass # Or handle dim mismatch
        return output

class CaNetConv(nn.Module):
    """ CaNet Convolutional Layer """
    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(CaNetConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.in_features = in_features # Store in_features
        self.residual = residual
        self.K = K
        self.device = device
        self.variant = variant # Note: variant=True GCN path might be slow/memory intensive

        if backbone_type == 'gcn':
             # Input to matmul is [K, N, D*2], weight should be [K, D*2, D_out] or [D*2, D_out]
             # Original code uses [K, in_features*2, out_features] which implies shared weights across nodes but different per K
             self.weights = Parameter(torch.FloatTensor(K, in_features*2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features)) # GAT weights per head K
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1)) # GAT attention weights per head K
        else:
            raise ValueError("Unsupported backbone type")

        if self.residual:
             # Add a linear layer for residual connection if dims mismatch
             if self.in_features != self.out_features:
                 self.residual_linear = nn.Linear(in_features, out_features)
             else:
                 self.residual_linear = nn.Identity() # If dims match, identity is fine

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if hasattr(self, 'weights'):
             self.weights.data.uniform_(-stdv, stdv)
        if hasattr(self, 'residual_linear') and isinstance(self.residual_linear, nn.Linear):
            self.residual_linear.reset_parameters()
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        # Helper for GAT sparse multiplication
        adj_sparse = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj_sparse, h)

    def forward(self, x, adj, e, weights=None):
        if weights is None:
            weights = self.weights # Use internal weights if not passed externally

        N = x.shape[0]
        if self.backbone_type == 'gcn':
            if not self.variant:
                # Standard GCN calculation using sparse tensor utility
                hi = gcn_conv(x, adj)
            else:
                # Variant using torch.sparse.mm (potentially slower)
                # Requires adj to be a sparse tensor
                # This part needs careful checking for efficiency and correctness
                adj_val = torch.ones(adj.shape[1], device=self.device)
                adj_sparse = torch.sparse_coo_tensor(adj, adj_val, size=(N, N)).to(self.device)
                hi = torch.sparse.mm(adj_sparse, x)

            hi = torch.cat([hi, x], dim=1) # Concatenate with original features [N, D*2]
            hi = hi.unsqueeze(0).expand(self.K, -1, -1) # Repeat for K experts [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D_out]
            outputs = outputs.transpose(0, 1) # [N, K, D_out]

        elif self.backbone_type == 'gat':
            xi = x.unsqueeze(0).expand(self.K, -1, -1) # [K, N, D_in]
            h = torch.matmul(xi, weights) # Apply linear transformation per head [K, N, D_out]

            adj_no_self_loops, _ = remove_self_loops(adj)
            adj_with_self_loops, _ = add_self_loops(adj_no_self_loops, num_nodes=N)
            row, col = adj_with_self_loops

            # Prepare for attention calculation
            h_row = h[:, row, :] # [K, E, D_out]
            h_col = h[:, col, :] # [K, E, D_out]
            edge_h = torch.cat((h_row, h_col), dim=2) # [K, E, 2*D_out]

            # Calculate attention logits per edge per head
            # self.a is [K, 2*D_out, 1] -> Unsqueeze K for broadcasting
            # edge_h is [K, E, 2*D_out], a_expanded is [K, 1, 2*D_out, 1]
            # Need matmul([K, E, 2*D_out], [K, 2*D_out, 1]) -> [K, E, 1] ?
            # Let's adjust self.a or the matmul
            # Option 1: Reshape a -> [K, 1, 2*D_out] and use bmm
            # a_reshaped = self.a.squeeze(-1).unsqueeze(1) # [K, 1, 2*D_out]
            # logits = self.leakyrelu(torch.bmm(a_reshaped, edge_h.transpose(1,2))).squeeze(1) # [K, E] Requires D adjustments
            # Option 2: Element-wise attention or simpler form? The original code seems to do this:
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2) # [K, E] This works if self.a is [K, 2D, 1]

            # Stabilize attention scores
            logits_max , _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits - logits_max) # [K, E] Unnormalized attention scores

            outputs_list = []
            eps = 1e-8
            size = torch.Size([N, N])
            # Aggregate features using attention scores per head
            for k in range(self.K):
                edge_e_k = edge_e[k, :] # [E] Attention scores for head k
                # Calculate normalization factor (sum of attention scores for each node)
                # Use specialspmm for sparse sum reduction
                # Need a vector of ones [N, 1] on the correct device
                ones_vec = torch.ones(N, 1, device=self.device, dtype=h.dtype)
                e_expsum_k = self.specialspmm(adj_with_self_loops, edge_e_k, size, ones_vec) + eps # [N, 1]
                assert not torch.isnan(e_expsum_k).any()

                # Apply attention weighted aggregation
                # Aggregate h[k] which is [N, D_out]
                hi_k = self.specialspmm(adj_with_self_loops, edge_e_k, size, h[k]) # [N, D_out]
                hi_k = torch.div(hi_k, e_expsum_k) # Normalize [N, D_out]
                outputs_list.append(hi_k)

            outputs = torch.stack(outputs_list, dim=1) # [N, K, D_out]

        # Combine outputs from K experts using attention scores 'e'
        # e should be [N, K] -> expand to [N, K, D_out]
        es = e.unsqueeze(2).expand(-1, -1, self.out_features) # [N, K, D_out]
        output = torch.sum(torch.mul(es, outputs), dim=1) # Weighted sum -> [N, D_out]

        if self.residual:
             # Apply residual connection after combining experts
             output = output + self.residual_linear(x) # Use linear layer for potential dim change

        return output

class CaNet(nn.Module):
    """ Causal Attention Network Model """
    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.tau = args.tau
        self.env_type = args.env_type
        self.K = args.K # Number of experts/environments
        self.device = device
        self.hidden_channels = args.hidden_channels # Use hidden_channels consistently

        self.convs = nn.ModuleList()
        self.env_enc = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.acts = nn.ModuleList() # Store activation functions if needed per layer

        # Input layer
        self.fcs.append(nn.Linear(d, self.hidden_channels))
        self.acts.append(nn.ReLU()) # Activation after input layer

        # Hidden layers
        for i in range(self.num_layers):
             # CaNetConv Layer
             self.convs.append(CaNetConv(self.hidden_channels, self.hidden_channels, self.K,
                                       backbone_type=args.backbone_type, residual=True, # Assuming residual=True is desired
                                       device=device, variant=args.variant))
             # Environment Encoder Layer
             if args.env_type == 'node':
                 # Simple linear layer per node to predict environment distribution
                 self.env_enc.append(nn.Linear(self.hidden_channels, self.K))
             elif args.env_type == 'graph':
                 # GCN-based layer to predict environment distribution
                 # Needs careful input handling (adj, x0)
                 self.env_enc.append(GraphConvolutionBase(self.hidden_channels, self.K, residual=True)) # Assuming residual
             else:
                 raise NotImplementedError(f"env_type '{args.env_type}' not supported")
             self.acts.append(nn.ReLU()) # Activation after each hidden layer

        # Output layer
        self.fcs.append(nn.Linear(self.hidden_channels, c))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
             if hasattr(fc, 'reset_parameters'):
                fc.reset_parameters()
        for enc in self.env_enc:
             if hasattr(enc, 'reset_parameters'):
                enc.reset_parameters()


    def forward(self, x, adj, training=False):
        # Initial transformation
        h = F.dropout(x, self.dropout, training=training)
        h = self.fcs[0](h)
        h = self.acts[0](h)
        h0 = h.clone() # Keep copy of first layer features if using graph env_enc

        reg = 0 # Regularization loss accumulator
        for i, con in enumerate(self.convs): # Iterate through hidden layers
             h_in = F.dropout(h, self.dropout, training=training) # Dropout before conv and env_enc

             # Calculate environment attention weights 'e'
             if self.env_type == 'node':
                 logit = self.env_enc[i](h_in)
             elif self.env_type == 'graph':
                 # GraphConvolutionBase needs x, adj, x0
                 logit = self.env_enc[i](h_in, adj, h0)
             else:
                  raise NotImplementedError

             if training:
                 # Use Gumbel-Softmax for differentiable sampling during training
                 e = F.gumbel_softmax(logit, tau=self.tau, dim=-1, hard=False) # Use hard=False for REINFORCE-like gradient
                 # Calculate regularization loss for this layer
                 reg += self.reg_loss(e, logit)
             else:
                 # Use Softmax for deterministic attention during inference
                 e = F.softmax(logit, dim=-1)

             # Apply CaNetConv layer using the calculated attention 'e'
             h = con(h_in, adj, e) # Pass attention 'e' to the conv layer
             h = self.acts[i+1](h) # Apply activation function

        # Final layer
        h = F.dropout(h, self.dropout, training=training)
        out = self.fcs[-1](h) # Output logits

        if training:
            return out, reg / self.num_layers # Return logits and average regularization loss
        else:
            return out # Return only logits during inference

    def reg_loss(self, z, logit):
        """ Regularization loss (consistency promoting). """
        # Assumes z is the result of softmax/gumbel_softmax (probabilities)
        # Assumes logit is the raw output before softmax
        log_pi = F.log_softmax(logit, dim=-1) # Calculate log probabilities
        # Cross-entropy between z and the distribution defined by logits
        # This encourages the sampled 'e' (z) to be close to the predicted distribution mean
        loss = torch.mean(torch.sum(torch.mul(z, log_pi), dim=1)) # Mean over nodes
        # The original implementation uses this formula. It might be aiming to maximize entropy
        # or minimize KL divergence depending on interpretation. Let's stick to it.
        return loss # Note: The original CaNet returns this value. Depending on optimization goal, might need negation.

    def sup_loss_calc(self, y, pred, criterion):
        """ Calculates the supervised classification loss. """
        # Adapt based on expected y shape and criterion type
        # Assuming CrossEntropyLoss for multi-class classification
        if isinstance(criterion, nn.CrossEntropyLoss):
            # Expects preds: [N, C], y: [N]
            target = y.squeeze(-1) if y.dim() > 1 else y # Ensure target is [N]
            loss = criterion(pred, target)
        elif isinstance(criterion, nn.BCEWithLogitsLoss):
             # Expects preds: [N, C], y: [N, C] (one-hot or multi-label)
             # Check if y needs one-hot encoding
             if y.shape[1] == 1 and pred.shape[1] > 1:
                 true_label = F.one_hot(y.squeeze(-1), pred.shape[1]).float()
             else:
                 true_label = y.float()
             loss = criterion(pred, true_label)
        else:
            raise ValueError("Unsupported criterion type for sup_loss_calc")
        return loss

    # Modified loss_compute for the target script's setting
    def loss_compute(self, x, edge_index, y, train_mask, criterion, args):
         """ Computes the total loss for training """
         # Perform forward pass to get logits and regularization loss
         logits, reg_loss = self.forward(x, edge_index, training=True)

         # Calculate supervised loss ONLY on the labeled training nodes
         sup_loss = self.sup_loss_calc(y[train_mask], logits[train_mask], criterion)

         # Combine supervised loss and regularization loss
         loss = sup_loss + args.lamda * reg_loss # Use lamda hyperparameter

         return loss

# --- CaNet Components End ---


# --- Main Script Logic ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Graph OOD Adaptation with CaNet')
# Data args
parser.add_argument("--source", type=str, default='acmv9', help="Source domain dataset name")
parser.add_argument("--target", type=str, default='citationv1', help="Target domain dataset name")
parser.add_argument("--data_dir", type=str, default='data/', help="Directory where data is stored")
# Training args
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay (L2 penalty)")
parser.add_argument("--label_rate", type=float, default=0.05, help="Fraction of source nodes used for labels")
# CaNet specific args
parser.add_argument('--num_layers', type=int, default=2, help='Number of CaNetConv layers.')
parser.add_argument('--hidden_channels', type=int, default=512, help='Number of hidden units.') # Renamed from encoder_dim
parser.add_argument('--K', type=int, default=3, help='Number of experts/environments K.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.') # Renamed from drop_out
parser.add_argument('--lamda', type=float, default=1.0, help='Weight for CaNet regularization loss.') # Lambda hyperparameter
parser.add_argument('--tau', type=float, default=1.0, help='Temperature for Gumbel-Softmax.')
parser.add_argument('--backbone_type', type=str, default='gat', choices=['gcn', 'gat'], help='Backbone GNN type for CaNetConv.')
parser.add_argument('--env_type', type=str, default='node', choices=['node', 'graph'], help='Environment encoder type.')
parser.add_argument('--variant', action='store_true', help='Whether to use GCN variant.') # Default False

# Compatibility args (might be unused but kept for structure)
# parser.add_argument("--perturb", type=bool, default=False) # Perturbation is not used by CaNet here
# parser.add_argument("--perturb_value", type=float, default=0.5)

args = parser.parse_args()

# --- Seed and Setup ---
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # Can slow down, uncomment if needed
    # torch.backends.cudnn.benchmark = False

# --- Logging Setup ---
id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, K:{}, lambda:{}, tau:{}, layers:{}, hidden:{}" \
    .format(args.source, args.target, seed, args.label_rate, args.lr, args.weight_decay,
            args.K, args.lamda, args.tau, args.num_layers, args.hidden_channels)
print(id_str)

# --- Load Data ---
# Ensure DomainData loads x, y, edge_index, potentially train/val/test masks
source_dataset = DomainData(os.path.join(args.data_dir, args.source), name=args.source)
source_data = source_dataset[0]
source_data.num_classes = source_dataset.num_classes
source_data.num_features = source_dataset.num_features
source_data.num_nodes = source_data.x.shape[0] # Explicitly set num_nodes
print("Source Data:", source_data)

target_dataset = DomainData(os.path.join(args.data_dir, args.target), name=args.target)
target_data = target_dataset[0]
# Use source dataset's class/feature info assuming alignment
target_data.num_classes = source_dataset.num_classes
target_data.num_features = source_dataset.num_features
target_data.num_nodes = target_data.x.shape[0] # Explicitly set num_nodes
print("Target Data:", target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

# Create label mask for source data
source_train_size = int(source_data.num_nodes * args.label_rate)
label_mask = torch.zeros(source_data.num_nodes, dtype=torch.bool, device=device)
perm = torch.randperm(source_data.num_nodes, device=device)
label_mask[perm[:source_train_size]] = True


# --- Define Model ---
model = CaNet(d=source_data.num_features,
              c=source_data.num_classes,
              args=args,
              device=device).to(device)

print("Model:", model)
print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# --- Define Loss and Optimizer ---
# Use CrossEntropyLoss for classification (standard for Cora, PubMed, etc.)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


# --- Evaluation Function ---
def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    # Ensure labels and preds are on CPU for sklearn
    labels_cpu = labels.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1

# --- Test Function ---
def test(data, data_name): # Removed cache_name, perturb args
    model.eval()
    with torch.no_grad():
        # Use CaNet's forward pass for inference
        logits = model(data.x, data.edge_index, training=False)
        preds = logits.argmax(dim=1)
        labels = data.y

        # If data has a predefined test mask, use it. Otherwise evaluate on all nodes.
        if hasattr(data, 'test_mask') and data.test_mask is not None:
             mask = data.test_mask
             accuracy, macro_f1, micro_f1 = evaluate(preds[mask], labels[mask])
             print(f"{data_name} Test Results (Masked):")
        else:
             mask = None # Evaluate on all nodes if no mask
             accuracy, macro_f1, micro_f1 = evaluate(preds, labels)
             print(f"{data_name} Test Results (All Nodes):")

        print(f"  Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}")

    # Return metrics and optionally embeddings (logits or pre-logit features if needed)
    return accuracy, macro_f1, micro_f1, logits # Return logits as 'output'

# --- Training Function ---
def train(epoch):
    model.train()
    optimizer.zero_grad()

    # Calculate loss using CaNet's method on source data with label mask
    loss = model.loss_compute(source_data.x,
                              source_data.edge_index,
                              source_data.y,
                              label_mask, # Pass the mask for supervised loss
                              criterion,
                              args)

    loss.backward()
    optimizer.step()
    return loss.item()


# --- Training Loop ---
best_source_eval_acc = 0.0 # Evaluate source on its own test split if available, else all nodes
best_target_acc = 0.0
best_epoch = 0
best_macro_f1 = 0.0
best_micro_f1 = 0.0

print("\n--- Starting Training ---")
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    loss = train(epoch)

    # Evaluate on source and target domains
    # Determine if source_data has a test_mask
    source_eval_acc, _, _, _ = test(source_data, args.source) # Use source test mask if exists
    target_acc, macro_f1, micro_f1, output_target = test(target_data, args.target) # Target usually evaluated on all nodes

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, SrcAcc: {source_eval_acc:.4f}, "
          f"TgtAcc: {target_acc:.4f}, TgtMacroF1: {macro_f1:.4f}, TgtMicroF1: {micro_f1:.4f}, "
          f"Time: {epoch_time:.2f}s")

    # Simple best model saving strategy: save based on best target accuracy
    # Could also use source validation accuracy if a val split was available
    if target_acc > best_target_acc:
        best_target_acc = target_acc
        best_source_eval_acc = source_eval_acc
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch
        # Save model checkpoint if needed
        # torch.save(model.state_dict(), f"best_model_{args.source}_{args.target}.pth")
        # Save embeddings if needed (using output_target from test)
        # Note: Source embeddings (output_source) aren't computed unless test(source_data) returns them
        # _, _, _, output_source = test(source_data, args.source) # Compute if needed
        # with open('log/{}_{}_embeddings.pkl'.format(args.source, args.target), 'wb') as f:
        #    pickle.dump([output_source.cpu().numpy(), output_target.cpu().numpy()], f)
        # print(f"*** New best target accuracy: {best_target_acc:.4f} at epoch {best_epoch} ***")


print("\n--- Training Finished ---")
print("=============================================================")
line = "{}\n - Best Epoch: {}, Best Source Acc: {:.4f}, Best Target Acc: {:.4f}, Best Macro-F1: {:.4f}, Best Micro-F1: {:.4f}" \
    .format(id_str, best_epoch, best_source_eval_acc, best_target_acc, best_macro_f1, best_micro_f1)
print(line)

# --- Logging Results ---
log_dir = "log"
os.makedirs(log_dir, exist_ok=True) # Create log directory if it doesn't exist
log_file = os.path.join(log_dir, f"{args.source}-{args.target}.log")

with open(log_file, 'a') as f:
    log_entry = ("{} - Best Epoch: {:0>3d}, Best Src Acc: {:.5f}, Best Tgt Acc: {:.5f}, "
                 "Best Macro-F1: {:.5f}, Best Micro-F1: {:.5f}\t{}\n").format(
                    id_str, best_epoch, best_source_eval_acc, best_target_acc,
                    best_macro_f1, best_micro_f1,
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    f.write(log_entry)

print(f"Results logged to {log_file}")