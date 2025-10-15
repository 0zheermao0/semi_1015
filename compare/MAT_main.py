# coding=utf-8
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import time
import warnings
import pickle
import math
from sklearn.metrics import f1_score

# Assuming gnn package and data are structured as expected
# If these imports fail, ensure the 'gnn' directory and 'data' directory are accessible
# from gnn.cached_gcn_conv import CachedGCNConv # Not used in the provided snippet
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv

warnings.filterwarnings("ignore", category=UserWarning)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="MAT for Domain Generalization on Graphs")
parser.add_argument("--source", type=str, default='acmv9', help="Source domain dataset name")
parser.add_argument("--target", type=str, default='citationv1', help="Target domain dataset name (for testing only)")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate for model parameters")
parser.add_argument("--weight_decay", type=float, default=3e-3, help="Weight decay (L2 penalty)")
parser.add_argument("--drop_out", type=float, default=5e-1, help="Dropout rate")
parser.add_argument("--mat_lr", type=float, default=1e-1, help="Learning rate for adversarial perturbation update (MAT step size)")
parser.add_argument("--mat_norm_p", type=str, default='fro', help="Norm type for perturbation gradient normalization ('fro', 2, etc.)")
parser.add_argument("--mat_norm_clip", type=float, default=1.0, help="Maximum norm for the perturbation itself (clip value)") # Added clipping
parser.add_argument("--perturb", type=bool, default=True, help="Whether to use adversarial perturbation (MAT)")
parser.add_argument("--perturb_init_value", type=float, default=1e-3, help="Initial range for perturbation parameter initialization (-val, val)")
parser.add_argument("--encoder_dim", type=int, default=512, help="Dimension of GNN hidden layers")
parser.add_argument("--label_rate", type=float, default=0.05, help="Fraction of source nodes used as labeled training data")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--data_dir", type=str, default='./data', help="Directory where datasets are stored")
parser.add_argument("--cache_dir", type=str, default='./tmp', help="Directory to store precomputed PPMI matrices")
parser.add_argument("--log_dir", type=str, default='./log', help="Directory to store logs and embeddings")

args = parser.parse_args()

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = args.seed
encoder_dim = args.encoder_dim
use_perturb = args.perturb
perturb_init_value = args.perturb_init_value
label_rate = args.label_rate

# Create directories if they don't exist
os.makedirs(args.cache_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)


id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, mat_lr:{}, mat_clip:{:.1f}, dim: {}" \
    .format(args.source, args.target, seed, label_rate, args.learning_rate, args.weight_decay,
            args.mat_lr if use_perturb else 0, args.mat_norm_clip if use_perturb else 0, encoder_dim)
print("--- Experiment Configuration ---")
print(id_str)
print(f"Device: {device}")
print("-----------------------------")

# --- Reproducibility ---
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True # Can slow down training
# torch.backends.cudnn.benchmark = False

# --- Load Data ---
print(f"Loading source domain: {args.source}")
source_dataset = DomainData(os.path.join(args.data_dir, args.source), name=args.source)
source_data = source_dataset[0]
source_data.num_classes = source_dataset.num_classes
print(source_data)

print(f"Loading target domain: {args.target} (for testing only)")
target_dataset = DomainData(os.path.join(args.data_dir, args.target), name=args.target)
target_data = target_dataset[0]
target_data.num_classes = target_dataset.num_classes # Should be same as source
print(target_data)

# Ensure class consistency
assert source_data.num_classes == target_data.num_classes, "Source and target domains must have the same number of classes"
num_classes = source_data.num_classes
num_features = source_dataset.num_features # Use source dataset features

source_data = source_data.to(device)
# Target data is loaded but NOT moved to device globally, only within test()

# --- Create Label Mask for Source Domain ---
source_train_size = int(source_data.num_nodes * label_rate)
source_indices = list(range(source_data.num_nodes))
random.shuffle(source_indices)
label_mask = torch.zeros(source_data.num_nodes, dtype=torch.bool)
label_mask[source_indices[:source_train_size]] = True
label_mask = label_mask.to(device)
unlabeled_mask = ~label_mask
print(f"Source nodes: {source_data.num_nodes}, Labeled: {label_mask.sum().item()}, Unlabeled: {unlabeled_mask.sum().item()}")


# --- Utility Function ---
def index2dense(edge_index, nnode):
    """Converts edge_index to a dense adjacency matrix."""
    adj = torch.zeros((nnode, nnode), dtype=torch.float32) # Use float32 for potential matmul
    adj[edge_index[0], edge_index[1]] = 1
    return adj.to(device) # Move to device here

# --- Adversarial Perturbation Layer ---
class AddPerturb(nn.Module):
    """Adds a learnable adversarial perturbation."""
    def __init__(self, num_nodes, dim, init_val):
        super().__init__()
        # Initialize perturbation close to zero
        self.perturb = nn.Parameter(torch.FloatTensor(num_nodes, dim).uniform_(-init_val, init_val).to(device))
        self.perturb.requires_grad_(True)

    def forward(self, input):
        return input + self.perturb

    def clip(self, max_norm):
        """Clips the perturbation norm."""
        with torch.no_grad():
            norm = torch.norm(self.perturb, p=2) # L2 norm, can be changed
            if norm > max_norm:
                self.perturb.data.mul_(max_norm / norm)

# --- GNN Model with MAT ---
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_nodes, dropout_rate, perturb_init_val):
        super().__init__()
        self.dropout_rate = dropout_rate

        # PPMIConv layers
        self.conv1 = PPMIConv(in_channels, hidden_channels)
        self.conv2 = PPMIConv(hidden_channels, hidden_channels)

        # Perturbation layers (one for the output of each conv layer)
        self.perturb_layers = nn.ModuleList([
            AddPerturb(num_nodes, hidden_channels, perturb_init_val),
            AddPerturb(num_nodes, hidden_channels, perturb_init_val)
        ]) if use_perturb else nn.ModuleList() # Only create if perturb is True

    def forward(self, x, edge_index, cache_name, apply_perturb):
        # Layer 1
        x = self.conv1(x, edge_index, cache_name + "_conv1")
        if apply_perturb and len(self.perturb_layers) > 0:
             x = self.perturb_layers[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, cache_name + "_conv2")
        if apply_perturb and len(self.perturb_layers) > 1:
            x = self.perturb_layers[1](x)
        # No activation/dropout after the last layer before classifier usually

        return x

    def clip_perturbations(self, max_norm):
        """Clips all perturbation layers."""
        if use_perturb:
            for layer in self.perturb_layers:
                layer.clip(max_norm)

# --- Classifier ---
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


# --- ReNode Weight Calculation (from original code) ---
# Precompute source adjacency matrix for ReNode
# Load precomputed edge index if available (assuming it includes self-loops and is symmetric if needed by PPMI)
source_cache_path = os.path.join(args.cache_dir, args.source + '.pkl')
if os.path.exists(source_cache_path):
    print(f"Loading cached edge index for {args.source} from {source_cache_path}")
    with open(source_cache_path, 'rb') as f:
        # Assuming the pkl contains edge_index and potentially norm/weights
        # Adjust loading based on the actual content of your .pkl file
        loaded_data = pickle.load(f)
        if isinstance(loaded_data, (list, tuple)) and len(loaded_data) >= 1:
             # Assuming the first element is edge_index
             source_edge_index_for_renode = loaded_data[0].to(device)
             print("Loaded edge_index from pkl.")
        else:
             print("Warning: Unexpected format in pkl file. Using data.edge_index.")
             source_edge_index_for_renode = source_data.edge_index
else:
    print(f"Warning: Cache file {source_cache_path} not found. Using data.edge_index for ReNode.")
    source_edge_index_for_renode = source_data.edge_index

# Compute dense adjacency for ReNode calculation
source_data.adj_dense_for_renode = index2dense(source_edge_index_for_renode, source_data.num_nodes)

def get_renode_weight(adj_matrix, pseudo_label, num_classes):
    """Computes ReNode weights based on pseudo-labels and adjacency."""
    ppr_matrix = adj_matrix # Assuming adj_matrix is appropriate (e.g., A+I or normalized)

    gpr_matrix = []
    for iter_c in range(num_classes):
        class_mask = (pseudo_label == iter_c)
        if class_mask.sum() == 0: # Handle case where a class has no pseudo-labels
            iter_gpr = torch.zeros(ppr_matrix.shape[1], device=adj_matrix.device)
        else:
            iter_gpr = torch.mean(ppr_matrix[class_mask], dim=0).squeeze()
        gpr_matrix.append(iter_gpr)
    gpr_matrix = torch.stack(gpr_matrix, dim=1) # Shape: [n_nodes, n_classes]

    base_w = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)

    # Computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix, dim=1)
    gpr_rn = gpr_sum.unsqueeze(1) - gpr_matrix
    # Use float division
    rn_matrix = torch.mm(ppr_matrix, gpr_matrix) - torch.mm(ppr_matrix, gpr_rn) / (num_classes - 1.0)

    label_matrix = F.one_hot(pseudo_label, num_classes).float()
    rn_matrix = torch.sum(rn_matrix * label_matrix, dim=1)

    # Computing the ReNode Weight
    totoro_list = rn_matrix.tolist()
    # Handle potential NaNs or Infs if division by zero occurred (e.g., num_classes=1)
    totoro_list = [x if np.isfinite(x) else 0.0 for x in totoro_list]

    id2totoro = {i: totoro_list[i] for i in range(nnode)}
    # Sort descending, handle NaNs by placing them lower? (already replaced with 0)
    sorted_totoro = sorted(id2totoro.items(), key=lambda x: x[1], reverse=True)
    id2rank = {sorted_totoro[i][0]: i for i in range(nnode)}
    totoro_rank = [id2rank[i] for i in range(nnode)]

    # Avoid division by zero if nnode=1
    if nnode > 1:
        rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x * 1.0 * math.pi / (nnode - 1)))) for x in totoro_rank]
    else:
        rn_weight = [base_w + 0.5 * scale_w * 2] # Max weight if only one node

    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor).to(adj_matrix.device)

    return rn_weight


# --- Loss Functions ---
cross_entropy_loss = nn.CrossEntropyLoss().to(device) # For labeled data

def entropy_loss_weighted(logits, weights, pseudo_labels_one_hot):
    """Calculates weighted entropy loss for pseudo-labeled data."""
    log_softmax_out = F.log_softmax(logits, dim=-1)
    # Element-wise multiplication with pseudo-labels (selects the log-prob of the pseudo-label class)
    entropy = -pseudo_labels_one_hot * log_softmax_out
    # Sum across classes, multiply by weight, then average over nodes
    entropy_node = torch.sum(entropy, dim=1)
    weighted_entropy = weights * entropy_node
    entropy_loss = torch.mean(weighted_entropy)

    # Imposing diversity constraint (from original code)
    softmax_out = F.softmax(logits, dim=-1)
    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-7)) # Add epsilon for stability

    return entropy_loss


# --- Instantiate Models ---
encoder = GNN(
    in_channels=num_features,
    hidden_channels=encoder_dim,
    num_nodes=source_data.num_nodes, # Perturbation size depends on source graph size
    dropout_rate=args.drop_out,
    perturb_init_val=perturb_init_value
).to(device)

cls_model = Classifier(encoder_dim, num_classes).to(device)

models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])

# Exclude perturbation parameters from the main optimizer
model_params = [p for n, p in itertools.chain(encoder.named_parameters(), cls_model.named_parameters()) if 'perturb' not in n and p.requires_grad]
perturb_params = [p for n, p in encoder.named_parameters() if 'perturb' in n and p.requires_grad]

optimizer = torch.optim.Adam(model_params, lr=args.learning_rate, weight_decay=args.weight_decay)

# --- Training Function ---
def train(epoch):
    for model in models:
        model.train()

    # --- 1. Standard Forward Pass & Loss Calculation ---
    optimizer.zero_grad()

    # Encode source data (potentially with perturbation)
    encoded_source = encoder(source_data.x, source_data.edge_index, args.source, apply_perturb=use_perturb)
    source_logits = cls_model(encoded_source)

    # a) Classifier loss on labeled nodes
    cls_loss = cross_entropy_loss(source_logits[label_mask], source_data.y[label_mask])

    # b) Semi-supervised loss (ReNode + Entropy) on unlabeled nodes
    with torch.no_grad(): # Don't need gradients for pseudo-label generation
        pseudo_probs = F.softmax(source_logits[unlabeled_mask], dim=1)
        max_probs, pseudo_labels_unlabeled = torch.max(pseudo_probs, dim=1)

        # Combine true labels and pseudo labels for ReNode weight calculation
        all_pseudo_labels = torch.zeros_like(source_data.y)
        all_pseudo_labels[label_mask] = source_data.y[label_mask]
        all_pseudo_labels[unlabeled_mask] = pseudo_labels_unlabeled

        # Get ReNode weights for *all* nodes based on combined labels
        renode_weights_all = get_renode_weight(source_data.adj_dense_for_renode, all_pseudo_labels, num_classes)
        renode_weights_unlabeled = renode_weights_all[unlabeled_mask] # Select weights for unlabeled nodes

        # Convert pseudo-labels to one-hot for entropy loss calculation
        pseudo_labels_one_hot = F.one_hot(pseudo_labels_unlabeled, num_classes).float()

    # Calculate weighted entropy loss for unlabeled nodes
    semi_loss = entropy_loss_weighted(
        source_logits[unlabeled_mask],
        renode_weights_unlabeled,
        pseudo_labels_one_hot
    )

    # c) Total Loss
    # Weight semi_loss by epoch progress (as in original code)
    total_loss = cls_loss + (float(epoch) / args.epochs) * semi_loss

    # --- 2. Backward Pass (Calculate Gradients for Model and Perturbation) ---
    # Retain graph if we need to backprop through loss again for perturbation update,
    # but here we do it in one pass.
    total_loss.backward()

    # --- 3. Update Model Parameters (Gradient Descent) ---
    optimizer.step()

    # --- 4. Update Perturbation Parameters (Gradient Ascent - MAT step) ---
    if use_perturb and len(perturb_params) > 0:
        for pi in encoder.perturb_layers:
            if pi.perturb.grad is not None:
                # Normalize gradient: grad / ||grad||
                grad_norm = torch.norm(pi.perturb.grad.detach(), p=args.mat_norm_p)
                if grad_norm > 1e-8: # Avoid division by zero
                    normalized_grad = pi.perturb.grad.detach() / grad_norm
                else:
                    normalized_grad = torch.zeros_like(pi.perturb.grad.detach())

                # Ascent step: perturbation = perturbation + step_size * normalized_grad
                # Use a separate learning rate (mat_lr) for perturbation
                with torch.no_grad():
                    pi.perturb.data.add_(args.mat_lr * normalized_grad)
                pi.perturb.grad.zero_() # Zero gradients after update

        # Optional: Clip perturbation norm after update
        encoder.clip_perturbations(args.mat_norm_clip)


    return cls_loss.item(), semi_loss.item(), total_loss.item()


# --- Evaluation Function ---
def evaluate(logits, labels):
    """Calculates accuracy and F1 scores."""
    preds = logits.argmax(dim=1)
    corrects = preds.eq(labels).float()
    accuracy = corrects.mean()
    macro_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
    micro_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1


def test(data, cache_name, is_target=False):
    """Evaluates the model on the given data."""
    for model in models:
        model.eval()

    # Target data needs its own perturbation layer size if node count differs
    # For DG, we don't train perturbation on target, so we test *without* it.
    # If MAT aims to make the *encoder* robust, applying source-trained perturbation
    # doesn't make sense on target. We evaluate the robust encoder's raw performance.

    data = data.to(device) # Move data to device for testing

    with torch.no_grad():
        # Encode data WITHOUT perturbation for evaluation
        encoded_output = encoder(data.x, data.edge_index, cache_name, apply_perturb=False)
        logits = cls_model(encoded_output)

        # Use all nodes for evaluation if mask is not specified (typical for target domain)
        labels = data.y
        accuracy, macro_f1, micro_f1 = evaluate(logits, labels)

    return accuracy, macro_f1, micro_f1, encoded_output # Return embeddings too


# --- Main Training Loop ---
best_source_val_acc = 0.0 # Use a validation set if available, otherwise use source test acc. Here using source test.
best_target_test_acc = 0.0
best_epoch = 0
best_target_macro_f1 = 0.0
best_target_micro_f1 = 0.0

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    cls_l, semi_l, total_l = train(epoch)
    epoch_time = time.time() - epoch_start_time

    # --- Evaluation Phase ---
    # Using source test set as validation proxy - split source_data for proper validation if possible
    # Evaluate source domain (without perturbation for fair comparison)
    source_acc, source_macro_f1, source_micro_f1, _ = test(source_data, args.source)

    # Evaluate target domain (always without perturbation)
    target_acc, target_macro_f1, target_micro_f1, output_target_eval = test(target_data, args.target, is_target=True)

    print(f"Epoch: {epoch:03d}/{args.epochs} | Time: {epoch_time:.2f}s | "
          f"Loss: {total_l:.4f} (Cls: {cls_l:.4f}, Semi: {semi_l:.4f}) | "
          f"Source Acc: {source_acc:.4f} | Target Acc: {target_acc:.4f} | "
          f"Target Macro-F1: {target_macro_f1:.4f} | Target Micro-F1: {target_micro_f1:.4f}")

    # Model selection based on source domain performance (common practice in DG)
    # Alternatively, could use a validation split of the source domain if available.
    if source_acc > best_source_val_acc:
        best_source_val_acc = source_acc
        best_target_test_acc = target_acc
        best_target_macro_f1 = target_macro_f1
        best_target_micro_f1 = target_micro_f1
        best_epoch = epoch

        # Save embeddings from the best epoch
        # Re-run test to get embeddings (optional, could save from above run)
        _, _, _, output_source_best = test(source_data, args.source)
        _, _, _, output_target_best = test(target_data, args.target, is_target=True)
        try:
            emb_file = os.path.join(args.log_dir, f"{args.source}_to_{args.target}_seed{args.seed}_embeddings.pkl")
            with open(emb_file, 'wb') as f:
                pickle.dump([
                    output_source_best.cpu().numpy(),
                    output_target_best.cpu().numpy(),
                    source_data.y.cpu().numpy(), # Save labels too
                    target_data.y.cpu().numpy()
                    ], f)
            # print(f"Saved best embeddings to {emb_file}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")


# --- Final Results ---
total_training_time = time.time() - start_time
print("--- Training Finished ---")
print(f"Total Training Time: {total_training_time:.2f}s")
print("=============================================================")
print(f"Configuration: {id_str}")
print(f"Best Epoch (based on source acc): {best_epoch}")
print(f"Best Source Accuracy (validation proxy): {best_source_val_acc:.5f}")
print(f"Corresponding Target Test Accuracy: {best_target_test_acc:.5f}")
print(f"Corresponding Target Test Macro-F1: {best_target_macro_f1:.5f}")
print(f"Corresponding Target Test Micro-F1: {best_target_micro_f1:.5f}")
print("=============================================================")

# --- Logging ---
log_file = os.path.join(args.log_dir, f"{args.source}_to_{args.target}.log")
log_line = (f"Config: [{id_str}] - BestEpoch: {best_epoch:03d}, "
            f"BestSourceAcc: {best_source_val_acc:.5f}, TargetTestAcc: {best_target_test_acc:.5f}, "
            f"TargetMacroF1: {best_target_macro_f1:.5f}, TargetMicroF1: {best_target_micro_f1:.5f} - "
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    with open(log_file, 'a') as f:
        f.write(log_line)
    print(f"Results logged to {log_file}")
except Exception as e:
    print(f"Error writing to log file {log_file}: {e}")