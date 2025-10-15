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

# Import GCNConv from PyTorch Geometric
from torch_geometric.nn.conv import GCNConv
# Assuming gnn package and data are structured as expected
# If these imports fail, ensure the 'gnn' directory and 'data' directory are accessible
# from gnn.cached_gcn_conv import CachedGCNConv # Not used
from gnn.dataset.DomainData import DomainData
# from gnn.ppmi_conv import PPMIConv # Replaced with GCNConv

warnings.filterwarnings("ignore", category=UserWarning)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="LDAT for Domain Generalization on Graphs (GCN Version)")
parser.add_argument("--source", type=str, default='acmv9', help="Source domain dataset name")
parser.add_argument("--target", type=str, default='citationv1', help="Target domain dataset name (for testing only)")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=3e-3, help="Weight decay (L2 penalty)")
parser.add_argument("--drop_out", type=float, default=5e-1, help="Dropout rate")
parser.add_argument("--ldat_weight", type=float, default=0.1, help="Weight for LDAT loss term") # LDAT specific arg
parser.add_argument("--encoder_dim", type=int, default=512, help="Dimension of GNN hidden layers")
parser.add_argument("--label_rate", type=float, default=0.05, help="Fraction of source nodes used as labeled training data")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--data_dir", type=str, default='./data', help="Directory where datasets are stored")
parser.add_argument("--cache_dir", type=str, default='./tmp', help="Directory to store precomputed PPMI matrices (if needed by other parts)")
parser.add_argument("--log_dir", type=str, default='./log', help="Directory to store logs and embeddings")

args = parser.parse_args()

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = args.seed
encoder_dim = args.encoder_dim
label_rate = args.label_rate

# Create directories if they don't exist
os.makedirs(args.cache_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)


id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, ldat_w:{:.3f}, dim: {}" \
    .format(args.source, args.target, seed, label_rate, args.learning_rate, args.weight_decay,
            args.ldat_weight, encoder_dim)
print("--- Experiment Configuration (GCN Version) ---")
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
# torch.backends.cudnn.deterministic = True
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
# Ensure target data uses the same number of classes as source
target_data.num_classes = source_dataset.num_classes
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


# --- GCN Model ---
# Replaced the original GNN with a standard GCN using GCNConv
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Standard GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        # No activation/dropout after the last layer before classifier usually

        return x

# --- Classifier ---
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


# --- ReNode Weight Calculation (from original code) ---
# Precompute source adjacency matrix for ReNode
# Note: This part remains the same, as ReNode calculation depends on the adjacency matrix,
# not directly on the GNN architecture used for feature extraction.
source_cache_path = os.path.join(args.cache_dir, args.source + '.pkl')
if os.path.exists(source_cache_path):
    print(f"Loading cached edge index for {args.source} from {source_cache_path}")
    with open(source_cache_path, 'rb') as f:
        loaded_data = pickle.load(f)
        if isinstance(loaded_data, (list, tuple)) and len(loaded_data) >= 1:
             # Ensure the loaded edge_index is on the correct device
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
    # Ensure adj_matrix is on the correct device
    adj_matrix = adj_matrix.to(pseudo_label.device)
    ppr_matrix = adj_matrix # Using the dense adjacency directly

    gpr_matrix = []
    for iter_c in range(num_classes):
        class_mask = (pseudo_label == iter_c)
        if class_mask.sum() == 0:
            iter_gpr = torch.zeros(ppr_matrix.shape[1], device=adj_matrix.device)
        else:
            # Ensure ppr_matrix[class_mask] is 2D before mean
            class_rows = ppr_matrix[class_mask]
            if class_rows.dim() == 1: # Handle case where only one node is in the class
                 class_rows = class_rows.unsqueeze(0)
            iter_gpr = torch.mean(class_rows, dim=0).squeeze()
        gpr_matrix.append(iter_gpr)
    gpr_matrix = torch.stack(gpr_matrix, dim=1)

    base_w = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)

    gpr_sum = torch.sum(gpr_matrix, dim=1)
    gpr_rn = gpr_sum.unsqueeze(1) - gpr_matrix

    # Ensure tensors are on the same device for matrix multiplication
    ppr_matrix = ppr_matrix.to(gpr_matrix.device)
    gpr_rn = gpr_rn.to(gpr_matrix.device)

    if num_classes <= 1:
         # Avoid division by zero or negative numbers if num_classes is 1 or less
         rn_matrix = torch.mm(ppr_matrix, gpr_matrix)
    else:
        rn_matrix = torch.mm(ppr_matrix, gpr_matrix) - torch.mm(ppr_matrix, gpr_rn) / (num_classes - 1.0)


    label_matrix = F.one_hot(pseudo_label, num_classes).float().to(rn_matrix.device) # Move one_hot to device
    rn_matrix = torch.sum(rn_matrix * label_matrix, dim=1)

    totoro_list = rn_matrix.cpu().tolist() # Move to CPU before converting to list
    totoro_list = [x if np.isfinite(x) else 0.0 for x in totoro_list]

    id2totoro = {i: totoro_list[i] for i in range(nnode)}
    sorted_totoro = sorted(id2totoro.items(), key=lambda x: x[1], reverse=True)
    id2rank = {sorted_totoro[i][0]: i for i in range(nnode)}
    totoro_rank = [id2rank[i] for i in range(nnode)]

    if nnode > 1:
        rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x * 1.0 * math.pi / (nnode - 1)))) for x in totoro_rank]
    else:
        rn_weight = [base_w + 0.5 * scale_w * 2] # Handle case of single node

    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor).to(adj_matrix.device) # Move weight to device

    return rn_weight


# --- Loss Functions ---
cross_entropy_loss = nn.CrossEntropyLoss().to(device) # For labeled data

def entropy_loss_weighted(logits, weights, pseudo_labels_one_hot):
    """Calculates weighted entropy loss for pseudo-labeled data."""
    log_softmax_out = F.log_softmax(logits, dim=-1)
    entropy = -pseudo_labels_one_hot * log_softmax_out
    entropy_node = torch.sum(entropy, dim=1)
    # Ensure weights are on the same device as entropy_node
    weighted_entropy = weights.to(entropy_node.device) * entropy_node
    entropy_loss = torch.mean(weighted_entropy)

    softmax_out = F.softmax(logits, dim=-1)
    msoftmax = softmax_out.mean(dim=0)
    # Add small epsilon for numerical stability before log
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-9))

    return entropy_loss

def calculate_ldat_loss(logits, device):
    """Calculates LDAT loss (KL divergence from uniform prior)."""
    # Use all nodes for calculating average prediction
    p_source_pred = F.softmax(logits, dim=1).mean(dim=0)
    # Define uniform target prior
    p_target_prior = torch.ones_like(p_source_pred, device=device) / num_classes
    # KL divergence: KL(P_source || P_prior)
    # Add epsilon for numerical stability before log
    ldat_loss = F.kl_div(
        torch.log(p_source_pred + 1e-9), # Log probabilities of source predictions
        p_target_prior,                  # Target prior probabilities
        reduction='batchmean'            # Sum losses and divide by batch size (1 in this case)
                                         # 'sum' might also be appropriate, depends on scaling preference
    )
    return ldat_loss

# --- Instantiate Models ---
# Use the new GCN class for the encoder
encoder = GCN(
    in_channels=num_features,
    hidden_channels=encoder_dim,
    dropout_rate=args.drop_out
).to(device)

cls_model = Classifier(encoder_dim, num_classes).to(device)

models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

# --- Training Function ---
def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    # --- Forward Pass ---
    # Call the GCN encoder (no cache_name needed)
    encoded_source = encoder(source_data.x, source_data.edge_index)
    source_logits = cls_model(encoded_source)

    # --- Loss Calculation ---
    # a) Classifier loss on labeled nodes
    cls_loss = cross_entropy_loss(source_logits[label_mask], source_data.y[label_mask])

    # b) Semi-supervised loss (ReNode + Entropy) on unlabeled nodes
    with torch.no_grad():
        pseudo_probs = F.softmax(source_logits[unlabeled_mask], dim=1)
        max_probs, pseudo_labels_unlabeled = torch.max(pseudo_probs, dim=1)
        all_pseudo_labels = torch.zeros_like(source_data.y)
        all_pseudo_labels[label_mask] = source_data.y[label_mask]
        all_pseudo_labels[unlabeled_mask] = pseudo_labels_unlabeled
        # Ensure adj_dense_for_renode is on the correct device before passing
        renode_weights_all = get_renode_weight(source_data.adj_dense_for_renode.to(device), all_pseudo_labels, num_classes)
        renode_weights_unlabeled = renode_weights_all[unlabeled_mask]
        pseudo_labels_one_hot = F.one_hot(pseudo_labels_unlabeled, num_classes).float()

    semi_loss = entropy_loss_weighted(
        source_logits[unlabeled_mask],
        renode_weights_unlabeled,
        pseudo_labels_one_hot
    )

    # c) LDAT loss (align average source predictions to uniform prior)
    ldat_loss = calculate_ldat_loss(source_logits, device)

    # d) Total Loss
    semi_loss_weight = float(epoch) / args.epochs # Dynamic weight for semi-supervised loss
    total_loss = cls_loss + semi_loss_weight * semi_loss + args.ldat_weight * ldat_loss

    # --- Backward Pass and Optimization ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return cls_loss.item(), semi_loss.item(), ldat_loss.item(), total_loss.item()


# --- Evaluation Function ---
def evaluate(logits, labels):
    """Calculates accuracy and F1 scores."""
    preds = logits.argmax(dim=1)
    corrects = preds.eq(labels).float()
    accuracy = corrects.mean()
    # Ensure labels and preds are on CPU for scikit-learn functions
    labels_cpu = labels.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1


def test(data, cache_name=None): # Removed cache_name default, not needed by GCN
    """Evaluates the model on the given data."""
    for model in models:
        model.eval()

    data = data.to(device) # Move data to device for testing

    with torch.no_grad():
        # Encode data using GCN (no cache_name)
        encoded_output = encoder(data.x, data.edge_index)
        logits = cls_model(encoded_output)

        # Evaluate
        labels = data.y
        accuracy, macro_f1, micro_f1 = evaluate(logits, labels)

    return accuracy, macro_f1, micro_f1, encoded_output # Return embeddings too


# --- Main Training Loop ---
best_source_val_acc = 0.0 # Using source test acc as validation proxy
best_target_test_acc = 0.0
best_epoch = 0
best_target_macro_f1 = 0.0
best_target_micro_f1 = 0.0

print("\n--- Starting Training (GCN Encoder) ---")
start_time = time.time()

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    cls_l, semi_l, ldat_l, total_l = train(epoch)
    epoch_time = time.time() - epoch_start_time

    # --- Evaluation Phase ---
    # Pass data directly, no cache_name needed for test with GCN
    source_acc, source_macro_f1, source_micro_f1, _ = test(source_data)
    target_acc, target_macro_f1, target_micro_f1, output_target_eval = test(target_data)

    print(f"Epoch: {epoch:03d}/{args.epochs} | Time: {epoch_time:.2f}s | "
          f"Loss: {total_l:.4f} (Cls: {cls_l:.4f}, Semi: {semi_l:.4f}, LDAT: {ldat_l:.4f}) | "
          f"Source Acc: {source_acc:.4f} | Target Acc: {target_acc:.4f} | "
          f"Target Macro-F1: {target_macro_f1:.4f}")

    # Model selection based on source domain performance
    if source_acc > best_source_val_acc:
        best_source_val_acc = source_acc
        best_target_test_acc = target_acc
        best_target_macro_f1 = target_macro_f1
        best_target_micro_f1 = target_micro_f1
        best_epoch = epoch

        # Save embeddings from the best epoch
        # Rerun test to get embeddings for the best epoch state
        _, _, _, output_source_best = test(source_data)
        _, _, _, output_target_best = test(target_data)
        try:
            # Update filename to reflect GCN usage
            emb_file = os.path.join(args.log_dir, f"{args.source}_to_{args.target}_seed{args.seed}_LDAT_GCN_embeddings.pkl")
            with open(emb_file, 'wb') as f:
                pickle.dump([
                    output_source_best.cpu().numpy(),
                    output_target_best.cpu().numpy(),
                    source_data.y.cpu().numpy(),
                    target_data.y.cpu().numpy()
                    ], f)
            print(f"Best embeddings saved to {emb_file}") # Added confirmation message
        except Exception as e:
            print(f"Error saving embeddings: {e}")


# --- Final Results ---
total_training_time = time.time() - start_time
print("--- Training Finished ---")
print(f"Total Training Time: {total_training_time:.2f}s")
print("=============================================================")
print(f"Configuration (GCN): {id_str}")
print(f"Best Epoch (based on source acc): {best_epoch}")
print(f"Best Source Accuracy (validation proxy): {best_source_val_acc:.5f}")
print(f"Corresponding Target Test Accuracy: {best_target_test_acc:.5f}")
print(f"Corresponding Target Test Macro-F1: {best_target_macro_f1:.5f}")
print(f"Corresponding Target Test Micro-F1: {best_target_micro_f1:.5f}")
print("=============================================================")

# --- Logging ---
# Update log filename to reflect GCN usage
log_file = os.path.join(args.log_dir, f"{args.source}_to_{args.target}_LDAT_GCN.log")
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

