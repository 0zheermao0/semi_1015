# coding=utf-8
import os
import csv
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
import json
# Try importing ollama, but handle potential errors if not installed/configured
try:
    import ollama
    ollama_available = True
except ImportError:
    print("Warning: ollama library not found. LLM-related functions will be disabled.")
    ollama_available = False
except Exception as e:
    print(f"Warning: Error importing or configuring ollama: {e}. LLM functions disabled.")
    ollama_available = False


# --- W&B Integration ---
try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb library not found. Results will not be logged to Weights & Biases.")
    wandb_available = False
# --- End W&B Integration ---

from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
from gnn.moe import MoE
from common.graph_encoder import GraphEncoder as Graph2TextEncoder

warnings.filterwarnings("ignore", category=UserWarning)
# os.environ['OLLAMA_HOST'] = 'http://192.168.1.100:11434' # Keep if needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser()
# Basic Configs
parser.add_argument("--source", type=str, default='dblpv7', help="Source dataset name")
parser.add_argument("--target", type=str, default='citationv1', help="Target dataset name")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--label_rate", type=float, default=0.05, help="Fraction of source nodes used for labeled training")

# Model Hyperparameters
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=2e-3)
parser.add_argument("--drop_out", type=float, default=1e-1) # Note: Dropout not explicitly used in cls_model in original code
parser.add_argument("--encoder_dim", type=int, default=512, help="Dimension of the node embeddings")
parser.add_argument("--expert_num", type=int, default=3, help="Number of experts in MoE")
parser.add_argument("--gate_coef", type=float, default=3e-2, help="Coefficient for MoE gate loss (if used)")
parser.add_argument("--gnn_type", type=str, default='ppmi', choices=['ppmi', 'gcn'], help="Type of GNN expert to use (requires MoE class modification)") # For Exp 7

# Loss Weights & Strategy Configs
parser.add_argument("--select_weight", type=float, default=0.01, help="Weight for LLM selection loss")
parser.add_argument("--semi_weight", type=float, default=1.0, help="Weight for high-quality semi-supervised loss")
parser.add_argument("--uncertainty_k", type=int, default=50, help="Number of top uncertain nodes to select for potential LLM query")
parser.add_argument("--llm_interval", type=int, default=20, help="Interval of epochs to call LLM for expert selection")
parser.add_argument("--llm", type=str, default='qwen2.5:7b', help="Ollama model name for expert selection")

# Ablation Mode
parser.add_argument("--ablation_mode", type=str, default="none",
                    choices=["none", "no_moe", "no_renode", "no_llm_complete"],
                    help="Specify the ablation experiment mode")

# W&B Configs
parser.add_argument("--wandb_project", type=str, default="GNN-Domain-Adaptation-Ablation")
parser.add_argument("--wandb_entity", type=str, default=None, help="Your W&B username or team name")
parser.add_argument("--wandb_disabled", action="store_true", help="Disable W&B logging")

args = parser.parse_args()

# --- W&B Initialization ---
if wandb_available and not args.wandb_disabled:
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        mode="online" # or "disabled" if args.wandb_disabled
    )
    config = wandb.config # Use wandb.config for all hyperparameters
    # Construct a meaningful run name
    run_name_parts = [
        f"{config.source}-{config.target}",
        f"abl_{config.ablation_mode}",
    ]
    if config.ablation_mode == 'none':
         run_name_parts.append(f"sw{config.select_weight:.1f}")
         run_name_parts.append(f"smw{config.semi_weight:.1f}")
         run_name_parts.append(f"gc{config.gate_coef:.1e}")
         run_name_parts.append(f"exp{config.expert_num}")
         run_name_parts.append(f"gtype_{config.gnn_type}")
    elif config.ablation_mode == 'no_moe':
        run_name_parts.append(f"smw{config.semi_weight:.1f}") # Still relevant
    # Add other relevant params based on ablation mode if needed
    run_name_parts.append(f"seed{config.seed}")
    wandb.run.name = "-".join(run_name_parts)

else:
    # If wandb is not used, fall back to args Namespace
    config = args
    print("W&B logging disabled.")
# --- End W&B Initialization ---


# --- Seeding ---
seed = config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic=True # Can slow down training
# torch.backends.cudnn.benchmark = False

# --- Logging Setup ---
id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}, drop_out: {}, expert_num: {}, llm: {}, uncertainty_k: {}, gate_coef: {}, select_w: {}, semi_w: {}, ablation: {}, gnn_type: {}" \
    .format(config.source, config.target, config.seed, config.label_rate, config.learning_rate, config.weight_decay,
            config.encoder_dim, config.drop_out, config.expert_num, config.llm, config.uncertainty_k, config.gate_coef,
            config.select_weight, config.semi_weight, config.ablation_mode, config.gnn_type)
print("--- Experiment Configuration ---")
print(id_str)
print("-----------------------------")


# --- Data Loading ---
dataset = DomainData("data/{}".format(config.source), name=config.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print("Source Data:", source_data)

dataset = DomainData("data/{}".format(config.target), name=config.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print("Target Data:", target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

# Create label mask
source_train_size = int(source_data.size(0) * config.label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)


# --- Helper Functions (index2dense, evaluate) ---
def index2dense(edge_index, nnode):
    device = edge_index.device
    adj = torch.zeros((nnode, nnode), dtype=torch.float32, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    # For undirected graphs, ensure symmetry if needed by the algorithm (PPMI usually handles this)
    # adj = adj + adj.T
    # adj[adj > 1] = 1 # Clamp values to 1
    return adj

def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    # Ensure labels/preds are on CPU for sklearn
    labels_cpu = labels.cpu().detach().numpy() # Use numpy for sklearn
    preds_cpu = preds.cpu().detach().numpy()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1
# --- End Helper Functions ---

# --- PPMI Loading ---
# Load precomputed PPMI matrices if they exist
os.makedirs('tmp', exist_ok=True)
source_pkl_path = f'tmp/{config.source}.pkl'
target_pkl_path = f'tmp/{config.target}.pkl'

try:
    with open(source_pkl_path, 'rb') as f:
        source_edge_index, _ = pickle.load(f)
    source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)
    print(f"Loaded precomputed PPMI for {config.source}")
except FileNotFoundError:
    print(f"Warning: {source_pkl_path} not found. PPMI features may not be used if GNN requires them.")
    # Fallback or compute PPMI here if necessary for the chosen GNN type
    source_data.new_adj = index2dense(source_data.edge_index, source_data.num_nodes).to(device) # Simple adj fallback


try:
    with open(target_pkl_path, 'rb') as f:
        target_edge_index, _ = pickle.load(f)
    target_data.new_adj = index2dense(target_edge_index, target_data.num_nodes).to(device)
    print(f"Loaded precomputed PPMI for {config.target}")
except FileNotFoundError:
    print(f"Warning: {target_pkl_path} not found. PPMI features may not be used if GNN requires them.")
    target_data.new_adj = index2dense(target_data.edge_index, target_data.num_nodes).to(device) # Simple adj fallback


# --- Model Definition ---

# Define a simple GNN Encoder for the 'no_moe' ablation
class SimplePPMIEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2):
        super().__init__()
        # We assume PPMIConv uses the precomputed new_adj implicitly via cache name
        self.convs = nn.ModuleList()
        hidden_size = output_size # Or introduce a separate hidden dim
        self.convs.append(PPMIConv(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.convs.append(PPMIConv(hidden_size, output_size))
        # No explicit dropout in the original encoder, add if desired
        # self.dropout = nn.Dropout(p=config.drop_out)

    def forward(self, x, edge_index, cache_name):
        # Note: PPMIConv implementation in gnn/ppmi_conv.py needs to correctly
        # use the cache_name to fetch the precomputed `data.new_adj`
        for conv in self.convs[:-1]:
             # PPMIConv might not need edge_index if using precomputed adj
            x = F.relu(conv(x, cache_name))
            # x = self.dropout(x) # Add dropout if needed
        x = self.convs[-1](x, cache_name)
        return x

# Initialize Encoder based on ablation mode
if config.ablation_mode == 'no_moe':
    print("--- Ablation Mode: No MoE ---")
    encoder = SimplePPMIEncoder(input_size=source_data.num_features,
                                output_size=config.encoder_dim,
                                num_layers=2).to(device) # 2 layers as suggested
    # Ensure expert_num is consistent if other parts of the code rely on it
    config.expert_num = 1 # Explicitly set for clarity downstream
else:
    print("--- Ablation Mode: Using MoE ---")
    # Adjust num_hops based on expert_num
    num_hops_config = [1, 2, 3][:config.expert_num] if config.expert_num >= 3 else [1] * config.expert_num
    # Ensure num_hops_config has length equal to expert_num
    if len(num_hops_config) < config.expert_num:
        num_hops_config.extend([num_hops_config[-1]] * (config.expert_num - len(num_hops_config))) # Pad with last value

    encoder = MoE(input_size=source_data.num_features,
                  output_size=config.encoder_dim,
                  num_experts=config.expert_num,
                  k=1, # Default k=1 from original code
                  coef=config.gate_coef,
                  gnn_type=config.gnn_type, # Pass gnn_type for Exp 7
                  num_hops=num_hops_config).to(device)

# Classifier remains the same
cls_model = nn.Sequential(
    nn.Linear(config.encoder_dim, dataset.num_classes),
    # nn.Dropout(p=config.drop_out), # Add if dropout is desired here
).to(device)


# --- Optimizer ---
models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)


# --- Encoding and Prediction Functions ---
def encode(data, cache_name, mask=None):
    if config.ablation_mode == 'no_moe':
        encoded_output = encoder(data.x, data.edge_index, cache_name)
        # Return dummy values for MoE-specific outputs
        experts_outputs = None # Or encoded_output.unsqueeze(1) if needed downstream? Check usage.
        gate_loss = torch.tensor(0.0, device=device)
        clean_logits = None # Not produced by simple encoder
    else:
        # Original MoE encoding
        encoded_output, experts_outputs, gate_loss, clean_logits = encoder(data.x, data.edge_index, cache_name)

    if mask is not None:
        if encoded_output is not None:
             encoded_output = encoded_output[mask]
        if experts_outputs is not None:
             experts_outputs = experts_outputs[mask]
        if clean_logits is not None:
             clean_logits = clean_logits[mask]

    return encoded_output, experts_outputs, gate_loss, clean_logits

def predict(data, cache_name, mask=None):
    encoded_output, _, _, _ = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits

# --- Testing Function ---
def test(data, cache_name, mask=None):
    encoder.eval()
    cls_model.eval()
    with torch.no_grad():
        encoded_output, experts_outputs, _, clean_logits = encode(data, cache_name, mask)
        logits = cls_model(encoded_output) # Use cls_model directly on final encoding
        preds = logits.argmax(dim=1)
        labels = data.y if mask is None else data.y[mask]
        accuracy, macro_f1, micro_f1 = evaluate(preds, labels)

        # Log expert selection during testing for target data (Only for MoE)
        if config.ablation_mode != 'no_moe' and cache_name == config.target:
            try:
                moe_expert_indices, moe_expert_probs, _ = encoder.get_node_expert_assignment(data.x, data.edge_index)
                # Ensure log dir exists
                log_dir = os.path.join("log", wandb.run.id if wandb_available and not config.wandb_disabled else "local_run")
                os.makedirs(log_dir, exist_ok=True)
                expert_selection_log_path = os.path.join(log_dir, f"{config.target}-expert-selection-test.csv")

                with open(expert_selection_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Node ID', 'Selected Expert', 'Probability'])
                    if isinstance(moe_expert_indices, torch.Tensor) and moe_expert_indices.ndim >= 1:
                        num_nodes_sel = moe_expert_indices.shape[0]
                        num_k_sel = moe_expert_indices.shape[1] if moe_expert_indices.ndim > 1 else 1
                        for node_id in range(num_nodes_sel):
                            for k in range(num_k_sel):
                                expert_idx_tensor = moe_expert_indices[node_id, k] if moe_expert_indices.ndim > 1 else moe_expert_indices[node_id]
                                expert_prob_tensor = moe_expert_probs[node_id, k] if moe_expert_probs.ndim > 1 else moe_expert_probs[node_id]
                                writer.writerow([node_id, expert_idx_tensor.item(), expert_prob_tensor.item()])

                # Log artifact to W&B if enabled
                if wandb_available and not config.wandb_disabled:
                    artifact = wandb.Artifact(f'{config.target}_expert_selection_{config.ablation_mode}', type='analysis_results')
                    artifact.add_file(expert_selection_log_path)
                    wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: Could not log expert selection during test: {e}")

    return accuracy, macro_f1, micro_f1, encoded_output

# --- ReNode Weight Function ---
def get_renode_weight(data, pseudo_label):
    ppr_matrix = data.new_adj # Assumes new_adj is the PPR matrix or similar
    gpr_matrix = []
    for iter_c in range(data.num_classes):
        class_mask = (pseudo_label == iter_c)
        if class_mask.sum() == 0: # Handle case where a class has no pseudo-labels
             iter_gpr = torch.zeros(ppr_matrix.shape[1], device=ppr_matrix.device)
        else:
             iter_gpr = torch.mean(ppr_matrix[class_mask], dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    gpr_matrix = torch.stack(gpr_matrix, dim=0).transpose(0, 1) # Shape: [nnode, num_classes]
    base_w = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)

    # Handle cases with few classes
    if data.num_classes <= 1:
        # Cannot calculate gpr_rn reasonably, return uniform weights
        print("Warning: ReNode requires > 1 class. Returning uniform weights.")
        return torch.ones(nnode, device=ppr_matrix.device)

    gpr_sum = torch.sum(gpr_matrix, dim=1)
    gpr_rn = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix, gpr_matrix) - torch.mm(ppr_matrix, gpr_rn) / (data.num_classes - 1.0)

    label_matrix = F.one_hot(pseudo_label, gpr_matrix.size(1)).float()
    # Element-wise multiply and sum over classes
    rn_matrix = torch.sum(rn_matrix * label_matrix, dim=1)

    # Use torch for sorting and ranking for potential GPU acceleration
    sorted_indices = torch.argsort(rn_matrix, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(nnode, device=ranks.device)

    # Calculate weights using torch operations
    cos_term = torch.cos(ranks * math.pi / (nnode - 1)) if nnode > 1 else torch.ones_like(ranks)
    rn_weight = base_w + 0.5 * scale_w * (1 + cos_term)

    return rn_weight.float() # Ensure float

# --- Loss Function ---
loss_func = nn.CrossEntropyLoss().to(device)

# --- Entropy Function ---
def Entropy(input_logits, weight, target_onehot):
    # Expects logits as input
    log_softmax_out = F.log_softmax(input_logits, dim=-1)
    # Negative log likelihood using one-hot target
    entropy = -torch.sum(target_onehot * log_softmax_out, dim=1) # NLL per sample
    entropy_loss = torch.mean(weight * entropy) # Weighted average NLL

    # Im-balance loss (encourages uniform predictions across batch/dataset)
    softmax_out = torch.softmax(input_logits, dim=-1)
    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-9)) # Add negative entropy of mean softmax

    return entropy_loss

# --- Uncertainty and Neighbor Functions ---
def calculate_expert_uncertainty(experts_outputs, num_classes, cls_model, uncertainty_k=5):
    # Calculates uncertainty based on variance/disagreement among expert predictions
    # This function is only relevant for MoE setups.
    if experts_outputs is None: # Handle 'no_moe' case
        return None

    num_nodes, num_experts, _ = experts_outputs.shape
    experts_logits = torch.zeros(num_nodes, num_experts, num_classes, device=experts_outputs.device)

    # Get predictions from each expert
    for i in range(num_experts):
        experts_logits[:, i, :] = cls_model(experts_outputs[:, i, :])

    experts_probs = torch.softmax(experts_logits, dim=-1) # [N, num_experts, C]

    # Uncertainty Metric: Predictive Entropy of the Mean Prediction
    # 1. Average probabilities across experts: mean_probs = mean(experts_probs, dim=1) -> [N, C]
    # 2. Calculate entropy of this mean distribution: H = -sum(mean_probs * log(mean_probs)) -> [N]
    mean_probs = torch.mean(experts_probs, dim=1)
    uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9), dim=-1)

    # --- Alternative: Variance of logits or probs ---
    # prob_variance = torch.var(experts_probs, dim=1).mean(dim=-1) # Mean variance across classes
    # uncertainty = prob_variance

    # Select top K uncertain nodes
    k = min(uncertainty_k, uncertainty.size(0))
    if k > 0:
        _, topk_indices = torch.topk(uncertainty, k, dim=0)
        uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)
        uncertainty_mask[topk_indices] = True
    else:
        uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)

    return uncertainty_mask


def get_max_hop_neighbors(edge_index, num_nodes, mask):
    # Finds all nodes reachable from the initial 'mask' nodes (inefficient for large graphs)
    # Consider using k-hop subgraph methods from PyG for efficiency if needed.
    if not mask.any(): return mask.clone() # No starting nodes

    neighbor_mask = mask.clone()
    visited_mask = mask.clone()
    frontier = mask.clone()

    # Build adjacency list (more efficient for traversal than dense matrix)
    adj = [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.cpu()
    for i in range(edge_index.shape[1]):
        u, v = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
        adj[u].append(v)
        adj[v].append(u) # Assuming undirected

    current_frontier_indices = torch.where(frontier)[0].tolist()

    while current_frontier_indices:
        new_frontier_indices = []
        next_frontier_mask = torch.zeros_like(frontier)
        for node_idx in current_frontier_indices:
            for neighbor in adj[node_idx]:
                if not visited_mask[neighbor]:
                    visited_mask[neighbor] = True
                    next_frontier_mask[neighbor] = True
                    new_frontier_indices.append(neighbor)

        if not new_frontier_indices: # No new nodes found
            break

        neighbor_mask |= next_frontier_mask
        current_frontier_indices = new_frontier_indices
        # Safety break if something goes wrong
        if visited_mask.all(): break


    return neighbor_mask
# --- End Uncertainty and Neighbor Functions ---


# --- Training Function ---
train_step = 0 # Global step counter for W&B
def train(epoch):
    global train_step
    for model in models:
        model.train()
    optimizer.zero_grad()

    # --- Source Domain Forward Pass ---
    encoded_source, experts_outputs, source_gate_loss, source_clean_logits = encode(source_data, config.source)
    source_logits = cls_model(encoded_source) # Final prediction from aggregated output

    # --- Classifier Loss ---
    cls_loss = loss_func(source_logits[label_mask], source_data.y[label_mask])

    # --- LLM Expert Selection Logic (Exp 2 & 8 Ablations) ---
    select_loss = torch.tensor(0.0, device=device)
    llm_active = (config.ablation_mode not in ['no_moe', 'no_llm_complete']) and \
                 (config.select_weight > 0) and \
                 ollama_available

    if llm_active:
        uncertainty_mask = calculate_expert_uncertainty(experts_outputs, source_data.num_classes, cls_model, config.uncertainty_k)

        if uncertainty_mask is not None and uncertainty_mask.sum() > 0:
            # Use a persistent dictionary to store expert selections across epochs
            if not hasattr(train, 'expert_selections_cache'):
                train.expert_selections_cache = {}

            # Only call LLM every llm_interval epochs
            uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()
            if (epoch - 1) % config.llm_interval == 0:
                print(f"Epoch {epoch}: Querying LLM for {len(uncertainty_node_indices)} uncertain nodes...")
                # graph2text_encoder = Graph2TextEncoder() # Initialize if needed, maybe globally?
                current_epoch_selections = {}
                prompts = []
                # --- Prepare Prompts (Simplified Graph Description) ---
                # NOTE: The original Graph2TextEncoder might be too slow/complex.
                # Using a simpler description for demonstration.
                adj_for_prompt = source_data.edge_index.cpu().numpy() # Use original edges
                for node_id in uncertainty_node_indices:
                    neighbors = adj_for_prompt[1, adj_for_prompt[0] == node_id]
                    graph_desc = f"Node {node_id} has {len(neighbors)} direct neighbors: {neighbors.tolist()[:10]}..." # Basic description
                    expert_options = " ".join([f'{i}:{i+1}-hop' for i in range(config.expert_num)])
                    prompt = f"""You are a GNN expert selector. Given graph info: {graph_desc}.
                    Available experts: ({expert_options}).
                    - 1-hop: Good for direct neighbor signals.
                    - 2-hop: Good for indirect relationships.
                    - 3-hop: Good for wider context/community structure.
                    Choose the best expert index (0 to {config.expert_num - 1}) for node {node_id}.
                    Return ONLY JSON: {{"reason": "Your brief reason", "expert": index}}"""
                    prompts.append(prompt)

                # --- Query LLM ---
                for idx, node_id in enumerate(uncertainty_node_indices):
                    prompt = prompts[idx]
                    try:
                        response = ollama.generate(
                            model=config.llm,
                            prompt=prompt,
                            format='json',
                            options={'temperature': 0.1} # Low temp for consistency
                        )
                        json_output_str = response['response']
                        parsed_json = json.loads(json_output_str)
                        expert = parsed_json.get("expert")
                        if isinstance(expert, int) and 0 <= expert < config.expert_num:
                            current_epoch_selections[node_id] = expert
                        else:
                            print(f"Warning: Invalid expert {expert} from LLM for node {node_id}. Defaulting random.")
                            current_epoch_selections[node_id] = np.random.randint(0, config.expert_num)
                    except Exception as e:
                        # Handle JSON errors, connection errors etc.
                        print(f"Warning: LLM query failed for node {node_id}: {e}. Defaulting random.")
                        current_epoch_selections[node_id] = np.random.randint(0, config.expert_num)

                # Update cache
                train.expert_selections_cache.update(current_epoch_selections)
                # Print some selections
                print(f"LLM Selections (sample): {list(current_epoch_selections.items())[:5]}")

            # --- Calculate Selection Loss using cached selections ---
            # Use selections from cache for nodes that were uncertain in this epoch
            expert_selections_for_loss = {nid: train.expert_selections_cache[nid]
                                          for nid in uncertainty_node_indices
                                          if nid in train.expert_selections_cache}

            if expert_selections_for_loss:
                loss_node_indices = list(expert_selections_for_loss.keys())
                loss_node_indices_tensor = torch.tensor(loss_node_indices, device=device, dtype=torch.long)

                # Get MoE's routing decision (clean logits before softmax/aggregation) for these nodes
                # Assuming source_clean_logits is [N, num_experts] representing gate outputs
                if source_clean_logits is not None:
                    moe_expert_logits_for_loss = source_clean_logits[loss_node_indices_tensor] # Shape [N_loss, num_experts]

                    # Target distribution based on LLM choice (one-hot)
                    llm_expert_indices_list = [expert_selections_for_loss[nid] for nid in loss_node_indices]
                    llm_expert_indices_tensor = torch.tensor(llm_expert_indices_list, device=device, dtype=torch.long)
                    #llm_expert_dist_one_hot = F.one_hot(llm_expert_indices_tensor, num_classes=config.expert_num).float()

                    # Calculate CrossEntropy loss between MoE's routing logits and LLM's target index
                    select_loss = F.cross_entropy(moe_expert_logits_for_loss, llm_expert_indices_tensor)
                else:
                     print("Warning: source_clean_logits not available for select_loss calculation.")
                     select_loss = torch.tensor(0.0, device=device)

    # --- Semi-Supervised Loss (Exp 3 & 4 Ablations) ---
    high_quality_semi_loss = torch.tensor(0.0, device=device)
    if config.semi_weight > 0:
        unlabeled_mask = ~label_mask
        if unlabeled_mask.sum() > 0:
            with torch.no_grad(): # Pseudo-label generation doesn't need gradients
                s_plabel = source_logits.argmax(dim=1)
                s_plabel[label_mask] = source_data.y[label_mask] # Correct known labels
                s_plabel_onehot = F.one_hot(s_plabel, source_data.num_classes).float()

                # Get ReNode weights (or uniform weights for Exp 4)
                if config.ablation_mode == 'no_renode':
                    s_weight = torch.ones(source_data.num_nodes, device=device)
                    # print("Using uniform weights (ablation: no_renode)") # Optional print
                else:
                    s_weight = get_renode_weight(source_data, s_plabel).to(device)
                    # print("Using ReNode weights") # Optional print


            # --- High-Quality Node Selection (Only for MoE) ---
            if config.ablation_mode == 'no_moe':
                 # For No MoE, apply semi-loss to all unlabeled nodes
                 high_quality_mask_unlabeled = torch.ones_like(unlabeled_mask[unlabeled_mask]) # Mask of True for all unlabeled
                 print("Applying semi-loss to all unlabeled nodes (Ablation: no_moe)")
            else:
                # Original MoE-based high-quality selection
                moe_softmax = torch.softmax(source_logits[unlabeled_mask], dim=-1)
                moe_entropy = -torch.sum(moe_softmax * torch.log(moe_softmax + 1e-9), dim=1)

                # Calculate average expert entropy (only if experts_outputs exist)
                avg_expert_entropy = torch.full_like(moe_entropy, float('inf')) # Default to infinite entropy if no experts
                if experts_outputs is not None:
                    unlabeled_experts_outputs = experts_outputs[unlabeled_mask] # [N_unlabeled, num_experts, D]
                    num_unlabeled, num_experts_actual, _ = unlabeled_experts_outputs.shape
                    if num_experts_actual > 0:
                        expert_entropies_unlabeled = torch.zeros(num_unlabeled, num_experts_actual, device=device)
                        for i in range(num_experts_actual):
                            expert_logits_unlabeled = cls_model(unlabeled_experts_outputs[:, i, :])
                            expert_softmax_unlabeled = torch.softmax(expert_logits_unlabeled, dim=-1)
                            expert_entropies_unlabeled[:, i] = -torch.sum(expert_softmax_unlabeled * torch.log(expert_softmax_unlabeled + 1e-9), dim=1)
                        avg_expert_entropy = torch.mean(expert_entropies_unlabeled, dim=1)

                high_quality_mask_unlabeled = (moe_entropy < avg_expert_entropy)
                # print(f"HQ Semi: Selected {high_quality_mask_unlabeled.sum()}/{unlabeled_mask.sum()} nodes.") # Debug print


            # --- Calculate HQ Semi-Loss ---
            if high_quality_mask_unlabeled.sum() > 0:
                # Get indices of unlabeled nodes that are also high quality
                unlabeled_indices = torch.where(unlabeled_mask)[0]
                hq_indices_in_unlabeled = torch.where(high_quality_mask_unlabeled)[0]
                hq_indices_global = unlabeled_indices[hq_indices_in_unlabeled]


                # Calculate loss only on high quality subset
                high_quality_semi_loss = Entropy(
                    input_logits=source_logits[hq_indices_global],
                    weight=s_weight[hq_indices_global],
                    target_onehot=s_plabel_onehot[hq_indices_global]
                )

    # --- Gate Loss (Only for MoE, and often implicitly handled or added if needed) ---
    # The original code calculates gate_loss but doesn't add it to the final loss.
    # We keep it this way unless the MoE implementation requires explicit addition.
    # gate_loss = source_gate_loss # If returned by encode and needed
    gate_loss = torch.tensor(0.0, device=device) # Default to 0 if not used in final loss

    # --- Total Loss ---
    # epoch_weight = float(epoch) / config.epochs # Original epoch weighting
    epoch_weight = 1.0 # Simpler: use fixed weights first

    # Combine losses based on weights and ablation mode
    loss = cls_loss
    if config.ablation_mode not in ['no_moe', 'no_llm_complete'] and config.select_weight > 0:
         loss += epoch_weight * config.select_weight * select_loss
    if config.semi_weight > 0:
         loss += epoch_weight * config.semi_weight * high_quality_semi_loss
    # Add gate_loss here if required by the MoE implementation and ablation design
    # if config.ablation_mode != 'no_moe' and config.gate_coef > 0:
    #    loss += config.gate_coef * source_gate_loss # Example

    # --- Backpropagation ---
    loss.backward()
    # Optional: Gradient Clipping
    # torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()

    # --- W&B Logging for Training Step ---
    if wandb_available and not config.wandb_disabled:
        log_dict = {
            'train/epoch': epoch,
            'train/cls_loss': cls_loss.item(),
            'train/select_loss': select_loss.item(),
            'train/high_quality_semi_loss': high_quality_semi_loss.item(),
            'train/total_loss': loss.item(),
            # 'train/gate_loss': gate_loss.item(), # Log if relevant
        }
        wandb.log(log_dict, step=train_step)
    train_step += 1
    # --- End W&B Logging ---

    # Optional: Print losses periodically
    if epoch % 10 == 0 or epoch == 1:
         print(f"Epoch {epoch:03d} | ClsLoss: {cls_loss.item():.4f} | SelLoss: {select_loss.item():.4f} "
               f"| HQSemiLoss: {high_quality_semi_loss.item():.4f} | TotalLoss: {loss.item():.4f}")


# --- Training Loop ---
best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0
best_macro_f1 = 0.0
best_micro_f1 = 0.0

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(1, config.epochs + 1):
    train(epoch)
    # --- Evaluation Step ---
    try:
        # Evaluate on source test set (using mask if defined, else evaluate all?)
        # Original code used source_data.test_mask - checking if it exists
        source_test_mask = getattr(source_data, 'test_mask', None)
        source_acc, _, _, _ = test(source_data, config.source, mask=source_test_mask)

        # Evaluate on target set (usually evaluated on all nodes)
        target_acc, macro_f1, micro_f1, _ = test(target_data, config.target)

        # --- W&B Logging for Evaluation Step ---
        if wandb_available and not config.wandb_disabled:
             wandb.log({
                'eval/epoch': epoch,
                'eval/source_acc': source_acc,
                'eval/target_acc': target_acc,
                'eval/macro_f1': macro_f1,
                'eval/micro_f1': micro_f1,
             }, step=train_step) # Log against global step
        # --- End W&B Logging ---

        print(f"Epoch {epoch:03d} | SrcAcc: {source_acc:.4f} | TgtAcc: {target_acc:.4f} | MacroF1: {macro_f1:.4f} | MicroF1: {micro_f1:.4f}")

        # Update best metrics based on target accuracy
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            best_source_acc = source_acc
            best_macro_f1 = macro_f1
            best_micro_f1 = micro_f1
            best_epoch = epoch
            print(f"    *** New best target accuracy at epoch {epoch}: {best_target_acc:.4f} ***")
            # --- Save Best Model (Optional) ---
            # save_path = f'checkpoints/best_model_{wandb.run.id if wandb_available else "local"}.pt'
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # torch.save({'encoder': encoder.state_dict(), 'cls_model': cls_model.state_dict(), 'epoch': epoch}, save_path)


    except Exception as e:
        print(f"Error during training/evaluation at epoch {epoch}: {e}")
        # Optionally log error to W&B or break the loop
        if wandb_available and not config.wandb_disabled:
            wandb.log({'error': str(e)})
        raise e # Re-raise error to stop if critical

# --- End Training Loop ---

total_time = time.time() - start_time
print("\n--- Training Finished ---")
print(f"Total Training Time: {total_time:.2f} seconds")

# --- Final Results Logging ---
print("=============================================================")
final_id_str = "source: {}, target: {}, seed: {}, ablation: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}, expert_num: {}, gate_coef: {}, select_w: {}, semi_w: {}, gnn_type: {}" \
    .format(config.source, config.target, config.seed, config.ablation_mode, config.label_rate, config.learning_rate,
            config.weight_decay, config.encoder_dim, config.expert_num, config.gate_coef, config.select_weight,
            config.semi_weight, config.gnn_type)
line = "{}\n - Best Epoch: {}, Best Source Acc: {:.5f}, Best Target Acc: {:.5f}, Best Macro F1: {:.5f}, Best Micro F1: {:.5f}" \
    .format(final_id_str, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)
print(line)

# --- Log Best Metrics to W&B Summary ---
if wandb_available and not config.wandb_disabled:
    wandb.summary['best_epoch'] = best_epoch
    wandb.summary['best_source_acc'] = best_source_acc
    wandb.summary['best_target_acc'] = best_target_acc
    wandb.summary['best_macro_f1'] = best_macro_f1
    wandb.summary['best_micro_f1'] = best_micro_f1
    wandb.summary['total_training_time'] = total_time
# --- End W&B Summary Logging ---

# Log final results to a local file (optional)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"{config.source}-{config.target}-ablations.log")
with open(log_file_path, 'a') as f:
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    run_url = run.url if wandb_available and not config.wandb_disabled and run else "N/A"
    log_line = f"[{timestamp}] {final_id_str} - Best Epoch: {best_epoch:0>3d}, Best Tgt Acc: {best_target_acc:.5f}, Best Macro F1: {best_macro_f1:.5f}, Best Micro F1: {best_micro_f1:.5f} - W&B: {run_url}\n"
    f.write(log_line)

# --- Finish W&B Run ---
if wandb_available and not config.wandb_disabled:
    # Save the log file as a W&B artifact (optional)
    # log_artifact = wandb.Artifact('run_log', type='log_file')
    # log_artifact.add_file(log_file_path)
    # wandb.log_artifact(log_artifact)
    wandb.finish()
# --- End Finish W&B Run ---

print("--- Script Finished ---")