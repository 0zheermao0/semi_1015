# coding=utf-8
import os
import csv
import argparse # Keep argparse for non-sweep parameters or defaults
import random
import numpy as np
import torch
# import torch.functional as F # This seems unused, commented out
import torch.nn as nn # Use nn directly
import torch.nn.functional as F # Keep this one
import itertools
import time
import warnings
import pickle
import math
from sklearn.metrics import f1_score
import json
from common.openai_client import create_openai_client
import multiprocessing
from pathlib import Path # For checking CSV existence

# --- W&B Integration ---
import wandb
# --- End W&B Integration ---

# Assuming these modules are in the specified paths relative to the script
try:
    from gnn.cached_gcn_conv import CachedGCNConv
    from gnn.dataset.DomainData import DomainData
    from gnn.ppmi_conv import PPMIConv
    from gnn.moe import MoE
    from common.graph_encoder import GraphEncoder as Graph2TextEncoder
except ImportError as e:
    print(f"Error importing GNN modules: {e}")
    print("Please ensure the 'gnn' and 'common' directories are correctly placed relative to the script or in the Python path.")
    exit()

# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM # Seems unused

warnings.filterwarnings("ignore", category=UserWarning)
# os.environ['OLLAMA_HOST'] = 'http://192.168.1.100:11434' # Set Ollama host if needed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
# --- Basic Setup ---
parser.add_argument("--source", type=str, default='dblpv7')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=2) # Updated Default
parser.add_argument("--label_rate", type=float, default=0.05)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--drop_out", type=float, default=1e-1)
parser.add_argument("--hop", type=int, default=5) # Default hop for LLM prompt generation

# --- Hyperparameters (with updated defaults) ---
parser.add_argument("--learning_rate", type=float, default=1e-3) # Updated Default
parser.add_argument("--weight_decay", type=float, default=2e-3)
parser.add_argument("--uncertainty_k", type=int, default=100) # Updated Default
parser.add_argument("--gate_coef", type=float, default=3e-2) # Updated Default
parser.add_argument("--select_weight", type=float, default=0.8) # Updated Default
parser.add_argument("--semi_weight", type=float, default=1)
parser.add_argument("--div_weight", type=float, default=1.0) # Updated Default based on user request
parser.add_argument("--epochs", type=int, default=200)

# --- MoE Specific ---
parser.add_argument("--expert_num", type=int, default=3)

# --- LLM Specific ---
parser.add_argument("--llm", type=str, default='gpt-3.5-turbo') # Updated Default
parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (can also be set via OPENAI_API_KEY env var)")
parser.add_argument("--openai_base_url", type=str, default=None, help="OpenAI-compatible API base URL (can also be set via OPENAI_BASE_URL env var)")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for LLM")
parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens in LLM response")
parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
parser.add_argument("--llm_interval", type=int, default=10) # Note: LLM is pre-calculated

# --- Ablation Flags ---
parser.add_argument("--ablate_moe", action='store_true', help="Remove MoE module, use single GNN encoder")
parser.add_argument("--ablate_llm_select", action='store_true', help="Remove LLM-based expert selection mechanism and loss")
parser.add_argument("--ablate_hq_semi", action='store_true', help="Remove high-quality pseudo-labeling loss component")
parser.add_argument("--ablate_gate_loss", action='store_true', help="Remove MoE gate/diversity loss component")

# --- W&B specific arguments ---
parser.add_argument("--wandb_project", type=str, default="GNN-Ablation-Study") # Project name for ablations
parser.add_argument("--wandb_entity", type=str, default=None, help="Your W&B username or team name")


args = parser.parse_args()

# -------- START FIX: Modify args *before* wandb.init --------
# If MoE is ablated, force related parameters and ablation flags
if args.ablate_moe:
    print("INFO: MoE module is ablated. Adjusting related parameters before W&B init.")
    original_expert_num = args.expert_num
    args.expert_num = 1  # Set effective number of experts to 1

    # Automatically disable related components if MoE is removed
    if not args.ablate_gate_loss:
        args.ablate_gate_loss = True
        print("INFO: Gate loss automatically disabled due to MoE ablation.")
    if not args.ablate_llm_select:
        args.ablate_llm_select = True
        print("INFO: LLM selection automatically disabled due to MoE ablation.")
    if not args.ablate_hq_semi:
        args.ablate_hq_semi = True
        print("INFO: HQ semi-loss automatically disabled due to MoE ablation.")
# -------- END FIX --------

# --- W&B Initialization ---
# Pass the potentially modified 'args' to WandB
try:
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args # Pass the finalized argparse object here
    )
except Exception as e:
    print(f"Error initializing WandB: {e}")
    print("Please ensure you are logged in (`wandb login`) or set WANDB_API_KEY.")
    print("Running without WandB logging.")
    run = None # Set run to None to skip WandB logging calls later
    config = args # Use args directly if WandB fails

if run:
    # Access hyperparameters via `wandb.config` (which now reflects the modifications)
    config = wandb.config
    print(f"WandB run initialized: {run.url}")
else:
    # If WandB initialization failed, use args directly as config
    config = args
    print("WandB disabled. Using command-line args directly.")

# Initialize OpenAI client
llm_client = create_openai_client(config)
print(f"Initialized OpenAI-compatible client with model: {config.llm}")
if config.openai_base_url:
    print(f"Using custom base URL: {config.openai_base_url}")

# --- Seed Setting ---
seed = config.seed # Use config (or args if WandB failed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # Can slow down training
    # torch.backends.cudnn.benchmark = False
print(f"Seed set to: {seed}")
# --- End Seed Setting ---

# --- ID String and Run Name (incorporating ablation info) ---
ablation_str = ""
if config.ablate_moe: ablation_str += "-NoMoE"
# Note: These flags might be True either by command line OR because MoE was ablated
if config.ablate_llm_select and not config.ablate_moe: ablation_str += "-NoLLMSelect" # Avoid redundancy if NoMoE
if config.ablate_hq_semi and not config.ablate_llm_select and not config.ablate_moe: ablation_str += "-NoHQSemi" # Avoid redundancy
if config.ablate_gate_loss and not config.ablate_moe: ablation_str += "-NoGateLoss" # Avoid redundancy
if not ablation_str and not config.ablate_moe: ablation_str = "-FullModel" # Full only if MoE is present AND no other flags

id_str = "src:{},tgt:{},seed:{},lr:{:.1e},wd:{:.1e},dim:{},do:{:.1e},lbl_rt:{:.2f}{}".format(
    config.source, config.target, config.seed, config.learning_rate, config.weight_decay,
    config.encoder_dim, config.drop_out, config.label_rate, ablation_str
)
print(f"Run ID String: {id_str}")
if run:
    run_name = f"{config.source}-{config.target}-lr{config.learning_rate:.1e}-seed{config.seed}{ablation_str}"
    wandb.run.name = run_name
    print(f"WandB Run Name: {run_name}")
# --- End ID String ---

# --- Load Data ---
try:
    dataset_source = DomainData("data/{}".format(config.source), name=config.source)
    source_data = dataset_source[0]
    source_data.num_classes = dataset_source.num_classes
    print("Source Data:", source_data)

    dataset_target = DomainData("data/{}".format(config.target), name=config.target)
    target_data = dataset_target[0]
    # Ensure target uses the same number of classes as source
    target_data.num_classes = dataset_source.num_classes
    print("Target Data:", target_data)

    source_data = source_data.to(device)
    target_data = target_data.to(device)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the data directories (e.g., 'data/dblpv7', 'data/citationv1') exist.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- Create Label Mask ---
source_train_size = int(source_data.size(0) * config.label_rate)
if source_train_size == 0 and source_data.size(0) > 0:
    print(f"Warning: label_rate {config.label_rate} resulted in 0 training samples for source data size {source_data.size(0)}. Using 1 sample.")
    source_train_size = 1
elif source_train_size > source_data.size(0):
     print(f"Warning: label_rate {config.label_rate} resulted in {source_train_size} training samples, more than source data size {source_data.size(0)}. Clamping.")
     source_train_size = source_data.size(0)

label_mask_np = np.zeros(source_data.size(0), dtype=bool)
if source_train_size > 0:
    train_indices = np.random.choice(source_data.size(0), source_train_size, replace=False)
    label_mask_np[train_indices] = True
# label_mask_np = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool) # Old way
# np.random.shuffle(label_mask_np) # Shuffle is implicit in np.random.choice
label_mask = torch.tensor(label_mask_np).to(device)
print(f"Using {label_mask.sum().item()} labeled source nodes ({config.label_rate*100:.2f}%)")
# --- End Load Data ---


# --- Helper Functions ---
def index2dense(edge_index, nnode):
    device = edge_index.device
    # Handle case where nnode might be 0
    if nnode == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)
    adj = torch.zeros((nnode, nnode), dtype=torch.float32, device=device)
    if edge_index.numel() > 0: # Check if edge_index is not empty
        # Ensure indices are within bounds
        valid_mask = (edge_index[0] < nnode) & (edge_index[1] < nnode)
        edge_index_valid = edge_index[:, valid_mask]
        adj[edge_index_valid[0], edge_index_valid[1]] = 1.0
    return adj

def evaluate(preds, labels):
    if preds.numel() == 0 or labels.numel() == 0: # Handle empty tensors
        return 0.0, 0.0, 0.0
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean().item()
    labels_cpu = labels.cpu().detach().numpy()
    preds_cpu = preds.cpu().detach().numpy()
    # Calculate F1 scores safely
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1
# --- End Helper Functions ---


# --- ReNode Weight Function ---
def get_renode_weight(data, pseudo_label):
    # Ensure data has nodes before proceeding
    if data.num_nodes == 0:
        return torch.tensor([], dtype=torch.float32, device=pseudo_label.device)

    ppr_matrix = data.new_adj
    gpr_matrix = []
    for iter_c in range(data.num_classes):
        class_mask = (pseudo_label == iter_c)
        if class_mask.sum() == 0: # Handle case where a class has no nodes
             iter_gpr = torch.zeros(data.num_nodes, dtype=ppr_matrix.dtype, device=ppr_matrix.device)
        else:
             iter_gpr = torch.mean(ppr_matrix[class_mask], dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    # Check if gpr_matrix is empty or contains empty tensors before stacking
    if not gpr_matrix or any(t.numel() == 0 for t in gpr_matrix):
         return torch.ones(data.num_nodes, dtype=torch.float32, device=pseudo_label.device) # Return default weights


    gpr_matrix = torch.stack(gpr_matrix,dim=0).transpose(0,1)
    base_w  = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix

    # Handle division by zero if num_classes is 1
    if data.num_classes <= 1:
        rn_matrix = torch.mm(ppr_matrix, gpr_matrix) # Simplified calculation
    else:
        rn_matrix = torch.mm(ppr_matrix, gpr_matrix) - torch.mm(ppr_matrix, gpr_rn) / (data.num_classes - 1.0)

    label_matrix = F.one_hot(pseudo_label, num_classes=data.num_classes).float() # Use data.num_classes
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)

    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=True)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]

    # Avoid division by zero if nnode is 1
    if nnode <= 1:
        rn_weight = [base_w + 0.5 * scale_w] * nnode
    else:
        rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x * 1.0 * math.pi / (nnode - 1)))) for x in totoro_rank]

    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor).to(pseudo_label.device) # Move to correct device
    return rn_weight
# --- End ReNode Weight Function ---

# --- Uncertainty and Neighbor Functions ---
def calculate_expert_uncertainty(experts_outputs, num_classes, cls_model, uncertainty_k=5):
    # Handle case with no experts_outputs
    if experts_outputs is None or experts_outputs.numel() == 0:
        return torch.tensor([], dtype=torch.bool, device=device) # Match device

    num_nodes, num_experts, d_feature = experts_outputs.shape
    if num_nodes == 0:
        return torch.tensor([], dtype=torch.bool, device=experts_outputs.device)

    experts_logits = torch.zeros(num_nodes, num_experts, num_classes, device=experts_outputs.device)
    try:
        with torch.no_grad(): # No need for gradients here
            for i in range(num_experts):
                experts_logits[:, i, :] = cls_model(experts_outputs[:, i, :])
    except Exception as e:
        print(f"Error during cls_model call in calculate_expert_uncertainty: {e}")
        # Return an empty mask or handle error appropriately
        return torch.zeros(num_nodes, dtype=torch.bool, device=experts_outputs.device)


    experts_probs = torch.softmax(experts_logits, dim=-1)
    log_probs = torch.log(experts_probs + 1e-9) # Use 1e-9 for numerical stability
    joint_log_probs = torch.logsumexp(log_probs, dim=1) # LogSumExp across experts for each class
    total_confidence = torch.softmax(joint_log_probs, dim=-1) # Softmax across classes
    expected_ratios = joint_log_probs + torch.log(total_confidence + 1e-9)
    uncertainty = torch.logsumexp(expected_ratios, dim=-1) # LogSumExp across classes

    # Ensure k is not larger than the number of nodes
    k = min(uncertainty_k, uncertainty.size(0))
    if k > 0 and uncertainty.numel() > 0 : # Check uncertainty is not empty
        try:
             # Ensure uncertainty tensor is valid for topk
             if uncertainty.ndim == 0: # Handle scalar tensor case
                  uncertainty = uncertainty.unsqueeze(0)

             # Use largest=False to get smallest values (most uncertain) if uncertainty metric means higher is better
             # Assuming lower value means more uncertain based on context, so largest=True
             _, topk_indices = torch.topk(uncertainty, k, dim=0, largest=True)
             uncertainty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=experts_outputs.device)
             uncertainty_mask[topk_indices] = True
        except RuntimeError as e:
             print(f"Error during torch.topk in calculate_expert_uncertainty: {e}")
             print(f"Uncertainty shape: {uncertainty.shape}, k: {k}, num_nodes: {num_nodes}")
             uncertainty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=experts_outputs.device) # Fallback
    else:
        uncertainty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=experts_outputs.device)

    return uncertainty_mask

def get_max_hop_neighbors(edge_index, num_nodes, mask):
    # Handle empty graph or empty mask
    if num_nodes == 0 or mask.numel() == 0 or edge_index.numel() == 0:
        return torch.zeros_like(mask)

    neighbor_mask = mask.clone()
    current_nodes = torch.where(mask)[0]
    if current_nodes.numel() == 0: return neighbor_mask

    # Build adjacency list for efficiency (on the correct device)
    adj = [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.cpu().numpy() # Move to CPU for list building
    for i in range(edge_index_cpu.shape[1]):
        src, dst = edge_index_cpu[0, i], edge_index_cpu[1, i]
        if 0 <= src < num_nodes and 0 <= dst < num_nodes: # Bounds check
            adj[src].append(dst)
            adj[dst].append(src) # Assuming undirected

    # BFS to find all reachable nodes
    queue = current_nodes.tolist()
    visited_set = set(queue)

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if v not in visited_set:
                visited_set.add(v)
                neighbor_mask[v] = True # Update mask on original device
                queue.append(v)

    return neighbor_mask
# --- End Uncertainty and Neighbor Functions ---


# --- Build Prompt Function (Only needed if not ablating LLM) ---
# Define Graph2TextEncoder globally if needed for prompt building
if not args.ablate_llm_select and not args.ablate_moe:
     try:
          graph2text_encoder_global = Graph2TextEncoder()
     except NameError:
          print("Warning: Graph2TextEncoder class not found. Prompt generation will likely fail.")
          graph2text_encoder_global = None # Set to None to indicate failure

def build_prompt_for_node(args_tuple): # Renamed input arg
    node_id, source_data_cpu_dict, config_dict, expert_num_int, hop_int = args_tuple # Unpack

    # Reconstruct necessary parts of source_data from dict on CPU
    num_nodes = source_data_cpu_dict['num_nodes']
    edge_index = source_data_cpu_dict['edge_index'] # Should be CPU tensor

    if num_nodes == 0: return "" # Cannot generate prompt for empty graph

    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if 0 <= node_id < num_nodes: # Check node_id validity
         node_mask[node_id] = True
    else:
         print(f"Warning: Invalid node_id {node_id} for num_nodes {num_nodes} in build_prompt.")
         return "" # Cannot generate prompt for invalid node_id


    # Find k-hop neighbors on CPU using Adjacency List
    adj = [[] for _ in range(num_nodes)]
    edge_index_np = edge_index.numpy() # Already on CPU
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        # Basic bounds check
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
             adj[src].append(dst)
             adj[dst].append(src) # Assuming undirected

    visited = {node_id}
    current_level = {node_id}
    all_neighbors_in_hop = {node_id}

    for _ in range(hop_int): # Use hop_int from args
        next_level = set()
        if not current_level: # Stop if no new nodes were found in the previous hop
            break
        for u in current_level:
             # Check if u is valid index for adj list
             if 0 <= u < len(adj):
                  for v in adj[u]:
                       if v not in visited:
                            visited.add(v)
                            next_level.add(v)
                            all_neighbors_in_hop.add(v)
             else:
                  print(f"Warning: Invalid node index {u} encountered during neighbor search.")
        current_level = next_level

    # Set mask for all neighbors found within hop distance
    for n in all_neighbors_in_hop:
        if 0 <= n < num_nodes: # Bounds check before setting mask
             node_mask[n] = True

    # Use local graph2text_encoder instance if global failed or not needed
    if 'graph2text_encoder_global' in globals() and graph2text_encoder_global is not None:
         local_encoder = graph2text_encoder_global
    else:
         try: # Try to create locally if needed
              local_encoder = Graph2TextEncoder()
         except NameError:
               print("Error: Graph2TextEncoder not defined. Cannot generate graph description.")
               return "" # Return empty prompt


    try:
        graph_description = local_encoder.encode(
            edge_index, mask=node_mask, num_nodes=num_nodes, style='natural'
        )
    except Exception as e:
        print(f"Error during graph2text_encoder.encode for node {node_id}: {e}")
        graph_description = "[Error generating graph description]"


    # Use config_dict for expert_num (passed via args_tuple)
    # expert_num_int = config_dict.get('expert_num', 1) # Get from passed dict

    expert_desc = ' '.join([f'{i}:{i+1}-hop' for i in range(expert_num_int)])
    prompt = f"""
    You are an expert on GNN experts selector, given GNN experts: ({expert_desc})
    - 1-hop: Use when direct neighbors provide sufficient classification signals.
    - 2-hop: Use for indirect relationships.
    - 3-hop: Use for long-range dependencies or hierarchical structures.
    With node and its {hop_int}-hop neighorbood: {graph_description}
    and {expert_num_int}, These sub-graph represent paper connection relationships and each node is a paper.
    Now give out your choice on expert directly for node {node_id} classification.
    Please note:
    1. If the structure around the node is simple and there are few neighbors, it is recommended to choose 1-hop expert
    2. If the node has more 2-hop neighbors, it is recommended to choose 2-hop expert
    3. If the node is in a complex community structure, it is recommended to choose 3-hop expert
    4. Given the specific reason for the selection, it is necessary to be based on the actual structural characteristics of the node.
    return in json format directly:
    {{
        "reason": "your reason",
        "expert": 0, 1, ..., up to {expert_num_int - 1},
        "probability": 0.x
    }}
    """
    # print(f"Prompt for node {node_id}: {prompt[:200]}...") # Keep prompt short for printing
    return prompt
# --- End Helper Functions ---


# --- Define Simple Encoder for MoE Ablation ---
class SimplePPMIEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Using PPMIConv as the base layer, similar to MoE experts
        self.conv1 = PPMIConv(input_size, config.encoder_dim) # Adjust path if needed
        self.conv2 = PPMIConv(config.encoder_dim, output_size)

    def forward(self, x, edge_index, cache_name):
        # Note: PPMIConv expects edge_index to be the PPMI matrix's indices
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h # Return only the final embedding
# --- End Simple Encoder ---


# --- Model Definition ---
loss_func = nn.CrossEntropyLoss().to(device)

# Now this check uses the 'config' object which reflects the pre-init modifications
if config.ablate_moe:
    print("INFO: Using SimplePPMIEncoder (MoE ablated).")
    encoder = SimplePPMIEncoder(input_size=source_data.num_features, output_size=config.encoder_dim).to(device)
else:
    print("INFO: Using MoE encoder.")
    # Use config.expert_num which is the original value if MoE is not ablated
    if config.expert_num <= 0:
         raise ValueError("expert_num must be positive when MoE is enabled.")
    num_hops_config = list(range(1, config.expert_num + 1))
    print(f"INFO: MoE Hops Config: {num_hops_config}")

    encoder = MoE(input_size=source_data.num_features,
                  output_size=config.encoder_dim,
                  num_experts=config.expert_num, # Use the correct expert_num from config
                  k=1,
                  coef=config.gate_coef,
                  gnn_type='ppmi',
                  num_hops=num_hops_config).to(device)

# Classifier model remains the same
cls_model = nn.Sequential(
    nn.Linear(config.encoder_dim, dataset_source.num_classes), # Use source dataset's num_classes
    # nn.Dropout(p=config.drop_out), # Optional dropout
).to(device)
print(f"Models defined: Encoder={encoder.__class__.__name__}, Classifier")
# --- End Model Definition ---


# --- Load Precomputed PPMI Data ---
# Ensure tmp directory exists
os.makedirs('tmp', exist_ok=True)
source_pkl_path = f'tmp/{config.source}.pkl'
target_pkl_path = f'tmp/{config.target}.pkl'

# Load PPMI edge indices
try:
    with open(source_pkl_path, 'rb') as f:
        source_ppmi_edge_index, _ = pickle.load(f)
    source_ppmi_edge_index = source_ppmi_edge_index.to(device) # Move to device
    print(f"Loaded source PPMI from {source_pkl_path}, shape: {source_ppmi_edge_index.shape}")
except FileNotFoundError:
    print(f"ERROR: Source PPMI file not found at {source_pkl_path}. PPMIConv requires this.")
    exit()
except Exception as e:
    print(f"Error loading source PPMI file {source_pkl_path}: {e}")
    exit()

try:
    with open(target_pkl_path, 'rb') as f:
        target_ppmi_edge_index, _ = pickle.load(f)
    target_ppmi_edge_index = target_ppmi_edge_index.to(device) # Move to device
    print(f"Loaded target PPMI from {target_pkl_path}, shape: {target_ppmi_edge_index.shape}")
except FileNotFoundError:
    print(f"ERROR: Target PPMI file not found at {target_pkl_path}. PPMIConv requires this.")
    exit()
except Exception as e:
    print(f"Error loading target PPMI file {target_pkl_path}: {e}")
    exit()


# Store PPMI edge index in data objects for easy access in encode function
source_data.ppmi_edge_index = source_ppmi_edge_index
target_data.ppmi_edge_index = target_ppmi_edge_index

# Also compute dense adjacency for ReNode (using original edge_index, not PPMI)
source_data.new_adj = index2dense(source_data.edge_index, source_data.num_nodes).to(device)
target_data.new_adj = index2dense(target_data.edge_index, target_data.num_nodes).to(device)
print("Computed dense adjacency matrices for ReNode.")
# --- End PPMI Loading ---


# --- Optimizer ---
models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
print(f"Optimizer: Adam (lr={config.learning_rate}, wd={config.weight_decay})")
# --- End Optimizer ---


# --- Encode/Predict/Test Functions (Adapting for Ablation) ---
def encode(data, cache_name, mask=None):
    x = data.x.to(device)
    # Use PPMI edge index for encoder
    edge_index_for_encoder = data.ppmi_edge_index.to(device)

    # Handle empty data case
    if x.numel() == 0 or edge_index_for_encoder.numel() == 0:
         output_dim = config.encoder_dim
         encoded_output = torch.empty((0, output_dim), device=device)
         experts_outputs = None
         gate_loss = torch.tensor(0.0, device=device)
         clean_logits = None
         if mask is not None: # Apply mask to empty tensor if needed
              # This part might need careful handling depending on how mask relates to 0 nodes
              # Assuming mask is also empty or corresponds to the 0 nodes
              encoded_output = encoded_output[mask] if encoded_output.numel() > 0 else encoded_output
         return encoded_output, experts_outputs, gate_loss, clean_logits


    if config.ablate_moe:
        # Simple encoder only returns embeddings
        encoded_output = encoder(x, edge_index_for_encoder, cache_name)
        experts_outputs = None # No multiple experts
        gate_loss = torch.tensor(0.0, device=device) # No gate loss
        clean_logits = None # No gating logits
    else:
        # Original MoE encoder call
        encoded_output, experts_outputs, gate_loss, clean_logits = encoder(x, edge_index_for_encoder, cache_name)

    # Apply mask if provided
    if mask is not None:
        try:
            encoded_output = encoded_output[mask]
            if experts_outputs is not None: experts_outputs = experts_outputs[mask]
            if clean_logits is not None: clean_logits = clean_logits[mask]
        except IndexError as e:
            print(f"Error applying mask in encode: {e}")
            print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}")
            print(f"encoded_output shape: {encoded_output.shape if encoded_output is not None else 'None'}")
            # Decide how to handle - return unmasked or raise? For now, return unmasked with warning.
            print("Warning: Returning unmasked output due to mask error.")
            # Re-encode without mask if error occurred
            if config.ablate_moe:
                 encoded_output = encoder(x, edge_index_for_encoder, cache_name)
                 experts_outputs, gate_loss, clean_logits = None, torch.tensor(0.0, device=device), None
            else:
                 encoded_output, experts_outputs, gate_loss, clean_logits = encoder(x, edge_index_for_encoder, cache_name)


    return encoded_output, experts_outputs, gate_loss, clean_logits

def predict(data, cache_name, mask=None):
    encoded_output, _, _, _ = encode(data, cache_name, mask)
    # Handle case where encoded_output might be empty
    if encoded_output is None or encoded_output.numel() == 0:
         return torch.empty((0, data.num_classes), device=device)
    logits = cls_model(encoded_output)
    return logits

def test(data, cache_name, mask=None):
    encoder.eval()
    cls_model.eval()
    accuracy, macro_f1, micro_f1 = 0.0, 0.0, 0.0 # Default values
    final_encoded_output = None # Default

    with torch.no_grad():
        try:
            # Get potentially masked outputs
            encoded_output, experts_outputs, _, clean_logits = encode(data, cache_name, mask=mask)
            final_encoded_output = encoded_output # Store for returning

            # Get predictions from encoded output
            if encoded_output is not None and encoded_output.numel() > 0:
                 logits = cls_model(encoded_output)
                 preds = logits.argmax(dim=1)

                 # Get corresponding labels
                 labels = data.y if mask is None else data.y[mask]
                 labels = labels.to(device) # Ensure labels are on the correct device

                 # Ensure preds and labels align
                 if preds.shape == labels.shape:
                      accuracy, macro_f1, micro_f1 = evaluate(preds, labels)
                 else:
                      print(f"Warning: Shape mismatch in test - preds: {preds.shape}, labels: {labels.shape}. Mask applied? Mask shape: {mask.shape if mask is not None else 'None'}")

            else:
                 print(f"Warning: No encoded output generated for test ({cache_name}). Skipping evaluation.")


            # Log expert selection during testing for target data (if MoE is active and not masked)
            if not config.ablate_moe and cache_name == config.target and mask is None:
                 try:
                     # We need the gating logits for the full graph (unmasked)
                     _, _, _, full_clean_logits = encode(data, cache_name, mask=None)

                     if full_clean_logits is not None and full_clean_logits.numel() > 0:
                         moe_expert_probs = F.softmax(full_clean_logits, dim=-1)
                         moe_expert_indices = torch.argmax(moe_expert_probs, dim=1)

                         # Use a unique filename per run if WandB is active
                         run_id_suffix = f"-{run.id}" if run else ""
                         expert_selection_log_path = f"log/{config.target}-expert-selection{run_id_suffix}.csv"
                         os.makedirs(os.path.dirname(expert_selection_log_path), exist_ok=True)

                         with open(expert_selection_log_path, 'w', newline='') as f:
                             writer = csv.writer(f)
                             writer.writerow(['Node ID', 'Selected Expert', 'Probability'])
                             if isinstance(moe_expert_indices, torch.Tensor) and moe_expert_indices.ndim == 1:
                                 num_nodes_sel = moe_expert_indices.shape[0]
                                 # Ensure probs and indices match shape
                                 if moe_expert_probs.shape[0] == num_nodes_sel:
                                     probs_for_indices = moe_expert_probs[torch.arange(num_nodes_sel), moe_expert_indices]
                                     for node_id in range(num_nodes_sel):
                                         writer.writerow([node_id, moe_expert_indices[node_id].item(), probs_for_indices[node_id].item()])
                                 else:
                                      print("Warning: Shape mismatch between moe_expert_indices and moe_expert_probs in test logging.")

                         # Optional: Log artifact to W&B
                         # if run:
                         #    artifact_name = f'{config.target}_expert_selection{run_id_suffix}'
                         #    artifact = wandb.Artifact(artifact_name, type='analysis_results')
                         #    artifact.add_file(expert_selection_log_path)
                         #    wandb.log_artifact(artifact)

                 except Exception as e:
                      print(f"Warning: Could not log expert selection: {e}")

        except Exception as e:
            print(f"Error during test function for {cache_name}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback

    # Return the potentially masked embeddings obtained during this call
    return accuracy, macro_f1, micro_f1, final_encoded_output
# --- End Encode/Predict/Test Functions ---


# --- LLM Expert Selection Preprocessing (Conditional) ---
expert_selections = {} # Default empty dictionary
if not config.ablate_moe and not config.ablate_llm_select:
    print("INFO: LLM Expert Selection is ENABLED. Loading or generating selections...")
    expert_selections_path = f"log/{config.source}-{config.llm}-selections.json"
    prompts_path = f"log/{config.source}-prompts.pkl"
    os.makedirs('log', exist_ok=True)

    # --- Load Cached Selections ---
    if os.path.exists(expert_selections_path):
        try:
            with open(expert_selections_path, 'r') as f:
                expert_selections_raw = json.load(f)
            # Ensure keys are integers and values are within valid range
            valid_selections = {}
            max_expert_index = config.expert_num - 1
            for k, v in expert_selections_raw.items():
                 try:
                      node_id = int(k)
                      expert_idx = int(v)
                      if 0 <= expert_idx <= max_expert_index:
                           valid_selections[node_id] = expert_idx
                      else:
                           print(f"Warning: Invalid expert index {expert_idx} for node {node_id} in cache file. Ignoring.")
                 except ValueError:
                      print(f"Warning: Invalid key/value ('{k}':'{v}') in cache file. Ignoring.")
            expert_selections = valid_selections
            print(f"Loaded {len(expert_selections)} valid expert selections from {expert_selections_path}")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Error loading or parsing expert selections file {expert_selections_path}: {e}. Regenerating.")
            expert_selections = {} # Reset if file is corrupted or invalid

    # --- Generate Selections if Not Loaded ---
    if not expert_selections:
        print(f"Generating expert selections for all nodes via LLM ({config.llm})...")
        expert_selections = {} # Ensure it's empty before generating

        # --- Load or Generate Prompts ---
        prompts = []
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, 'rb') as f:
                    prompts = pickle.load(f)
                print(f"Loaded {len(prompts)} prompts from {prompts_path}")
                if len(prompts) != source_data.num_nodes:
                    print("Warning: Number of loaded prompts doesn't match number of source nodes. Regenerating prompts.")
                    prompts = [] # Force regeneration
            except Exception as e:
                print(f"Error loading prompts file {prompts_path}: {e}. Regenerating.")
                prompts = []

        if not prompts:
            print("Generating prompts...")
            # Pass necessary config values explicitly to multiprocessing function
            config_dict_for_prompt = {'expert_num': config.expert_num} # Pass necessary config

            # Prepare args for multiprocessing pool
            # Pass simplified source_data representation (on CPU) to avoid large object pickling issues
            source_data_cpu_dict = {
                 'num_nodes': source_data.num_nodes,
                 'edge_index': source_data.edge_index.cpu() # Pass edge index on CPU
                 # Add other necessary attributes if build_prompt_for_node needs them
            }
            args_list = [
                (node_id, source_data_cpu_dict, config_dict_for_prompt, int(config.expert_num), int(config.hop))
                for node_id in range(source_data.num_nodes)
            ]
            # Limit processes to avoid overwhelming system or OLLAMA server
            num_processes = min(multiprocessing.cpu_count(), 8) # Reduced default
            print(f"Using {num_processes} processes for prompt generation.")
            try:
                 # Set start method for multiprocessing if needed (e.g., 'spawn' on some systems)
                 # multiprocessing.set_start_method('spawn', force=True) # Uncomment if needed
                 with multiprocessing.Pool(processes=num_processes) as pool:
                      prompts = pool.map(build_prompt_for_node, args_list)
                 with open(prompts_path, 'wb') as f:
                      pickle.dump(prompts, f)
                 print(f"Generated and saved {len(prompts)} prompts to {prompts_path}")
            except Exception as e:
                 print(f"Error during prompt generation with multiprocessing: {e}")
                 print("Try running sequentially (this will be slow).")
                 prompts = [build_prompt_for_node(arg_tuple) for arg_tuple in args_list]


        # --- Call OpenAI for each prompt ---
        print(f"Calling OpenAI-compatible API ({config.llm}) for {len(prompts)} prompts...")
        for node_id, prompt in enumerate(prompts):
            if not prompt: # Skip if prompt generation failed
                 print(f"Skipping node {node_id} due to empty prompt.")
                 expert_selections[node_id] = np.random.randint(0, config.expert_num) # Default random
                 continue

            try:
                response = llm_client.generate(
                    prompt=prompt,
                    model=config.llm,
                    temperature=config.temperature,
                    max_tokens=min(config.max_tokens, 1024),  # Reduced to 1024 as in original
                    format='json'
                )
                json_output_str = response['response']
                try:
                    parsed_json = json.loads(json_output_str)
                    expert = parsed_json.get("expert")
                    if isinstance(expert, int) and 0 <= expert < config.expert_num:
                        expert_selections[node_id] = expert
                    else:
                        print(f"Warning: Invalid expert value '{expert}' for node {node_id}. Response: '{json_output_str[:100]}...'. Defaulting to random.")
                        expert_selections[node_id] = np.random.randint(0, config.expert_num)
                except json.JSONDecodeError as e:
                    print(f"Node {node_id}: JSONDecodeError: {e}. Response: '{json_output_str[:100]}...'. Defaulting to random.")
                    expert_selections[node_id] = np.random.randint(0, config.expert_num)
                except Exception as e_parse: # Catch other potential parsing errors
                     print(f"Node {node_id}: Error parsing LLM response JSON: {e_parse}. Response: '{json_output_str[:100]}...'. Defaulting to random.")
                     expert_selections[node_id] = np.random.randint(0, config.expert_num)

            except Exception as e_openai:
                print(f"Node {node_id}: OpenAI API error: {e_openai}. Check OpenAI API connection/status. Defaulting to random.")
                expert_selections[node_id] = np.random.randint(0, config.expert_num)
                # Optional: Add a delay or break if OpenAI errors persist
                # time.sleep(2)

            # Log progress and save intermediate results
            if (node_id + 1) % 100 == 0:
                print(f"Processed {node_id+1}/{len(prompts)} nodes for LLM selection...")
                try:
                     with open(expert_selections_path, 'w') as f:
                        json.dump(expert_selections, f, indent=4) # Add indent for readability
                except Exception as e_save:
                     print(f"Error saving intermediate expert selections: {e_save}")

        # Final save
        try:
             with open(expert_selections_path, 'w') as f:
                 json.dump(expert_selections, f, indent=4)
             print(f"Expert selections generation complete. Saved to {expert_selections_path}")
        except Exception as e_save:
             print(f"Error saving final expert selections: {e_save}")

else:
    if config.ablate_moe:
        print("INFO: LLM Expert Selection is SKIPPED (MoE ablated).")
    elif config.ablate_llm_select:
        print("INFO: LLM Expert Selection is SKIPPED (ablated by flag).")
# --- End LLM Preprocessing ---


# --- Entropy Function ---
def Entropy(input, weight, label):
    # (Keep this function as is)
    softmax_out = nn.Softmax(dim=-1)(input)
    entropy = -label * torch.log(softmax_out + 1e-5)
    entropy_loss = torch.mean(weight * torch.sum(entropy, dim=1))
    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return entropy_loss
# def Entropy(input_logits, weight, target_prob):
#     # Ensure inputs are valid
#     if input_logits.numel() == 0 or weight.numel() == 0 or target_prob.numel() == 0:
#         return torch.tensor(0.0, device=input_logits.device)

#     # Calculate softmax and log softmax safely
#     softmax_out = nn.Softmax(dim=-1)(input_logits)
#     log_softmax_out = torch.log(softmax_out + 1e-9) # Use log(softmax) for stability

#     # Calculate cross-entropy component: - sum(target * log(pred))
#     # Ensure weight and target_prob dimensions match input_logits batch size
#     batch_size = input_logits.shape[0]
#     if weight.shape[0] != batch_size or target_prob.shape[0] != batch_size:
#          print(f"Warning: Mismatch shapes in Entropy - input: {input_logits.shape}, weight: {weight.shape}, target: {target_prob.shape}")
#          # Attempt to resize or return 0, depending on expected behavior
#          # For now, return 0 to avoid crashing
#          return torch.tensor(0.0, device=input_logits.device)


#     entropy = -target_prob * log_softmax_out
#     # Sum over classes, then take weighted mean over batch
#     weighted_node_entropy = torch.mean(weight * torch.sum(entropy, dim=1))

#     # Calculate batch entropy component: sum(mean_prob * log(mean_prob))
#     msoftmax = softmax_out.mean(dim=0) # Average softmax prediction across batch
#     batch_entropy = torch.sum(msoftmax * torch.log(msoftmax + 1e-9)) # Should be negative sum

#     # Total entropy loss (minimize weighted node entropy, maximize batch entropy)
#     entropy_loss = weighted_node_entropy - batch_entropy
#     return entropy_loss
# --- End Entropy Function ---


# --- Training Function (Adapting for Ablation) ---
def train(epoch):
    for model in models:
        model.train()

    # --- Forward Pass ---
    try:
        encoded_source, experts_outputs, source_gate_loss, source_clean_logits = encode(source_data, config.source)
        # Handle potential empty outputs if data is empty
        if encoded_source.numel() == 0:
             print(f"Warning: Empty encoded_source at epoch {epoch}. Skipping training step.")
             return # Skip rest of training step

        source_logits = cls_model(encoded_source)

    except Exception as e:
        print(f"Error during forward pass in train epoch {epoch}: {e}")
        import traceback
        traceback.print_exc()
        return # Skip training step if forward pass fails


    # --- Initialize Losses ---
    cls_loss = torch.tensor(0.0, device=device)
    select_loss = torch.tensor(0.0, device=device)
    high_quality_semi_loss = torch.tensor(0.0, device=device)
    gate_loss = source_gate_loss if source_gate_loss is not None else torch.tensor(0.0, device=device)

    # --- Classifier Loss ---
    labeled_nodes_mask = label_mask
    if labeled_nodes_mask.sum() > 0:
         cls_loss = loss_func(source_logits[labeled_nodes_mask], source_data.y[labeled_nodes_mask])

    # --- LLM Selection Loss (Only if MoE and LLM Select are active) ---
    if not config.ablate_moe and not config.ablate_llm_select and experts_outputs is not None and source_clean_logits is not None:
        # Check if necessary components have data
        if experts_outputs.numel() > 0 and source_clean_logits.numel() > 0:
             uncertainty_mask = calculate_expert_uncertainty(experts_outputs, source_data.num_classes, cls_model, config.uncertainty_k)
             uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()

             if uncertainty_node_indices and expert_selections: # Need uncertain nodes and LLM selections
                 num_experts = config.expert_num
                 # Get LLM expert indices, default to -1 if not found
                 llm_expert_indices_list = [expert_selections.get(node_id_int, -1) for node_id_int in uncertainty_node_indices]

                 # Filter out nodes where LLM selection failed (or wasn't available) indicated by -1
                 valid_indices_mask = torch.tensor([idx != -1 for idx in llm_expert_indices_list], device=device)
                 uncertainty_node_indices_tensor = torch.tensor(uncertainty_node_indices, device=device, dtype=torch.long)[valid_indices_mask]
                 llm_expert_indices_tensor = torch.tensor(llm_expert_indices_list, device=device, dtype=torch.long)[valid_indices_mask]

                 if uncertainty_node_indices_tensor.numel() > 0: # Check if any valid uncertain nodes remain
                     try:
                         # Get MoE's predicted expert distribution for these uncertain nodes
                         moe_expert_dist = source_clean_logits[uncertainty_node_indices_tensor]

                         # Calculate cross-entropy loss: encourage MoE gate to predict the LLM's choice index
                         select_loss = F.cross_entropy(moe_expert_dist, llm_expert_indices_tensor)
                     except IndexError as e:
                         print(f"IndexError calculating select_loss: {e}")
                         print(f"source_clean_logits shape: {source_clean_logits.shape}, uncertainty_node_indices_tensor max: {uncertainty_node_indices_tensor.max() if uncertainty_node_indices_tensor.numel() > 0 else 'N/A'}")
                         select_loss = torch.tensor(0.0, device=device) # Reset loss


    # --- High-Quality Semi-Supervised Loss (Only if MoE, LLM Select, and HQ Semi are active) ---
    calculate_hq_semi = not config.ablate_moe and not config.ablate_llm_select and not config.ablate_hq_semi
    if calculate_hq_semi and experts_outputs is not None and experts_outputs.numel() > 0 :
        # --- Pseudo labeling (required for any semi-supervised loss) ---
        with torch.no_grad():
             # Use full source_logits for pseudo labels
             if source_logits.numel() > 0:
                  _, s_plabel = torch.max(source_logits, dim=1)
                  s_plabel[label_mask] = source_data.y[label_mask] # Correct known labels
                  s_weight = get_renode_weight(source_data, s_plabel).to(device)
                  s_plabel_onehot = F.one_hot(s_plabel, num_classes=source_data.num_classes).float()
             else:
                  # Handle case where source_logits is empty
                  s_plabel = torch.tensor([], dtype=torch.long, device=device)
                  s_weight = torch.tensor([], dtype=torch.float, device=device)
                  s_plabel_onehot = torch.tensor([], dtype=torch.float, device=device)


        unlabeled_mask = ~label_mask
        if unlabeled_mask.sum() > 0 and expert_selections and s_plabel.numel() > 0: # Need unlabeled nodes, LLM selections, and pseudo-labels
            num_experts = experts_outputs.shape[1]
            unlabeled_indices = torch.where(unlabeled_mask)[0]

            if unlabeled_indices.numel() > 0:
                # Get LLM expert selections for unlabeled nodes, default to -1
                llm_expert_indices_unlabeled_list = [expert_selections.get(idx.item(), -1) for idx in unlabeled_indices]
                # Create mask for nodes that have a valid LLM selection
                valid_llm_selection_mask = torch.tensor([idx != -1 for idx in llm_expert_indices_unlabeled_list], device=device)

                # Filter unlabeled nodes and their LLM expert choices
                unlabeled_indices_filtered = unlabeled_indices[valid_llm_selection_mask]
                llm_expert_indices_unlabeled_tensor = torch.tensor(llm_expert_indices_unlabeled_list, device=device, dtype=torch.long)[valid_llm_selection_mask]


                if unlabeled_indices_filtered.numel() > 0: # Check if any valid unlabeled nodes remain
                    try:
                         unlabeled_experts_outputs = experts_outputs[unlabeled_indices_filtered] # [num_unlabeled_valid, num_experts, d_feature]
                         num_unlabeled_valid = unlabeled_experts_outputs.shape[0]
                         expert_entropies_unlabeled = torch.zeros(num_unlabeled_valid, num_experts, device=device)

                         # Calculate entropy for each expert's prediction on these unlabeled nodes
                         with torch.no_grad(): # Don't need gradients here
                              for i in range(num_experts):
                                   expert_logits_unlabeled = cls_model(unlabeled_experts_outputs[:, i, :])
                                   expert_softmax_unlabeled = nn.Softmax(dim=-1)(expert_logits_unlabeled)
                                   # Calculate entropy: -sum(p * log(p))
                                   expert_entropies_unlabeled[:, i] = -torch.sum(expert_softmax_unlabeled * torch.log(expert_softmax_unlabeled + 1e-9), dim=1)

                         # Entropy of the LLM-selected expert for each valid node
                         llm_expert_entropy = expert_entropies_unlabeled[torch.arange(num_unlabeled_valid), llm_expert_indices_unlabeled_tensor]

                         # Max entropy among other experts
                         other_expert_mask = torch.ones_like(expert_entropies_unlabeled, dtype=torch.bool)
                         other_expert_mask[torch.arange(num_unlabeled_valid), llm_expert_indices_unlabeled_tensor] = False # Mask out the LLM expert
                         # Use -inf for stability when calculating max, handle case where all other experts might have NaN/Inf entropy
                         other_expert_entropy = expert_entropies_unlabeled.masked_fill(~other_expert_mask, -float('inf'))
                         max_other_entropy, _ = other_expert_entropy.max(dim=1)

                         # High-quality nodes: LLM expert entropy is lower than max entropy of others
                         high_quality_mask_within_unlabeled = llm_expert_entropy < max_other_entropy
                         num_high_quality = high_quality_mask_within_unlabeled.sum().item()

                         if num_high_quality > 0:
                              # Indices of high-quality nodes relative to the *filtered* unlabeled set
                              hq_indices_relative = torch.where(high_quality_mask_within_unlabeled)[0]
                              # Get corresponding logits, weights, and pseudo-labels
                              hq_logits = source_logits[unlabeled_indices_filtered][hq_indices_relative]
                              hq_weights = s_weight[unlabeled_indices_filtered][hq_indices_relative]
                              hq_plabels_onehot = s_plabel_onehot[unlabeled_indices_filtered][hq_indices_relative]

                              # Calculate entropy loss only for these high-quality pseudo-labels
                              high_quality_semi_loss = Entropy(hq_logits, hq_weights, hq_plabels_onehot)
                              # print(f"Epoch {epoch}: Found {num_high_quality} high-quality nodes. HQ Semi Loss: {high_quality_semi_loss.item():.4f}")

                    except IndexError as e:
                         print(f"IndexError calculating HQ Semi loss: {e}")
                         # Potentially log shapes here for debugging
                         high_quality_semi_loss = torch.tensor(0.0, device=device) # Reset loss
                    except Exception as e_hq: # Catch other unexpected errors
                         print(f"Unexpected error calculating HQ Semi loss: {e_hq}")
                         high_quality_semi_loss = torch.tensor(0.0, device=device)


    # --- Determine Loss Weights based on Ablation Flags ---
    # These flags are now correctly set in 'config' before the loop starts
    effective_select_weight = 0.0 if config.ablate_llm_select else config.select_weight
    effective_hq_semi_weight = 0.0 if config.ablate_hq_semi else config.semi_weight
    effective_div_weight = 0.0 if config.ablate_gate_loss else config.div_weight

    # Ensure weights are only applied if the corresponding component is active
    # (Implicitly handled by config flags, but double-check select/hq semi conditions)
    if config.ablate_moe: # If MoE is gone, these must be zero
         effective_select_weight = 0.0
         effective_hq_semi_weight = 0.0
         effective_div_weight = 0.0

    # --- Total Loss ---
    # epoch_weight = min(1.0, float(epoch) / max(1, config.epochs // 2)) # Annealing weight (ramp up over first half)
    epoch_weight = float(epoch) / config.epochs
    # epoch_weight = 1.0 # No annealing

    loss = (cls_loss
            + epoch_weight * effective_hq_semi_weight * high_quality_semi_loss
            + epoch_weight * effective_select_weight * select_loss # Anneal select loss as well
            + effective_div_weight * gate_loss)


    # --- W&B Logging ---
    if run: # Only log if WandB run is active
        log_dict = {
            'epoch': epoch,
            'train/cls_loss': cls_loss.item(),
            'train/gate_loss': gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
            'train/select_loss': select_loss.item() if isinstance(select_loss, torch.Tensor) else select_loss,
            'train/high_quality_semi_loss': high_quality_semi_loss.item() if isinstance(high_quality_semi_loss, torch.Tensor) else high_quality_semi_loss,
            'train/total_loss': loss.item(),
            # Log effective weights
            #'train/used_select_weight': effective_select_weight * epoch_weight,
            #'train/used_hq_semi_weight': effective_hq_semi_weight * epoch_weight,
            #'train/used_div_weight': effective_div_weight,
        }
        wandb.log(log_dict)
    # --- End W&B Logging ---

    # Print losses (optional)
    # print(f"Epoch {epoch:03d} Ls: Cls={cls_loss.item():.4f}, Gate={gate_loss.item():.4f}, Sel={select_loss.item():.4f}, HQ={high_quality_semi_loss.item():.4f} Tot={loss.item():.4f}")

    # --- Backpropagation ---
    optimizer.zero_grad()
    try:
        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
    except RuntimeError as e:
        print(f"Error during backward pass or optimizer step at epoch {epoch}: {e}")
        # Potentially skip step or break loop depending on error severity
    # --- End Backpropagation ---
# --- End Training Function ---


# --- Training Loop ---
best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0
best_macro_f1 = 0.0
best_micro_f1 = 0.0

print(f"\n--- Starting Training for {config.epochs} Epochs ---")
for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train(epoch)
    epoch_train_time = time.time() - epoch_start_time

    # --- Evaluation Step ---
    eval_start_time = time.time()
    try:
        # Evaluate on source test set (if mask exists)
        source_test_mask = getattr(source_data, 'test_mask', None)
        if source_test_mask is not None and source_test_mask.sum() > 0:
             source_acc, _, _, _ = test(source_data, config.source, mask=source_test_mask)
        else:
             # Fallback: evaluate on whole source graph if no test mask or empty mask
             if source_test_mask is None:
                  print(f"Epoch {epoch}: Warning: No source_data.test_mask found. Evaluating on full source graph.")
             elif source_test_mask.sum() == 0:
                  print(f"Epoch {epoch}: Warning: source_data.test_mask is empty. Evaluating on full source graph.")
             source_acc, _, _, _ = test(source_data, config.source, mask=None) # Eval full graph


        # Evaluate on target set (usually full graph)
        target_acc, macro_f1, micro_f1, output_target = test(target_data, config.target, mask=None) # Evaluate on full target graph
        epoch_eval_time = time.time() - eval_start_time

        print(f"Epoch: {epoch:03d}/{config.epochs} | "
              f"Train Time: {epoch_train_time:.2f}s | Eval Time: {epoch_eval_time:.2f}s | "
              f"Src Acc: {source_acc:.4f} | Tgt Acc: {target_acc:.4f} | "
              f"Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")


        # --- W&B Logging for Evaluation ---
        if run:
            wandb.log({
                'epoch': epoch, # Log epoch here too for alignment
                'eval/source_acc': source_acc,
                'eval/target_acc': target_acc,
                'eval/macro_f1': macro_f1,
                'eval/micro_f1': micro_f1,
                'time/train_epoch': epoch_train_time,
                'time/eval_epoch': epoch_eval_time,
            })
        # --- End W&B Logging ---

        # --- Update Best Metrics ---
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            best_source_acc = source_acc # Store corresponding source acc
            best_macro_f1 = macro_f1
            best_micro_f1 = micro_f1
            best_epoch = epoch
            print(f"    *** New best target accuracy at epoch {epoch}: {best_target_acc:.4f} ***")
            # --- Save Best Model Checkpoint (Optional) ---
            # if run: # Only save if W&B is active to associate with run
            #    log_dir = Path('log')
            #    log_dir.mkdir(parents=True, exist_ok=True)
            #    checkpoint_path = log_dir / f'best_model_{run.id}.pt'
            #    state = {'epoch': epoch,
            #             'encoder_state_dict': encoder.state_dict(),
            #             'cls_model_state_dict': cls_model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'best_target_acc': best_target_acc}
            #    torch.save(state, checkpoint_path)
            #    # Optional: Save checkpoint to W&B artifacts
            #    # artifact = wandb.Artifact(f'best_model_{run.id}', type='model')
            #    # artifact.add_file(str(checkpoint_path))
            #    # wandb.log_artifact(artifact)
            #    print(f"    Saved new best model checkpoint to {checkpoint_path}")
            # --- End Checkpoint Saving ---


    except Exception as e:
        print(f"Error during evaluation at epoch {epoch}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        if run: wandb.log({'error': str(e)})
        # Decide whether to break or continue
        print("Continuing training despite evaluation error.")
        # break # Stop training if evaluation fails critically
# --- End Training Loop ---

print("\n--- Training Finished ---")

# --- Final Results ---
print("=============================================================")
# Recreate final_id_str using the config object
final_ablation_str = ""
if config.ablate_moe: final_ablation_str += "-NoMoE"
if config.ablate_llm_select and not config.ablate_moe: final_ablation_str += "-NoLLMSelect"
if config.ablate_hq_semi and not config.ablate_llm_select and not config.ablate_moe: final_ablation_str += "-NoHQSemi"
if config.ablate_gate_loss and not config.ablate_moe: final_ablation_str += "-NoGateLoss"
if not final_ablation_str and not config.ablate_moe: final_ablation_str = "-FullModel"

final_id_str = "src:{},tgt:{},seed:{},lr:{:.1e},wd:{:.1e},dim:{},do:{:.1e},lbl_rt:{:.2f}{}".format(
    config.source, config.target, config.seed, config.learning_rate, config.weight_decay,
    config.encoder_dim, config.drop_out, config.label_rate, final_ablation_str
)
line = "{}\n - Best Epoch: {}, Best Source Acc: {:.5f}, Best Target Acc: {:.5f}, Best Macro F1: {:.5f}, Best Micro F1: {:.5f}" \
    .format(final_id_str, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)
print(line)

# --- Log Best Metrics to W&B Summary ---
if run:
    wandb.summary['best_epoch'] = best_epoch
    wandb.summary['best_source_acc'] = best_source_acc
    wandb.summary['best_target_acc'] = best_target_acc
    wandb.summary['best_macro_f1'] = best_macro_f1
    wandb.summary['best_micro_f1'] = best_micro_f1
    wandb.summary['final_id_str'] = final_id_str # Log the identifier string
    # Log ablation flags to summary for easy filtering in W&B
    wandb.summary['ablate_moe'] = config.ablate_moe
    wandb.summary['ablate_llm_select'] = config.ablate_llm_select
    wandb.summary['ablate_hq_semi'] = config.ablate_hq_semi
    wandb.summary['ablate_gate_loss'] = config.ablate_gate_loss
    print("Logged final summary metrics to WandB.")
# --- End W&B Summary Logging ---


# --- CSV Logging ---
log_dir = Path('log')
log_dir.mkdir(parents=True, exist_ok=True)
csv_log_path = log_dir / "ablation_results.csv"

# Define header based on config keys + results + ablation flags
# Need to get config keys (handle case where WandB failed)
if run:
     config_dict = config.as_dict() # Get config as dict from WandB
else:
     config_dict = vars(config) # Get config as dict from argparse namespace

config_keys = list(config_dict.keys())

# Clean up keys - remove internal W&B keys if present
config_keys = [k for k in config_keys if not k.startswith('_') and not k == 'wandb']
# Ensure ablation flags are included
for flag in ['ablate_moe', 'ablate_llm_select', 'ablate_hq_semi', 'ablate_gate_loss']:
    if flag not in config_keys:
        config_keys.append(flag)

result_keys = ['best_epoch', 'best_source_acc', 'best_target_acc', 'best_macro_f1', 'best_micro_f1', 'wandb_url', 'timestamp']
header = sorted(config_keys) + sorted(result_keys) # Sort for consistent column order

# Data dictionary for this run
log_data = {k: config_dict.get(k, '') for k in config_keys} # Get config values safely
log_data.update({
    'best_epoch': best_epoch,
    'best_source_acc': f"{best_source_acc:.5f}", # Format float
    'best_target_acc': f"{best_target_acc:.5f}", # Format float
    'best_macro_f1': f"{best_macro_f1:.5f}",
    'best_micro_f1': f"{best_micro_f1:.5f}",
    'wandb_url': run.url if run else 'N/A',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
})

file_exists = csv_log_path.is_file()

try:
    with open(csv_log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore') # Ignore extra fields not in header
        if not file_exists or os.path.getsize(csv_log_path) == 0: # Check size too
            writer.writeheader() # Write header only if file is new or empty
        writer.writerow(log_data)
    print(f"Results appended to {csv_log_path}")
except IOError as e:
    print(f"Error writing to CSV {csv_log_path}: {e}")
except Exception as e_csv: # Catch other potential errors
     print(f"Unexpected error writing to CSV: {e_csv}")
# --- End CSV Logging ---

# --- Save Final Embeddings (Optional) ---
# Rerun test on full target graph to get final embeddings after all epochs
print("\nGetting final target embeddings after last epoch...")
try:
    final_target_acc, final_macro_f1, final_micro_f1, final_output_target = test(target_data, config.target, mask=None)
    print(f"Final Target Metrics (Epoch {config.epochs}): Acc={final_target_acc:.4f}, MacroF1={final_macro_f1:.4f}, MicroF1={final_micro_f1:.4f}")

    if final_output_target is not None and final_output_target.numel() > 0:
        run_id_suffix = f"-{run.id}" if run else ""
        final_target_embeddings_path = log_dir / f"{config.target}-final-embeddings{run_id_suffix}.pt"
        final_target_labels_path = log_dir / f"{config.target}-final-labels{run_id_suffix}.pt"

        print(f"Saving final target embeddings ({final_output_target.shape}) to {final_target_embeddings_path}")
        torch.save(final_output_target.cpu(), final_target_embeddings_path)

        # Ensure target labels exist before saving
        if hasattr(target_data, 'y') and target_data.y is not None:
             print(f"Saving final target labels ({target_data.y.shape}) to {final_target_labels_path}")
             torch.save(target_data.y.cpu(), final_target_labels_path)
        else:
             print("Warning: target_data.y not found, cannot save final labels.")

        # --- Optional: Save embeddings/labels as W&B artifacts ---
        # if run:
        #    try:
        #        embedding_artifact_name = f'{config.target}_final_embeddings{run_id_suffix}'
        #        embedding_artifact = wandb.Artifact(embedding_artifact_name, type='model_outputs')
        #        embedding_artifact.add_file(str(final_target_embeddings_path))
        #        wandb.log_artifact(embedding_artifact)
        #
        #        if Path(final_target_labels_path).exists(): # Check if labels were saved
        #             label_artifact_name = f'{config.target}_final_labels{run_id_suffix}'
        #             label_artifact = wandb.Artifact(label_artifact_name, type='model_outputs')
        #             label_artifact.add_file(str(final_target_labels_path))
        #             wandb.log_artifact(label_artifact)
        #        print("Logged final embeddings/labels artifacts to WandB.")
        #    except Exception as e_art:
        #        print(f"Error logging final artifacts to WandB: {e_art}")
        # --- End Optional Artifact Logging ---
    else:
         print("Skipping saving final embeddings (output was None or empty).")

except Exception as e_final_test:
     print(f"Error during final test/embedding saving: {e_final_test}")
# --- End Save Final Embeddings ---


# --- Finish W&B Run ---
if run:
    wandb.finish()
    print("WandB run finished.")
# --- End Finish W&B Run ---

print("\nScript execution completed.")