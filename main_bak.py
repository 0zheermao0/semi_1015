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
import ollama

# --- W&B Integration ---
import wandb
# --- End W&B Integration ---

from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
from gnn.moe import MoE
from common.graph_encoder import GraphEncoder as Graph2TextEncoder
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM # Seems unused

warnings.filterwarnings("ignore", category=UserWarning)
# os.environ['OLLAMA_HOST'] = 'http://192.168.1.100:11434' # Keep if needed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
# Keep argparse for arguments you might want to set outside of a sweep or as defaults
parser.add_argument("--source", type=str, default='dblpv7')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--weight_decay", type=float, default=2e-3)
parser.add_argument("--drop_out", type=float, default=1e-1)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--label_rate", type=float, default=0.05)
parser.add_argument("--expert_num", type=int, default=3)
parser.add_argument("--llm", type=str, default='qwen2.5:7b')
parser.add_argument("--uncertainty_k", type=int, default=100)
parser.add_argument("--gate_coef", type=float, default=1e-1)
parser.add_argument("--select_weight", type=float, default=1)
parser.add_argument("--semi_weight", type=float, default=1)
parser.add_argument("--llm_interval", type=int, default=20, help="Interval of epochs to call LLM for expert selection")
# --- W&B specific arguments (optional, can be set in wandb.init or sweep config) ---
parser.add_argument("--wandb_project", type=str, default="GNN-Domain-Adaptation-Sweep")
parser.add_argument("--wandb_entity", type=str, default=None, help="Your W&B username or team name") # Or set directly in wandb.init

args = parser.parse_args()

# --- W&B Initialization ---
# Initialize a W&B run.
# project: Name of the project in W&B.
# entity: Your W&B username or team name (optional, defaults to your default entity).
# config: Pass the argparse Namespace. W&B will log these hyperparameters.
#         If running with `wandb agent`, the Sweep controller will override values in `wandb.config`.
run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=args # Pass argparse object here
)
# Now, access hyperparameters via `wandb.config` instead of `args`
# This ensures that parameters from a W&B Sweep are used.
config = wandb.config # Use config alias for convenience
# --- End W&B Initialization ---

seed = config.seed # Use wandb.config for seed as well, for reproducibility tracked by W&B

# Use config for hyperparameters from now on
id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}, drop_out: {}, expert_num: {}, llm: {}, uncertainty_k: {}, gate_coef: {}" \
    .format(config.source, config.target, config.seed, config.label_rate, config.learning_rate, config.weight_decay,
            config.encoder_dim, config.drop_out, config.expert_num, config.llm, config.uncertainty_k, config.gate_coef)
print(id_str)
wandb.run.name = f"{config.source}-{config.target}-lr{config.learning_rate:.1e}-do{config.drop_out:.1e}-seed{config.seed}" # Example run name

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = False

dataset = DomainData("data/{}".format(config.source), name=config.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print(source_data)

dataset = DomainData("data/{}".format(config.target), name=config.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

source_train_size = int(source_data.size(0) * config.label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)

# --- Helper Functions (index2dense, GradReverse, GRL, encode, predict, evaluate) ---
# (Keep these functions as they are, they don't need direct W&B integration)
# def index2dense(edge_index,nnode=2708):
#     indx = edge_index.cpu().detach().numpy()
#     adj = np.zeros((nnode,nnode),dtype = 'int8')
#     adj[(indx[0],indx[1])]=1
#     new_adj = torch.from_numpy(adj).float()
#     return new_adj

def index2dense(edge_index, nnode=2708):
    # edge_index: shape [2, num_edges]
    device = edge_index.device
    adj = torch.zeros((nnode, nnode), dtype=torch.float32, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def encode(data, cache_name, mask=None):
    # Ensure 'encoder' is accessible (defined later, might need refactoring or pass as arg if needed earlier)
    encoded_output, experts_outputs, gate_loss, clean_logits = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
        experts_outputs = experts_outputs[mask]
        clean_logits = clean_logits[mask]
    return encoded_output, experts_outputs, gate_loss, clean_logits

def predict(data, cache_name, mask=None):
    # Ensure 'cls_model' is accessible
    encoded_output, _, _, _ = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits

def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    # Ensure labels/preds are on CPU for sklearn
    labels_cpu = labels.cpu().detach()
    preds_cpu = preds.cpu().detach()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0) # Added zero_division=0
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0) # Added zero_division=0
    return accuracy, macro_f1, micro_f1
# --- End Helper Functions ---


def test(data, cache_name, mask=None):
    encoded_output, experts_outputs, _, clean_logits = encode(data, cache_name, mask)
    logits = predict(data, cache_name, mask) # predict already calls encode
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy, macro_f1, micro_f1 = evaluate(preds, labels)

    # Log expert selection during testing for target data (Maybe log as artifact)
    if cache_name == config.target: # Use config.target
        moe_expert_indices, moe_expert_probs, _ = encoder.get_node_expert_assignment(data.x, data.edge_index)
        expert_selection_log_path = f"log/{config.target}-expert-selection.csv"
        os.makedirs(os.path.dirname(expert_selection_log_path), exist_ok=True)
        with open(expert_selection_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Node ID', 'Selected Expert', 'Probability'])
            # Check if indices/probs are tensors and get dimensions safely
            if isinstance(moe_expert_indices, torch.Tensor) and moe_expert_indices.ndim >= 1:
                num_nodes_sel = moe_expert_indices.shape[0]
                num_k_sel = moe_expert_indices.shape[1] if moe_expert_indices.ndim > 1 else 1
                for node_id in range(num_nodes_sel):
                    for k in range(num_k_sel):
                        expert_idx_tensor = moe_expert_indices[node_id, k] if moe_expert_indices.ndim > 1 else moe_expert_indices[node_id]
                        expert_prob_tensor = moe_expert_probs[node_id, k] if moe_expert_probs.ndim > 1 else moe_expert_probs[node_id]
                        writer.writerow([node_id, expert_idx_tensor.item(), expert_prob_tensor.item()])
        # --- W&B Artifact Logging (Optional) ---
        # Log the expert selection CSV as an artifact
        # artifact = wandb.Artifact(f'{config.target}_expert_selection', type='analysis_results')
        # artifact.add_file(expert_selection_log_path)
        # wandb.log_artifact(artifact)
            # --- End W&B Artifact Logging ---

    return accuracy, macro_f1, micro_f1, encoded_output # Return output for saving embeddings


def get_renode_weight(data, pseudo_label):
    # (Keep this function as is)
    ppr_matrix = data.new_adj
    gpr_matrix = []
    for iter_c in range(data.num_classes):
        iter_gpr = torch.mean(ppr_matrix[pseudo_label==iter_c],dim=0).squeeze()
        gpr_matrix.append(iter_gpr)
    gpr_matrix = torch.stack(gpr_matrix,dim=0).transpose(0,1)
    base_w  = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix =  torch.mm(ppr_matrix,gpr_matrix) - torch.mm(ppr_matrix,gpr_rn)/(data.num_classes-1.0)
    label_matrix = F.one_hot(pseudo_label, gpr_matrix.size(1)).float()
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=True)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(nnode-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    return rn_weight


loss_func = nn.CrossEntropyLoss().to(device)

# Use config for model hyperparameters
# num_hops_config = [1, 2, 3][:config.expert_num] if config.expert_num >= 3 else [1] * config.expert_num
num_hops_config = [1, 2, 3][:config.expert_num] if config.expert_num >= 3 else [1] * config.expert_num
encoder = MoE(input_size=source_data.num_features, output_size=config.encoder_dim, num_experts=config.expert_num, k=1
          , coef=config.gate_coef, gnn_type='ppmi', num_hops=num_hops_config).to(device)

cls_model = nn.Sequential(
    nn.Linear(config.encoder_dim, dataset.num_classes),
    # Maybe add dropout here if needed, using config.drop_out?
    # nn.Dropout(p=config.drop_out), # Example
).to(device)

# --- Watch Models with W&B (Optional, logs gradients and parameters) ---
# wandb.watch(encoder, log_freq=100)
# wandb.watch(cls_model, log_freq=100)
# --- End Watch Models ---


# Load precomputed PPMI matrices if they exist
# Make sure tmp directory exists
os.makedirs('tmp', exist_ok=True)
source_pkl_path = f'tmp/{config.source}.pkl'
target_pkl_path = f'tmp/{config.target}.pkl'

# Check if files exist before loading
if os.path.exists(source_pkl_path):
    with open(source_pkl_path, 'rb') as f:
        source_edge_index, _ = pickle.load(f) # Assuming norm is not needed later
else:
    print(f"Warning: {source_pkl_path} not found. PPMI may not be properly initialized.")
    # Handle missing file, maybe compute PPMI here if needed or raise error
    source_edge_index = source_data.edge_index # Fallback or compute

if os.path.exists(target_pkl_path):
    with open(target_pkl_path, 'rb') as f:
        target_edge_index, _ = pickle.load(f) # Assuming norm is not needed later
else:
    print(f"Warning: {target_pkl_path} not found. PPMI may not be properly initialized.")
    target_edge_index = target_data.edge_index # Fallback or compute

# Convert edge indices to dense adjacency matrices
source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)
target_data.new_adj = index2dense(target_edge_index, target_data.num_nodes).to(device)


models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
# optimizer_moe = torch.optim.Adam(encoder.gate_gnn.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
# print 哪些模块的参数被更新
for name, param in encoder.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# optimizer = torch.optim.Adam(params, lr=config.learning_rate)

# --- Remove Manual CSV Logging ---
# The W&B logging replaces the need for manual hyperparameter and metric CSV logging.
# log_csv_path = "log/{}-{}-metrics.csv".format(config.source, config.target)
# ... (code for writing headers and hyperparameters removed) ...
# --- End Remove Manual CSV Logging ---

epochs = 200 # Consider making this a config parameter: config.epochs


def Entropy(input, weight, label):
    # (Keep this function as is)
    softmax_out = nn.Softmax(dim=-1)(input)
    entropy = -label * torch.log(softmax_out + 1e-5)
    entropy_loss = torch.mean(weight * torch.sum(entropy, dim=1))
    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return entropy_loss

# def calculate_expert_uncertainty(experts_outputs, num_classes, cls_model, uncertainty_k=5):
#     # (Keep this function as is, but use config.uncertainty_k)
#     import scipy.special # Keep import local if only used here
#     num_nodes, num_experts, d_feature = experts_outputs.shape
#     experts_logits = torch.zeros(num_nodes, num_experts, num_classes, device=experts_outputs.device)
#     for i in range(num_experts):
#         experts_logits[:, i, :] = cls_model(experts_outputs[:, i, :])
#     experts_probs = torch.softmax(experts_logits, dim=-1)
#     log_probs = torch.log(experts_probs + 1e-9)
#     joint_log_probs = torch.logsumexp(log_probs, dim=1)
#     total_confidence = torch.softmax(joint_log_probs, dim=-1)
#     expected_ratios = joint_log_probs + torch.log(total_confidence + 1e-9)
#     # Use numpy for logsumexp on CPU
#     expected_ratios_np = scipy.special.logsumexp(expected_ratios.cpu().detach().numpy(), axis=-1)
#     uncertainty = torch.from_numpy(expected_ratios_np).float().to(experts_outputs.device)
#     k = min(uncertainty_k, uncertainty.size(0)) # Use uncertainty_k from args/config
#     if k > 0:
#         _, topk_indices = torch.topk(uncertainty, k, dim=0)
#         uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)
#         uncertainty_mask[topk_indices] = True
#     else:
#         uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)
#     return uncertainty_mask

def calculate_expert_uncertainty(experts_outputs, num_classes, cls_model, uncertainty_k=5):
    num_nodes, num_experts, d_feature = experts_outputs.shape
    experts_logits = torch.zeros(num_nodes, num_experts, num_classes, device=experts_outputs.device)
    for i in range(num_experts):
        experts_logits[:, i, :] = cls_model(experts_outputs[:, i, :])
    experts_probs = torch.softmax(experts_logits, dim=-1)
    log_probs = torch.log(experts_probs + 1e-9)
    joint_log_probs = torch.logsumexp(log_probs, dim=1)
    total_confidence = torch.softmax(joint_log_probs, dim=-1)
    expected_ratios = joint_log_probs + torch.log(total_confidence + 1e-9)
    # 纯 PyTorch 实现 logsumexp
    uncertainty = torch.logsumexp(expected_ratios, dim=-1)
    k = min(uncertainty_k, uncertainty.size(0))
    if k > 0:
        _, topk_indices = torch.topk(uncertainty, k, dim=0)
        uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)
        uncertainty_mask[topk_indices] = True
    else:
        uncertainty_mask = torch.zeros_like(uncertainty, dtype=torch.bool)
    return uncertainty_mask

def get_max_hop_neighbors(edge_index, num_nodes, mask):
    # (Keep this function as is)
    neighbor_mask = mask.clone()
    current_nodes = torch.where(mask)[0]
    if current_nodes.numel() == 0: return neighbor_mask # Handle empty mask case

    adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1], device=edge_index.device), (num_nodes, num_nodes)).to_dense()
    adj_matrix = (adj_matrix + adj_matrix.T).clamp(0, 1) # Ensure symmetry for undirected

    visited_mask = mask.clone()
    frontier = mask.clone()

    while frontier.any():
        # Sparse matrix multiplication might be more efficient for large graphs if adj is sparse
        new_neighbors_float = torch.matmul(adj_matrix, frontier.float())
        new_neighbors_mask = (new_neighbors_float > 0) & (~visited_mask)

        if not new_neighbors_mask.any():
            break

        neighbor_mask |= new_neighbors_mask
        visited_mask |= new_neighbors_mask
        frontier = new_neighbors_mask

    return neighbor_mask

def train(epoch):
    for model in models:
        model.train()
    # optimizer.zero_grad()

    # Set rate for GradReverse (ensure this is the intended way to set it)
    # GradReverse.rate = min((epoch + 1) / epochs, 0.05)

    encoded_source, experts_outputs, source_gate_loss, source_clean_logits = encode(source_data, config.source)
    source_logits = cls_model(encoded_source)

    # Use config for uncertainty_k
    uncertainty_mask = calculate_expert_uncertainty(experts_outputs, source_data.num_classes, cls_model, config.uncertainty_k)

    max_hop_neighbors_mask = get_max_hop_neighbors(source_data.edge_index, source_data.num_nodes, uncertainty_mask)
    combined_mask = uncertainty_mask | max_hop_neighbors_mask

    # Initialize losses to avoid potential unbound errors
    select_loss = torch.tensor(0.0, device=device)
    high_quality_semi_loss = torch.tensor(0.0, device=device)
    # semi_loss = torch.tensor(0.0, device=device) # Initialize base semi_loss too

    # Use a persistent dictionary to store expert selections across epochs
    if not hasattr(train, 'expert_selections_cache'):
        train.expert_selections_cache = {}

    # Check if combined_mask has any True values before proceeding with LLM part
    if combined_mask.sum() > 0:
        # Only call LLM every llm_interval epochs
        uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()
        if (epoch-1) % config.llm_interval == 0:
            graph2text_encoder = Graph2TextEncoder()
            # ...existing code...
            # Use LLM for expert selection only if there are uncertain nodes
            uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()
            if not uncertainty_node_indices:
                print(f"Epoch {epoch}: No high uncertainty nodes found, skipping LLM expert selection.")
                expert_selections = {}
            else:
                print(f"Epoch {epoch}: Found {len(uncertainty_node_indices)} high uncertainty nodes. Querying LLM...")
                expert_selections = {}
                prompts = []
                for node_id in uncertainty_node_indices:
                    # 构造mask: 只包含当前node和其邻居
                    node_mask = torch.zeros(source_data.num_nodes, dtype=torch.bool, device=source_data.edge_index.device)
                    node_mask[node_id] = True
                    # 找到所有邻居
                    neighbors = set()
                    edge_index_np = source_data.edge_index.cpu().numpy()
                    for i in range(edge_index_np.shape[1]):
                        src, dst = edge_index_np[0, i], edge_index_np[1, i]
                        if src == node_id:
                            neighbors.add(dst)
                        if dst == node_id:
                            neighbors.add(src)
                    for n in neighbors:
                        node_mask[n] = True
                    # 用mask生成graph_description
                    graph_description = graph2text_encoder.encode(source_data.edge_index, mask=node_mask, num_nodes=source_data.num_nodes)
                    prompt = f"""
                    You are an expert on GNN experts selector, given node and its neighorbood: {graph_description}
                    and {config.expert_num} GNN experts: (0:1-hop, 1:2-hop, 2:3-hop, ...) {' '.join([f'{i}:{i+1}-hop' for i in range(config.expert_num)])}
                    - 1-hop: Use when direct neighbors provide sufficient classification signals.
                    - 2-hop: Use for indirect relationships.
                    - 3-hop: Use for long-range dependencies or hierarchical structures.
                    give out your choice on expert directly for node {node_id}.
                    Please note:
                    1. If the structure around the node is simple and there are few neighbors, it is recommended to choose 1-hop expert
                    2. If the node has more 2-hop neighbors, it is recommended to choose 2-hop expert
                    3. If the node is in a complex community structure, it is recommended to choose 3-hop expert
                    4. Given the specific reason for the selection, it is necessary to be based on the actual structural characteristics of the node.
                    return in json format directly:
                    {{
                        "reason": "your reason",
                        "expert": 0, 1, or 2, ..., up to {config.expert_num - 1},
                        "probability": 0.x
                    }}
                    """
                    prompts.append(prompt)
                # Use config.llm
                for idx, node_id in enumerate(uncertainty_node_indices):
                    prompt = prompts[idx]
                    try:
                        response = ollama.generate(
                            model=config.llm,
                            prompt=prompt,
                            format='json',
                            options={
                                'temperature': 0
                            } # Adjust temperature as needed
                        )
                        json_output_str = response['response']
                        try:
                            parsed_json = json.loads(json_output_str)
                            expert = parsed_json.get("expert") # Get expert value
                            # Validate expert is int and within range
                            if isinstance(expert, int) and 0 <= expert < config.expert_num:
                                expert_selections[node_id] = expert
                            else:
                                print(f"Warning: Invalid expert value {expert} for node {node_id}. Defaulting to random.")
                                expert_selections[node_id] = np.random.randint(0, config.expert_num)
                        except json.JSONDecodeError as e:
                            print(f"Epoch {epoch}: JSONDecodeError for node {node_id}: {e}. Response: '{json_output_str}'. Defaulting to random.")
                            expert_selections[node_id] = np.random.randint(0, config.expert_num)
                    except Exception as e:
                        print(f"Epoch {epoch}: Ollama error for node {node_id}: {e}. Defaulting to random.")
                        expert_selections[node_id] = np.random.randint(0, config.expert_num) # Defaulting to random
                # Update the cache with new selections
                train.expert_selections_cache.update(expert_selections)

                # 打印每次选择的 uncertainty node id 及 llm 的 choice
                for node_id in uncertainty_node_indices:
                    expert = expert_selections.get(node_id, None)
                    print(f"Epoch {epoch}: LLM selected expert {expert} for uncertainty node {node_id}")
# ...existing code...

        else:
            print(f"Epoch {epoch}: Not calling LLM (interval not reached). Using cached expert selections.")
            expert_selections = train.expert_selections_cache

        # Calculate select_loss if expert_selections is not empty
        
        if expert_selections:
            # moe_expert_indices, moe_expert_probs, moe_full_gates = encoder.get_node_expert_assignment(source_data.x, source_data.edge_index, k=config.expert_num) # Ensure k matches num_experts if needed
            num_experts = config.expert_num
            llm_expert_dist = torch.zeros(len(uncertainty_node_indices), num_experts, device=device)
            # Gather MoE probabilities for the specific uncertain nodes
            uncertainty_node_indices_tensor = torch.tensor(uncertainty_node_indices, device=device, dtype=torch.long)
            # Directly index the gates tensor from get_node_expert_assignment for the uncertain nodes
            # The 'gates' returned by get_node_expert_assignment IS the probability distribution
            # _, _, moe_full_gates = encoder.get_node_expert_assignment(source_data.x, source_data.edge_index, k=config.expert_num) # Get the full [N, num_experts] gate distribution
            moe_expert_dist = source_clean_logits[uncertainty_node_indices_tensor] # Shape: [num_uncertain, num_experts]

            # Build llm_expert_dist (target distribution)
            llm_expert_indices_list = [expert_selections.get(node_id_int, 0) for node_id_int in uncertainty_node_indices]
            llm_expert_indices_tensor = torch.tensor(llm_expert_indices_list, device=device, dtype=torch.long)
            # Create one-hot encoding for LLM selections
            llm_expert_dist = F.one_hot(llm_expert_indices_tensor, num_classes=num_experts).float()

            # print(f"Epoch {epoch}: LLM expert distribution shape: {llm_expert_dist.shape}, MoE expert distribution shape: {moe_expert_dist.shape}")

            # temperature = 2.0 # Or use a config parameter
            # llm_soft = torch.softmax(llm_expert_dist / temperature, dim=-1)
            # moe_soft = torch.softmax(moe_expert_dist / temperature, dim=-1)

            # print(f"Epoch {epoch}: LLM expert distribution: {llm_soft}, MoE expert distribution: {moe_soft}")

            # Ensure no log(0)
            # select_loss = torch.nn.functional.kl_div(
            #     (moe_soft + 1e-9).log(), # input (log prob)
            #     llm_soft,      # target (prob) - detached correctly
            #     reduction='batchmean',
            #     log_target=False        # target is not log prob
            # ) * (temperature ** 2) # Scaling factor for temperature
            select_loss = torch.nn.functional.cross_entropy(
                moe_expert_dist, llm_expert_dist
            )


    # Classifier loss:
    cls_loss = loss_func(source_logits[label_mask], source_data.y[label_mask])

    # Pseudo labeling and semi-supervised loss:
    # with torch.no_grad(): # Don't need gradients for pseudo-label generation
    _, s_plabel = torch.max(source_logits, dim=1)
    s_plabel[label_mask] = source_data.y[label_mask] # Correct known labels
    s_weight = get_renode_weight(source_data, s_plabel).to(device)
    s_plabel_onehot = F.one_hot(s_plabel, source_data.num_classes).float() # Use float

    # Calculate base semi_loss only if there are unlabeled nodes
    # if (~label_mask).sum() > 0:
    #     semi_loss = Entropy(source_logits[~label_mask], s_weight[~label_mask], s_plabel_onehot[~label_mask])
    # else:
    #     semi_loss = torch.tensor(0.0, device=device)

    # Calculate high-quality semi-loss
    # with torch.no_grad(): # Calculations for mask don't need gradient
        # Calculate MoE entropy on unlabeled data
    if (~label_mask).sum() > 0:
        moe_softmax = nn.Softmax(dim=-1)(source_logits[~label_mask])
        moe_entropy = -torch.sum(moe_softmax * torch.log(moe_softmax + 1e-5), dim=1)

        # Calculate average expert entropy on unlabeled data
        num_experts = experts_outputs.shape[1]
        expert_entropies = torch.zeros(experts_outputs.shape[0], num_experts, device=device) # Size should match experts_outputs
        # Iterate only over unlabeled nodes for efficiency
        unlabeled_indices = torch.where(~label_mask)[0]
        if unlabeled_indices.numel() > 0:
                # expert_outputs for unlabeled nodes only: experts_outputs[unlabeled_indices]
                # shape: [num_unlabeled, num_experts, d_feature]
                unlabeled_experts_outputs = experts_outputs[unlabeled_indices]
                num_unlabeled = unlabeled_experts_outputs.shape[0]
                expert_entropies_unlabeled = torch.zeros(num_unlabeled, num_experts, device=device)

                for i in range(num_experts):
                    expert_logits_unlabeled = cls_model(unlabeled_experts_outputs[:, i, :])
                    expert_softmax_unlabeled = nn.Softmax(dim=-1)(expert_logits_unlabeled)
                    expert_entropies_unlabeled[:, i] = -torch.sum(expert_softmax_unlabeled * torch.log(expert_softmax_unlabeled + 1e-5), dim=1)

                avg_expert_entropy = torch.mean(expert_entropies_unlabeled, dim=1)
                high_quality_mask_unlabeled = moe_entropy < avg_expert_entropy

                # Calculate loss only on high quality subset
                if high_quality_mask_unlabeled.sum() > 0:
                    high_quality_semi_loss = Entropy(source_logits[unlabeled_indices][high_quality_mask_unlabeled],
                                                s_weight[unlabeled_indices][high_quality_mask_unlabeled],
                                                s_plabel_onehot[unlabeled_indices][high_quality_mask_unlabeled])
                else:
                    high_quality_semi_loss = torch.tensor(0.0, device=device)
        else: # No unlabeled nodes
                high_quality_semi_loss = torch.tensor(0.0, device=device)

    else: # No unlabeled nodes
            high_quality_semi_loss = torch.tensor(0.0, device=device)


    gate_loss = source_gate_loss # Currently only source gate loss

    # Calculate total loss
    # Ensure all components are tensors
    epoch_weight = float(epoch) / epochs
    # epoch_weight = 0

    loss = cls_loss + epoch_weight * (config.semi_weight * high_quality_semi_loss + config.select_weight * select_loss) #+ semi_loss
    # loss = gate_loss + epoch_weight * (config.semi_weight * high_quality_semi_loss) + config.select_weight * select_loss
    # loss = cls_loss + gate_loss + (config.semi_weight * high_quality_semi_loss)

    # --- W&B Logging for Training Step ---
    wandb.log({
        'epoch': epoch,
        'train/cls_loss': cls_loss.item(),
        # 'train/gate_loss': gate_loss.item(),
        'train/select_loss': select_loss.item(), # Handle non-tensor case
        # 'train/semi_loss': semi_loss.item(),
        'train/high_quality_semi_loss': high_quality_semi_loss.item(),
        'train/total_loss': loss.item(),
    })
    # --- End W&B Logging ---

    # Print losses (optional, as they are logged to W&B)
    print(f"Epoch {epoch} Losses: Cls={cls_loss.item():.4f}, Select={select_loss.item() if isinstance(select_loss, torch.Tensor) else select_loss:.4f}, HQ Semi={high_quality_semi_loss.item() if isinstance(high_quality_semi_loss, torch.Tensor) else high_quality_semi_loss:.4f}, Total={loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    # print(f'grad of moe router: {encoder.w_gate.grad}, grad of cls: {cls_model[0].weight.grad}, {cls_model[0].bias.grad}')
    # Gradient clipping (optional, but can help stability)
    # torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    # optimizer_moe.step() # Update MoE gate separately if needed


# --- Training Loop ---
best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0
best_macro_f1 = 0.0
best_micro_f1 = 0.0

for epoch in range(1, epochs + 1): # Run for `epochs` epochs (e.g., 1 to 200)
    train(epoch)
    # Use try-except for evaluation robustness
    try:
        source_correct, _, _, output_source = test(source_data, config.source, source_data.test_mask)
        target_correct, macro_f1, micro_f1, output_target = test(target_data, config.target) # No mask for target test? Assuming full graph test

        print(f"Epoch: {epoch}, Source Acc: {source_correct:.4f}, Target Acc: {target_correct:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

        # --- W&B Logging for Evaluation Step ---
        wandb.log({
            'epoch': epoch, # Log epoch here too for alignment
            'eval/source_acc': source_correct, # .item() if tensor
            'eval/target_acc': target_correct, # .item() if tensor
            'eval/macro_f1': macro_f1,
            'eval/micro_f1': micro_f1
        })
        # --- End W&B Logging ---

        # Update best metrics based on target accuracy
        if target_correct > best_target_acc:
            best_target_acc = target_correct
            best_source_acc = source_correct # Store corresponding source acc
            best_macro_f1 = macro_f1
            best_micro_f1 = micro_f1
            best_epoch = epoch
            print(f"*** New best target accuracy at epoch {epoch}: {best_target_acc:.4f} ***")


    except Exception as e:
        print(f"Error during evaluation at epoch {epoch}: {e}")
        # Optionally log error to W&B or break the loop
        # wandb.log({'error': str(e)})
        break # Stop training if evaluation fails critically

# --- End Training Loop ---

print("=============================================================")
final_id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}, drop_out: {}, expert_num: {}, llm: {}, uncertainty_k: {}, gate_coef: {}" \
    .format(config.source, config.target, config.seed, config.label_rate, config.learning_rate, config.weight_decay,
            config.encoder_dim, config.drop_out, config.expert_num, config.llm, config.uncertainty_k, config.gate_coef) # Recreate id with final config values
line = "{}\n - Best Epoch: {}, Best Source Acc: {:.5f}, Best Target Acc: {:.5f}, Best Macro F1: {:.5f}, Best Micro F1: {:.5f}" \
    .format(final_id_str, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)
print(line)

# --- Log Best Metrics to W&B Summary ---
# wandb.summary stores key-value pairs for the run's final summary
wandb.summary['best_epoch'] = best_epoch
wandb.summary['best_source_acc'] = best_source_acc
wandb.summary['best_target_acc'] = best_target_acc
wandb.summary['best_macro_f1'] = best_macro_f1
wandb.summary['best_micro_f1'] = best_micro_f1
# --- End W&B Summary Logging ---


# Log final results to a local file (Optional, W&B summary is usually sufficient)
os.makedirs('log', exist_ok=True) # Ensure log dir exists
log_file_path = "log/{}-{}.log".format(config.source, config.target)
with open(log_file_path, 'a') as f:
    log_line = "{} - Best Epoch: {:0>3d}, Best Target Acc: {:.5f}, Best Macro F1: {:.5f}, Best Micro F1: {:.5f}\t" \
               .format(final_id_str, best_epoch, best_target_acc, best_macro_f1, best_micro_f1) + time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + f" W&B Run: {run.url}\n" # Add W&B run URL
    f.write(log_line)

# --- Save Log File to W&B (Optional) ---
# wandb.save(log_file_path)
# --- End Save Log File ---

# --- Finish W&B Run ---
wandb.finish()
# --- End Finish W&B Run ---
