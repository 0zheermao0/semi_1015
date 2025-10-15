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
import ollama
import multiprocessing

# --- W&B Integration ---
import wandb
# --- End W&B Integration ---

from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
from gnn.moe import MoE
from common.graph_encoder import GraphEncoder as Graph2TextEncoder

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
# 基本参数
parser.add_argument("--source", type=str, default='citationv1')
parser.add_argument("--target", type=str, default='acmv9')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=2e-3)
parser.add_argument("--drop_out", type=float, default=1e-1)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--label_rate", type=float, default=0.05)
parser.add_argument("--expert_num", type=int, default=3)
parser.add_argument("--llm", type=str, default='qwen2.5:7b')
parser.add_argument("--uncertainty_k", type=int, default=100)
parser.add_argument("--gate_coef", type=float, default=3e-2)
parser.add_argument("--select_weight", type=float, default=0.03)
parser.add_argument("--semi_weight", type=float, default=1)
parser.add_argument("--llm_interval", type=int, default=20, help="Interval of epochs to call LLM for expert selection")
parser.add_argument("--hop", type=int, default=5)
# 消融实验控制参数
parser.add_argument("--disable_moe", action='store_true', default=True, help="Disable MoE, use single expert")
parser.add_argument("--disable_llm_select", action='store_true', default=True, help="Disable LLM selection distillation")
parser.add_argument("--disable_high_quality_semi", action='store_true', default=True, help="Disable high-quality pseudo-label training")
# W&B 相关参数
parser.add_argument("--wandb_project", type=str, default="GNN-Domain-Adaptation-Ablation")
parser.add_argument("--wandb_entity", type=str, default=None, help="Your W&B username or team name")

args = parser.parse_args()

# --- W&B Initialization ---
run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=args
)
config = wandb.config
# --- End W&B Initialization ---

seed = config.seed

# 构建运行名称，包含消融选项
ablation_str = ""
if config.disable_moe:
    ablation_str += "noMoE_"
if config.disable_llm_select:
    ablation_str += "noLLMSelect_"
if config.disable_high_quality_semi:
    ablation_str += "noHQSemi_"
if not ablation_str:
    ablation_str = "full_"
wandb.run.name = f"{ablation_str}{config.source}-{config.target}-lr{config.learning_rate:.1e}-seed{config.seed}"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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

def index2dense(edge_index, nnode=2708):
    device = edge_index.device
    adj = torch.zeros((nnode, nnode), dtype=torch.float32, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def encode(data, cache_name, mask=None):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if config.disable_moe:
        encoded_output = encoder(x, edge_index, cache_name)
        experts_outputs = torch.zeros(x.size(0), 1, encoded_output.size(-1), device=device)
        gate_loss = torch.tensor(0.0, device=device)
        clean_logits = torch.zeros(x.size(0), 1, device=device)
    else:
        encoded_output, experts_outputs, gate_loss, clean_logits = encoder(x, edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
        experts_outputs = experts_outputs[mask]
        clean_logits = clean_logits[mask]
    return encoded_output, experts_outputs, gate_loss, clean_logits

def predict(data, cache_name, mask=None):
    encoded_output, _, _, _ = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits

def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    labels_cpu = labels.cpu().detach()
    preds_cpu = preds.cpu().detach()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1

def test(data, cache_name, mask=None):
    encoded_output, experts_outputs, _, clean_logits = encode(data, cache_name, mask)
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy, macro_f1, micro_f1 = evaluate(preds, labels)

    if cache_name == config.target:
        if not config.disable_moe:
            moe_expert_indices, moe_expert_probs, _ = encoder.get_node_expert_assignment(data.x, data.edge_index)
            expert_selection_log_path = f"log/{config.target}-expert-selection.csv"
            os.makedirs(os.path.dirname(expert_selection_log_path), exist_ok=True)
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

    return accuracy, macro_f1, micro_f1, encoded_output

def get_renode_weight(data, pseudo_label):
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

def build_prompt_for_node(args):
    node_id, source_data, config_dict, expert_num, hop = args
    node_mask = torch.zeros(source_data.num_nodes, dtype=torch.bool)
    node_mask[node_id] = True

    max_hop = hop
    edge_index_np = source_data.edge_index.cpu().numpy()
    adj = [[] for _ in range(source_data.num_nodes)]
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj[src].append(dst)
        adj[dst].append(src)

    visited = set([node_id])
    current_level = set([node_id])
    for hop in range(1, max_hop + 1):
        next_level = set()
        for current in current_level:
            for neighbor in adj[current]:
                if neighbor not in visited:
                    next_level.add(neighbor)
                    visited.add(neighbor)
        current_level = next_level
    for n in visited:
        node_mask[n] = True

    graph_description = graph2text_encoder.encode(
        source_data.edge_index, mask=node_mask, num_nodes=source_data.num_nodes, style='natural'
    )
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
    print(f"Prompt for node {node_id}: {prompt}")
    return prompt

loss_func = nn.CrossEntropyLoss().to(device)

# 根据 disable_moe 参数决定是否使用 MoE
if config.disable_moe:
    print("MoE is disabled, using single expert.")
    num_hops_config = [1]  # 单一专家，使用 1-hop
    encoder = PPMIConv(in_channels=source_data.num_features, out_channels=config.encoder_dim, path_len=1).to(device)
else:
    print("MoE is enabled.")
    num_hops_config = [1, 2, 3][:config.expert_num] if config.expert_num >= 3 else [1] * config.expert_num
    encoder = MoE(input_size=source_data.num_features, output_size=config.encoder_dim, num_experts=config.expert_num, k=1
              , coef=config.gate_coef, gnn_type='ppmi', num_hops=num_hops_config).to(device)

cls_model = nn.Sequential(
    nn.Linear(config.encoder_dim, dataset.num_classes),
).to(device)

# 加载预计算的 PPMI 矩阵
os.makedirs('tmp', exist_ok=True)
source_pkl_path = f'tmp/{config.source}.pkl'
target_pkl_path = f'tmp/{config.target}.pkl'

if os.path.exists(source_pkl_path):
    with open(source_pkl_path, 'rb') as f:
        source_edge_index, _ = pickle.load(f)
else:
    print(f"Warning: {source_pkl_path} not found. PPMI may not be properly initialized.")
    source_edge_index = source_data.edge_index

if os.path.exists(target_pkl_path):
    with open(target_pkl_path, 'rb') as f:
        target_edge_index, _ = pickle.load(f)
else:
    print(f"Warning: {target_pkl_path} not found. PPMI may not be properly initialized.")
    target_edge_index = target_data.edge_index

source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)
target_data.new_adj = index2dense(target_edge_index, target_data.num_nodes).to(device)

models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)

epochs = 200

def Entropy(input, weight, label):
    softmax_out = nn.Softmax(dim=-1)(input)
    entropy = -label * torch.log(softmax_out + 1e-5)
    entropy_loss = torch.mean(weight * torch.sum(entropy, dim=1))
    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return entropy_loss

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
    neighbor_mask = mask.clone()
    current_nodes = torch.where(mask)[0]
    if current_nodes.numel() == 0: return neighbor_mask

    adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1], device=edge_index.device), (num_nodes, num_nodes)).to_dense()
    adj_matrix = (adj_matrix + adj_matrix.T).clamp(0, 1)

    visited_mask = mask.clone()
    frontier = mask.clone()

    while frontier.any():
        new_neighbors_float = torch.matmul(adj_matrix, frontier.float())
        new_neighbors_mask = (new_neighbors_float > 0) & (~visited_mask)

        if not new_neighbors_mask.any():
            break

        neighbor_mask |= new_neighbors_mask
        visited_mask |= new_neighbors_mask
        frontier = new_neighbors_mask

    return neighbor_mask

# LLM 专家选择预处理
expert_selections_path = f"log/{config.source}-{config.llm}-selections.json"
prompts_path = f"log/{config.source}-prompts.pkl"
if os.path.exists(expert_selections_path):
    with open(expert_selections_path, 'r') as f:
        expert_selections = json.load(f)
    expert_selections = {int(k): v for k, v in expert_selections.items()}
    print(f"Loaded expert selections from {expert_selections_path}, total: {len(expert_selections)}")
else:
    print(f"Generating expert selections for all nodes via LLM...")
    expert_selections = {}
    graph2text_encoder = Graph2TextEncoder()
    if os.path.exists(prompts_path):
        with open(prompts_path, 'rb') as f:
            prompts = pickle.load(f)
        print(f"Loaded prompts from {prompts_path}")
    else:
        prompts = []
        graph2text_encoder = Graph2TextEncoder()
        expert_num = int(config.expert_num)
        hop = int(config.hop)
        args_list = [
            (node_id, source_data.cpu(), {}, expert_num, hop)
            for node_id in range(source_data.num_nodes)
        ]
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 16)) as pool:
            prompts = pool.map(build_prompt_for_node, args_list)
        with open(prompts_path, 'wb') as f:
            pickle.dump(prompts, f)
        print(f"Prompts saved to {prompts_path}")
    for node_id, prompt in enumerate(prompts):
        try:
            response = ollama.generate(
                model=config.llm,
                prompt=prompt,
                format='json',
                options={'temperature': 0, 'num_ctx': 40960, 'num_predict': 4096}
            )
            json_output_str = response['response']
            try:
                parsed_json = json.loads(json_output_str)
                expert = parsed_json.get("expert")
                if isinstance(expert, int) and 0 <= expert < config.expert_num:
                    expert_selections[node_id] = expert
                else:
                    print(f"Warning: Invalid expert value {expert} for node {node_id}. Defaulting to random.")
                    expert_selections[node_id] = np.random.randint(0, config.expert_num)
            except json.JSONDecodeError as e:
                print(f"Node {node_id}: JSONDecodeError: {e}. Response: '{json_output_str}'. Defaulting to random.")
                expert_selections[node_id] = np.random.randint(0, config.expert_num)
        except Exception as e:
            print(f"Node {node_id}: Ollama error: {e}. Defaulting to random.")
            expert_selections[node_id] = np.random.randint(0, config.expert_num)
        if (node_id+1) % 100 == 0:
            print(f"Processed {node_id+1}/{source_data.num_nodes} nodes...")
    with open(expert_selections_path, 'w') as f:
        json.dump(expert_selections, f)
    print(f"Expert selections saved to {expert_selections_path}")

def train(epoch):
    for model in models:
        model.train()

    encoded_source, experts_outputs, source_gate_loss, source_clean_logits = encode(source_data, config.source)
    source_logits = cls_model(encoded_source)

    if config.disable_moe:
        uncertainty_mask = torch.zeros(source_data.num_nodes, dtype=torch.bool, device=device)
    else:
        uncertainty_mask = calculate_expert_uncertainty(experts_outputs, source_data.num_classes, cls_model, config.uncertainty_k)

    max_hop_neighbors_mask = get_max_hop_neighbors(source_data.edge_index, source_data.num_nodes, uncertainty_mask)
    combined_mask = uncertainty_mask | max_hop_neighbors_mask

    select_loss = torch.tensor(0.0, device=device)
    high_quality_semi_loss = torch.tensor(0.0, device=device)

    expert_selections_local = expert_selections if not config.disable_llm_select else {}

    if not config.disable_moe and not config.disable_llm_select:
        uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()
        if uncertainty_node_indices:
            num_experts = config.expert_num
            llm_expert_dist = torch.zeros(len(uncertainty_node_indices), num_experts, device=device)
            uncertainty_node_indices_tensor = torch.tensor(uncertainty_node_indices, device=device, dtype=torch.long)
            moe_expert_dist = source_clean_logits[uncertainty_node_indices_tensor]
            llm_expert_indices_list = [expert_selections_local.get(node_id_int, 0) for node_id_int in uncertainty_node_indices]
            llm_expert_indices_tensor = torch.tensor(llm_expert_indices_list, device=device, dtype=torch.long)
            llm_expert_dist = F.one_hot(llm_expert_indices_tensor, num_classes=num_experts).float()
            select_loss = torch.nn.functional.cross_entropy(
                moe_expert_dist, llm_expert_dist
            )

    cls_loss = loss_func(source_logits[label_mask], source_data.y[label_mask])

    _, s_plabel = torch.max(source_logits, dim=1)
    s_plabel[label_mask] = source_data.y[label_mask]
    s_weight = get_renode_weight(source_data, s_plabel).to(device)
    s_plabel_onehot = F.one_hot(s_plabel, source_data.num_classes).float()

    if not config.disable_high_quality_semi and (~label_mask).sum() > 0 and not config.disable_moe:
        num_experts = experts_outputs.shape[1]
        unlabeled_indices = torch.where(~label_mask)[0]
        if unlabeled_indices.numel() > 0 and expert_selections_local:
            llm_expert_indices = []
            for idx in unlabeled_indices.tolist():
                llm_expert_indices.append(expert_selections_local.get(idx, 0))
            llm_expert_indices_tensor = torch.tensor(llm_expert_indices, device=device, dtype=torch.long)
            unlabeled_experts_outputs = experts_outputs[unlabeled_indices]
            num_unlabeled = unlabeled_experts_outputs.shape[0]
            expert_entropies_unlabeled = torch.zeros(num_unlabeled, num_experts, device=device)
            for i in range(num_experts):
                expert_logits_unlabeled = cls_model(unlabeled_experts_outputs[:, i, :])
                expert_softmax_unlabeled = nn.Softmax(dim=-1)(expert_logits_unlabeled)
                expert_entropies_unlabeled[:, i] = -torch.sum(expert_softmax_unlabeled * torch.log(expert_softmax_unlabeled + 1e-5), dim=1)
            llm_expert_entropy = expert_entropies_unlabeled[torch.arange(num_unlabeled), llm_expert_indices_tensor]
            mask = torch.ones_like(expert_entropies_unlabeled, dtype=torch.bool)
            mask[torch.arange(num_unlabeled), llm_expert_indices_tensor] = False
            other_expert_entropy = expert_entropies_unlabeled.masked_fill(~mask, float('-inf'))
            max_other_entropy, _ = other_expert_entropy.max(dim=1)
            high_quality_mask_unlabeled = llm_expert_entropy < max_other_entropy
            if high_quality_mask_unlabeled.sum() > 0:
                high_quality_semi_loss = Entropy(source_logits[unlabeled_indices][high_quality_mask_unlabeled],
                                            s_weight[unlabeled_indices][high_quality_mask_unlabeled],
                                            s_plabel_onehot[unlabeled_indices][high_quality_mask_unlabeled])
            else:
                high_quality_semi_loss = torch.tensor(0.0, device=device)
        else:
            high_quality_semi_loss = torch.tensor(0.0, device=device)
    else:
        high_quality_semi_loss = torch.tensor(0.0, device=device)

    gate_loss = source_gate_loss if not config.disable_moe else torch.tensor(0.0, device=device)

    epoch_weight = float(epoch) / epochs

    loss = cls_loss + epoch_weight * (config.semi_weight * high_quality_semi_loss + config.select_weight * select_loss)

    wandb.log({
        'epoch': epoch,
        'train/cls_loss': cls_loss.item(),
        'train/select_loss': select_loss.item(),
        'train/high_quality_semi_loss': high_quality_semi_loss.item(),
        'train/total_loss': loss.item(),
    })

    print(f"Epoch {epoch} Losses: Cls={cls_loss.item():.4f}, Select={select_loss.item() if isinstance(select_loss, torch.Tensor) else select_loss:.4f}, HQ Semi={high_quality_semi_loss.item() if isinstance(high_quality_semi_loss, torch.Tensor) else high_quality_semi_loss:.4f}, Total={loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 训练循环
best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0
best_macro_f1 = 0.0
best_micro_f1 = 0.0

for epoch in range(1, epochs + 1):
    train(epoch)
    try:
        source_correct, _, _, output_source = test(source_data, config.source, source_data.test_mask)
        target_correct, macro_f1, micro_f1, output_target = test(target_data, config.target)

        print(f"Epoch: {epoch}, Source Acc: {source_correct:.4f}, Target Acc: {target_correct:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

        wandb.log({
            'epoch': epoch,
            'eval/source_acc': source_correct,
            'eval/target_acc': target_correct,
            'eval/macro_f1': macro_f1,
            'eval/micro_f1': micro_f1
        })

        if target_correct > best_target_acc:
            best_target_acc = target_correct
            best_source_acc = source_correct
            best_macro_f1 = macro_f1
            best_micro_f1 = micro_f1
            best_epoch = epoch
            print(f"*** New best target accuracy at epoch {epoch}: {best_target_acc:.4f} ***")

    except Exception as e:
        print(f"Error during evaluation at epoch {epoch}: {e}")
        break

print("=============================================================")
final_id_str = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, dim: {}, drop_out: {}, expert_num: {}, llm: {}, uncertainty_k: {}, gate_coef: {}".format(
    config.source, config.target, config.seed, config.label_rate, config.learning_rate, config.weight_decay,
    config.encoder_dim, config.drop_out, config.expert_num, config.llm, config.uncertainty_k, config.gate_coef)
line = "{}\n - Best Epoch: {}, Best Source Acc: {:.5f}, Best Target Acc: {:.5f}, Best Macro F1: {:.5f}, Best Micro F1: {:.5f}".format(
    final_id_str, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)
print(line)

wandb.summary['best_epoch'] = best_epoch
wandb.summary['best_source_acc'] = best_source_acc
wandb.summary['best_target_acc'] = best_target_acc
wandb.summary['best_macro_f1'] = best_macro_f1
wandb.summary['best_micro_f1'] = best_micro_f1

os.makedirs('log', exist_ok=True)
log_file_path = "log/{}-{}-ablation.log".format(config.source, config.target)
with open(log_file_path, 'a') as f:
    log_line = "{} - Best Epoch: {:0>3d}, Best Target Acc: {:.5f}, Best Macro F1: {:.5f}, Best Micro F1: {:.5f}\t".format(
        final_id_str, best_epoch, best_target_acc, best_macro_f1, best_micro_f1) + time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + f" W&B Run: {run.url}\n"
    f.write(log_line)

final_target_embeddings_path = f"log/{config.target}-final-embeddings-ablation.pt"
final_target_labels_path = f"log/{config.target}-final-labels-ablation.pt"

if 'output_target' in locals() and output_target is not None:
    print(f"Saving final target embeddings to {final_target_embeddings_path}")
    torch.save(output_target.cpu(), final_target_embeddings_path)
    print(f"Saving final target labels to {final_target_labels_path}")
    torch.save(target_data.y.cpu(), final_target_labels_path)
else:
    print("Re-running test to get final target embeddings...")
    _, _, _, final_output_target = test(target_data, config.target)
    print(f"Saving final target embeddings to {final_target_embeddings_path}")
    torch.save(final_output_target.cpu(), final_target_embeddings_path)
    print(f"Saving final target labels to {final_target_labels_path}")
    torch.save(target_data.y.cpu(), final_target_labels_path)

wandb.finish()
