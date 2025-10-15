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
from common.openai_client import create_openai_client
import multiprocessing

# --- W&B Integration ---
import wandb
# --- End W&B Integration ---

# Import existing modules
from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
from gnn.moe import MoE
from common.graph_encoder import GraphEncoder as Graph2TextEncoder

# Import new enhanced modules
from gnn.gat_soft_prompt_moe import GATSoftPromptMoE
from common.llm_cache import EnhancedLLMCacheManager, LLMResponseParser
from common.loss_scheduler import DynamicLossScheduler, AdaptiveLossScheduler
from common.advanced_losses import DPOLoss, EnhancedConsistencyLoss, MMDDiversityLoss, GateEntropyRegularizer
from common.memory_efficient import MemoryEfficientProcessor, GradientAccumulator, CheckpointManager
from utils.expert_analyzer import ExpertSpecializationAnalyzer
from utils.visualization import EmbeddingVisualizer

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

# Core parameters
parser.add_argument("--source", type=str, default='dblpv7')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=2e-5)
parser.add_argument("--drop_out", type=float, default=0.1)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--label_rate", type=float, default=0.05)
parser.add_argument("--expert_num", type=int, default=6)

# LLM parameters
parser.add_argument("--llm", type=str, default='gpt-3.5-turbo')
parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (can also be set via OPENAI_API_KEY env var)")
parser.add_argument("--openai_base_url", type=str, default=None, help="OpenAI-compatible API base URL (can also be set via OPENAI_BASE_URL env var)")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for LLM")
parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens in LLM response")
parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
parser.add_argument("--uncertainty_k", type=int, default=100)
parser.add_argument("--llm_interval", type=int, default=1, help="Interval of epochs to call LLM for expert selection")
parser.add_argument("--hop", type=int, default=3)
parser.add_argument("--node_limit", type=int, default=50)

# MoE architecture parameters
parser.add_argument("--moe_architecture", type=str, default='original', choices=['original', 'gat_soft_prompt', 'both'],
                    help="MoE architecture to use")
parser.add_argument("--prompt_length", type=int, default=8, help="Length of soft prompts for GAT+Soft-prompt MoE")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads for GAT experts")

# Loss parameters
parser.add_argument("--alpha", type=float, default=0.61, help="DPO temperature parameter")
parser.add_argument("--beta", type=float, default=0.11, help="Consistency loss weight")
parser.add_argument("--consistency_weight", type=float, default=1.0, help="Consistency loss weight")
parser.add_argument("--div_weight", type=float, default=0.01, help="Diversity loss weight")
parser.add_argument("--select_weight", type=float, default=1.0, help="Selection loss weight")
parser.add_argument("--gate_coef", type=float, default=0.1, help="Gate loss coefficient")

# Training parameters
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=0, help="Batch size for memory-efficient processing (0=full graph)")
parser.add_argument("--use_memory_efficient", action='store_true', help="Use memory-efficient processing")
parser.add_argument("--adaptive_scheduler", action='store_true', help="Use adaptive loss scheduling")

# Analysis parameters
parser.add_argument("--analyze_experts", action='store_true', help="Enable expert specialization analysis")
parser.add_argument("--visualize_embeddings", action='store_true', help="Enable embedding visualization")

# W&B parameters
parser.add_argument("--wandb_project", type=str, default="Enhanced-MoE-Domain-Adaptation")
parser.add_argument("--wandb_entity", type=str, default=None)

args = parser.parse_args()

# --- W&B Initialization ---
run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=args
)
config = wandb.config

# Set random seeds
seed = config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("=" * 80)
print("ENHANCED LLM-GUIDED MoE FOR SEMI-SUPERVISED DOMAIN GENERALIZATION")
print("=" * 80)
print(f"Source: {config.source}, Target: {config.target}")
print(f"MoE Architecture: {config.moe_architecture}")
print(f"Experts: {config.expert_num}, LLM Interval: {config.llm_interval}")
print(f"Memory Efficient: {config.use_memory_efficient}, Adaptive Scheduler: {config.adaptive_scheduler}")
print("=" * 80)

# Load datasets
dataset = DomainData("data/{}".format(config.source), name=config.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print(f"Source dataset: {source_data}")

dataset = DomainData("data/{}".format(config.target), name=config.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print(f"Target dataset: {target_data}")

source_data = source_data.to(device)
target_data = target_data.to(device)

# Create label mask
source_train_size = int(source_data.size(0) * config.label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)

# Helper functions
def index2dense(edge_index, nnode=2708):
    device = edge_index.device
    adj = torch.zeros((nnode, nnode), dtype=torch.float32, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def encode(data, cache_name, model, mask=None):
    """Enhanced encoding function supporting dual MoE architectures."""
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Check model type and call appropriate forward method
    if hasattr(model, 'experts') and hasattr(model, 'gate_network'):
        # GAT+Soft-prompt MoE
        encoded_output, experts_outputs, aux_loss, clean_logits = model(x, edge_index)
    else:
        # Original MoE
        encoded_output, experts_outputs, aux_loss, clean_logits = model(x, edge_index, cache_name)

    if mask is not None:
        encoded_output = encoded_output[mask]
        experts_outputs = experts_outputs[mask]
        clean_logits = clean_logits[mask]

    return encoded_output, experts_outputs, aux_loss, clean_logits

def predict(data, cache_name, model, cls_model, mask=None):
    """Enhanced prediction function."""
    encoded_output, _, _, _ = encode(data, cache_name, model, mask)
    logits = cls_model(encoded_output)
    return logits

def evaluate(preds, labels):
    """Evaluate predictions."""
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    labels_cpu = labels.cpu().detach()
    preds_cpu = preds.cpu().detach()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1

def test(data, cache_name, model, cls_model, mask=None):
    """Enhanced test function."""
    encoded_output, experts_outputs, _, clean_logits = encode(data, cache_name, model, mask)
    logits = predict(data, cache_name, model, cls_model, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy, macro_f1, micro_f1 = evaluate(preds, labels)

    return accuracy, macro_f1, micro_f1, encoded_output, experts_outputs, clean_logits

def calculate_expert_uncertainty(experts_outputs, num_classes, cls_model, uncertainty_k=5):
    """Calculate expert uncertainty."""
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

def build_enhanced_prompt_for_node(node_id, source_data, experts_outputs, cls_model, expert_num, hop=5, node_limit=100):
    """Build enhanced LLM prompt for expert selection."""
    node_mask = torch.zeros(source_data.num_nodes, dtype=torch.bool, device=source_data.edge_index.device)
    node_mask[node_id] = True

    # Traverse neighborhood
    max_hop = hop
    edge_index = source_data.edge_index.cpu()
    adj = [[] for _ in range(source_data.num_nodes)]
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[src].append(dst)
        adj[dst].append(src)

    visited = set([node_id])
    current_level = set([node_id])
    for hop in range(1, max_hop + 1):
        if len(visited) >= node_limit:
            break
        next_level = set()
        for current in current_level:
            for neighbor in adj[current]:
                if neighbor not in visited:
                    next_level.add(neighbor)
                    visited.add(neighbor)
        current_level = next_level

    for n in visited:
        node_mask[n] = True

    # Get expert predictions
    node_expert_outputs = experts_outputs[node_id]
    expert_predictions = []
    for expert_idx in range(expert_num):
        expert_logits = cls_model(node_expert_outputs[expert_idx].unsqueeze(0))
        expert_probs = F.softmax(expert_logits, dim=1)
        predicted_class = torch.argmax(expert_probs).item()
        confidence = expert_probs[0][predicted_class].item()
        expert_predictions.append(f"Expert {expert_idx}: predicted class {predicted_class} with confidence {confidence:.3f}")

    # Graph description
    graph2text_encoder = Graph2TextEncoder()
    graph_description = graph2text_encoder.encode(
        source_data.edge_index, mask=node_mask, num_nodes=source_data.num_nodes, style='natural'
    )

    # Enhanced prompt following LLM_Guided_MoE_Reproduction_Prompt.md
    prompt = f"""You are an expert GNN analyst. For node {node_id} with the following neighborhood structure:
    {graph_description}

    Current expert predictions:
    {chr(10).join(expert_predictions)}

    Available experts: {', '.join([f'Expert {i}' for i in range(expert_num)])}

    Analyze the graph structure and expert predictions to determine the most suitable expert.

    Consider:
    1. Structural complexity (simple vs. complex neighborhoods)
    2. Prediction confidence patterns
    3. Expert specialization characteristics

    Based on the reproduction guide, rank all experts from most to least suitable for this node.

    Return your analysis in JSON format:
    {{
        "reasoning": "detailed analysis of structure and predictions",
        "expert": 0,
        "confidence": 0.85,
        "ranking": [0, 1, 2, 3, 4, 5]
    }}

    CRITICAL: You MUST include the "ranking" field with ALL {expert_num} experts ranked from best (0) to worst ({expert_num-1}).
    """
    return prompt

# Initialize models based on architecture
print("Initializing models...")

# Create encoder(s) based on architecture
encoders = {}
if config.moe_architecture in ['original', 'both']:
    # Create num_hops configuration that matches expert_num
    if config.expert_num <= 3:
        num_hops_config = [1] * config.expert_num
    else:
        # For more than 3 experts, create a pattern: [1, 2, 3, 1, 2, 3, ...]
        base_hops = [1, 2, 3]
        num_hops_config = []
        for i in range(config.expert_num):
            num_hops_config.append(base_hops[i % len(base_hops)])

    print(f"Created num_hops_config: {num_hops_config} for {config.expert_num} experts")
    encoders['original'] = MoE(
        input_size=source_data.num_features,
        output_size=config.encoder_dim,
        num_experts=config.expert_num,
        k=1,
        coef=config.gate_coef,
        gnn_type='ppmi',
        num_hops=num_hops_config
    ).to(device)
    print(f"Created Original MoE with {config.expert_num} experts")

if config.moe_architecture in ['gat_soft_prompt', 'both']:
    encoders['gat_soft_prompt'] = GATSoftPromptMoE(
        input_size=source_data.num_features,
        output_size=config.encoder_dim,
        num_experts=config.expert_num,
        prompt_length=config.prompt_length,
        num_heads=config.num_heads,
        dropout=config.drop_out,
        k=1,
        coef=config.gate_coef,
        noisy_gating=False
    ).to(device)
    print(f"Created GAT+Soft-prompt MoE with {config.expert_num} experts")

# Classifier model (shared)
cls_model = nn.Sequential(
    nn.Linear(config.encoder_dim, dataset.num_classes),
).to(device)

# Initialize loss functions and schedulers
loss_functions = {
    'dpo': DPOLoss(alpha=config.alpha),
    'consistency': EnhancedConsistencyLoss(beta=config.beta),
    'mmd_diversity': MMDDiversityLoss(gamma=1.0, device=str(device)),
    'gate_entropy': GateEntropyRegularizer(weight=0.01)
}

if config.adaptive_scheduler:
    scheduler = AdaptiveLossScheduler(config.epochs, config.__dict__)
else:
    scheduler = DynamicLossScheduler(config.epochs, config.__dict__)

# Initialize OpenAI client
llm_client = create_openai_client(config)
print(f"Initialized OpenAI-compatible client with model: {config.llm}")
if config.openai_base_url:
    print(f"Using custom base URL: {config.openai_base_url}")

# Initialize enhanced cache manager
cache_manager = EnhancedLLMCacheManager(cache_dir="llm_cache")
response_parser = LLMResponseParser()

# Initialize analysis tools if requested
analyzer = None
visualizer = None
if config.analyze_experts:
    analyzer = ExpertSpecializationAnalyzer(save_dir="analysis_results")
if config.visualize_embeddings:
    visualizer = EmbeddingVisualizer(save_dir="visualizations")

# Initialize memory-efficient processor if requested
memory_processor = None
if config.use_memory_efficient:
    memory_processor = MemoryEfficientProcessor(device, max_memory_gb=8.0)

# Load precomputed PPMI matrices
os.makedirs('tmp', exist_ok=True)
source_pkl_path = f'tmp/{config.source}.pkl'
target_pkl_path = f'tmp/{config.target}.pkl'

if os.path.exists(source_pkl_path):
    with open(source_pkl_path, 'rb') as f:
        source_edge_index, _ = pickle.load(f)
else:
    print(f"Warning: {source_pkl_path} not found. Using original edge_index.")
    source_edge_index = source_data.edge_index

if os.path.exists(target_pkl_path):
    with open(target_pkl_path, 'rb') as f:
        target_edge_index, _ = pickle.load(f)
else:
    print(f"Warning: {target_pkl_path} not found. Using original edge_index.")
    target_edge_index = target_data.edge_index

source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)
target_data.new_adj = index2dense(target_edge_index, target_data.num_nodes).to(device)

# Training setup
expert_selections_cache = {}

def train_epoch(epoch, encoder_name, encoder, optimizer):
    """Enhanced training epoch."""
    encoder.train()
    cls_model.train()

    # Forward pass
    cache_name = f"{config.source}_{encoder_name}"
    encoded_source, experts_outputs, aux_loss, clean_logits = encode(source_data, cache_name, encoder)
    source_logits = cls_model(encoded_source)

    # Calculate uncertainty
    uncertainty_mask = calculate_expert_uncertainty(experts_outputs, source_data.num_classes, cls_model, config.uncertainty_k)
    uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()

    # Debug output
    print(f"[DEBUG] Epoch {epoch}: Found {len(uncertainty_node_indices)} uncertain nodes")
    print(f"[DEBUG] LLM interval check: epoch % {config.llm_interval} == {epoch % config.llm_interval}")
    print(f"[DEBUG] Should call LLM: {len(uncertainty_node_indices) > 0 and (epoch % config.llm_interval == 0)}")

    # LLM-guided expert selection
    dpo_loss = torch.tensor(0.0, device=device)
    consistency_loss = torch.tensor(0.0, device=device)
    dpo_preferences = []
    consistency_rankings = {}

    if uncertainty_node_indices and (epoch % config.llm_interval == 0):
        print(f"Epoch {epoch}: Calling LLM for {len(uncertainty_node_indices)} uncertain nodes...")

        for node_id in uncertainty_node_indices:
            cache_key = f"{encoder_name}_{node_id}"

            if cache_key not in expert_selections_cache:
                try:
                    prompt = build_enhanced_prompt_for_node(
                        node_id, source_data, experts_outputs, cls_model, config.expert_num,
                        config.hop, config.node_limit
                    )

                    response = llm_client.generate(
                        prompt=prompt,
                        model=config.llm,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        format='json'
                    )

                    json_output_str = response['response']
                    parsed_response = response_parser.parse_llm_response_robust(json_output_str, config.expert_num)

                    expert_selections_cache[cache_key] = parsed_response

                    # Debug: Print the parsed response to see what we got
                    print(f"[DEBUG] Node {node_id} parsed response: {parsed_response}")

                    # Generate ranking if missing but have expert selection
                    if 'ranking' not in parsed_response and 'expert' in parsed_response:
                        preferred_expert = parsed_response['expert']
                        # Create ranking: preferred expert first, then others
                        ranking = [preferred_expert] + [i for i in range(config.expert_num) if i != preferred_expert]
                        parsed_response['ranking'] = ranking
                        print(f"[DEBUG] Node {node_id} generated ranking from expert: {ranking}")

                    # Extract DPO preferences
                    if 'ranking' in parsed_response:
                        dpo_prefs = response_parser.parse_dpo_preferences(json_output_str, config.expert_num)
                        dpo_preferences.extend(dpo_prefs)
                        print(f"[DEBUG] Node {node_id} extracted {len(dpo_prefs)} DPO preferences")
                    else:
                        print(f"[DEBUG] Node {node_id} no 'ranking' field in parsed response: {list(parsed_response.keys())}")

                    # Extract consistency rankings
                    if 'ranking' in parsed_response:
                        consistency_rankings[node_id] = parsed_response['ranking']
                        print(f"[DEBUG] Node {node_id} consistency ranking: {parsed_response['ranking']}")
                    else:
                        print(f"[DEBUG] Node {node_id} no 'ranking' for consistency")

                except Exception as e:
                    print(f"Node {node_id}: Error processing LLM response: {e}")
                    expert_selections_cache[cache_key] = {"expert": 0, "confidence": 0.5, "strategy": "error_fallback"}
                    consistency_rankings[node_id] = list(range(config.expert_num))

        # Calculate DPO loss
        print(f"[DEBUG] DPO preferences collected: {len(dpo_preferences)}")
        if dpo_preferences:
            uncertainty_tensor = torch.tensor(uncertainty_node_indices, dtype=torch.long)
            dpo_loss = loss_functions['dpo'](clean_logits, dpo_preferences, uncertainty_tensor)
            print(f"[DEBUG] DPO loss calculated: {dpo_loss.item()}")
        else:
            print(f"[DEBUG] No DPO preferences, DPO loss remains 0")

        # Calculate consistency loss
        print(f"[DEBUG] Consistency rankings collected: {len(consistency_rankings)}")
        if consistency_rankings:
            uncertainty_tensor = torch.tensor(list(consistency_rankings.keys()), dtype=torch.long)
            consistency_loss = loss_functions['consistency'](experts_outputs, consistency_rankings, uncertainty_tensor)
            print(f"[DEBUG] Consistency loss calculated: {consistency_loss.item()}")
        else:
            print(f"[DEBUG] No consistency rankings, consistency loss remains 0")

    # Classification loss
    cls_loss = F.cross_entropy(source_logits[label_mask], source_data.y[label_mask])

    # Diversity loss (for GAT+Soft-prompt MoE)
    diversity_loss = torch.tensor(0.0, device=device)
    if encoder_name == 'gat_soft_prompt' and hasattr(encoder, 'diversity_loss'):
        diversity_loss = loss_functions['mmd_diversity'](experts_outputs, clean_logits)

    # Gate entropy regularization
    gate_entropy_loss = loss_functions['gate_entropy'](clean_logits)

    # Get loss weights from scheduler
    weights = scheduler.get_loss_weights(epoch)

    # Total loss
    total_loss = (
        weights['cls_weight'] * cls_loss +
        weights['select_weight'] * dpo_loss +
        weights['consistency_weight'] * consistency_loss +
        weights['diversity_weight'] * diversity_loss +
        weights['gate_weight'] * gate_entropy_loss +
        aux_loss
    )

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(cls_model.parameters()), 1.0)
    optimizer.step()

    # Log losses
    loss_dict = {
        'cls_loss': cls_loss.item(),
        'dpo_loss': dpo_loss.item(),
        'consistency_loss': consistency_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'gate_entropy_loss': gate_entropy_loss.item(),
        'aux_loss': aux_loss.item(),
        'total_loss': total_loss.item(),
        'uncertain_nodes': len(uncertainty_node_indices)
    }

    # Add weight information
    loss_dict.update({f'{k}_weight': v for k, v in weights.items()})

    wandb.log({f'{encoder_name}/train/{k}': v for k, v in loss_dict.items()}, step=epoch)

    print(f"Epoch {epoch} [{encoder_name}] "
          f"Cls: {cls_loss.item():.4f}, DPO: {dpo_loss.item():.4f}, "
          f"Cons: {consistency_loss.item():.4f}, Div: {diversity_loss.item():.4f}, "
          f"Total: {total_loss.item():.4f}")

    return total_loss.item(), loss_dict

# Training loop
print("Starting training...")

# Initialize optimizers for each encoder
optimizers = {}
for encoder_name, encoder in encoders.items():
    optimizers[encoder_name] = torch.optim.Adam(
        list(encoder.parameters()) + list(cls_model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

# Training statistics
best_metrics = {encoder_name: {'target_acc': 0.0, 'epoch': 0} for encoder_name in encoders.keys()}
training_history = {encoder_name: [] for encoder_name in encoders.keys()}

for epoch in range(1, config.epochs + 1):
    print(f"\n--- Epoch {epoch}/{config.epochs} ---")

    epoch_results = {}

    # Train each encoder
    for encoder_name, encoder in encoders.items():
        loss, loss_dict = train_epoch(epoch, encoder_name, encoder, optimizers[encoder_name])
        epoch_results[encoder_name] = loss_dict
        training_history[encoder_name].append(loss_dict)

    # Evaluation
    if epoch % 10 == 0 or epoch == config.epochs:
        print("\n--- Evaluation ---")

        for encoder_name, encoder in encoders.items():
            encoder.eval()
            cls_model.eval()

            with torch.no_grad():
                source_correct, _, _, _, _, _ = test(source_data, f"{config.source}_{encoder_name}", encoder, cls_model, source_data.test_mask)
                target_correct, macro_f1, micro_f1, embeddings, expert_outputs, clean_logits = test(target_data, f"{config.target}_{encoder_name}", encoder, cls_model)

                print(f"[{encoder_name}] Source Acc: {source_correct:.4f}, Target Acc: {target_correct:.4f}, "
                      f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

                wandb.log({
                    f'{encoder_name}/eval/source_acc': source_correct.item(),
                    f'{encoder_name}/eval/target_acc': target_correct.item(),
                    f'{encoder_name}/eval/macro_f1': macro_f1.item(),
                    f'{encoder_name}/eval/micro_f1': micro_f1.item()
                }, step=epoch)

                # Update best metrics
                if target_correct > best_metrics[encoder_name]['target_acc']:
                    best_metrics[encoder_name] = {
                        'target_acc': target_correct.item(),
                        'source_acc': source_correct.item(),
                        'epoch': epoch,
                        'macro_f1': macro_f1.item(),
                        'micro_f1': micro_f1.item()
                    }
                    print(f"*** New best target accuracy for {encoder_name}: {target_correct:.4f} at epoch {epoch} ***")

                # Expert analysis
                if config.analyze_experts and analyzer and epoch % 50 == 0:
                    expert_indices, expert_probs, _ = encoder.get_node_expert_assignment(
                        target_data.x, target_data.edge_index
                    )

                    # Convert to torch tensors if needed
                    if not isinstance(expert_indices, torch.Tensor):
                        expert_indices = torch.tensor(expert_indices)
                    if not isinstance(expert_probs, torch.Tensor):
                        expert_probs = torch.tensor(expert_probs)

                    usage_results = analyzer.analyze_expert_usage(
                        expert_indices, expert_probs, target_data.y, config.expert_num
                    )
                    diversity_results = analyzer.analyze_expert_diversity(expert_outputs, clean_logits)

                    wandb.log({
                        f'{encoder_name}/expert/load_balance': usage_results['load_balance_score'],
                        f'{encoder_name}/expert/diversity_score': diversity_results['diversity_score'],
                        f'{encoder_name}/expert/usage_entropy': usage_results['usage_entropy']
                    }, step=epoch)

                # Embedding visualization
                if config.visualize_embeddings and visualizer and epoch % 50 == 0:
                    visualizer.visualize_embeddings_tsne(
                        embeddings, target_data.y,
                        title=f"{encoder_name} t-SNE Epoch {epoch}"
                    )

# Final results
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

for encoder_name, metrics in best_metrics.items():
    print(f"\n{encoder_name.upper()} ARCHITECTURE:")
    print(f"  Best Epoch: {metrics['epoch']}")
    print(f"  Best Target Accuracy: {metrics['target_acc']:.5f}")
    print(f"  Corresponding Source Accuracy: {metrics['source_acc']:.5f}")
    print(f"  Macro F1: {metrics['macro_f1']:.5f}")
    print(f"  Micro F1: {metrics['micro_f1']:.5f}")

# Save final models and analysis
for encoder_name, encoder in encoders.items():
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    torch.save(encoder.state_dict(), f"{model_save_dir}/{config.source}-{config.target}-{encoder_name}-encoder.pt")
    torch.save(cls_model.state_dict(), f"{model_save_dir}/{config.source}-{config.target}-{encoder_name}-classifier.pt")

# Generate comprehensive analysis report
if analyzer:
    for encoder_name, encoder in encoders.items():
        encoder.eval()
        with torch.no_grad():
            _, expert_outputs, clean_logits = encode(target_data, f"{config.target}_{encoder_name}", encoder)
            expert_indices, expert_probs, _ = encoder.get_node_expert_assignment(
                target_data.x, target_data.edge_index
            )

            # Convert to torch tensors
            if not isinstance(expert_indices, torch.Tensor):
                expert_indices = torch.tensor(expert_indices)
            if not isinstance(expert_probs, torch.Tensor):
                expert_probs = torch.tensor(expert_probs)

            analyzer.analyze_expert_usage(expert_indices, expert_probs, target_data.y, config.expert_num)
            analyzer.analyze_expert_diversity(expert_outputs, clean_logits)
            report = analyzer.generate_comprehensive_report()

# Save training history
os.makedirs("log", exist_ok=True)
with open("log/training_history.json", 'w') as f:
    json.dump(training_history, f, indent=2, default=str)

wandb.finish()
print("\nTraining completed successfully!")