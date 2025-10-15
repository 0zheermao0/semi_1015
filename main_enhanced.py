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

# DyCon-inspired loss parameters (following DyCon's hyperparameter settings)
parser.add_argument("--alpha", type=float, default=2.0, help="DPO temperature parameter (DyCon-style)")
parser.add_argument("--beta", type=float, default=0.8, help="Uncertainty weighting for consistency loss (DyCon's beta)")
parser.add_argument("--beta_min", type=float, default=0.5, help="Minimum adaptive beta (DyCon)")
parser.add_argument("--beta_max", type=float, default=5.0, help="Maximum adaptive beta (DyCon)")
parser.add_argument("--consistency_weight", type=float, default=0.1, help="Consistency loss weight (DyCon)")
parser.add_argument("--div_weight", type=float, default=0.5, help="Diversity loss weight")
parser.add_argument("--select_weight", type=float, default=1.0, help="Selection loss weight")
parser.add_argument("--gate_coef", type=float, default=0.1, help="Gate loss coefficient")
parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay for teacher model (DyCon)")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="Consistency ramp-up epochs (DyCon)")

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

# DyCon-inspired adaptive beta function
def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
    """Adaptive beta computation following DyCon's approach."""
    ratio = min_beta / max_beta
    exponent = epoch / total_epochs
    beta = max_beta * (ratio ** exponent)
    return beta

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

# DyCon-inspired Teacher-EMA setup
def update_ema_variables(student_model, ema_model, alpha, global_step):
    """Update EMA teacher model following DyCon's approach."""
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, student_param in zip(ema_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, student_param.data)

# Create teacher models (EMA copies of student models)
teacher_encoders = {}
for encoder_name, encoder in encoders.items():
    teacher_encoder = type(encoder)(**{k: v for k, v in encoder.__dict__.items() if not k.startswith('_')})
    teacher_encoder.load_state_dict(encoder.state_dict())
    # Freeze teacher parameters initially
    for param in teacher_encoder.parameters():
        param.detach_()
    teacher_encoders[encoder_name] = teacher_encoder.to(device)

# Training setup
expert_selections_cache = {}
global_step = 0

def train_epoch(epoch, encoder_name, student_encoder, teacher_encoder, optimizer):
    """DyCon-inspired training epoch with student-teacher architecture."""
    global global_step
    student_encoder.train()
    teacher_encoder.eval()
    cls_model.train()

    # Forward pass for student
    cache_name = f"{config.source}_{encoder_name}"
    student_encoded, student_experts_outputs, student_aux_loss, student_clean_logits = encode(source_data, cache_name, student_encoder)
    student_logits = cls_model(student_encoded)

    # Forward pass for teacher with noise (DyCon approach)
    with torch.no_grad():
        # Add noise to teacher input for consistency regularization
        noise = torch.clamp(torch.randn_like(source_data.x) * 0.1, -0.2, 0.2)
        noisy_source_data = source_data.clone()
        noisy_source_data.x = source_data.x + noise

        teacher_encoded, teacher_experts_outputs, teacher_aux_loss, teacher_clean_logits = encode(noisy_source_data, cache_name, teacher_encoder)
        teacher_logits = cls_model(teacher_encoded)

    # Calculate uncertainty using student expert outputs
    uncertainty_mask = calculate_expert_uncertainty(student_experts_outputs, source_data.num_classes, cls_model, config.uncertainty_k)
    uncertainty_node_indices = torch.where(uncertainty_mask)[0].tolist()

    # Debug output
    print(f"[DEBUG] Epoch {epoch}: Found {len(uncertainty_node_indices)} uncertain nodes")
    print(f"[DEBUG] LLM interval check: epoch % {config.llm_interval} == {epoch % config.llm_interval}")
    print(f"[DEBUG] Should call LLM: {len(uncertainty_node_indices) > 0 and (epoch % config.llm_interval == 0)}")

    # DyCon-inspired consistency loss (student-teacher uncertainty-weighted)
    # Use adaptive beta following DyCon's approach
    current_beta = adaptive_beta(epoch, config.epochs, config.beta_max, config.beta_min)
    loss_functions['consistency'].beta = current_beta  # Update beta for current epoch
    consistency_loss = loss_functions['consistency'](student_logits, teacher_logits)

    # LLM-guided expert selection (DPO ranking for challenging samples)
    dpo_loss = torch.tensor(0.0, device=device)
    dpo_preferences = []

    if uncertainty_node_indices and (epoch % config.llm_interval == 0):
        print(f"Epoch {epoch}: Calling LLM for {len(uncertainty_node_indices)} uncertain nodes...")

        for node_id in uncertainty_node_indices:
            cache_key = f"{encoder_name}_{node_id}"

            if cache_key not in expert_selections_cache:
                try:
                    prompt = build_enhanced_prompt_for_node(
                        node_id, source_data, student_experts_outputs, cls_model, config.expert_num,
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

                    # Extract DPO preferences (challenging sample ranking)
                    if 'ranking' in parsed_response:
                        # Create DPO preferences from the parsed ranking directly
                        ranking = parsed_response['ranking']
                        dpo_prefs = []
                        for i in range(len(ranking) - 1):
                            for j in range(i + 1, len(ranking)):
                                dpo_prefs.append((ranking[i], ranking[j]))
                        dpo_preferences.extend(dpo_prefs)
                        print(f"[DEBUG] Node {node_id} extracted {len(dpo_prefs)} DPO preferences from ranking: {ranking}")
                    else:
                        print(f"[DEBUG] Node {node_id} no 'ranking' field in parsed response: {list(parsed_response.keys())}")

                except Exception as e:
                    print(f"Node {node_id}: Error processing LLM response: {e}")
                    expert_selections_cache[cache_key] = {"expert": 0, "confidence": 0.5, "strategy": "error_fallback"}

        # Calculate DPO loss for challenging samples
        print(f"[DEBUG] DPO preferences collected: {len(dpo_preferences)}")
        if dpo_preferences:
            uncertainty_tensor = torch.tensor(uncertainty_node_indices, dtype=torch.long)
            dpo_loss = loss_functions['dpo'](student_clean_logits, dpo_preferences, uncertainty_tensor)
            print(f"[DEBUG] DPO loss calculated: {dpo_loss.item()}")
        else:
            print(f"[DEBUG] No DPO preferences, DPO loss remains 0")

    # Classification loss
    cls_loss = F.cross_entropy(student_logits[label_mask], source_data.y[label_mask])

    # Diversity loss (for both Original and GAT+Soft-prompt MoE)
    diversity_loss = torch.tensor(0.0, device=device)
    # Calculate diversity loss for all MoE architectures when we have expert outputs
    if student_experts_outputs is not None and student_clean_logits is not None:
        diversity_loss = loss_functions['mmd_diversity'](student_experts_outputs, student_clean_logits)
        print(f"[DEBUG] Diversity loss calculated: {diversity_loss.item()}")

    # Gate entropy regularization
    gate_entropy_loss = loss_functions['gate_entropy'](student_clean_logits)

    # Get loss weights from scheduler (DyCon-style consistency ramp-up)
    weights = scheduler.get_loss_weights(epoch)
    consistency_weight = weights['consistency_weight'] * min(1.0, (epoch / 100.0))  # Gradual ramp-up

    # Total loss (following DyCon's formulation)
    total_loss = (
        weights['cls_weight'] * cls_loss +
        weights['select_weight'] * dpo_loss +
        consistency_weight * consistency_loss +  # DyCon-style weighted consistency
        weights['diversity_weight'] * diversity_loss +
        weights['gate_weight'] * gate_entropy_loss +
        student_aux_loss + teacher_aux_loss
    )

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(student_encoder.parameters()) + list(cls_model.parameters()), 1.0)
    optimizer.step()

    # Update EMA teacher model (DyCon approach)
    update_ema_variables(student_encoder, teacher_encoder, alpha=config.ema_decay, global_step=global_step)
    global_step += 1

    # Log losses and performance metrics
    loss_dict = {
        'cls_loss': cls_loss.item(),
        'dpo_loss': dpo_loss.item(),
        'consistency_loss': consistency_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'gate_entropy_loss': gate_entropy_loss.item(),
        'aux_loss': aux_loss.item(),
        'total_loss': total_loss.item(),
        'uncertain_nodes': len(uncertainty_node_indices),
        'epoch': epoch
    }

    # Add weight information
    loss_dict.update({f'{k}_weight': v for k, v in weights.items()})

    wandb.log({f'{encoder_name}/train/{k}': v for k, v in loss_dict.items()}, step=epoch)

    # Track loss evolution for performance analysis
    performance_metrics['loss_evolution'][encoder_name]['cls'].append(cls_loss.item())
    performance_metrics['loss_evolution'][encoder_name]['dpo'].append(dpo_loss.item())
    performance_metrics['loss_evolution'][encoder_name]['consistency'].append(consistency_loss.item())
    performance_metrics['loss_evolution'][encoder_name]['diversity'].append(diversity_loss.item())
    performance_metrics['loss_evolution'][encoder_name]['total'].append(total_loss.item())

    # Enhanced training output with performance indicators
    print(f"Epoch {epoch:3d} [{encoder_name:12s}] "
          f"Cls: {cls_loss.item():6.4f} | DPO: {dpo_loss.item():6.4f} | "
          f"Cons: {consistency_loss.item():6.4f} | Div: {diversity_loss.item():6.4f} | "
          f"Gate: {gate_entropy_loss.item():6.4f} | Total: {total_loss.item():6.4f} | "
          f"Uncertain: {len(uncertainty_node_indices):3d}")

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

# Training statistics and performance tracking
best_metrics = {encoder_name: {'target_acc': 0.0, 'epoch': 0, 'source_acc': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0} for encoder_name in encoders.keys()}
training_history = {encoder_name: [] for encoder_name in encoders.keys()}
performance_metrics = {
    'loss_evolution': {encoder_name: {'cls': [], 'dpo': [], 'consistency': [], 'diversity': [], 'total': []} for encoder_name in encoders.keys()},
    'accuracy_evolution': {encoder_name: {'source': [], 'target': []} for encoder_name in encoders.keys()},
    'convergence_metrics': {encoder_name: {'improvement_rate': [], 'stability_score': 0.0} for encoder_name in encoders.keys()}
}

# Early stopping and learning rate scheduling
early_stopping_patience = 50
early_stopping_counter = {encoder_name: 0 for encoder_name in encoders.keys()}
best_validation_score = {encoder_name: 0.0 for encoder_name in encoders.keys()}
lr_patience = 10
lr_counter = {encoder_name: 0 for encoder_name in encoders.keys()}

for epoch in range(1, config.epochs + 1):
    print(f"\n--- Epoch {epoch}/{config.epochs} ---")

    epoch_results = {}

    # Train each encoder
    for encoder_name, student_encoder in encoders.items():
        teacher_encoder = teacher_encoders[encoder_name]
        loss, loss_dict = train_epoch(epoch, encoder_name, student_encoder, teacher_encoder, optimizers[encoder_name])
        epoch_results[encoder_name] = loss_dict
        training_history[encoder_name].append(loss_dict)

    # Evaluation with comprehensive performance analysis
    if epoch % 10 == 0 or epoch == config.epochs:
        print("\n" + "="*80)
        print(f"EVALUATION AT EPOCH {epoch}")
        print("="*80)

        for encoder_name, encoder in encoders.items():
            encoder.eval()
            cls_model.eval()

            with torch.no_grad():
                # Source domain evaluation
                source_correct, source_macro_f1, source_micro_f1, _, _, _ = test(
                    source_data, f"{config.source}_{encoder_name}", encoder, cls_model, source_data.test_mask
                )

                # Target domain evaluation
                target_correct, target_macro_f1, target_micro_f1, embeddings, expert_outputs, clean_logits = test(
                    target_data, f"{config.target}_{encoder_name}", encoder, cls_model
                )

                # Track accuracy evolution
                performance_metrics['accuracy_evolution'][encoder_name]['source'].append(source_correct.item())
                performance_metrics['accuracy_evolution'][encoder_name]['target'].append(target_correct.item())

                # Calculate convergence metrics
                if len(performance_metrics['accuracy_evolution'][encoder_name]['target']) > 1:
                    recent_improvement = (performance_metrics['accuracy_evolution'][encoder_name]['target'][-1] -
                                          performance_metrics['accuracy_evolution'][encoder_name]['target'][-2])
                    performance_metrics['convergence_metrics'][encoder_name]['improvement_rate'].append(recent_improvement)

                # Comprehensive evaluation output
                print(f"\n🔍 {encoder_name.upper()} ARCHITECTURE PERFORMANCE:")
                print(f"   Source Domain  - Acc: {source_correct.item():6.4f} | Macro F1: {source_macro_f1.item():6.4f} | Micro F1: {source_micro_f1.item():6.4f}")
                print(f"   Target Domain  - Acc: {target_correct.item():6.4f} | Macro F1: {target_macro_f1.item():6.4f} | Micro F1: {target_micro_f1.item():6.4f}")

                # Domain adaptation metrics
                domain_gap = abs(source_correct.item() - target_correct.item())
                adaptation_ratio = target_correct.item() / (source_correct.item() + 1e-8)
                print(f"   Domain Gap: {domain_gap:6.4f} | Adaptation Ratio: {adaptation_ratio:6.4f}")

                # Update best metrics with comprehensive tracking
                current_score = target_correct.item() + 0.1 * target_macro_f1.item() + 0.1 * target_micro_f1.item()
                if current_score > best_validation_score[encoder_name]:
                    best_validation_score[encoder_name] = current_score
                    best_metrics[encoder_name] = {
                        'target_acc': target_correct.item(),
                        'source_acc': source_correct.item(),
                        'target_macro_f1': target_macro_f1.item(),
                        'target_micro_f1': target_micro_f1.item(),
                        'domain_gap': domain_gap,
                        'adaptation_ratio': adaptation_ratio,
                        'epoch': epoch
                    }
                    print(f"   🏆 NEW BEST PERFORMANCE! Combined Score: {current_score:.4f}")

                    # Early stopping counter reset
                    early_stopping_counter[encoder_name] = 0
                    lr_counter[encoder_name] = 0
                else:
                    early_stopping_counter[encoder_name] += 1
                    lr_counter[encoder_name] += 1

                # Log to W&B with enhanced metrics
                wandb.log({
                    f'{encoder_name}/eval/source_acc': source_correct.item(),
                    f'{encoder_name}/eval/target_acc': target_correct.item(),
                    f'{encoder_name}/eval/source_macro_f1': source_macro_f1.item(),
                    f'{encoder_name}/eval/target_macro_f1': target_macro_f1.item(),
                    f'{encoder_name}/eval/source_micro_f1': source_micro_f1.item(),
                    f'{encoder_name}/eval/target_micro_f1': target_micro_f1.item(),
                    f'{encoder_name}/eval/domain_gap': domain_gap,
                    f'{encoder_name}/eval/adaptation_ratio': adaptation_ratio,
                    f'{encoder_name}/eval/best_score': current_score
                }, step=epoch)

                # Expert analysis for model interpretability
                if config.analyze_experts and analyzer and epoch % 50 == 0:
                    try:
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

                        print(f"   🧠 Expert Analysis - Load Balance: {usage_results['load_balance_score']:.4f} | "
                              f"Diversity: {diversity_results['diversity_score']:.4f} | "
                              f"Usage Entropy: {usage_results['usage_entropy']:.4f}")

                        wandb.log({
                            f'{encoder_name}/expert/load_balance': usage_results['load_balance_score'],
                            f'{encoder_name}/expert/diversity_score': diversity_results['diversity_score'],
                            f'{encoder_name}/expert/usage_entropy': usage_results['usage_entropy']
                        }, step=epoch)
                    except Exception as e:
                        print(f"   ⚠️  Expert analysis failed: {e}")

                # Embedding visualization for convergence monitoring
                if config.visualize_embeddings and visualizer and epoch % 50 == 0:
                    try:
                        visualizer.visualize_embeddings_tsne(
                            embeddings, target_data.y,
                            title=f"{encoder_name} t-SNE Epoch {epoch}"
                        )
                        print(f"   📊 Embedding visualization saved for epoch {epoch}")
                    except Exception as e:
                        print(f"   ⚠️  Embedding visualization failed: {e}")

                # Learning rate scheduling for convergence optimization
                if lr_counter[encoder_name] >= lr_patience:
                    for param_group in optimizers[encoder_name].param_groups:
                        param_group['lr'] *= 0.5
                    print(f"   📉 Learning rate reduced to {optimizers[encoder_name].param_groups[0]['lr']:.6f} for {encoder_name}")
                    lr_counter[encoder_name] = 0

                # Early stopping check
                if early_stopping_counter[encoder_name] >= early_stopping_patience:
                    print(f"\n⏹️  Early stopping triggered for {encoder_name} at epoch {epoch}")
                    print(f"   No improvement for {early_stopping_patience} consecutive evaluations")
                    break

# Final comprehensive results and analysis
print("\n" + "=" * 100)
print("🏆 FINAL TRAINING RESULTS AND PERFORMANCE ANALYSIS")
print("=" * 100)

print(f"\n📊 TRAINING CONFIGURATION:")
print(f"   Source Domain: {config.source} → Target Domain: {config.target}")
print(f"   MoE Architecture: {config.moe_architecture}")
print(f"   Expert Count: {config.expert_num} | LLM Interval: {config.llm_interval}")
print(f"   Total Epochs: {config.epochs} | Early Stopping Patience: {early_stopping_patience}")

# Results summary table
print(f"\n📈 PERFORMANCE SUMMARY:")
print("-" * 100)
print(f"{'Architecture':<15} {'Best Epoch':<10} {'Source Acc':<12} {'Target Acc':<12} {'Macro F1':<10} {'Micro F1':<10} {'Domain Gap':<12} {'Adaptation':<12}")
print("-" * 100)

for encoder_name, metrics in best_metrics.items():
    print(f"{encoder_name.upper():<15} {metrics['epoch']:<10} {metrics['source_acc']:<12.4f} {metrics['target_acc']:<12.4f} "
          f"{metrics.get('target_macro_f1', 0.0):<10.4f} {metrics.get('target_micro_f1', 0.0):<10.4f} "
          f"{metrics.get('domain_gap', 0.0):<12.4f} {metrics.get('adaptation_ratio', 0.0):<12.4f}")

print("-" * 100)

# Convergence analysis
print(f"\n📈 CONVERGENCE ANALYSIS:")
for encoder_name in encoders.keys():
    if encoder_name in performance_metrics['convergence_metrics']:
        improvement_rates = performance_metrics['convergence_metrics'][encoder_name]['improvement_rate']
        if improvement_rates:
            avg_improvement = np.mean(improvement_rates)
            stability_score = 1.0 - np.std(improvement_rates) if len(improvement_rates) > 1 else 1.0
            print(f"   {encoder_name.upper()}: Avg Improvement: {avg_improvement:+.4f} | Stability: {stability_score:.4f}")

# Loss evolution summary
print(f"\n📉 LOSS EVOLUTION SUMMARY:")
for encoder_name in encoders.keys():
    if encoder_name in performance_metrics['loss_evolution']:
        evolution = performance_metrics['loss_evolution'][encoder_name]
        print(f"\n   {encoder_name.upper()} Loss Trends:")
        print(f"      Classification: {evolution['cls'][0]:.4f} → {evolution['cls'][-1]:.4f} "
              f"(Δ: {evolution['cls'][-1] - evolution['cls'][0]:+.4f})")
        print(f"      DPO:          {evolution['dpo'][0]:.4f} → {evolution['dpo'][-1]:.4f} "
              f"(Δ: {evolution['dpo'][-1] - evolution['dpo'][0]:+.4f})")
        print(f"      Consistency:  {evolution['consistency'][0]:.4f} → {evolution['consistency'][-1]:.4f} "
              f"(Δ: {evolution['consistency'][-1] - evolution['consistency'][0]:+.4f})")
        print(f"      Diversity:    {evolution['diversity'][0]:.4f} → {evolution['diversity'][-1]:.4f} "
              f"(Δ: {evolution['diversity'][-1] - evolution['diversity'][0]:+.4f})")
        print(f"      Total:       {evolution['total'][0]:.4f} → {evolution['total'][-1]:.4f} "
              f"(Δ: {evolution['total'][-1] - evolution['total'][0]:+.4f})")

# Performance comparison with baseline
print(f"\n🎯 PERFORMANCE COMPARISON:")
for encoder_name, metrics in best_metrics.items():
    domain_gap = metrics.get('domain_gap', 0.0)
    adaptation_ratio = metrics.get('adaptation_ratio', 0.0)

    if domain_gap < 0.1:
        gap_status = "✅ Excellent"
    elif domain_gap < 0.2:
        gap_status = "🟡 Good"
    else:
        gap_status = "⚠️ Needs Improvement"

    if adaptation_ratio > 0.9:
        adapt_status = "✅ Excellent"
    elif adaptation_ratio > 0.8:
        adapt_status = "🟡 Good"
    else:
        adapt_status = "⚠️ Needs Improvement"

    print(f"   {encoder_name.upper()}: Domain Gap {gap_status} | Adaptation {adapt_status}")

print(f"\n🔧 TRAINING EFFICIENCY:")
for encoder_name, epochs_trained in [(name, len(history)) for name, history in training_history.items()]:
    efficiency = epochs_trained / config.epochs * 100
    print(f"   {encoder_name.upper()}: {epochs_trained}/{config.epochs} epochs ({efficiency:.1f}% of planned)")

# Recommendations based on results
print(f"\n💡 TRAINING RECOMMENDATIONS:")
best_encoder = max(best_metrics.items(), key=lambda x: x[1].get('target_acc', 0))
print(f"   Best Performing Architecture: {best_encoder[0].upper()}")
print(f"   Target Achievement: {best_encoder[1]['target_acc']:.4f}")

if best_encoder[1].get('domain_gap', 0.0) > 0.15:
    print("   ⚠️  Consider: Increased domain alignment techniques")
if best_encoder[1].get('adaptation_ratio', 0.0) < 0.8:
    print("   ⚠️  Consider: Enhanced transfer learning strategies")

print(f"\n🚀 MODEL SAVED FOR DEPLOYMENT:")
model_save_dir = "models"
for encoder_name in encoders.keys():
    print(f"   {encoder_name.upper()}: {model_save_dir}/{config.source}-{config.target}-{encoder_name}-encoder.pt")
print(f"   CLASSIFIER: {model_save_dir}/{config.source}-{config.target}-classifier.pt")

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

# Save comprehensive training data
os.makedirs("log", exist_ok=True)

# Save basic training history
with open("log/training_history.json", 'w') as f:
    json.dump(training_history, f, indent=2, default=str)

# Save detailed performance metrics
performance_data = {
    'best_metrics': best_metrics,
    'performance_metrics': performance_metrics,
    'training_config': {
        'source': config.source,
        'target': config.target,
        'moe_architecture': config.moe_architecture,
        'expert_num': config.expert_num,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'llm_interval': config.llm_interval,
        'uncertainty_k': config.uncertainty_k,
        'early_stopping_patience': early_stopping_patience,
        'seed': config.seed
    }
}

with open("log/performance_analysis.json", 'w') as f:
    json.dump(performance_data, f, indent=2, default=str)

print(f"\n💾 Training data saved to log/ directory")
print(f"   📄 training_history.json: Basic loss and accuracy history")
print(f"   📄 performance_analysis.json: Comprehensive performance analysis and metrics")

wandb.finish()
print("\nTraining completed successfully!")