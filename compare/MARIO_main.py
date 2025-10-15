# coding=utf-8
import os
from argparse import ArgumentParser
# Remove unused GNN imports if they are not needed elsewhere
# from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import pickle
import math
from sklearn.metrics import f1_score
from torch_geometric.utils import dropout_adj # For MARIO augmentation

warnings.filterwarnings("ignore", category=UserWarning)

# --- MARIO Helper Functions ---
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

# MARIO Contrastive Loss
def mario_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    loss = -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    return loss.mean()

# MARIO Clustering Loss Helpers
def sinkhorn(scores, epsilon=0.05, n_iters=3): # Epsilon adjusted from MARIO code example
    Q = torch.exp(scores / epsilon).t() # Q is K-by-B
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q.t()

def weak_cmi(z1, z2, q1, q2): # Clustering loss component
    N = z1.shape[0]
    f = lambda x: torch.exp(x / 1.0) # Use tau=1 for CMI part as implicit in MARIO code?
    between_sim = f(sim(z1, z2))

    p1, y1 = torch.max(q1, dim=1)
    p2, y2 = torch.max(q2, dim=1)

    mask = (y1 == y2).float() # Use float mask for multiplication

    conditional_mask = (y1.repeat(N,1) == y2.reshape(-1,1).repeat(1, N)).float()
    diag_mask = torch.eye(N, device=z1.device).bool()
    conditional_mask = conditional_mask.masked_fill(diag_mask, 1.0) # Ensure diagonal is considered

    neg_sim = torch.sum(torch.mul(between_sim, conditional_mask), dim=1)

    # Add small epsilon to prevent log(0) or division by zero
    epsilon = 1e-8
    ccl = -torch.log(
        between_sim.diag() / (neg_sim + epsilon) + epsilon)

    return (ccl * mask).mean() # Return mean loss over masked elements

# MLP Projector from MARIO
class Two_MLP_BN(torch.nn.Module):
    def __init__(self, hidden, mlp_hid, mlp_out):
        super(Two_MLP_BN, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, mlp_hid),
            nn.BatchNorm1d(mlp_hid),
            nn.ReLU(),
            nn.Linear(mlp_hid, mlp_out)
        )
    def forward(self, feat):
        return self.proj(feat)

# -----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acmv9')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=5e-3) #5e-3
parser.add_argument("--weight_decay", type=float, default=1e-3) #2e-3
parser.add_argument("--drop_out", type=float, default=5e-1)

parser.add_argument("--perturb", type=bool, default=True) # Keep user's perturbation?
parser.add_argument("--perturb_value", type=float, default=0.5)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--label_rate", type=float, default=0.05)

# --- MARIO Hyperparameters ---
parser.add_argument("--mario_tau", type=float, default=0.5, help="Temperature for MARIO contrastive loss")
parser.add_argument("--num_clusters", type=int, default=100, help="Number of prototypes for MARIO clustering")
parser.add_argument("--prototypes_lr", type=float, default=1e-5, help="Learning rate for MARIO prototypes/projector")
parser.add_argument("--prototypes_iters", type=int, default=10, help="Number of optimization steps for prototypes per epoch")
parser.add_argument("--cmi_coefficient", type=float, default=0.1, help="Weight for MARIO CMI/clustering loss")
parser.add_argument("--contrastive_coefficient", type=float, default=1.0, help="Weight for MARIO contrastive loss")
parser.add_argument("--mask_feat_rate", type=float, default=0.3, help="MARIO feature drop rate")
parser.add_argument("--mask_edge_rate", type=float, default=0.4, help="MARIO edge drop rate")
parser.add_argument("--mlp_hidden_dim", type=int, default=512, help="Hidden dim for MARIO projector MLP")
# --- End MARIO Hyperparameters ---

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim
use_perturb = args.perturb
perturb_value = args.perturb_value
label_rate = args.label_rate

# Extract MARIO args
mario_tau = args.mario_tau
num_clusters = args.num_clusters
prototypes_lr = args.prototypes_lr
prototypes_iters = args.prototypes_iters
cmi_coefficient = args.cmi_coefficient
contrastive_coefficient = args.contrastive_coefficient
mask_feat_rate = args.mask_feat_rate
mask_edge_rate = args.mask_edge_rate
mlp_hidden_dim = args.mlp_hidden_dim


id = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, perturb:{:.3f}, dim: {}, mario_tau: {:.2f}, clusters:{}, proto_lr:{}, cmi_w:{:.2f}, contr_w:{:.2f}" \
    .format(args.source, args.target, seed, label_rate, args.learning_rate, args.weight_decay, perturb_value,
            encoder_dim, mario_tau, num_clusters, prototypes_lr, cmi_coefficient, contrastive_coefficient)
print(id)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print(source_data)

dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print(target_data)

source_data = source_data.to(device)

source_train_size = int(source_data.size(0) * label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)


def index2dense(edge_index,nnode=2708):
    indx = edge_index.cpu().detach().numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj

class add_perturb(nn.Module):
    def __init__(self, dim1, dim2, beta):
        super(add_perturb, self).__init__()
        self.perturb = nn.Parameter(torch.FloatTensor(dim1, dim2).normal_(-beta, beta).to(device))
        self.perturb.requires_grad_(True)

    def forward(self, input):
        return input + self.perturb


class GNN(torch.nn.Module):
    def __init__(self, num_features, num_nodes, base_model=None, **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(args.drop_out) for _ in weights]

        # Keep user's perturbation layer
        self.perturb_layers = nn.ModuleList([
            add_perturb(num_nodes, encoder_dim, perturb_value),
            add_perturb(num_nodes, encoder_dim, perturb_value)
        ])

        self.conv_layers = nn.ModuleList([
            PPMIConv(num_features, encoder_dim,
                      weight=weights[0],
                      bias=biases[0],
                      **kwargs),
            PPMIConv(encoder_dim, encoder_dim,
                      weight=weights[1],
                      bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name, perturb):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if perturb:
                # Only apply perturbation if requested (e.g., for original embedding)
                x = self.perturb_layers[i](x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


# --- No GRL needed for MARIO adaptation as requested ---
# class GradReverse(torch.autograd.Function): ...
# class GRL(nn.Module): ...

# Remove user's ReNode weighting and Entropy loss for simplicity, replace with MARIO losses
# def get_renode_weight(data, pseudo_label): ...
# def Entropy(input, weight, label): ...


# Adjusted encode and predict to handle MARIO components if needed later
# Note: Classification uses encoder output directly
def encode(data, cache_name, perturb=False, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name, perturb)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output

def predict(data, cache_name, perturb=False, mask=None):
    data = data.to(device)
    encoded_output = encode(data, cache_name, perturb, mask)
    logits = cls_model(encoded_output) # Classify based on encoder output
    return logits

def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    # Ensure labels and preds are on CPU before passing to sklearn
    labels_cpu = labels.cpu().detach()
    preds_cpu = preds.cpu().detach()
    macro_f1 = f1_score(labels_cpu, preds_cpu, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cpu, preds_cpu, average='micro', zero_division=0)
    return accuracy, macro_f1, micro_f1


def test(data, cache_name, perturb=False, mask=None):
    data = data.to(device)
    encoder.eval()
    cls_model.eval()
    projector.eval() # MARIO component
    prototypes.eval() # MARIO component

    encoded_output = encode(data, cache_name, perturb=False) # Test without perturbation usually
    logits = predict(data, cache_name, perturb=False, mask=mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]

    # Ensure labels and preds are on CPU for evaluate function
    preds_cpu = preds # Already on CPU if mask is used in predict? Check predict.
    labels_cpu = labels

    accuracy, macro_f1, micro_f1 = evaluate(preds_cpu, labels_cpu)
    return accuracy, macro_f1, micro_f1, encoded_output


loss_func = nn.CrossEntropyLoss().to(device)

# Initialize models
encoder = GNN(num_features=dataset.num_features, num_nodes=source_data.size(0)).to(device)

# MARIO Projector
projector = Two_MLP_BN(encoder_dim, mlp_hidden_dim, encoder_dim).to(device)

# MARIO Prototypes (Linear layer without bias)
prototypes = nn.Linear(encoder_dim, num_clusters, bias=False).to(device)
torch.nn.init.xavier_uniform_(prototypes.weight.data) # Initialize prototypes

# Classifier
cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

# Optimizer for Encoder and Classifier
models = [encoder, cls_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

# Optimizer for MARIO Projector and Prototypes
pro_optimizer = torch.optim.Adam([
    {'params': projector.parameters()},
    {'params': prototypes.parameters()}
], lr=prototypes_lr, weight_decay=1e-5) # Weight decay from MARIO example

# --- Remove Domain Discriminator ---
# domain_model = nn.Sequential(...)

# Only load source domain edge info if needed by PPMIConv's caching mechanism
# If PPMIConv doesn't need precomputed adj, this might be unnecessary
# with open ('tmp/'+args.source+'.pkl', 'rb') as f:
#     source_edge_index, norm = pickle.load(f)
# source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)

epochs = 200


# MARIO Prototype Update Function (adapted from MARIO's unsupervised.py update_prototypes logic)
def update_mario_prototypes(h1_no_grad, h2_no_grad):
    projector.train()
    prototypes.train()
    for _ in range(prototypes_iters):
        pro_optimizer.zero_grad()

        # Project embeddings (already computed outside, passed as no_grad)
        z1 = projector(h1_no_grad)
        z2 = projector(h2_no_grad)

        # Normalize projections and prototypes
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        with torch.no_grad():
            w = prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            prototypes.weight.copy_(w)

        # Compute scores and assignments
        scores1 = prototypes(z1) # B*K
        scores2 = prototypes(z2) # B*K
        with torch.no_grad():
            q1 = sinkhorn(scores1)
            q2 = sinkhorn(scores2)

        # Compute clustering loss for optimization step
        # Note: MARIO paper uses SwAV-style loss here (-0.5 * (q1*logsoftmax(score2) + q2*logsoftmax(score1)))
        loss_clus1 = -0.5 * torch.sum(q1 * F.log_softmax(scores2 / mario_tau, dim=1), dim=1)
        loss_clus2 = -0.5 * torch.sum(q2 * F.log_softmax(scores1 / mario_tau, dim=1), dim=1)
        online_clus_loss = (loss_clus1 + loss_clus2).mean()

        online_clus_loss.backward()
        pro_optimizer.step()

    # Final normalization of prototypes after updates
    with torch.no_grad():
        w = prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        prototypes.weight.copy_(w)


def train(epoch):
    encoder.train()
    cls_model.train()
    projector.train() # MARIO component
    prototypes.train() # MARIO component

    optimizer.zero_grad()

    # --- MARIO Augmentation ---
    x_aug1 = drop_feature(source_data.x, mask_feat_rate)
    edge_index_aug1, _ = dropout_adj(source_data.edge_index, p=mask_edge_rate)

    x_aug2 = drop_feature(source_data.x, mask_feat_rate)
    edge_index_aug2, _ = dropout_adj(source_data.edge_index, p=mask_edge_rate)
    # --------------------------

    # --- Forward Pass ---
    # Original embedding (with potential perturbation for cls loss)
    h_orig = encoder(source_data.x, source_data.edge_index, args.source, use_perturb)

    # Augmented view embeddings (no perturbation for contrastive/clustering)
    h1 = encoder(x_aug1, edge_index_aug1, args.source, False)
    h2 = encoder(x_aug2, edge_index_aug2, args.source, False)

    # Projections for MARIO losses
    z1 = projector(h1)
    z2 = projector(h2)
    # --------------------------

    # --- Loss Calculation ---
    # 1. Supervised Classification Loss
    cls_loss = loss_func(cls_model(h_orig[label_mask]), source_data.y[label_mask])

    # 2. MARIO Contrastive Loss
    contrast_loss = mario_contrastive_loss(z1, z2, mario_tau)

    # 3. MARIO Clustering Loss (using weak_cmi)
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    with torch.no_grad():
        # Get assignments based on current prototypes
        scores1_no_grad = prototypes(z1_norm)
        scores2_no_grad = prototypes(z2_norm)
        q1 = sinkhorn(scores1_no_grad)
        q2 = sinkhorn(scores2_no_grad)
    cluster_loss = weak_cmi(z1_norm, z2_norm, q1, q2)
    # --------------------------

    # --- Combine Losses ---
    # Use coefficients from args, maybe scale MARIO losses based on epoch
    # Example: lineary scale MARIO losses up during training
    mario_loss_weight = min(1.0, float(epoch) / max(1, epochs // 2)) # Scale up over first half
    loss = cls_loss + \
           mario_loss_weight * contrastive_coefficient * contrast_loss + \
           mario_loss_weight * cmi_coefficient * cluster_loss
    # --------------------------

    # --- Backward and Optimize Encoder/Classifier ---
    optimizer.zero_grad() # Zero gradients for main optimizer
    loss.backward()       # Calculate gradients for combined loss

    # Apply user's perturbation update (if enabled)
    if use_perturb:
        with torch.no_grad(): # Ensure perturbation update doesn't affect MARIO optimizer
            for pi in encoder.perturb_layers:
                 # Check if grad exists before detaching
                 if pi.perturb.grad is not None and torch.norm(pi.perturb.grad.detach(), p='fro') > 1e-8:
                    # Normalize gradient to prevent explosion
                    grad_norm = torch.norm(pi.perturb.grad.detach(), p='fro')
                    normalized_grad = pi.perturb.grad.detach() / grad_norm
                    # Update perturbation (consider smaller step size if needed)
                    x_perturb_data = pi.perturb.detach() - args.learning_rate * normalized_grad # Use main LR or a separate one?
                    pi.perturb.data = x_perturb_data.data
                    # Zero grad specific to perturbation parameter *after* update
                    pi.perturb.grad.zero_()
                 elif pi.perturb.grad is not None:
                     # Zero grad even if norm is small or None
                     pi.perturb.grad.zero_()


    optimizer.step()      # Update encoder and classifier
    # --------------------------

    # --- Optimize MARIO Prototypes/Projector ---
    with torch.no_grad(): # Get embeddings without tracking grads for prototype update
        h1_no_grad = encoder(x_aug1, edge_index_aug1, args.source, False)
        h2_no_grad = encoder(x_aug2, edge_index_aug2, args.source, False)
    update_mario_prototypes(h1_no_grad, h2_no_grad)
    # --------------------------


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0
best_micro_f1 = 0.0 # Added micro f1 tracking

start_time = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(epoch)
    # Evaluate source test mask accuracy (as in original code) for model selection criteria
    # Note: the original code uses source_data.test_mask, let's assume that's intended.
    # Test function now returns acc, macro_f1, micro_f1, embeddings
    source_correct, _, _, output_source = test(source_data, args.source, mask=source_data.test_mask) # Test on source test split
    target_correct, macro_f1, micro_f1, output_target = test(target_data, args.target) # Test on whole target graph

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch: {epoch:03d}, Source Acc: {source_correct:.4f}, Target Acc: {target_correct:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Time: {epoch_time:.2f}s")

    # Select best model based on source domain accuracy (as in original code)
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch
        # Saving embeddings based on best *source* performance
        # Ensure log directory exists
        os.makedirs('log', exist_ok=True)
        try:
            with open ('log/{}_{}_embeddings.pkl'.format(args.source, args.target),'wb') as f:
                # Ensure outputs are numpy arrays on CPU
                output_source_np = output_source.cpu().detach().numpy() if torch.is_tensor(output_source) else output_source
                output_target_np = output_target.cpu().detach().numpy() if torch.is_tensor(output_target) else output_target
                pickle.dump([output_source_np, output_target_np], f)
        except Exception as e:
            print(f"Error saving embeddings: {e}")


total_time = time.time() - start_time
print("=============================================================")
print(f"Total Training Time: {total_time:.2f}s")
line = ("{}\n - Best Epoch: {}, Best Source Acc: {:.4f}, Corresponding Target Acc: {:.4f}, "
        "Corresponding Macro F1: {:.4f}, Corresponding Micro F1: {:.4f}") \
    .format(id, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)
print(line)

# Ensure log directory exists
os.makedirs('log', exist_ok=True)
try:
    with open("log/{}-{}.log".format(args.source, args.target), 'a') as f:
        # Updated log format to include micro f1
        log_line = "{} - Epoch: {:0>3d}, Best Source Acc: {:.5f}, Target Acc: {:.5f}, Macro F1: {:.5f}, Micro F1: {:.5f}\t".format(
            id, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1
            ) + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n"
        f.write(log_line)
except Exception as e:
    print(f"Error writing log file: {e}")