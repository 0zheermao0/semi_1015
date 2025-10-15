# LLM指导的图神经网络专家混合模型：完整复现指南

## 1. 问题定义与核心创新

### 1.1 研究问题
**半监督图域适应 (Semi-supervised Graph Domain Adaptation)**
给定一个标注充足的源域图和一个无标注的目标域图，学习一个能够很好地泛化到目标域的图表示模型。

### 1.2 核心创新点

#### 🔥 **主要贡献：LLM指导的MoE架构**
- **专家选择的认知智能**：使用大语言模型(LLM)分析图节点的复杂特征，为不确定性节点提供专家选择建议
- **双轨MoE架构**：支持异构专家(Original)和统一专家(GAT+Soft-prompt)两种模式
- **自适应损失平衡**：动态调整多目标损失权重，解决训练过程中的冲突优化问题

#### 🧠 **技术创新细节**
1. **认知引导机制**：将图结构信息转换为自然语言描述，利用LLM的推理能力
2. **专家特化策略**：通过软提示让统一架构的专家学习不同的表示偏好
3. **不确定性感知采样**：基于专家分歧度智能选择需要LLM指导的节点

## 2. 数学建模与理论基础

### 2.1 问题形式化

设：
- 源域图 $G_s = (V_s, E_s, X_s, Y_s)$，其中 $|V_s| = n_s$
- 目标域图 $G_t = (V_t, E_t, X_t)$，其中 $|V_t| = n_t$，无标注
- 特征维度：$X \in \mathbb{R}^{n \times d}$
- 标签空间：$Y \in \{1, 2, ..., C\}$

目标：学习函数 $f: \mathbb{R}^d \rightarrow \mathbb{R}^C$，最小化目标域损失

### 2.2 LLM指导的专家选择模型

#### 专家网络定义
$$\text{Expert}_i(x, A) = \text{GNN}_i(x, A; \theta_i), \quad i = 1, ..., N$$

#### LLM偏好生成
对于不确定性节点 $v$，构建提示 $P(v)$：
```
节点特征: [归一化的节点特征向量]
邻居结构: [hop-1邻居的统计信息]
专家输出: [各专家的预测概率分布]
任务: 基于以上信息，为该节点排序专家，格式：{"ranking": [0,2,1,3], "confidence": 0.85}
```

#### 选择损失函数 (DPO版本)
$$\mathcal{L}_{\text{select}} = -\frac{1}{|U|}\sum_{v \in U} \sum_{i,j \in \mathcal{P}_v} \log \sigma(\alpha (s_i^v - s_j^v))$$

其中：
- $U$：不确定性节点集合
- $\mathcal{P}_v$：LLM生成的偏好对
- $s_i^v$：门控网络对专家i的得分
- $\alpha$：温度参数

### 2.3 一致性损失
$$\mathcal{L}_{\text{consistency}} = \frac{1}{|U|}\sum_{v \in U} \sum_{(i,j) \in \mathcal{R}_v} \text{KL}(P_i^v \| P_j^v)$$

其中 $P_i^v$ 是专家i在节点v的预测分布，$\mathcal{R}_v$ 是排名对

### 2.4 多样性损失 (MMD正则化)
$$\mathcal{L}_{\text{diversity}} = -\frac{2}{N(N-1)}\sum_{i<j} \text{MMD}^2(\mathcal{D}_i, \mathcal{D}_j) + \text{MMD}^2(\mathcal{C}_i, \mathcal{C}_j)$$

其中 $\mathcal{D}_i$ 和 $\mathcal{C}_i$ 分别是分发矩阵和组合矩阵

### 2.5 总损失函数
$$\mathcal{L}_{\text{total}} = w_{\text{cls}} \mathcal{L}_{\text{cls}} + w_{\text{select}} \mathcal{L}_{\text{select}} + w_{\text{consistency}} \mathcal{L}_{\text{consistency}} + w_{\text{diversity}} \mathcal{L}_{\text{diversity}} + w_{\text{gate}} \mathcal{L}_{\text{gate}}$$

## 3. 网络架构设计

### 3.1 整体架构图
```
输入: (X, A)
    ↓
[特征编码器]
    ↓
[专家混合层] ← LLM指导 ← [不确定性采样]
    ├── 专家1: GNN/hop=1 或 GAT+prompt1
    ├── 专家2: GNN/hop=2 或 GAT+prompt2
    ├── ...
    └── 专家N: GNN/hop=N 或 GAT+promptN
    ↓
[门控机制] ← [学习到的权重]
    ↓
[特征融合]
    ↓
[分类器] → 输出预测
```

### 3.2 双MoE架构实现

#### 架构1: Original MoE (异构专家)
```python
class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts=6,
                 gnn_type='ppmi', gnn_mode='power_adj'):
        # 专家配置：不同GNN类型或不同跳数
        self.experts = [
            PPMIConv(input_size, hidden_dim),  # hop=1
            PPMIConv(input_size, hidden_dim),  # hop=2
            PPMIConv(input_size, hidden_dim),  # hop=3
            GCNConv(input_size, hidden_dim),   # 多层1
            GCNConv(input_size, hidden_dim),   # 多层2
            SAGEConv(input_size, hidden_dim),  # 多层3
        ]
```

#### 架构2: GAT+Soft-prompt MoE (统一专家)
```python
class GATSoftPromptMoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts=6,
                 prompt_length=8, num_heads=4):
        # 统一的三层GAT架构
        self.experts = nn.ModuleList([
            UnifiedGATExpert(input_size, hidden_dim, output_size,
                           prompt_length=prompt_length, expert_id=i)
            for i in range(num_experts)
        ])
```

#### Soft-prompt实现
```python
class SoftPrompt(nn.Module):
    def __init__(self, prompt_length, hidden_dim, expert_id):
        # 可学习的prompt向量
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_dim) * 0.1
        )
        # 专家特化变换
        self.specialization_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 融合prompt信息到节点特征
        prompt_vector = self.prompt_embeddings.mean(dim=0)
        specialized = self.specialization_transform(x)
        fusion_weight = torch.sigmoid(self.fusion_weight)
        return (1-fusion_weight) * specialized + fusion_weight * prompt_vector
```

### 3.3 门控机制设计

#### 双模式门控
```python
class GateRouter(nn.Module):
    def __init__(self, input_dim, num_experts, gate_mode='embedding'):
        if gate_mode == 'embedding':
            self.gate = nn.Linear(input_dim, num_experts)
        elif gate_mode == 'gat':
            self.gate = GATConv(input_dim, num_experts, heads=1)

    def forward(self, x, edge_index):
        logits = self.gate(x, edge_index)
        return F.softmax(logits, dim=-1)
```

## 4. 训练策略与超参数

### 4.1 动态损失调度
```python
class DynamicLossScheduler:
    def get_cls_weight(self, epoch):
        # 分类损失权重：前期高，后期稳定
        return 1.0 if epoch < 50 else 0.8

    def get_consistency_weight(self, epoch):
        # 一致性损失：渐进式增加
        warmup_epochs = int(0.3 * total_epochs)
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        # 余弦调度
        return 0.1 + 0.9 * 0.5 * (1 + cos(π * (epoch-warmup_epochs) / (total_epochs-warmup_epochs)))

    def get_diversity_weight(self, epoch):
        # 多样性损失：周期性增强
        cycle_length = 20
        cycle_progress = (epoch % cycle_length) / cycle_length
        if cycle_progress < 0.5:
            return 0.01 + 0.04 * (2 * cycle_progress)
        else:
            return 0.01 + 0.04 * (2 * (1 - cycle_progress))
```

### 4.2 关键超参数配置

#### 基础配置
```yaml
# 模型参数
input_dim: 50-100          # 根据数据集调整
output_dim: 512            # 编码器输出维度
num_experts: 6             # 专家数量
hidden_dim: 1024           # 分类器隐藏维度
num_layers: 4              # 分类器层数

# 训练参数
learning_rate: 5e-4        # 学习率
weight_decay: 2e-5         # 权重衰减
dropout: 0.1               # Dropout率
epochs: 200                # 训练轮数
batch_size: full_graph     # 全图训练

# LLM参数
llm_model: "qwen2.5:7b"   # 或其他开源LLM
llm_interval: 1            # LLM调用间隔
uncertainty_k: 100         # 不确定性节点数量
max_context_length: 12000  # LLM上下文长度

# 损失权重
alpha: 0.61                # DPO温度参数
beta: 0.11                 # 一致性损失权重
div_weight: 0.01           # 多样性损失权重
consistency_weight: 1.0    # 一致性损失权重

# 优化器配置
optimizer: "Adam"
lr_scheduler: "adaptive"   # 或 "cosine"
lr_patience: 8
lr_factor: 0.6
warmup_epochs: 15
early_stop_patience: 80
```

#### 数据集特定配置
```yaml
# dblpv7 → citationv1
source: "dblpv7"
target: "citationv1"
label_rate: 0.05           # 5%标注率
gnn_type: "ppmi"           # PPMI卷积
gate_mode: "embedding"     # 门控模式

# acmv9 → citationv1
source: "acmv9"
target: "citationv1"
label_rate: 0.05
gnn_type: "ppmi"
gate_mode: "gat"
```

### 4.3 训练流程伪代码

```python
def train_epoch(model, data, optimizer, epoch):
    # 1. 前向传播
    encoded_output, expert_outputs, gate_loss, clean_logits = model(data.x, data.edge_index)
    logits = classifier(encoded_output)

    # 2. 不确定性采样
    uncertainty_mask = calculate_uncertainty(expert_outputs, classifier, k=uncertainty_k)
    uncertainty_nodes = torch.where(uncertainty_mask)[0].tolist()

    # 3. LLM指导 (每隔llm_interval轮)
    if epoch % llm_interval == 0:
        for node in uncertainty_nodes:
            if node not in cache:
                prompt = build_llm_prompt(node, data, expert_outputs, classifier)
                llm_response = query_llm(prompt)
                ranking, confidence = parse_response(llm_response)
                cache[node] = {"ranking": ranking, "confidence": confidence}

    # 4. 计算损失
    cls_loss = cross_entropy(logits[label_mask], data.y[label_mask])
    select_loss = dpo_loss(cache, uncertainty_nodes, clean_logits, alpha)
    consistency_loss = kl_consistency_loss(cache, expert_outputs, classifier, beta)
    diversity_loss = mmd_diversity_loss(data.x, expert_outputs, clean_logits)

    # 5. 动态权重
    w_cls = scheduler.get_cls_weight(epoch)
    w_consistency = scheduler.get_consistency_weight(epoch)
    w_diversity = scheduler.get_diversity_weight(epoch)

    # 6. 总损失
    total_loss = w_cls * cls_loss + w_consistency * consistency_loss + \
                 w_diversity * diversity_loss + select_loss + gate_loss

    # 7. 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return total_loss.item()
```

## 5. 实现细节与工程技巧

### 5.1 内存优化策略

#### 大图处理
```python
def encode_memory_efficient(model, data):
    """内存高效编码，避免OOM"""
    try:
        # 尝试标准编码
        return model(data.x, data.edge_index)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # 分批处理
            batch_size = 1000
            outputs = []
            for i in range(0, data.x.size(0), batch_size):
                batch_data = create_subgraph(data, i, i+batch_size)
                batch_output = model(batch_data.x, batch_data.edge_index)
                outputs.append(batch_output)
            return torch.cat(outputs, dim=0)
```

#### 缓存管理
```python
class EnhancedLLMCacheManager:
    def __init__(self, cache_dir="llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load_expert_preferences(self, dataset, config):
        """支持动态专家数的缓存加载"""
        config_hash = self._generate_config_hash(config)
        cache_path = self._get_cache_path(dataset, config_hash)

        if cache_path.exists():
            return torch.load(cache_path)
        else:
            # 尝试兼容性缓存
            return self._find_compatible_cache(dataset, config['expert_num'])
```

### 5.2 数值稳定性技巧

#### 门控熵正则化
```python
def calculate_gate_entropy_regularization(clean_logits, temperature=1.0):
    """防止门控过度集中"""
    gate_probs = F.softmax(clean_logits / temperature, dim=-1)
    gate_entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=-1)
    return -torch.mean(gate_entropy)  # 最大化熵
```

#### PPMI卷积数值稳定
```python
class PPMIConv(nn.Module):
    def forward(self, x, edge_index, edge_attr=None):
        # 预处理：边权重归一化
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), device=x.device)

        # 数值稳定的PPMI计算
        ppmi_matrix = self._compute_ppmi(edge_index, edge_attr, x.size(0))
        ppmi_matrix = torch.clamp(ppmi_matrix, min=1e-8, max=1e8)

        return torch.spmm(ppmi_matrix, x)
```

### 5.3 错误处理与鲁棒性

#### LLM响应解析
```python
def parse_llm_response_robust(response, expert_num):
    """鲁棒的LLM响应解析"""
    # 方法1: 直接JSON解析
    try:
        parsed = json.loads(response)
        return validate_ranking(parsed["ranking"], expert_num)
    except:
        pass

    # 方法2: 正则表达式提取
    import re
    pattern = r'\[\s*(\d+\s*,\s*)*\d+\s*\]'
    matches = re.findall(pattern, response)
    for match in matches:
        try:
            ranking = [int(x) for x in re.findall(r'\d+', match)]
            if validate_ranking(ranking, expert_num):
                return ranking
        except:
            continue

    # 方法3: 随机回退
    return list(range(expert_num))
```

## 6. 实验验证方案

### 6.1 数据集准备

#### 学术网络数据集
```python
class DomainData(InMemoryDataset):
    """标准学术网络数据集加载器"""

    def __init__(self, root, name):
        # 数据集统计
        # dblpv7: 19,997节点, 63,626边, 4类
        # citationv1: 9,227节点, 37,735边, 3类
        # acmv9: 11,464节点, 54,886边, 3类

        self.data = torch.load(f"{root}/{name}/processed/data.pt")
        self.num_classes = len(torch.unique(self.data.y))
```

#### 数据预处理
```python
def preprocess_graph(data):
    """图数据预处理"""
    # 1. 特征归一化
    data.x = F.normalize(data.x, p=2, dim=1)

    # 2. 构建PPMI矩阵
    if config.gnn_type == 'ppmi':
        ppmi_edge_index, ppmi_edge_weight = compute_ppmi(data.edge_index, data.num_nodes)
        data.ppmi_edge_index = ppmi_edge_index
        data.ppmi_edge_weight = ppmi_edge_weight

    # 3. 生成训练掩码
    label_mask = create_label_mask(data.y.size(0), label_rate=0.05)
    data.train_mask = label_mask

    return data
```

### 6.2 评估指标

#### 主要指标
```python
def evaluate_model(model, data):
    """模型评估"""
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)

        # 目标域准确率
        target_mask = data.test_mask  # 假设目标域
        accuracy = (pred[target_mask] == data.y[target_mask]).float().mean()

        # F1分数
        macro_f1 = f1_score(data.y[target_mask].cpu(),
                          pred[target_mask].cpu(), average='macro')
        micro_f1 = f1_score(data.y[target_mask].cpu(),
                          pred[target_mask].cpu(), average='micro')

    return accuracy, macro_f1, micro_f1
```

### 6.3 消融实验设计

#### 实验矩阵
```python
ablation_configs = {
    # LLM机制消融
    "full_method": {"llm_interval": 1},
    "no_llm_random": {"llm_interval": -1, "expert_selection": "random"},
    "no_llm_uniform": {"llm_interval": -1, "expert_selection": "uniform"},
    "llm_static": {"llm_interval": 999},

    # 架构消融
    "original_moe": {"moe_architecture": "original"},
    "gat_soft_prompt": {"moe_architecture": "gat_soft_prompt"},
    "single_expert": {"expert_num": 1},

    # 损失消融
    "no_consistency": {"consistency_weight": 0.0},
    "no_diversity": {"div_weight": 0.0},
    "no_dpo": {"alpha": 0.0},

    # GNN类型消融
    "ppmi_conv": {"gnn_type": "ppmi"},
    "gcn_conv": {"gnn_type": "gcn"},
    "sage_conv": {"gnn_type": "sage"},
}
```

### 6.4 基线对比

#### 竞争方法
1. **源域训练**: 仅在源域训练，直接测试目标域
2. **CaNet**: 对抗域适应网络
3. **LDAT**: 长尾域适应 transformer
4. **MARIO**: 多重表示学习和对齐
5. **SGDA**: 半监督图域适应

#### 实验设置
```yaml
# 对比实验配置
datasets: [
    ["dblpv7", "citationv1"],
    ["dblpv7", "acmv9"],
    ["citationv1", "acmv9"]
]

seeds: [1, 2, 3, 42, 100]  # 多种子实验
label_rates: [0.05, 0.1]   # 不同标注率
metrics: ["accuracy", "macro_f1", "micro_f1"]
```

## 7. 部署与优化

### 7.1 环境配置

#### 依赖包
```bash
# 核心依赖
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0

# LLM相关
openai>=1.0.0
transformers>=4.30.0
accelerate>=0.20.0

# 实验工具
wandb>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0

# 可选GPU加速
# pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

#### GPU配置
```python
# 内存监控
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# 自动混合精度
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(data.x, data.edge_index)
    loss = compute_loss(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 7.2 分布式训练

#### 多GPU训练
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)

    # 包装模型
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 数据加载器需要分布式采样器
    dataset = YourDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
```

### 7.3 生产部署

#### 模型保存与加载
```python
def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'config': config.__dict__,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_accuracy']
```

#### 推理优化
```python
class InferenceModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.model.eval()

    def predict(self, data):
        with torch.no_grad():
            # 批量推理
            batch_size = 1000
            predictions = []

            for i in range(0, len(data.x), batch_size):
                batch_x = data.x[i:i+batch_size]
                batch_edge_index = self.extract_subgraph_edges(data.edge_index, i, i+batch_size)

                output = self.model(batch_x, batch_edge_index)
                pred = output.argmax(dim=1)
                predictions.append(pred)

            return torch.cat(predictions, dim=0)
```

## 8. 结果分析与可视化

### 8.1 专家特化分析

#### 专家使用统计
```python
def analyze_expert_specialization(model, data):
    """分析专家特化模式"""
    expert_indices, expert_probs = model.get_node_expert_assignment(data.x, data.edge_index)

    # 专家使用分布
    expert_usage = torch.zeros(model.num_experts)
    for i in range(model.num_experts):
        expert_usage[i] = (expert_indices == i).float().mean()

    # 按类别分析专家偏好
    class_expert_matrix = torch.zeros(data.num_classes, model.num_experts)
    for node_id in range(len(data.y)):
        class_id = data.y[node_id]
        for expert_id in expert_indices[node_id]:
            class_expert_matrix[class_id, expert_id] += 1

    return {
        'expert_usage': expert_usage,
        'class_expert_matrix': class_expert_matrix,
        'entropy': -torch.sum(expert_usage * torch.log(expert_usage + 1e-9))
    }
```

### 8.2 训练过程可视化

#### 损失曲线
```python
def plot_training_curves(log_file):
    """绘制训练曲线"""
    df = pd.read_csv(log_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 损失曲线
    axes[0,0].plot(df['epoch'], df['train_loss'], label='Train')
    axes[0,0].plot(df['epoch'], df['val_loss'], label='Val')
    axes[0,0].set_title('Loss Curves')
    axes[0,0].legend()

    # 准确率曲线
    axes[0,1].plot(df['epoch'], df['source_acc'], label='Source')
    axes[0,1].plot(df['epoch'], df['target_acc'], label='Target')
    axes[0,1].set_title('Accuracy')
    axes[0,1].legend()

    # 专家多样性
    axes[1,0].plot(df['epoch'], df['diversity_loss'])
    axes[1,0].set_title('Expert Diversity')

    # 门控熵
    axes[1,1].plot(df['epoch'], df['gate_entropy'])
    axes[1,1].set_title('Gate Entropy')

    plt.tight_layout()
    plt.savefig('training_curves.png')
```

### 8.3 嵌入空间可视化

#### t-SNE可视化
```python
def visualize_embeddings(model, data):
    """可视化学习到的嵌入"""
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)

    # t-SNE降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

    # 绘制
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=data.y.cpu().numpy(), cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Learned Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('embeddings_tsne.png')
```

## 9. 常见问题与调试指南

### 9.1 训练问题

#### 梯度爆炸/消失
```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 层归一化
self.layer_norm = nn.LayerNorm(hidden_dim)

# 残差连接
output = x + self.transform(x)
```

#### 内存不足
```python
# 1. 减小批次大小
batch_size = 500  # 从1000减少到500

# 2. 梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 检查点技术
from torch.utils.checkpoint import checkpoint
output = checkpoint(self.expensive_function, input_tensor)
```

### 9.2 性能优化

#### 计算效率
```python
# 1. 稀疏矩阵优化
import torch_sparse
adj = torch.sparse_coo_tensor(edge_index, edge_weight, (n, n))

# 2. JIT编译
@torch.jit.script
def fast_gcn_forward(x, edge_index, weight):
    return torch.spmm(edge_index, x @ weight)

# 3. 预计算和缓存
@lru_cache(maxsize=128)
def compute_ppmi_matrix(edge_index_tuple, num_nodes):
    # 缓存PPMI矩阵计算结果
    pass
```

### 9.3 调试技巧

#### 中间结果检查
```python
def debug_forward(model, data):
    """调试前向传播过程"""
    print(f"Input shape: {data.x.shape}")

    # 检查每个专家的输出
    with torch.no_grad():
        for i, expert in enumerate(model.experts):
            expert_out = expert(data.x, data.edge_index)
            print(f"Expert {i} output: mean={expert_out.mean():.4f}, "
                  f"std={expert_out.std():.4f}, "
                  f"nan_count={torch.isnan(expert_out).sum()}")

    # 检查门控权重
    gate_logits = model.gate_router(data.x)
    gate_probs = F.softmax(gate_logits, dim=-1)
    print(f"Gate probs: mean={gate_probs.mean():.4f}, "
          f"entropy={-torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=-1).mean():.4f}")
```

## 10. 复现检查清单

### 10.1 代码实现检查
- [ ] 双MoE架构正确实现
- [ ] LLM提示构建逻辑正确
- [ ] 损失函数计算无误
- [ ] 动态权重调度正确
- [ ] 缓存机制工作正常

### 10.2 实验配置检查
- [ ] 数据集预处理正确
- [ ] 超参数设置合理
- [ ] 评估指标计算正确
- [ ] 多种子实验设置
- [ ] 基线对比公平

### 10.3 性能基准
```yaml
expected_performance:
  dblpv7_to_citationv1:
    target_accuracy: 0.85-0.90
    macro_f1: 0.80-0.85
    training_time: "1-2 hours on single GPU"

  dblpv7_to_acmv9:
    target_accuracy: 0.80-0.85
    macro_f1: 0.75-0.80
    training_time: "1-2 hours on single GPU"
```

### 10.4 可复现性保证
- [ ] 随机种子设置
- [ ] 环境依赖版本固定
- [ ] 数据划分方式一致
- [ ] 模型初始化相同
- [ ] 训练流程确定

---

## 总结

本复现指南提供了LLM指导的图神经网络专家混合模型的完整技术细节，包括：

1. **理论基础**：完整的数学建模和算法推导
2. **架构设计**：双MoE架构的详细实现
3. **训练策略**：动态损失调度和优化技巧
4. **工程实现**：内存优化、错误处理、部署方案
5. **实验验证**：全面的评估方案和对比实验
6. **调试指南**：常见问题和解决方案

遵循本指南，研究者可以完整复现该论文的核心贡献，并在此基础上进行进一步的创新和改进。