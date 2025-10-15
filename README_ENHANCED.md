# Enhanced LLM-Guided MoE for Semi-Supervised Domain Generalization

This repository implements an enhanced version of the LLM-guided Mixture of Experts (MoE) model for semi-supervised domain generalization on graphs. The implementation includes dual MoE architectures, advanced loss functions, and comprehensive analysis tools following the detailed specifications in the implementation guide.

## 🚀 Key Features

### **Dual MoE Architectures**
- **Original MoE**: Heterogeneous experts with different GNN types/hop configurations
- **GAT+Soft-prompt MoE**: Unified experts with learnable prompt-based specialization
- **Comparative Analysis**: Side-by-side evaluation of both architectures

### **Advanced Learning Mechanisms**
- **DPO (Direct Preference Optimization)**: Pairwise preference learning from LLM rankings
- **Enhanced Consistency Loss**: KL divergence-based consistency between expert predictions
- **MMD Diversity Regularization**: Maximum Mean Discrepancy for expert diversity
- **Dynamic Loss Scheduling**: Adaptive weight adjustment throughout training

### **Memory & Performance Optimization**
- **Memory-Efficient Processing**: Batch processing for large graphs
- **Gradient Accumulation**: Effective large-batch training with limited memory
- **Intelligent Caching**: Enhanced LLM response caching with compatibility management
- **Robust Error Handling**: Multiple fallback strategies for LLM parsing

### **Comprehensive Analysis Tools**
- **Expert Specialization Analysis**: Load balancing, diversity metrics, class specialization
- **t-SNE Visualization**: Interactive and static embedding visualizations
- **Training Dynamics Monitoring**: Real-time tracking of losses and expert usage
- **Performance Profiling**: Memory usage and computational efficiency analysis

## 📁 Project Structure

```
MoEDG/
├── gnn/
│   ├── moe.py                    # Original MoE implementation
│   ├── gat_soft_prompt_moe.py    # NEW: GAT+Soft-prompt MoE
│   ├── ppmi_conv.py              # PPMI convolution implementation
│   └── dataset/
│       └── DomainData.py         # Domain data loader
├── common/
│   ├── llm_cache.py              # NEW: Enhanced LLM cache management
│   ├── loss_scheduler.py         # NEW: Dynamic loss scheduling
│   ├── advanced_losses.py        # NEW: DPO, consistency, MMD losses
│   ├── memory_efficient.py       # NEW: Memory-efficient processing
│   ├── graph_encoder.py          # Graph to text encoding
│   └── ...                       # Other utilities
├── utils/
│   ├── expert_analyzer.py        # NEW: Expert specialization analysis
│   └── visualization.py          # NEW: t-SNE and other visualizations
├── main.py                       # Original training script
├── main_enhanced.py              # NEW: Enhanced training with dual MoE
└── README_ENHANCED.md            # This file
```

## 🛠️ Installation

### Dependencies
```bash
# Core dependencies
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install torch-scatter>=2.1.0
pip install torch-sparse>=0.6.0

# LLM integration
pip install ollama
pip install openai>=1.0.0
pip install transformers>=4.30.0

# Analysis and visualization
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install plotly>=5.0.0
pip install pandas>=1.3.0

# Experiment tracking
pip install wandb>=0.15.0
```

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull qwen2.5:7b
```

## 🚀 Quick Start

### Basic Usage (Original MoE)
```bash
python main.py \
    --source dblpv7 \
    --target citationv1 \
    --expert_num 6 \
    --llm_interval 10 \
    --uncertainty_k 100
```

### Enhanced Usage (Dual MoE with Analysis)
```bash
# Run both architectures with full analysis
python main_enhanced.py \
    --source dblpv7 \
    --target citationv1 \
    --moe_architecture both \
    --expert_num 6 \
    --prompt_length 8 \
    --num_heads 4 \
    --llm_interval 5 \
    --adaptive_scheduler \
    --analyze_experts \
    --visualize_embeddings \
    --use_memory_efficient \
    --epochs 200

# Run only GAT+Soft-prompt architecture
python main_enhanced.py \
    --source dblpv7 \
    --target citationv1 \
    --moe_architecture gat_soft_prompt \
    --expert_num 6 \
    --prompt_length 12 \
    --adaptive_scheduler \
    --analyze_experts
```

## 🔧 Configuration

### Core Parameters
- `--moe_architecture`: Choose from `original`, `gat_soft_prompt`, `both`
- `--expert_num`: Number of experts (default: 6)
- `--llm_interval`: Interval for LLM calls (default: 5)
- `--uncertainty_k`: Number of uncertain nodes to process (default: 100)

### Advanced Parameters
- `--prompt_length`: Soft prompt length for GAT experts (default: 8)
- `--num_heads`: Number of attention heads (default: 4)
- `--alpha`: DPO temperature parameter (default: 0.61)
- `--beta`: Consistency loss weight (default: 0.11)
- `--div_weight`: Diversity loss weight (default: 0.01)

### Analysis Parameters
- `--analyze_experts`: Enable expert specialization analysis
- `--visualize_embeddings`: Enable t-SNE visualizations
- `--use_memory_efficient`: Use memory-efficient processing
- `--adaptive_scheduler`: Use adaptive loss scheduling

## 📊 Analysis and Visualization

### Expert Specialization Analysis
The enhanced system provides comprehensive expert analysis:

```python
from utils.expert_analyzer import ExpertSpecializationAnalyzer

analyzer = ExpertSpecializationAnalyzer(save_dir="analysis_results")

# Analyze expert usage patterns
usage_results = analyzer.analyze_expert_usage(
    expert_indices, expert_probs, node_labels, num_experts
)

# Analyze expert diversity
diversity_results = analyzer.analyze_expert_diversity(
    expert_outputs, gates
)

# Generate comprehensive report
report = analyzer.generate_comprehensive_report()
```

### Embedding Visualization
Create t-SNE and PCA visualizations:

```python
from utils.visualization import EmbeddingVisualizer

visualizer = EmbeddingVisualizer(save_dir="visualizations")

# t-SNE visualization
visualizer.visualize_embeddings_tsne(
    embeddings, labels, title="Node Embeddings t-SNE"
)

# Compare expert embeddings
visualizer.visualize_expert_embeddings(
    expert_outputs, expert_indices, labels
)

# Plot training dynamics
visualizer.plot_training_dynamics(metrics_history)
```

## 🧠 Advanced Features

### Memory-Efficient Processing
For large graphs, use memory-efficient processing:

```python
from common.memory_efficient import MemoryEfficientProcessor

processor = MemoryEfficientProcessor(device, max_memory_gb=8.0)

# Adaptive batch size determination
optimal_batch_size = processor.adaptive_batch_size(data, model)

# Process in batches
results = processor.process_in_batches(
    model, data, batch_size=optimal_batch_size, strategy='cluster'
)
```

### Enhanced LLM Caching
Intelligent caching with compatibility management:

```python
from common.llm_cache import EnhancedLLMCacheManager

cache_manager = EnhancedLLMCacheManager(cache_dir="llm_cache")

# Load with compatibility checks
preferences = cache_manager.load_expert_preferences(dataset, config)

# Save with metadata
cache_manager.save_expert_preferences(dataset, config, preferences, metadata)
```

### Dynamic Loss Scheduling
Adaptive loss weighting throughout training:

```python
from common.loss_scheduler import AdaptiveLossScheduler

scheduler = AdaptiveLossScheduler(total_epochs=200, config=config_dict)

# Get current weights
weights = scheduler.get_loss_weights(epoch)

# Update with training history
scheduler.update_loss_history(epoch, loss_components)
```

## 📈 Expected Performance

Based on the implementation guide, expected performance improvements:

| Dataset Pair | Original MoE | GAT+Soft-prompt MoE | Improvement |
|--------------|-------------|-------------------|-------------|
| dblpv7→citationv1 | 0.85-0.90 | 0.87-0.92 | +2-3% |
| dblpv7→acmv9 | 0.80-0.85 | 0.82-0.87 | +2-3% |
| citationv1→acmv9 | 0.75-0.80 | 0.77-0.82 | +2-3% |

## 🔍 Analysis Results

### Expert Specialization Metrics
- **Load Balance Score**: Measures expert usage balance (0-1, higher is better)
- **Diversity Score**: Measures expert diversity (0-1, higher is better)
- **Usage Entropy**: Measures expert selection entropy
- **Class Specialization**: Analyzes expert preference for different classes

### Training Dynamics
- **Loss Curves**: Multi-component loss evolution
- **Expert Usage**: Expert activation patterns over time
- **Memory Usage**: Real-time memory consumption monitoring
- **Convergence Analysis**: Training stability and convergence metrics

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors**
```bash
# Use memory-efficient processing
python main_enhanced.py --use_memory_efficient --batch_size 1000

# Reduce number of experts
python main_enhanced.py --expert_num 4
```

**LLM Response Parsing Errors**
The system includes robust fallback mechanisms:
- Multiple parsing strategies
- Graceful degradation to random selection
- Comprehensive error logging

**Cache Compatibility Issues**
- Automatic cache compatibility detection
- Configuration-based cache invalidation
- Fallback to re-computation when needed

### Performance Optimization

**For Large Datasets**
```bash
# Use clustering-based batching
--batch_size 2000 --use_memory_efficient

# Reduce LLM call frequency
--llm_interval 10 --uncertainty_k 50

# Use adaptive scheduling
--adaptive_scheduler
```

**For Faster Training**
```bash
# Reduce experts
--expert_num 4

# Disable analysis (for training only)
--analyze_experts False --visualize_embeddings False

# Use original architecture only
--moe_architecture original
```

## 📚 Citation

If you use this enhanced implementation, please cite:

```bibtex
@article{llm_guided_moe_2024,
  title={LLM-Guided Mixture of Experts for Semi-Supervised Domain Generalization on Graphs},
  author={[Your Name]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Enhancement
- Additional MoE architectures (Transformer-based, Graph Attention variants)
- Advanced LLM integration techniques (few-shot prompting, chain-of-thought)
- Distributed training support
- Additional analysis metrics and visualizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original MoE implementation and baseline code
- Ollama for local LLM inference
- PyTorch Geometric for graph neural networks
- W&B for experiment tracking

---

**Note**: This enhanced implementation builds upon the original LLM-guided MoE codebase while incorporating advanced techniques from recent research in expert systems, preference learning, and graph neural networks.