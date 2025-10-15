import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import pandas as pd
from pathlib import Path


class ExpertSpecializationAnalyzer:
    """
    Comprehensive analyzer for expert specialization patterns in MoE models.
    Provides insights into expert usage, specialization, and diversity.
    """

    def __init__(self, save_dir: str = "analysis_results"):
        """
        Initialize the expert analyzer.

        Args:
            save_dir: Directory to save analysis results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Store analysis results
        self.analysis_results = {}

    def analyze_expert_usage(self, expert_indices: torch.Tensor, expert_probs: torch.Tensor,
                           node_labels: Optional[torch.Tensor] = None,
                           num_experts: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze expert usage patterns and specialization.

        Args:
            expert_indices: Expert indices for each node [num_nodes, k]
            expert_probs: Expert probabilities [num_nodes, k]
            node_labels: Optional node labels for class-based analysis
            num_experts: Total number of experts

        Returns:
            Dictionary with expert usage analysis results
        """
        if num_experts is None:
            num_experts = expert_indices.max().item() + 1

        # Flatten for easier analysis
        all_indices = expert_indices.flatten().cpu().numpy()
        all_probs = expert_probs.flatten().cpu().numpy()

        # Expert usage distribution
        expert_usage = np.zeros(num_experts)
        for i in range(num_experts):
            expert_usage[i] = np.sum(all_indices == i)

        expert_usage_normalized = expert_usage / np.sum(expert_usage)

        # Expert probability statistics
        expert_prob_stats = {}
        for i in range(num_experts):
            expert_mask = all_indices == i
            if np.any(expert_mask):
                probs = all_probs[expert_mask]
                expert_prob_stats[i] = {
                    'mean': np.mean(probs),
                    'std': np.std(probs),
                    'min': np.min(probs),
                    'max': np.max(probs),
                    'median': np.median(probs)
                }

        # Load balancing analysis
        load_balance_score = 1.0 - np.std(expert_usage_normalized) / np.mean(expert_usage_normalized)

        # Entropy of expert usage
        usage_entropy = -np.sum(expert_usage_normalized * np.log(expert_usage_normalized + 1e-8))

        results = {
            'expert_usage': expert_usage,
            'expert_usage_normalized': expert_usage_normalized,
            'expert_prob_stats': expert_prob_stats,
            'load_balance_score': load_balance_score,
            'usage_entropy': usage_entropy,
            'num_experts': num_experts
        }

        # Class-based analysis if labels are provided
        if node_labels is not None:
            class_analysis = self._analyze_class_specialization(
                expert_indices, node_labels, num_experts
            )
            results['class_analysis'] = class_analysis

        self.analysis_results['expert_usage'] = results
        return results

    def _analyze_class_specialization(self, expert_indices: torch.Tensor, node_labels: torch.Tensor,
                                    num_experts: int) -> Dict[str, Any]:
        """
        Analyze expert specialization across different classes.

        Args:
            expert_indices: Expert indices for each node [num_nodes, k]
            node_labels: Node labels [num_nodes]
            num_experts: Number of experts

        Returns:
            Class specialization analysis results
        """
        unique_labels = torch.unique(node_labels).cpu().numpy()
        num_classes = len(unique_labels)

        # Create class-expert matrix
        class_expert_matrix = np.zeros((num_classes, num_experts))
        class_node_counts = np.zeros(num_classes)

        for i, label in enumerate(unique_labels):
            class_mask = node_labels == label
            class_node_indices = torch.where(class_mask)[0]
            class_node_counts[i] = len(class_node_indices)

            # Count expert usage for this class
            class_expert_indices = expert_indices[class_mask].flatten().cpu().numpy()
            for j in range(num_experts):
                class_expert_matrix[i, j] = np.sum(class_expert_indices == j)

        # Normalize by class size
        class_expert_matrix_normalized = class_expert_matrix / class_node_counts.reshape(-1, 1)

        # Compute specialization metrics
        # Expert specialization: how much each expert focuses on specific classes
        expert_specialization = np.zeros(num_experts)
        for j in range(num_experts):
            expert_dist = class_expert_matrix[:, j]
            expert_dist = expert_dist / (expert_dist.sum() + 1e-8)
            expert_specialization[j] = -np.sum(expert_dist * np.log(expert_dist + 1e-8))

        # Class specialization: how much each class uses specific experts
        class_specialization = np.zeros(num_classes)
        for i in range(num_classes):
            class_dist = class_expert_matrix_normalized[i, :]
            class_dist = class_dist / (class_dist.sum() + 1e-8)
            class_specialization[i] = -np.sum(class_dist * np.log(class_dist + 1e-8))

        # Dominant expert for each class
        dominant_experts = np.argmax(class_expert_matrix_normalized, axis=1)

        # Expert dominance: which class each expert is most associated with
        expert_dominant_classes = np.argmax(class_expert_matrix, axis=1)

        return {
            'class_expert_matrix': class_expert_matrix,
            'class_expert_matrix_normalized': class_expert_matrix_normalized,
            'expert_specialization': expert_specialization,
            'class_specialization': class_specialization,
            'dominant_experts': dominant_experts,
            'expert_dominant_classes': expert_dominant_classes,
            'unique_labels': unique_labels
        }

    def analyze_expert_diversity(self, expert_outputs: torch.Tensor, gates: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze diversity between expert representations.

        Args:
            expert_outputs: Expert outputs [num_nodes, num_experts, feature_dim]
            gates: Gating weights [num_nodes, num_experts]

        Returns:
            Diversity analysis results
        """
        num_nodes, num_experts, feature_dim = expert_outputs.shape

        # Compute pairwise similarities between experts
        expert_similarities = np.zeros((num_experts, num_experts))
        expert_activations = []

        for i in range(num_experts):
            # Get activated nodes for expert i
            mask_i = gates[:, i] > 0
            if not mask_i.any():
                expert_activations.append([])
                continue

            expert_i_repr = expert_outputs[mask_i, i].cpu().numpy()
            expert_activations.append(len(expert_i_repr))

            for j in range(num_experts):
                if i == j:
                    expert_similarities[i, j] = 1.0
                    continue

                mask_j = gates[:, j] > 0
                if not mask_j.any():
                    expert_similarities[i, j] = 0.0
                    continue

                expert_j_repr = expert_outputs[mask_j, j].cpu().numpy()

                # Compute cosine similarity
                if len(expert_i_repr) > 0 and len(expert_j_repr) > 0:
                    # Use mean representations for simplicity
                    mean_i = np.mean(expert_i_repr, axis=0)
                    mean_j = np.mean(expert_j_repr, axis=0)

                    # Cosine similarity
                    similarity = np.dot(mean_i, mean_j) / (
                        np.linalg.norm(mean_i) * np.linalg.norm(mean_j) + 1e-8
                    )
                    expert_similarities[i, j] = similarity
                else:
                    expert_similarities[i, j] = 0.0

        # Diversity metrics
        # Average pairwise similarity (lower is more diverse)
        mask = ~np.eye(num_experts, dtype=bool)
        avg_similarity = np.mean(expert_similarities[mask])

        # Diversity score (1 - average similarity)
        diversity_score = 1.0 - avg_similarity

        # Expert activation statistics
        activation_stats = {
            'mean_activation': np.mean(expert_activations),
            'std_activation': np.std(expert_activations),
            'min_activation': np.min(expert_activations),
            'max_activation': np.max(expert_activations)
        }

        # Load imbalance
        activation_distribution = np.array(expert_activations)
        activation_distribution = activation_distribution / (np.sum(activation_distribution) + 1e-8)
        load_imbalance = np.std(activation_distribution)

        return {
            'expert_similarities': expert_similarities,
            'avg_similarity': avg_similarity,
            'diversity_score': diversity_score,
            'expert_activations': expert_activations,
            'activation_stats': activation_stats,
            'load_imbalance': load_imbalance
        }

    def visualize_expert_usage(self, save_plot: bool = True) -> None:
        """Visualize expert usage patterns."""
        if 'expert_usage' not in self.analysis_results:
            print("No expert usage analysis available. Run analyze_expert_usage first.")
            return

        results = self.analysis_results['expert_usage']
        expert_usage = results['expert_usage_normalized']
        expert_prob_stats = results['expert_prob_stats']

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Expert Usage Analysis', fontsize=16)

        # 1. Expert usage distribution
        experts = np.arange(len(expert_usage))
        axes[0, 0].bar(experts, expert_usage)
        axes[0, 0].set_title('Expert Usage Distribution')
        axes[0, 0].set_xlabel('Expert ID')
        axes[0, 0].set_ylabel('Usage Fraction')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Expert probability statistics
        prob_means = [expert_prob_stats[i]['mean'] for i in range(len(expert_prob_stats))]
        prob_stds = [expert_prob_stats[i]['std'] for i in range(len(expert_prob_stats))]

        axes[0, 1].errorbar(experts, prob_means, yerr=prob_stds, fmt='o', capsize=5)
        axes[0, 1].set_title('Expert Probability Statistics')
        axes[0, 1].set_xlabel('Expert ID')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Load balance score and entropy
        balance_score = results['load_balance_score']
        usage_entropy = results['usage_entropy']

        axes[1, 0].bar(['Load Balance\nScore', 'Usage\nEntropy'], [balance_score, usage_entropy])
        axes[1, 0].set_title('Load Balance and Entropy')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Expert usage pie chart
        if len(expert_usage) > 0:
            axes[1, 1].pie(expert_usage, labels=[f'Expert {i}' for i in range(len(expert_usage))],
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Expert Usage Proportion')

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'expert_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_class_specialization(self, save_plot: bool = True) -> None:
        """Visualize class-based expert specialization."""
        if 'expert_usage' not in self.analysis_results or 'class_analysis' not in self.analysis_results['expert_usage']:
            print("No class analysis available. Run analyze_expert_usage with node_labels first.")
            return

        class_analysis = self.analysis_results['expert_usage']['class_analysis']
        class_expert_matrix = class_analysis['class_expert_matrix_normalized']
        unique_labels = class_analysis['unique_labels']

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(class_expert_matrix,
                   xticklabels=[f'Expert {i}' for i in range(class_expert_matrix.shape[1])],
                   yticklabels=[f'Class {label}' for label in unique_labels],
                   annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Class-Expert Specialization Matrix')
        plt.xlabel('Expert ID')
        plt.ylabel('Class')
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'class_specialization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_expert_diversity(self, save_plot: bool = True) -> None:
        """Visualize expert diversity analysis."""
        if 'expert_diversity' not in self.analysis_results:
            print("No diversity analysis available. Run analyze_expert_diversity first.")
            return

        results = self.analysis_results['expert_diversity']
        similarities = results['expert_similarities']
        diversity_score = results['diversity_score']

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Expert Diversity Analysis', fontsize=16)

        # 1. Similarity heatmap
        num_experts = similarities.shape[0]
        sns.heatmap(similarities,
                   xticklabels=[f'Expert {i}' for i in range(num_experts)],
                   yticklabels=[f'Expert {i}' for i in range(num_experts)],
                   annot=True, cmap='coolwarm', center=0.5, ax=axes[0])
        axes[0].set_title('Expert Pairwise Similarities')
        axes[0].set_xlabel('Expert ID')
        axes[0].set_ylabel('Expert ID')

        # 2. Diversity metrics
        metrics = ['Diversity\nScore', 'Load\nImbalance']
        values = [diversity_score, results['load_imbalance']]

        axes[1].bar(metrics, values)
        axes[1].set_title('Diversity Metrics')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'expert_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.

        Returns:
            Dictionary containing all analysis results and summary statistics
        """
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'summary_statistics': {},
            'detailed_results': self.analysis_results
        }

        # Generate summary statistics
        if 'expert_usage' in self.analysis_results:
            usage_results = self.analysis_results['expert_usage']
            report['summary_statistics'].update({
                'load_balance_score': usage_results['load_balance_score'],
                'usage_entropy': usage_results['usage_entropy'],
                'total_nodes_analyzed': int(np.sum(usage_results['expert_usage']))
            })

        if 'expert_diversity' in self.analysis_results:
            diversity_results = self.analysis_results['expert_diversity']
            report['summary_statistics'].update({
                'diversity_score': diversity_results['diversity_score'],
                'avg_expert_similarity': diversity_results['avg_similarity'],
                'load_imbalance': diversity_results['load_imbalance']
            })

        # Save report
        report_path = self.save_dir / 'expert_analysis_report.json'
        import json
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._convert_numpy_to_python(report)
            json.dump(json_report, f, indent=2)

        print(f"Comprehensive analysis report saved to {report_path}")
        return report

    def _convert_numpy_to_python(self, obj: Any) -> Any:
        """Convert numpy arrays and types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def compare_experts_over_time(self, history_results: List[Dict[str, Any]]) -> None:
        """
        Compare expert analysis results over time.

        Args:
            history_results: List of analysis results from different time points
        """
        if len(history_results) < 2:
            print("Need at least 2 analysis results for comparison.")
            return

        # Extract metrics over time
        time_points = range(len(history_results))
        load_balances = []
        diversities = []
        entropies = []

        for result in history_results:
            if 'expert_usage' in result:
                load_balances.append(result['expert_usage']['load_balance_score'])
                entropies.append(result['expert_usage']['usage_entropy'])
            else:
                load_balances.append(0)
                entropies.append(0)

            if 'expert_diversity' in result:
                diversities.append(result['expert_diversity']['diversity_score'])
            else:
                diversities.append(0)

        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Expert Analysis Over Time', fontsize=16)

        metrics = [
            ('Load Balance Score', load_balances),
            ('Diversity Score', diversities),
            ('Usage Entropy', entropies)
        ]

        for i, (title, values) in enumerate(metrics):
            axes[i].plot(time_points, values, marker='o', linewidth=2)
            axes[i].set_title(title)
            axes[i].set_xlabel('Time Point')
            axes[i].set_ylabel(title)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'expert_analysis_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()