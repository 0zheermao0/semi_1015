import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class EmbeddingVisualizer:
    """
    Comprehensive visualization tool for graph embeddings and expert analysis.
    Supports t-SNE, PCA, and interactive visualizations.
    """

    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize the embedding visualizer.

        Args:
            save_dir: Directory to save visualization results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Color palettes
        self.color_palettes = {
            'tab10': plt.cm.tab10,
            'tab20': plt.cm.tab20,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'set1': plt.cm.Set1,
            'set2': plt.cm.Set2
        }

    def visualize_embeddings_tsne(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None,
                                 node_names: Optional[List[str]] = None, perplexity: int = 30,
                                 n_iter: int = 1000, random_state: int = 42,
                                 title: str = "t-SNE Visualization of Embeddings",
                                 save_plot: bool = True, interactive: bool = False) -> None:
        """
        Create t-SNE visualization of node embeddings.

        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Optional node labels for coloring
            node_names: Optional node names for annotations
            perplexity: t-SNE perplexity parameter
            n_iter: Number of t-SNE iterations
            random_state: Random state for reproducibility
            title: Plot title
            save_plot: Whether to save the plot
            interactive: Whether to create interactive plot
        """
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings

        # Apply t-SNE
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                   random_state=random_state, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings_np)

        # Create visualization
        if interactive:
            self._create_interactive_plot(embeddings_2d, labels, node_names, title, save_plot)
        else:
            self._create_static_plot(embeddings_2d, labels, node_names, title, save_plot)

    def visualize_embeddings_pca(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None,
                                node_names: Optional[List[str]] = None, title: str = "PCA Visualization of Embeddings",
                                save_plot: bool = True, interactive: bool = False) -> None:
        """
        Create PCA visualization of node embeddings.

        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Optional node labels for coloring
            node_names: Optional node names for annotations
            title: Plot title
            save_plot: Whether to save the plot
            interactive: Whether to create interactive plot
        """
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings

        # Apply PCA
        print("Computing PCA embedding...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_np)

        # Add explained variance to title
        explained_variance = pca.explained_variance_ratio_
        title += f" (Explained Variance: {explained_variance[0]:.3f}, {explained_variance[1]:.3f})"

        # Create visualization
        if interactive:
            self._create_interactive_plot(embeddings_2d, labels, node_names, title, save_plot)
        else:
            self._create_static_plot(embeddings_2d, labels, node_names, title, save_plot)

    def _create_static_plot(self, embeddings_2d: np.ndarray, labels: Optional[torch.Tensor],
                           node_names: Optional[List[str]], title: str, save_plot: bool) -> None:
        """Create static 2D visualization."""
        plt.figure(figsize=(12, 10))

        if labels is not None:
            # Convert labels to numpy
            if isinstance(labels, torch.Tensor):
                labels_np = labels.detach().cpu().numpy()
            else:
                labels_np = np.array(labels)

            # Handle continuous and discrete labels
            if len(np.unique(labels_np)) <= 20:  # Discrete labels
                # Create scatter plot with colors based on labels
                unique_labels = np.unique(labels_np)
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

                for i, label in enumerate(unique_labels):
                    mask = labels_np == label
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=[colors[i]], label=f'Class {label}', alpha=0.7)

                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:  # Continuous labels
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                    c=labels_np, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Label Value')
        else:
            # No labels - single color scatter plot
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

        # Add node names if provided (only for small graphs)
        if node_names is not None and len(node_names) <= 100:
            for i, name in enumerate(node_names):
                plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           fontsize=8, alpha=0.8)

        plt.title(title, fontsize=14)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / f"{title.replace(' ', '_').lower()}.png",
                       dpi=300, bbox_inches='tight')
        plt.show()

    def _create_interactive_plot(self, embeddings_2d: np.ndarray, labels: Optional[torch.Tensor],
                               node_names: Optional[List[str]], title: str, save_plot: bool) -> None:
        """Create interactive 2D visualization using Plotly."""
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1]
        })

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels_np = labels.detach().cpu().numpy()
            else:
                labels_np = np.array(labels)
            df['label'] = labels_np

        if node_names is not None:
            df['node_name'] = node_names

        # Create interactive scatter plot
        if labels is not None:
            if len(np.unique(labels_np)) <= 20:  # Discrete labels
                fig = px.scatter(df, x='x', y='y', color='label',
                               hover_data=['node_name'] if node_names is not None else None,
                               title=title)
            else:  # Continuous labels
                fig = px.scatter(df, x='x', y='y', color='label',
                               color_continuous_scale='viridis',
                               hover_data=['node_name'] if node_names is not None else None,
                               title=title)
        else:
            fig = px.scatter(df, x='x', y='y',
                           hover_data=['node_name'] if node_names is not None else None,
                           title=title)

        fig.update_layout(
            width=800,
            height=600,
            xaxis_title='Component 1',
            yaxis_title='Component 2'
        )

        if save_plot:
            fig.write_html(self.save_dir / f"{title.replace(' ', '_').lower()}.html")
        fig.show()

    def visualize_expert_embeddings(self, expert_outputs: torch.Tensor, expert_indices: torch.Tensor,
                                   labels: Optional[torch.Tensor] = None,
                                   save_plot: bool = True) -> None:
        """
        Visualize embeddings from different experts separately.

        Args:
            expert_outputs: Expert outputs [num_nodes, num_experts, feature_dim]
            expert_indices: Expert indices for each node [num_nodes, k]
            labels: Optional node labels
            save_plot: Whether to save plots
        """
        num_experts = expert_outputs.shape[1]

        # Create subplots for each expert
        fig, axes = plt.subplots(2, (num_experts + 1) // 2, figsize=(20, 10))
        if num_experts == 1:
            axes = [axes]
        elif num_experts <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for expert_id in range(num_experts):
            ax = axes[expert_id]

            # Get embeddings for this expert
            expert_embeddings = expert_outputs[:, expert_id, :].detach().cpu().numpy()

            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            embeddings_2d = tsne.fit_transform(expert_embeddings)

            # Plot with labels if available
            if labels is not None:
                labels_np = labels.detach().cpu().numpy()
                unique_labels = np.unique(labels_np)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for i, label in enumerate(unique_labels):
                    mask = labels_np == label
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                             c=[colors[i]], label=f'Class {label}', alpha=0.7, s=20)

                if expert_id == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)

            ax.set_title(f'Expert {expert_id}')
            ax.grid(True, alpha=0.3)

        # Remove unused subplots
        for expert_id in range(num_experts, len(axes)):
            fig.delaxes(axes[expert_id])

        plt.suptitle('t-SNE Visualization of Expert Embeddings', fontsize=16)
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'expert_embeddings_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_expert_specialization(self, expert_outputs: torch.Tensor, gates: torch.Tensor,
                                      labels: Optional[torch.Tensor] = None) -> None:
        """
        Visualize expert specialization patterns.

        Args:
            expert_outputs: Expert outputs [num_nodes, num_experts, feature_dim]
            gates: Gating weights [num_nodes, num_experts]
            labels: Optional node labels
        """
        num_experts = expert_outputs.shape[1]

        # Get dominant expert for each node
        dominant_experts = torch.argmax(gates, dim=1)

        # Create t-SNE visualization colored by dominant expert
        all_embeddings = expert_outputs.mean(dim=1)  # Average across experts

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings.detach().cpu().numpy())

        plt.figure(figsize=(12, 10))

        # Color by dominant expert
        dominant_experts_np = dominant_experts.detach().cpu().numpy()
        colors = plt.cm.tab10(np.linspace(0, 1, num_experts))

        for expert_id in range(num_experts):
            mask = dominant_experts_np == expert_id
            if np.any(mask):
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[colors[expert_id]], label=f'Expert {expert_id}', alpha=0.7)

        plt.title('t-SNE Visualization Colored by Dominant Expert', fontsize=14)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'expert_specialization_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Create confidence plot
        plt.figure(figsize=(10, 6))
        gate_probs = F.softmax(gates, dim=1)
        max_probs, _ = torch.max(gate_probs, dim=1)

        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                   c=max_probs.detach().cpu().numpy(), cmap='viridis', alpha=0.7)
        plt.colorbar(label='Gate Confidence')
        plt.title('t-SNE Visualization Colored by Gate Confidence', fontsize=14)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'gate_confidence_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_comparison_plot(self, embeddings_dict: Dict[str, torch.Tensor],
                            labels: Optional[torch.Tensor] = None,
                            method: str = 'tsne') -> None:
        """
        Create comparison plots for multiple embedding sets.

        Args:
            embeddings_dict: Dictionary of embedding sets with names as keys
            labels: Optional node labels
            method: Dimensionality reduction method ('tsne' or 'pca')
        """
        num_embeddings = len(embeddings_dict)
        fig, axes = plt.subplots(1, num_embeddings, figsize=(6 * num_embeddings, 6))

        if num_embeddings == 1:
            axes = [axes]

        for i, (name, embeddings) in enumerate(embeddings_dict.items()):
            # Convert to numpy and apply dimensionality reduction
            embeddings_np = embeddings.detach().cpu().numpy()

            if method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            else:
                reducer = PCA(n_components=2, random_state=42)

            embeddings_2d = reducer.fit_transform(embeddings_np)

            # Plot
            if labels is not None:
                labels_np = labels.detach().cpu().numpy()
                unique_labels = np.unique(labels_np)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for j, label in enumerate(unique_labels):
                    mask = labels_np == label
                    axes[i].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                 c=[colors[j]], label=f'Class {label}', alpha=0.7, s=20)

                if i == 0:
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                axes[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)

            axes[i].set_title(f'{name.upper()} - {method.upper()}')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / f'{method}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_dynamics(self, metrics_history: Dict[str, List[float]],
                             save_plot: bool = True) -> None:
        """
        Plot training dynamics over time.

        Args:
            metrics_history: Dictionary of metric histories
            save_plot: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Loss metrics
        loss_metrics = ['train_loss', 'val_loss', 'cls_loss', 'select_loss']
        available_losses = [m for m in loss_metrics if m in metrics_history]

        if available_losses:
            for metric in available_losses:
                axes[0].plot(metrics_history[metric], label=metric)
            axes[0].set_title('Loss Metrics')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No loss metrics available', ha='center', va='center')
            axes[0].set_title('Loss Metrics')

        # Accuracy metrics
        acc_metrics = ['train_acc', 'val_acc', 'target_acc']
        available_accs = [m for m in acc_metrics if m in metrics_history]

        if available_accs:
            for metric in available_accs:
                axes[1].plot(metrics_history[metric], label=metric)
            axes[1].set_title('Accuracy Metrics')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No accuracy metrics available', ha='center', va='center')
            axes[1].set_title('Accuracy Metrics')

        # Expert usage metrics
        expert_metrics = ['load_balance_score', 'diversity_score', 'usage_entropy']
        available_expert = [m for m in expert_metrics if m in metrics_history]

        if available_expert:
            for metric in available_expert:
                axes[2].plot(metrics_history[metric], label=metric)
            axes[2].set_title('Expert Usage Metrics')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No expert metrics available', ha='center', va='center')
            axes[2].set_title('Expert Usage Metrics')

        # Loss weights
        weight_metrics = ['cls_weight', 'consistency_weight', 'diversity_weight']
        available_weights = [m for m in weight_metrics if m in metrics_history]

        if available_weights:
            for metric in available_weights:
                axes[3].plot(metrics_history[metric], label=metric)
            axes[3].set_title('Loss Weights')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Weight')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'No weight metrics available', ha='center', va='center')
            axes[3].set_title('Loss Weights')

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()