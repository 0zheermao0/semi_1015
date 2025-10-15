import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from sklearn.cluster import KMeans
import gc
import warnings


class MemoryEfficientProcessor:
    """
    Memory-efficient processing utilities for large graphs.
    Implements batch processing, gradient accumulation, and memory optimization.
    """

    def __init__(self, device: torch.device, max_memory_gb: float = 8.0):
        """
        Initialize the memory-efficient processor.

        Args:
            device: Device to run computations on
            max_memory_gb: Maximum allowed memory usage in GB
        """
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage statistics
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': max(0, self.max_memory_gb - allocated),
                'utilization': allocated / self.max_memory_gb
            }
        else:
            return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': self.max_memory_gb, 'utilization': 0}

    def estimate_memory_requirements(self, num_nodes: int, num_edges: int,
                                   feature_dim: int, expert_num: int,
                                   batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate memory requirements for processing.

        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges in the graph
            feature_dim: Feature dimensionality
            expert_num: Number of experts
            batch_size: Batch size for processing

        Returns:
            Dictionary with memory estimates in GB
        """
        # Base memory for features
        features_memory = num_nodes * feature_dim * 4  # float32

        # Memory for edge index
        edge_index_memory = num_edges * 2 * 4  # int32

        # Memory for expert outputs (num_nodes * expert_num * output_dim)
        output_dim = feature_dim  # Assume same as input for estimation
        expert_outputs_memory = num_nodes * expert_num * output_dim * 4

        # Memory for gate logits
        gate_memory = num_nodes * expert_num * 4

        # Memory for gradients (approximately 2x parameters)
        gradient_memory = (features_memory + expert_outputs_memory + gate_memory) * 2

        # Total memory
        total_memory = features_memory + edge_index_memory + expert_outputs_memory + gate_memory + gradient_memory

        # Convert to GB
        memory_gb = {
            'features_gb': features_memory / 1024**3,
            'edge_index_gb': edge_index_memory / 1024**3,
            'expert_outputs_gb': expert_outputs_memory / 1024**3,
            'gate_gb': gate_memory / 1024**3,
            'gradients_gb': gradient_memory / 1024**3,
            'total_gb': total_memory / 1024**3
        }

        if batch_size is not None:
            # Estimate batched memory
            scale_factor = batch_size / num_nodes
            for key in memory_gb:
                if key != 'total_gb':
                    memory_gb[f'batched_{key}'] = memory_gb[key] * scale_factor

            memory_gb['batched_total_gb'] = memory_gb['total_gb'] * scale_factor

        return memory_gb

    def create_batches(self, data: Data, max_batch_size: int,
                      strategy: str = 'random') -> List[Data]:
        """
        Create batches from large graph data.

        Args:
            data: PyTorch Geometric Data object
            max_batch_size: Maximum size of each batch
            strategy: Batching strategy ('random', 'cluster', 'degree', 'balanced')

        Returns:
            List of Data batches
        """
        num_nodes = data.num_nodes
        batches = []

        if strategy == 'random':
            # Random batching
            indices = torch.randperm(num_nodes)
            for i in range(0, num_nodes, max_batch_size):
                batch_indices = indices[i:i + max_batch_size]
                batch_data = self._create_subgraph_batch(data, batch_indices)
                batches.append(batch_data)

        elif strategy == 'cluster':
            # Feature-based clustering for batching
            features_np = data.x.detach().cpu().numpy()
            n_clusters = max(1, num_nodes // max_batch_size)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_np)

            for cluster_id in range(n_clusters):
                cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id))[0]
                if len(cluster_indices) > 0:
                    batch_data = self._create_subgraph_batch(data, cluster_indices)
                    batches.append(batch_data)

        elif strategy == 'degree':
            # Degree-based batching (similar degree nodes together)
            if hasattr(data, 'edge_index'):
                degrees = torch.zeros(num_nodes, dtype=torch.long)
                degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
                degrees += data.edge_index[1].bincount(minlength=num_nodes)
            else:
                degrees = torch.zeros(num_nodes)

            # Sort by degree and create batches
            sorted_indices = torch.argsort(degrees, descending=True)
            for i in range(0, num_nodes, max_batch_size):
                batch_indices = sorted_indices[i:i + max_batch_size]
                batch_data = self._create_subgraph_batch(data, batch_indices)
                batches.append(batch_data)

        elif strategy == 'balanced':
            # Balanced batching with mixed strategies
            features_np = data.x.detach().cpu().numpy()
            if hasattr(data, 'edge_index'):
                degrees = torch.zeros(num_nodes)
                degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1]))
                degrees += data.edge_index[1].bincount(minlength=num_nodes)
                degrees_np = degrees.numpy()
            else:
                degrees_np = np.zeros(num_nodes)

            # Create balanced batches considering both features and degrees
            node_scores = features_np.mean(axis=1) + 0.1 * degrees_np
            sorted_indices = np.argsort(node_scores)

            for i in range(0, num_nodes, max_batch_size):
                batch_indices = torch.tensor(sorted_indices[i:i + max_batch_size])
                batch_data = self._create_subgraph_batch(data, batch_indices)
                batches.append(batch_data)

        else:
            raise ValueError(f"Unknown batching strategy: {strategy}")

        return batches

    def _create_subgraph_batch(self, data: Data, node_indices: torch.Tensor) -> Data:
        """
        Create a subgraph batch for the given node indices.

        Args:
            data: Original data
            node_indices: Node indices to include in the batch

        Returns:
            Subgraph batch
        """
        # Get subgraph
        subset, edge_index, edge_attr, mapping = subgraph(
            node_indices, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes
        )

        # Create new data object
        batch_data = Data(
            x=data.x[subset],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y[subset] if hasattr(data, 'y') else None,
            num_nodes=len(subset)
        )

        # Add additional attributes if they exist
        for attr in ['train_mask', 'test_mask', 'val_mask']:
            if hasattr(data, attr):
                setattr(batch_data, attr, getattr(data, attr)[subset])

        return batch_data

    def process_in_batches(self, model: nn.Module, data: Data,
                          batch_size: int, strategy: str = 'random',
                          accumulate_gradients: bool = True) -> Dict[str, Any]:
        """
        Process model in batches with memory management.

        Args:
            model: Model to process
            data: Input data
            batch_size: Batch size for processing
            strategy: Batching strategy
            accumulate_gradients: Whether to accumulate gradients

        Returns:
            Dictionary with processing results
        """
        model.train()
        batches = self.create_batches(data, batch_size, strategy)

        total_loss = 0.0
        all_outputs = []
        all_expert_outputs = []
        batch_count = len(batches)

        print(f"Processing {data.num_nodes} nodes in {batch_count} batches of size ~{batch_size}")

        for batch_idx, batch_data in enumerate(batches):
            # Check memory usage
            memory_usage = self.get_memory_usage()
            if memory_usage['utilization'] > 0.9:
                print(f"Warning: High memory usage ({memory_usage['utilization']:.1%})")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Move batch to device
            batch_data = batch_data.to(self.device)

            # Forward pass
            try:
                if hasattr(model, 'forward') and 'experts_outputs' in model.forward.__code__.co_varnames:
                    # MoE model with expert outputs
                    output, experts_outputs, aux_loss, clean_logits = model(
                        batch_data.x, batch_data.edge_index, batch_data.edge_attr
                    )
                    all_expert_outputs.append(experts_outputs.detach().cpu())
                else:
                    # Regular model
                    output = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
                    aux_loss = torch.tensor(0.0, device=self.device)

                all_outputs.append(output.detach().cpu())

                # Dummy loss for demonstration (replace with actual loss computation)
                if hasattr(batch_data, 'y') and batch_data.y is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(output, batch_data.y) + aux_loss
                else:
                    loss = aux_loss

                total_loss += loss.item()

                # Gradient accumulation
                if accumulate_gradients:
                    loss = loss / batch_count  # Scale loss for accumulation

                # Backward pass
                loss.backward()

                if not accumulate_gradients or (batch_idx + 1) % 5 == 0:
                    # Update gradients periodically to avoid memory buildup
                    # (In practice, this would be handled by the optimizer)
                    pass

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in batch {batch_idx}. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Clear batch from GPU
            del batch_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Combine results
        if all_outputs:
            combined_output = torch.cat(all_outputs, dim=0)
        else:
            combined_output = torch.empty(0, data.x.shape[1])

        if all_expert_outputs:
            combined_expert_outputs = torch.cat(all_expert_outputs, dim=0)
        else:
            combined_expert_outputs = torch.empty(0, 0, data.x.shape[1])

        results = {
            'output': combined_output,
            'expert_outputs': combined_expert_outputs,
            'total_loss': total_loss / max(batch_count, 1),
            'num_batches': batch_count,
            'memory_usage': self.get_memory_usage()
        }

        return results

    def adaptive_batch_size(self, data: Data, model: nn.Module,
                           initial_batch_size: int = 1000) -> int:
        """
        Automatically determine optimal batch size based on memory constraints.

        Args:
            data: Input data
            model: Model to test
            initial_batch_size: Starting batch size

        Returns:
            Optimal batch size
        """
        print("Determining optimal batch size...")

        low, high = 100, initial_batch_size
        optimal_size = initial_batch_size

        # Binary search for optimal batch size
        while low <= high:
            mid = (low + high) // 2

            # Create small test batch
            test_indices = torch.randperm(data.num_nodes)[:mid]
            test_data = self._create_subgraph_batch(data, test_indices).to(self.device)

            try:
                # Test forward pass
                with torch.no_grad():
                    if hasattr(model, 'forward') and 'experts_outputs' in model.forward.__code__.co_varnames:
                        output, experts_outputs, aux_loss, clean_logits = model(
                            test_data.x, test_data.edge_index
                        )
                    else:
                        output = model(test_data.x, test_data.edge_index)

                # Check memory usage
                memory_usage = self.get_memory_usage()
                if memory_usage['utilization'] < 0.8:  # Safe to increase
                    optimal_size = mid
                    low = mid + 1
                else:
                    high = mid - 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise e

            finally:
                del test_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"Optimal batch size determined: {optimal_size}")
        return optimal_size

    def monitor_memory(self, log_interval: int = 100) -> None:
        """
        Monitor and log memory usage.

        Args:
            log_interval: Logging interval in steps
        """
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0

        self._step_counter += 1

        if self._step_counter % log_interval == 0:
            memory_usage = self.get_memory_usage()
            print(f"Step {self._step_counter}: Memory Usage - "
                  f"Allocated: {memory_usage['allocated_gb']:.2f}GB, "
                  f"Reserved: {memory_usage['reserved_gb']:.2f}GB, "
                  f"Utilization: {memory_usage['utilization']:.1%}")

            # Warning if approaching limit
            if memory_usage['utilization'] > 0.9:
                warnings.warn(f"High memory usage detected: {memory_usage['utilization']:.1%}")

    def cleanup_memory(self) -> None:
        """Clean up memory and garbage collect."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GradientAccumulator:
    """
    Gradient accumulation utility for effective large batch training.
    """

    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_loss = 0.0

    def accumulate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Accumulate loss and scale appropriately.

        Args:
            loss: Current loss value

        Returns:
            Scaled loss
        """
        scaled_loss = loss / self.accumulation_steps
        self.accumulated_loss += loss.item()
        return scaled_loss

    def should_step(self) -> bool:
        """
        Check if optimizer should step.

        Returns:
            True if should step, False otherwise
        """
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def reset(self) -> None:
        """Reset accumulator state."""
        self.current_step = 0
        self.accumulated_loss = 0.0

    def get_accumulated_loss(self) -> float:
        """Get accumulated loss."""
        return self.accumulated_loss / max(self.current_step, 1)


class CheckpointManager:
    """
    Checkpoint manager for memory-efficient training.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metadata: Optional[Dict] = None) -> str:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            metadata: Additional metadata

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {}
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Clean up old checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))

        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint}")