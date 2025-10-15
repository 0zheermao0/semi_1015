# coding=utf-8
import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Argument parser for source and target datasets
parser = argparse.ArgumentParser(description='Visualization of LLM Selection, MoE Encoder, and t-SNE Embeddings')
parser.add_argument("--source", type=str, default='acmv9', help="Source dataset name")
parser.add_argument("--target", type=str, default='citationv1', help="Target dataset name")
parser.add_argument("--llm", type=str, default='qwen2.5:7b', help="LLM model name used for expert selection")
parser.add_argument("--output_dir", type=str, default='plots', help="Directory to save output plots")
# Add a try-except block for environments where parse_args might be called multiple times (e.g. Jupyter notebooks)
try:
    args = parser.parse_args()
except SystemExit:
    # Fallback for environments like Jupyter where parse_args might not work as expected
    # You can set default values here if needed, or re-raise if it's a genuine CLI execution error
    class Args:
        source = 'acmv9'
        target = 'citationv1'
        llm = 'qwen2.5:7b'
        output_dir = 'plots'
    args = Args()


# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define a green-based colormap for scientific visualization
green_cmap = sns.light_palette("seagreen", as_cmap=True)
# Custom green colormap for t-SNE scatter plot
greens = plt.get_cmap('Greens')
green_colormap = ListedColormap(greens(np.linspace(0.2, 0.9, 256)))

def load_llm_selections():
    """Load LLM expert selections from JSON file."""
    selections_path = f"log/{args.target}-{args.llm}-selections.json"
    if not os.path.exists(selections_path):
        # Create dummy data if file doesn't exist for demonstration
        print(f"Warning: LLM selections file not found at {selections_path}. Generating dummy data.")
        # Assuming 100 nodes and 4 experts for dummy data
        num_dummy_nodes = 100
        num_dummy_experts = 4
        return {i: np.random.randint(0, num_dummy_experts) for i in range(num_dummy_nodes)}
        # raise FileNotFoundError(f"LLM selections file not found at {selections_path}") # Original behavior
    with open(selections_path, 'r') as f:
        selections = json.load(f)
    # Convert keys to integers
    selections = {int(k): v for k, v in selections.items()}
    return selections

def load_moe_expert_selections_distilled():
    """Load distilled MoE expert selections from CSV file for the target dataset."""
    csv_path = f"log/{args.source}-{args.target}-expert-selection-distill.csv"
    if not os.path.exists(csv_path):
        # Create dummy data if file doesn't exist for demonstration
        print(f"Warning: Distilled MoE expert selection CSV not found at {csv_path}. Generating dummy data.")
        num_dummy_nodes = 100
        num_dummy_experts = 4
        data = []
        for i in range(num_dummy_nodes):
            # Simulate probabilities for each expert, ensuring one is dominant
            probs = np.random.rand(num_dummy_experts)
            probs /= probs.sum() # Normalize to sum to 1 (roughly)
            chosen_expert = np.random.randint(0, num_dummy_experts)
            for expert_idx in range(num_dummy_experts):
                 data.append((i, expert_idx, probs[expert_idx] if expert_idx == chosen_expert else probs[expert_idx]*0.1 )) # Make chosen expert more probable
        return data
        # raise FileNotFoundError(f"Distilled MoE expert selection CSV not found at {csv_path}") # Original behavior
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            node_id = int(row[0])
            expert = int(row[1])
            prob = float(row[2])
            data.append((node_id, expert, prob))
    return data

def load_moe_expert_selections_non_distilled():
    """Load non-distilled MoE expert selections from CSV file for the target dataset."""
    csv_path = f"log/{args.source}-{args.target}-expert-selection.csv"
    if not os.path.exists(csv_path):
        # Create dummy data if file doesn't exist for demonstration
        print(f"Warning: Non-distilled MoE expert selection CSV not found at {csv_path}. Generating dummy data.")
        num_dummy_nodes = 100
        num_dummy_experts = 4
        data = []
        for i in range(num_dummy_nodes):
            probs = np.random.rand(num_dummy_experts)
            probs /= probs.sum()
            chosen_expert = np.random.randint(0, num_dummy_experts)
            for expert_idx in range(num_dummy_experts):
                 data.append((i, expert_idx, probs[expert_idx] if expert_idx == chosen_expert else probs[expert_idx]*0.1 ))
        return data
        # raise FileNotFoundError(f"Non-distilled MoE expert selection CSV not found at {csv_path}") # Original behavior
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            node_id = int(row[0])
            expert = int(row[1])
            prob = float(row[2])
            data.append((node_id, expert, prob))
    return data

def load_embeddings():
    """Load final embeddings and labels for target dataset."""
    embeddings_path = f"log/{args.source}-{args.target}-final-embeddings.pt"
    labels_path = f"log/{args.source}-{args.target}-final-labels.pt"
    if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
        # Create dummy data if files don't exist for demonstration
        print(f"Warning: Embeddings or labels not found. Generating dummy data.")
        num_dummy_samples = 100
        embedding_dim = 64
        num_dummy_classes = 5
        embeddings = torch.randn(num_dummy_samples, embedding_dim)
        labels = torch.randint(0, num_dummy_classes, (num_dummy_samples,))
        return embeddings.numpy(), labels.numpy()
        # raise FileNotFoundError(f"Embeddings or labels not found at {embeddings_path} or {labels_path}") # Original behavior
    embeddings = torch.load(embeddings_path).detach().numpy()
    labels = torch.load(labels_path).detach().numpy()
    return embeddings, labels

def plot_heatmap(data, title, filename, annot=True, cmap=None, subplot=None):
    """Plot heatmap with a specified color scheme."""
    if subplot:
        ax = subplot
    else:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    if cmap is None:
        cmap = green_cmap
        
    # Transpose data to swap x and y axes
    # Data input to this function has nodes/node_groups as rows, experts as columns
    # Transposing makes experts rows (y-axis) and nodes/node_groups columns (x-axis)
    data_transposed = data.T
        
    sns.heatmap(data_transposed, cmap=cmap, annot=annot, fmt='.2f', cbar_kws={'label': 'Frequency/Probability'}, ax=ax)
    ax.set_title(title, fontsize=16, pad=15) # Reduced font size for combined plot
    ax.set_xlabel('Node Group / Selection', fontsize=12) # Reduced font size
    ax.set_ylabel('Expert ID', fontsize=12) # Reduced font size

    # Adjust tick label sizes
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    if not subplot:
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {output_path}")
    return ax

def plot_tsne(embeddings, labels, title, filename, sample_size=None):
    """Plot t-SNE visualization with a top conference style color scheme and optional random sampling."""
    # Perform t-SNE dimensionality reduction with adjusted parameters
    tsne = TSNE(n_components=2, random_state=30, perplexity=100, max_iter=1000, learning_rate=200)
    
    # Random sampling if sample_size is specified
    if sample_size is not None and sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                    c=[colors[i]], 
                    label=f'Class {label}', alpha=0.6, s=30)
    
    plt.title(title, fontsize=28, pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=24)
    plt.ylabel('t-SNE Dimension 2', fontsize=24)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")

def create_llm_selection_heatmap(selections, subplot=None):
    """Create a heatmap for LLM expert selections, condensing nodes by averaging groups of 20."""
    if not selections: # Handle empty selections
        print("Warning: LLM selections data is empty. Skipping heatmap.")
        if subplot: # Draw an empty plot with title if it's a subplot
            subplot.set_title('LLM Selection (No Data)', fontsize=16, pad=15)
            subplot.set_xlabel('Node Group / Selection', fontsize=12)
            subplot.set_ylabel('Expert ID', fontsize=12)
        return subplot if subplot else None

    node_ids = sorted(selections.keys())
    num_nodes = len(node_ids)
    if num_nodes == 0:
        print("Warning: LLM selections has 0 nodes. Skipping heatmap.")
        if subplot:
            subplot.set_title('LLM Selection (No Nodes)', fontsize=16, pad=15)
            subplot.set_xlabel('Node Group / Selection', fontsize=12)
            subplot.set_ylabel('Expert ID', fontsize=12)
        return subplot if subplot else None

    num_experts = max(selections.values(), default=-1) + 1 # Handle case where max expert is 0
    if num_experts <= 0: # If no experts or only expert 0, ensure valid dimension
        num_experts = 1 if selections else 0 # If selections exist but max is -1 (empty), still treat as 0 experts. If selections has items, min 1 expert.

    selection_matrix = np.zeros((num_nodes, num_experts if num_experts > 0 else 1)) # Ensure num_experts is at least 1 for matrix creation
    
    if num_experts > 0:
        for idx, node_id in enumerate(node_ids):
            expert = selections[node_id]
            if 0 <= expert < num_experts:
                 selection_matrix[idx, expert] = 1  # One-hot encoding

    data_to_plot = selection_matrix
    if num_nodes > 500:
        group_size = 500
        num_groups = (num_nodes + group_size - 1) // group_size  # Ceiling division
        # Ensure num_experts for condensed_matrix is at least 1
        condensed_matrix = np.zeros((num_groups, num_experts if num_experts > 0 else 1))
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, num_nodes)
            group = selection_matrix[start_idx:end_idx, :]
            if group.shape[0] > 0: # Ensure group is not empty
                 condensed_matrix[i, :] = group.mean(axis=0)
        data_to_plot = condensed_matrix
    
    purple_cmap = sns.light_palette("purple", as_cmap=True)
    # Ensure data_to_plot is not empty before plotting
    if data_to_plot.shape[0] == 0 or data_to_plot.shape[1] == 0:
        print("Warning: Data for LLM selection heatmap is empty after processing. Skipping plot.")
        if subplot:
            subplot.set_title('LLM Selection (Empty Data)', fontsize=16, pad=15)
        return subplot if subplot else None

    return plot_heatmap(data_to_plot, 'LLM Selection', f'{args.target}-{args.llm}_heatmap.png', annot=False, cmap=purple_cmap, subplot=subplot)

def create_moe_encoder_heatmap_distilled(moe_data, subplot=None):
    """Create a heatmap for distilled MoE encoder selections, condensing nodes by averaging groups of 20."""
    if not moe_data: # Handle empty data
        print("Warning: Distilled MoE data is empty. Skipping heatmap.")
        if subplot:
            subplot.set_title('Distilled Router (No Data)', fontsize=16, pad=15)
            subplot.set_xlabel('Node Group / Selection', fontsize=12)
            subplot.set_ylabel('Expert ID', fontsize=12)
        return subplot if subplot else None

    df = pd.DataFrame(moe_data, columns=['NodeID', 'Expert', 'Probability'])
    if df.empty:
        print("Warning: Distilled MoE DataFrame is empty. Skipping heatmap.")
        if subplot:
            subplot.set_title('Distilled Router (Empty DF)', fontsize=16, pad=15)
        return subplot if subplot else None

    pivot_prob = df.pivot_table(values='Probability', index='NodeID', columns='Expert', aggfunc='mean', fill_value=0)
    pivot_prob = pivot_prob.sort_index()

    if pivot_prob.empty:
        print("Warning: Distilled MoE pivot table is empty. Skipping heatmap.")
        if subplot:
            subplot.set_title('Distilled Router (Empty Pivot)', fontsize=16, pad=15)
        return subplot if subplot else None

    data_to_plot_df = pivot_prob
    if len(pivot_prob) > 500:
        group_size = 500
        grouper = np.arange(len(pivot_prob)) // group_size
        data_to_plot_df = pivot_prob.groupby(grouper).mean()
    
    # Ensure data_to_plot_df is not empty before converting to numpy and plotting
    if data_to_plot_df.empty:
        print("Warning: Data for distilled MoE heatmap is empty after processing. Skipping plot.")
        if subplot:
            subplot.set_title('Distilled Router (Empty Data)', fontsize=16, pad=15)
        return subplot if subplot else None

    return plot_heatmap(data_to_plot_df.to_numpy(), 'Distilled Router Selection', f'{args.source}-{args.target}-moe_encoder_distilled_heatmap.png', annot=False, subplot=subplot)

def create_moe_encoder_heatmap_non_distilled(moe_data, subplot=None):
    """Create a heatmap for non-distilled MoE encoder selections, condensing nodes by averaging groups of 20."""
    if not moe_data: # Handle empty data
        print("Warning: Non-distilled MoE data is empty. Skipping heatmap.")
        if subplot:
            subplot.set_title('Non-Distilled Router (No Data)', fontsize=16, pad=15)
            subplot.set_xlabel('Node Group / Selection', fontsize=12)
            subplot.set_ylabel('Expert ID', fontsize=12)
        return subplot if subplot else None
        
    df = pd.DataFrame(moe_data, columns=['NodeID', 'Expert', 'Probability'])
    if df.empty:
        print("Warning: Non-distilled MoE DataFrame is empty. Skipping heatmap.")
        if subplot:
            subplot.set_title('Non-Distilled Router (Empty DF)', fontsize=16, pad=15)
        return subplot if subplot else None

    pivot_prob = df.pivot_table(values='Probability', index='NodeID', columns='Expert', aggfunc='mean', fill_value=0)
    pivot_prob = pivot_prob.sort_index()

    if pivot_prob.empty:
        print("Warning: Non-distilled MoE pivot table is empty. Skipping heatmap.")
        if subplot:
            subplot.set_title('Non-Distilled Router (Empty Pivot)', fontsize=16, pad=15)
        return subplot if subplot else None
        
    data_to_plot_df = pivot_prob
    if len(pivot_prob) > 500:
        group_size = 500
        grouper = np.arange(len(pivot_prob)) // group_size
        data_to_plot_df = pivot_prob.groupby(grouper).mean()

    if data_to_plot_df.empty:
        print("Warning: Data for non-distilled MoE heatmap is empty after processing. Skipping plot.")
        if subplot:
            subplot.set_title('Non-Distilled Router (Empty Data)', fontsize=16, pad=15)
        return subplot if subplot else None

    return plot_heatmap(data_to_plot_df.to_numpy(), 'Non-Distilled Router Selection', f'{args.source}-{args.target}-moe_encoder_non_distilled_heatmap.png', annot=False, subplot=subplot)

def main():
    # Load data
    print("Loading LLM selections...")
    llm_selections = load_llm_selections()
    print("Loading distilled MoE expert selections...")
    moe_data_distilled = load_moe_expert_selections_distilled()
    print("Loading non-distilled MoE expert selections...")
    moe_data_non_distilled = load_moe_expert_selections_non_distilled()
    print("Loading embeddings and labels...")
    embeddings, labels = load_embeddings()
    
    # # Plot combined heatmaps
    # print("Generating combined heatmap...")
    # # Increased figure height slightly for better label spacing, width adjusted for potentially fewer x-ticks
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7)) # Adjusted figsize
    
    # create_llm_selection_heatmap(llm_selections, subplot=ax1)
    # if ax1: ax1.text(0.5, -0.1, '(a)', transform=ax1.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center') # Adjusted text position and size
    
    # create_moe_encoder_heatmap_distilled(moe_data_distilled, subplot=ax2)
    # if ax2: ax2.text(0.5, -0.1, '(b)', transform=ax2.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center')
    
    # create_moe_encoder_heatmap_non_distilled(moe_data_non_distilled, subplot=ax3)
    # if ax3: ax3.text(0.5, -0.1, '(c)', transform=ax3.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center')
    
    # plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0) # Added padding arguments
    # # Add a main title to the figure
    # fig.suptitle(f'Expert Selection Heatmaps ({args.source} to {args.target})', fontsize=20, y=1.03)


    # combined_output_path = os.path.join(args.output_dir, f'{args.source}-{args.target}-combined_heatmap.pdf')
    # plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    # plt.close(fig) # Ensure the correct figure is closed
    # print(f"Saved combined heatmap to {combined_output_path}")
    
    # Plot t-SNE visualization
    print("Generating t-SNE visualization...")
    if embeddings is not None and labels is not None and len(embeddings) > 1:
        plot_tsne(embeddings, labels, 
                  f't-SNE of Encoder Embeddings ({args.target})',
                  f'{args.source}-{args.target}-tsne_embeddings.pdf',
                  sample_size=1000)  # 可以调整采样数量
    else:
        print("Skipping t-SNE plot due to missing or insufficient data.")


if __name__ == "__main__":
    # Create dummy log directory and files for testing if they don't exist
    if not os.path.exists("log"):
        os.makedirs("log")

    # Note: The dummy file creation is for making the script runnable without actual data.
    # In a real scenario, these files would be inputs.
    # Example: Create a dummy selections.json if it doesn't exist
    dummy_selections_path = f"log/{args.target}-{args.llm}-selections.json"
    if not os.path.exists(dummy_selections_path):
        with open(dummy_selections_path, 'w') as f:
            json.dump({i: np.random.randint(0,4) for i in range(100)}, f)

    dummy_distill_csv = f"log/{args.source}-{args.target}-expert-selection-distill.csv"
    if not os.path.exists(dummy_distill_csv):
        with open(dummy_distill_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['NodeID', 'Expert', 'Probability'])
            for i in range(100): # 100 nodes
                for j in range(4): # 4 experts
                    writer.writerow([i, j, np.random.rand()*0.25 if j != i % 4 else 0.5 + np.random.rand()*0.5]) # Make one expert more probable

    dummy_nondistill_csv = f"log/{args.source}-{args.target}-expert-selection.csv"
    if not os.path.exists(dummy_nondistill_csv):
         with open(dummy_nondistill_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['NodeID', 'Expert', 'Probability'])
            for i in range(100): # 100 nodes
                for j in range(4): # 4 experts
                    writer.writerow([i, j, np.random.rand()*0.25 if j != i % 4 else 0.5 + np.random.rand()*0.5])
    
    dummy_embeddings_path = f"log/{args.source}-{args.target}-final-embeddings.pt"
    if not os.path.exists(dummy_embeddings_path):
        torch.save(torch.randn(100, 64), dummy_embeddings_path)

    dummy_labels_path = f"log/{args.source}-{args.target}-final-labels.pt"
    if not os.path.exists(dummy_labels_path):
        torch.save(torch.randint(0, 5, (100,)), dummy_labels_path)

    main()
