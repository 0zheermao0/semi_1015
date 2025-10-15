import matplotlib.pyplot as plt
import numpy as np

# Define the hyperparameter values (common for all subplots)
hyperparameter_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# --- Data for Subplot (a): select_weight ---
dataset_f1_scores_select_weight = {
    "A→C": [0.771460548405148,0.767207610520425, 0.774370453273643,0.722551762730833, 0.712143256855064],
    "C→D": [0.70, 0.75, 0.78, 0.82, 0.80],
    "D→C": [0.55, 0.62, 0.68, 0.70, 0.67],
    # "Dataset D": [0.78, 0.80, 0.81, 0.79, 0.83]
}

# --- Data for Subplot (b): semi_weight ---
dataset_f1_scores_semi_weight = {
    "A→C": [0.768998321208729, 0.774370453273643, 0.766536094012311, 0.770565193060996, 0.771460548405148],
    "C→D": [0.77, 0.80, 0.83, 0.81, 0.79],
    "D→C": [0.60, 0.65, 0.70, 0.73, 0.71],
    # "Dataset D": [0.81, 0.79, 0.77, 0.80, 0.82]
}

# --- Data for Subplot (c): another_hyperparam ---
# Using new example data for the third hyperparameter
dataset_f1_scores_another_hyperparam = {
    "A→C": [0.774370453273643, 0.73609, 0.73408, 0.71662, 0.774370453273643],
    "C→D": [0.75, 0.72, 0.70, 0.68, 0.71],
    "D→C": [0.50, 0.55, 0.60, 0.65, 0.62],
    # "Dataset D": [0.70, 0.73, 0.76, 0.74, 0.77]
}


# Define a color palette similar to the example image (shades of blue)
colors = {
    "A→C": "#90E0EF",  # Light Cyan/Blue
    "C→D": "#00B4D8",  # Medium Blue
    "D→C": "#0077B6",  # Darker Blue
    "Dataset D": "#03045E"   # Darkest Blue/Navy
}

# Create the plot with three subplots
plt.style.use('seaborn-v0_8-whitegrid') # Using a style for a nice grid
fig, axes = plt.subplots(1, 3, figsize=(24, 6)) # 1 row, 3 columns for subplots

# --- Configure Subplot (a): select_weight ---
ax1 = axes[0]
for dataset_name, f1_scores in dataset_f1_scores_select_weight.items():
    ax1.plot(
        hyperparameter_values,
        f1_scores,
        marker='o',                 # Circular markers
        linestyle='-',              # Solid line
        linewidth=2,                # Line width
        color=colors[dataset_name], # Assign color from the palette
        label=dataset_name          # Label for the legend
    )

ax1.set_xlabel("(a)", fontsize=12)
ax1.set_ylabel("Micro F1", fontsize=12)
ax1.set_title("$\\lambda_{u}$ (select_weight Value)", fontsize=14, loc='center') # Subplot title

ax1.set_xticks(hyperparameter_values)
ax1.set_xticklabels([f"{val:.1f}" for val in hyperparameter_values])

all_scores_select = [score for scores in dataset_f1_scores_select_weight.values() for score in scores]
min_score_select = min(all_scores_select) if all_scores_select else 0
max_score_select = max(all_scores_select) if all_scores_select else 1
ax1.set_ylim(min_score_select - 0.05, max_score_select + 0.05)
ax1.legend(fontsize=10, title="dataset")
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Configure Subplot (b): semi_weight ---
ax2 = axes[1]
for dataset_name, f1_scores in dataset_f1_scores_semi_weight.items():
    ax2.plot(
        hyperparameter_values,
        f1_scores,
        marker='o',                 # Circular markers
        linestyle='-',              # Solid line
        linewidth=2,                # Line width
        color=colors[dataset_name], # Assign color from the palette
        label=dataset_name
    )

ax2.set_xlabel("(b)", fontsize=12)
ax2.set_title("$\\lambda_{dl}$ (semi_weight Value)", fontsize=14, loc='center') # Subplot title

ax2.set_xticks(hyperparameter_values)
ax2.set_xticklabels([f"{val:.1f}" for val in hyperparameter_values])

all_scores_semi = [score for scores in dataset_f1_scores_semi_weight.values() for score in scores]
min_score_semi = min(all_scores_semi) if all_scores_semi else 0
max_score_semi = max(all_scores_semi) if all_scores_semi else 1
ax2.set_ylim(min_score_semi - 0.05, max_score_semi + 0.05)
# ax2.legend(fontsize=10, title="数据集") # Legend can be omitted if shared or placed once
ax2.grid(True, linestyle='--', alpha=0.7)

# --- Configure Subplot (c): another_hyperparam ---
ax3 = axes[2]
for dataset_name, f1_scores in dataset_f1_scores_another_hyperparam.items():
    ax3.plot(
        hyperparameter_values,
        f1_scores,
        marker='o',                 # Circular markers
        linestyle='-',              # Solid line
        linewidth=2,                # Line width
        color=colors[dataset_name], # Assign color from the palette
        label=dataset_name
    )

ax3.set_xlabel("(c)", fontsize=12) # Example name
ax3.set_title("$\\lambda_{ind}$ (diversity_weight Value)", fontsize=14, loc='center') # Subplot title

ax3.set_xticks(hyperparameter_values)
ax3.set_xticklabels([f"{val:.1f}" for val in hyperparameter_values])

all_scores_another = [score for scores in dataset_f1_scores_another_hyperparam.values() for score in scores]
min_score_another = min(all_scores_another) if all_scores_another else 0
max_score_another = max(all_scores_another) if all_scores_another else 1
ax3.set_ylim(min_score_another - 0.05, max_score_another + 0.05)
# ax3.legend(fontsize=10, title="数据集") # Legend can be omitted
ax3.grid(True, linestyle='--', alpha=0.7)


# Add an overall title to the figure
# fig.suptitle("不同超参数对 Micro F1 得分的影响", fontsize=16, y=1.00) # Adjust y for title position

# Improve layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0, 1, 0.95]) # rect adjusts for suptitle

# Show the plot
plt.savefig("plots/hyperparameter_effects.pdf", dpi=300, bbox_inches='tight') # Save the figure
