import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Define the hyperparameter values (common for all subplots)
hyperparameter_values_list = [0.2, 0.4, 0.6, 0.8, 1.0]

# Data dictionaries
data_sources = {
    "$\\lambda_u$": {
        "A→C": [0.771460548405148,0.767207610520425, 0.774370453273643,0.762551762730833, 0.712143256855064],
        "C→D": [0.71937, 0.70824, 0.7097, 0.69238, 0.62619],
        "D→C": [0.71819, 0.71405, 0.71964, 0.72166, 0.72132],
    },
    "$\\lambda_{dl}$": {
        "A→C": [0.768998321208729, 0.774370453273643, 0.766536094012311, 0.770565193060996, 0.771460548405148],
        "C→D": [0.71034, 0.71116, 0.7097, 0.70569, 0.7066],
        "D→C": [0.71863, 0.71214, 0.69871, 0.69525, 0.67822],
    },
    "$\\lambda_{ind}$": {
        "A→C": [0.764370453273643, 0.76609, 0.75408, 0.75662, 0.774370453273643],
        "C→D": [0.70587, 0.70204, 0.71025, 0.70973, 0.71156],
        "D→C": [0.70073, 0.71405, 0.72278, 0.71718, 0.71863],
    }
}

dataset_names = ["A→C", "C→D", "D→C"]
hyperparam_type_names = list(data_sources.keys())

# Determine overall min and max F1 scores for consistent color scaling across heatmaps
all_f1_scores = []
for hp_type in hyperparam_type_names:
    for dataset in dataset_names:
        if dataset in data_sources[hp_type]: 
            if data_sources[hp_type][dataset]: 
                 all_f1_scores.extend(data_sources[hp_type][dataset])

min_f1 = min(all_f1_scores) if all_f1_scores else 0.0 
max_f1 = max(all_f1_scores) if all_f1_scores else 1.0 


# Create the figure and subplots
n_datasets = len(dataset_names)
if n_datasets == 3:
    n_rows = 1
    n_cols = 3
    figsize = (19, 5.5) 
    cbar_orientation = "vertical"
    cbar_ax_rect = [0.93, 0.15, 0.015, 0.7] 
elif n_datasets == 2:
    n_rows = 1
    n_cols = 2
    figsize = (13, 5.5)
    cbar_orientation = "vertical"
    cbar_ax_rect = [0.92, 0.15, 0.02, 0.7]
elif n_datasets == 1:
    n_rows = 1
    n_cols = 1
    figsize = (8, 6) 
    cbar_orientation = "vertical"
    cbar_ax_rect = [0.86, 0.15, 0.03, 0.7] 
else: 
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols
    figsize = (6 * n_cols, 5 * n_rows + 1) 
    cbar_orientation = "horizontal"
    cbar_ax_rect = [0.15, 0.08, 0.7, 0.03] 


fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
axes_flat = axes.flatten()

# Using a known valid style, like 'seaborn-v0_8-whitegrid' or 'seaborn-v0_8-colorblind'
# 'seaborn-v0_8-colorblind' is excellent for accessibility and publications
plt.style.use('seaborn-v0_8-colorblind') 
# print(plt.style.available) # To list available styles if needed for debugging

for i, dataset_name in enumerate(dataset_names):
    ax = axes_flat[i]
    
    heatmap_data = []
    for hp_type in hyperparam_type_names:
        if dataset_name in data_sources[hp_type]:
            heatmap_data.append(data_sources[hp_type][dataset_name])
        else:
            heatmap_data.append([np.nan] * len(hyperparameter_values_list)) 
            
    df_heatmap = pd.DataFrame(heatmap_data, index=hyperparam_type_names, columns=hyperparameter_values_list)
    
    sns.heatmap(df_heatmap, ax=ax, annot=True, fmt=".3f", cmap="YlGnBu",
                linewidths=.5, linecolor='gray', cbar=False, 
                vmin=min_f1, vmax=max_f1, annot_kws={"size": 9})

    ax.set_title(f"Dataset: {dataset_name}", fontsize=14)
    ax.set_xlabel("Hyperparameter Value", fontsize=20)
    ax.set_ylabel("Hyperparameter Type", fontsize=20)
    ax.tick_params(axis='x', labelsize=20, rotation=0) 
    ax.tick_params(axis='y', labelsize=20, rotation=0)


for k in range(n_datasets, n_rows * n_cols):
    fig.delaxes(axes_flat[k])

source_ax_for_cbar = None # Initialize to None
if n_datasets > 0 : 
    for ax_s in axes_flat[:n_datasets]: 
        if len(ax_s.collections) > 0 : 
            source_ax_for_cbar = ax_s
            break
    
    if source_ax_for_cbar: # Proceed only if a valid source axes for colorbar is found
        cbar_ax = fig.add_axes(cbar_ax_rect) 
        cb = plt.colorbar(source_ax_for_cbar.collections[0], cax=cbar_ax, orientation=cbar_orientation)
        cb.set_label('Micro-F1 Score', fontsize=24)
        cb.ax.tick_params(labelsize=18)


# fig.suptitle('Heatmap of Hyperparameter Impact on Micro-F1 Score', fontsize=20, fontweight='bold')

if cbar_orientation == "vertical":
    # Adjust right boundary for vertical colorbar only if source_ax_for_cbar exists
    right_boundary = 0.90 if n_datasets > 0 and source_ax_for_cbar else 0.98
    fig.tight_layout(rect=[0.03, 0.03, right_boundary, 0.94]) 
else: # horizontal
    # Adjust bottom boundary for horizontal colorbar only if source_ax_for_cbar exists
    bottom_boundary = 0.12 if n_datasets > 0 and source_ax_for_cbar else 0.03
    fig.tight_layout(rect=[0.03, bottom_boundary, 0.97, 0.94]) 

plt.show()
plt.savefig("plots/heatmap_hyperparameter_impact.pdf", dpi=300, bbox_inches='tight')
# print("Heatmap visualization code executed with 'seaborn-v0_8-colorblind' style.")
# print("If in a suitable environment, the plot will be displayed.")
