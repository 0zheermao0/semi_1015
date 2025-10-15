# Author: Joey
# Email: zengjiayi666@gmail.com
# Date: :call strftime("%Y-%m-%d %H:%M") # Original date placeholder
# Description: Generates a bar chart with an overlaid line plot representing the average of bar groups.
# Version: 1.2 (Legend moved inside plot)

import matplotlib.pyplot as plt
import numpy as np
import datetime # Added for strftime
import os

# --- Data Estimation ---
# Data for Chart (a): Llama Models
labels_a = ['3', '4', '5', '6', '7']
ac_a_values = np.array([0.75762, 0.76508, 0.77023, 0.76564, 0.77333])
ad_a_values = np.array([0.70711, 0.71232, 0.71717, 0.72111, 0.70522])
ca_a_values = np.array([0.70611, 0.70535, 0.70816, 0.71212, 0.71111])
cd_a_values  = np.array([0.733, 0.72122, 0.72654, 0.73031, 0.73511])
data_a = [ac_a_values, ad_a_values, ca_a_values, cd_a_values]

# Calculate average for each group in Chart (a)
avg_a_values = np.mean(data_a, axis=0)

# Data for Chart (b): Expert Number
labels_b = ['3', '6', '9', '12', '15']
ac_b_values = np.array([0.75762, 0.77635, 0.77265, 0.77536, 0.76631])
ad_b_values = np.array([0.70711, 0.71298, 0.71536, 0.72191, 0.71491])
ca_b_values = np.array([0.70611, 0.71411, 0.71321, 0.71839, 0.71977])
cd_b_values  = np.array([0.733, 0.74200, 0.73812, 0.73519, 0.72111])
data_b = [ac_b_values, ad_b_values, ca_b_values, cd_b_values]

# Calculate average for each group in Chart (b)
avg_b_values = np.mean(data_b, axis=0)

# --- Plotting Configuration ---
# Colors (approximating the image's blue shades from light to dark)
colors = ['#a1c9f4',  # For qx (lightest)
          '#72aee6',  # For qy
          '#4a8dd8',  # For qz
          '#2c6db5']  # For q (darkest)
line_color = '#907caa' # Color for the average line
line_marker = 'o' # Marker for the average line points

bar_group_names = ['$A->C$', '$A->D$', '$C->A$', '$C->D$']
num_bar_groups = len(bar_group_names)
bar_width = 0.18  # Width of a single bar
# group_padding = 0.8 # This variable was defined but not used

# --- Create Figure and Axes ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Adjusted figsize for better layout

# --- Plot (a): Llama Models ---
ax1 = axes[0]
x_a_indices = np.arange(len(labels_a))

# Calculate offsets for grouped bars
total_width_for_group = bar_width * num_bar_groups
# Start position of the first bar in a group
start_offset = -total_width_for_group / 2 + bar_width / 2

for i in range(num_bar_groups):
    offset = start_offset + i * bar_width
    ax1.bar(x_a_indices + offset, data_a[i], bar_width, label=bar_group_names[i], color=colors[i], zorder=2) # Set zorder for bars

# Plot the average line for Chart (a)
ax1.plot(x_a_indices, avg_a_values, color=line_color, marker=line_marker, linestyle='-', linewidth=2, label='Avg.', zorder=3) # Set zorder for line

ax1.set_ylabel('Micro F1', fontsize=14) # Increased font size
ax1.set_xlabel('(a) Expert Number (Types)', fontsize=22) # Increased font size
ax1.set_xticks(x_a_indices)
ax1.set_xticklabels(labels_a, fontsize=18) # Increased font size
ax1.set_ylim(0.6, 0.8) # Adjusted Y limit to better visualize differences
ax1.set_yticks(np.arange(0.6, 0.81, 0.05)) # Adjusted Y ticks (0.81 to include 0.8)
ax1.tick_params(axis='y', labelsize=18) # Increased font size
# --- MODIFIED LEGEND PLACEMENT ---
# Moved legend inside the plot, adjusted ncol and fontsize for better fit
ax1.legend(loc='upper left', ncol=3, fontsize=12, frameon=True, framealpha=0.8)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7, zorder=1) # Ensure grid is behind bars and line
ax1.set_axisbelow(True)

# --- Plot (b): Expert Number ---
ax2 = axes[1]
x_b_indices = np.arange(len(labels_b))

for i in range(num_bar_groups):
    offset = start_offset + i * bar_width
    ax2.bar(x_b_indices + offset, data_b[i], bar_width, label=bar_group_names[i], color=colors[i], zorder=2) # Set zorder for bars

# Plot the average line for Chart (b)
ax2.plot(x_b_indices, avg_b_values, color=line_color, marker=line_marker, linestyle='-', linewidth=2, label='Avg.', zorder=3) # Set zorder for line

ax2.set_ylabel('Micro F1', fontsize=14) # Increased font size
ax2.set_xlabel('(b) Expert Number (Instances)', fontsize=22) # Increased font size
ax2.set_xticks(x_b_indices)
ax2.set_xticklabels(labels_b, fontsize=18) # Increased font size
ax2.set_ylim(0.6, 0.8) # Adjusted Y limit to better visualize differences
ax2.set_yticks(np.arange(0.6, 0.81, 0.05)) # Adjusted Y ticks (0.81 to include 0.8)
ax2.tick_params(axis='y', labelsize=18) # Increased font size
# --- MODIFIED LEGEND PLACEMENT ---
# Moved legend inside the plot, adjusted ncol and fontsize for better fit
# Using 'best' for this plot as an alternative, or stick to 'upper left'
ax2.legend(loc='upper left', ncol=3, fontsize=12, frameon=True, framealpha=0.8) # Consistent 'upper left'
ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7, zorder=1) # Ensure grid is behind bars and line
ax2.set_axisbelow(True)

# --- Final Adjustments ---
# Adjusted rect to [0, 0, 1, 1] as the legend is now inside.
# pad still provides overall padding for the figure.
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95]) # Keeping original rect for now, can be changed to [0,0,1,1] if no suptitle

# Add a main title to the figure
# Get current date and time for the title
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
# fig.suptitle(f'Analysis of Model Performance - {current_time}', fontsize=16) # Uncomment if you want a main title

# --- Save and Show ---
# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

file_path = f'plots/bar_chart_with_avg_line.pdf'
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {file_path}")
plt.show()
