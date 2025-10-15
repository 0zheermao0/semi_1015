import matplotlib.pyplot as plt
import numpy as np

def plot_radar_matplotlib(data_values, axis_labels, title, polygon_color, y_max_value, ax):
    """
    Generates a single radar chart on a given matplotlib Axes object.

    Args:
        data_values (list): List of numerical scores for the axes.
        axis_labels (list): List of strings for the axis labels.
        title (str): Title for the chart.
        polygon_color (str): Hex color string for the polygon (e.g., '#F39D51').
        y_max_value (float): The maximum value for the radial axis, used for scaling grid lines.
        ax (matplotlib.axes.Axes): The Axes object to plot on.
    """
    num_vars = len(axis_labels)

    # Calculate angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The plot is circular, so we need to "complete the loop"
    data_values_closed = list(data_values) + [data_values[0]]
    angles_closed = list(angles) + [angles[0]]

    # Polar plot settings
    ax.set_theta_offset(np.pi / 2)  # Start at the top
    ax.set_theta_direction(-1)    # Clockwise

    # Set axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=22) # Corresponds to R's axis.label.size (approx)
                                                # R's axis.label.offset is harder to match directly,
                                                # use ax.tick_params(pad=...) if needed.

    # Radial (y) axis settings
    # Corresponds to R's values.radar = c("", "", "") and gridline colors
    # We'll set 3 grid lines: min, mid, max
    # Assuming max is y_max_value, min and mid are fractional
    grid_values = [y_max_value * 0.33, y_max_value * 0.66, y_max_value]
    ax.set_yticks(grid_values)
    ax.set_yticklabels(["", "", ""]) # No labels on radial axis ticks

    ax.set_ylim(0, y_max_value)

    # Styling from R code
    # background.circle.colour = "white" / panel.background = element_blank()
    ax.set_facecolor("white")

    # grid.line.width = 0.2
    # gridline.min.colour = "grey", gridline.mid.colour = "grey"
    # axis.line.colour = "grey" (radial lines)
    ax.yaxis.grid(True, linestyle='solid', color='grey', linewidth=0.2) # Concentric grid lines
    ax.xaxis.grid(True, linestyle='solid', color='grey', linewidth=0.2) # Radial grid lines

    # gridline.max.colour = "black" (outermost grid line)
    # The spine acts as the outermost grid line in this setup
    ax.spines['polar'].set_edgecolor('black')
    ax.spines['polar'].set_linewidth(0.2)


    # Plot data
    # group.line.width = 0.5
    # group.point.size = 0.6 (very small in R, using a small marker size)
    ax.plot(angles_closed, data_values_closed, color=polygon_color, linewidth=0.5, linestyle='solid', marker='o', markersize=2) # markersize is approx to R's point.size

    # fill = TRUE, fill.alpha = 0.4
    ax.fill(angles_closed, data_values_closed, polygon_color, alpha=0.4)

    # Title
    # plot.title = element_text(size = 8, vjust = 1, hjust = 0.5, color="black")
    # Matplotlib's title positioning is slightly different.
    # `pad` can simulate `vjust`. `loc='center'` for `hjust=0.5`.
    ax.set_title(title, size=16*3, color="black", va='top', y=-0.2) # y adjusts vertical position

    # Remove outer frame unless it's the polar spine we styled
    # ax.spines['polar'].set_visible(True) # Ensure it's visible if other spines are off

def main():
    # --- User Data ---
    # Define your LLM names (these will be the axes of the radar chart)
    llm_names = ["qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "gemma3-4b", "gemma3-12b"]
    num_axes = len(llm_names)

    # Performance scores of "Your Method"
    # Replace these with your actual performance data.
    # Assuming scores are on a scale, e.g., 0-100 or 0-1.
    # Let's assume a max possible score of 100 for this example.
    performance_max_value = 80 # Adjust if your scale is different (e.g., 1.0 for normalized scores)

    # Data for "Your Method on Dataset A->C"
    # These are example values. Replace with your actual scores for each LLM.
    method_scores_dataset_ac = [71, 76, 77, 70.7, 76] # Scores for LLM1, LLM2, ..., LLM5

    # Data for "Your Method on Dataset A->D"
    # These are example values. Replace with your actual scores for each LLM.
    method_scores_dataset_ad = [68, 71, 72, 67, 73] # Scores for LLM1, LLM2, ..., LLM5

    # Check if data length matches number of axes
    if len(method_scores_dataset_ac) != num_axes or len(method_scores_dataset_ad) != num_axes:
        raise ValueError(f"Number of scores must match the number of LLMs ({num_axes}).")

    # --- Plotting ---
    # Create a figure with two subplots (side-by-side)
    # This mimics the layout of multiple charts in your example.
    # If you want more charts (e.g., to match the 3 in example, add another dataset/method)
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'})
    # If you have 3 items to plot like the example:
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'})

    # For 2 plots:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), subplot_kw={'projection': 'polar'})
    if num_axes == 0 : # Should not happen with llm_names defined
        print("No axes defined.")
        return
    if not isinstance(axs, np.ndarray): # If only one subplot, axs is not an array
        axs = [axs]


    # Plot for Dataset A->C
    plot_radar_matplotlib(
        data_values=method_scores_dataset_ac,
        axis_labels=llm_names,
        title="(a) A->C",
        polygon_color='#F39D51', # Orange, similar to PG3:High
        y_max_value=performance_max_value,
        ax=axs[0]
    )

    # Plot for Dataset A->D
    plot_radar_matplotlib(
        data_values=method_scores_dataset_ad,
        axis_labels=llm_names,
        title="(b) A->D",
        polygon_color='#83BDE3', # Blue, similar to PG2:Low
        y_max_value=performance_max_value,
        ax=axs[1]
    )

    # Example for a third plot (if you have data for it, e.g., another method or dataset)
    # method_scores_dataset_x = [50, 60, 70, 65, 75]
    # plot_radar_matplotlib(
    #     data_values=method_scores_dataset_x,
    #     axis_labels=llm_names,
    #     title="My Method - Dataset X",
    #     polygon_color='#78C679', # Green, similar to PG1:Low
    #     y_max_value=performance_max_value,
    #     ax=axs[2] # Assuming you changed subplots to (1,3)
    # )


    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(pad=2.0) # Add some padding

    # Set overall figure background to white (R: plot.background = element_blank())
    fig.patch.set_facecolor('white')

    plt.show()
    plt.savefig("plots/radar_chart.pdf", dpi=300, bbox_inches='tight') # Save the figure

if __name__ == '__main__':
    main()