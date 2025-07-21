import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
import understandingOU
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.ticker as mticker

def parse_edp_blocks(file_path):
    data = []
    with open(file_path, 'r') as f:
        rows, cols, edp = None, None, None
        for line in f:
            line = line.strip()
            if line.startswith("OU Size:"):
                size_str = line.split("OU Size:")[1].strip()
                rows, cols = map(int, size_str.split("x"))
            elif line.startswith("EDP"):
                parts = line.split(",")
                try:
                    edp_val = float(parts[-1])
                    data.append((rows, cols, edp_val))
                except:
                    pass  # Skip malformed EDP rows
    return pd.DataFrame(data, columns=["Rows", "Cols", "EDP"])

def plot_edp_heatmap_log(df_edp, save_path=None, log_scale: bool = True):
    """
    Plot a heatmap of EDP values on a logarithmic scale.

    Parameters:
    - df_edp: DataFrame containing 'Rows', 'Cols', and 'EDP' columns.
    """
    filename = os.path.basename(save_path)
    chipletName = filename.split("_OU")[0]

    pivot = df_edp.pivot(index="Rows", columns="Cols", values="EDP")

    # Replace or filter zeros/NaNs for LogNorm safety
    pivot_safe = pivot.replace(0, np.nan).dropna(how="all").dropna(axis=1, how="all")

    vmin = 1e-07
    vmax = 1e-05

    plt.figure(figsize=(12, 8))
    plt.title(f"{chipletName} OU")
    
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        heatmap = plt.imshow(pivot_safe, origin="lower", cmap="turbo", aspect="auto", norm=norm)
    else:
        heatmap = plt.imshow(pivot_safe, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax)

    # Colorbar formatting
    cbar = plt.colorbar(heatmap)
    ticks = [1e-7, 1e-6, 1e-5]
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{x:.1e} edp")
    )

    if chipletName == "Shared":
        # After computing pivot_safe
        col_labels = pivot_safe.columns
        row_labels = pivot_safe.index

        # Define tick frequency (e.g., every 8th label)
        x_ticks = np.arange(0, len(col_labels), 7)
        y_ticks = np.arange(0, len(row_labels), 7)

        plt.xticks(ticks=x_ticks, labels=col_labels[x_ticks], rotation=45)
        plt.yticks(ticks=y_ticks, labels=row_labels[y_ticks])
        annotations = understandingOU.generate_annotation_coords(start=28, step=64, max_dim=764)  # Adjust step for Shared chiplet
    elif chipletName == "Accumulator":
        # After computing pivot_safe
        col_labels = pivot_safe.columns
        row_labels = pivot_safe.index

        # Define tick frequency (e.g., every 2th label)
        x_ticks = np.arange(0, len(col_labels), 2)
        y_ticks = np.arange(0, len(row_labels), 2)

        plt.xticks(ticks=x_ticks, labels=col_labels[x_ticks], rotation=45)
        plt.yticks(ticks=y_ticks, labels=row_labels[y_ticks])
        annotations = understandingOU.generate_annotation_coords(start=12, step=24, max_dim=256)  # Adjust step for Accumulator chiplet
    else:
        plt.xticks(ticks=np.arange(len(pivot_safe.columns)), labels=pivot_safe.columns, rotation=45)
        plt.yticks(ticks=np.arange(len(pivot_safe.index)), labels=pivot_safe.index)

        if chipletName == "Adder":
            annotations = understandingOU.generate_annotation_coords(start=4, step=8, max_dim=64)
        else:
            annotations = understandingOU.generate_annotation_coords()
        

    for row, col in annotations:
        try:
            value = pivot_safe.loc[row, col]
            plt.text(pivot_safe.columns.get_loc(col), pivot_safe.index.get_loc(row),
                    f"{value:.2e}", color='black', ha='center', va='center', fontsize=8, fontweight='bold')
        except KeyError:
            continue  # In case the exact row/col is missing due to cl

    plt.xlabel("OU Columns")
    plt.ylabel("OU Rows")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

# Example usage
df_edp = parse_edp_blocks("HomoOULayerComputeResults/ADC_Less_OU_Sweep.csv")  # Replace with your actual path
plot_edp_heatmap_log(df_edp, save_path="OUHeatmaps/ADC_Less_OU_Heatmap.png")