import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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

def plot_edp_heatmap_log(df_edp, edp_max=None, save_path=None, shared=False):
    """
    Plot a heatmap of EDP values on a logarithmic scale.

    Parameters:
    - df_edp: DataFrame containing 'Rows', 'Cols', and 'EDP' columns.
    - edp_max: Optional float to override the maximum EDP value for the color scale.
    """
    pivot = df_edp.pivot(index="Rows", columns="Cols", values="EDP")

    # Replace or filter zeros/NaNs for LogNorm safety
    pivot_safe = pivot.replace(0, np.nan).dropna(how="all").dropna(axis=1, how="all")

    vmin = pivot_safe.min().min()
    vmax = pivot_safe.max().max()

    if edp_max is not None:
        vmax = edp_max

    # Guard against any remaining invalid vmin/vmax
    if vmin <= 0 or vmax <= 0:
        raise ValueError("LogNorm requires positive non-zero vmin and vmax")

    plt.figure(figsize=(12, 8))
    plt.title("Accumulator OU")
    norm = mcolors.LogNorm(vmin=vmin, vmax=edp_max if edp_max is not None else vmax)
    heatmap = plt.imshow(pivot_safe, origin="lower", cmap="viridis", aspect="auto", norm=norm)
    if shared:
        # After computing pivot_safe
        col_labels = pivot_safe.columns
        row_labels = pivot_safe.index

        # Define tick frequency (e.g., every 8th label)
        x_ticks = np.arange(0, len(col_labels), 7)
        y_ticks = np.arange(0, len(row_labels), 7)

        plt.xticks(ticks=x_ticks, labels=col_labels[x_ticks], rotation=45)
        plt.yticks(ticks=y_ticks, labels=row_labels[y_ticks])
    else:
        plt.xticks(ticks=np.arange(len(pivot_safe.columns)), labels=pivot_safe.columns, rotation=45)
        plt.yticks(ticks=np.arange(len(pivot_safe.index)), labels=pivot_safe.index)
    plt.xlabel("OU Columns")
    plt.ylabel("OU Rows")
    plt.colorbar(heatmap, label="EDP (log scale)")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

# Example usage
df_edp = parse_edp_blocks("HomoOULayerComputeResults/Accumulator_OU_Sweep.csv")  # Replace with your actual path
plot_edp_heatmap_log(df_edp, edp_max=1, save_path="OUHeatmaps/Accumulator_OU_Heatmap.png", shared=True)