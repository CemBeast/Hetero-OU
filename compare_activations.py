import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WORKLOAD_DIR = "./workloads"
POLY_DEGREE = 3  # degree for trendline fitting

def load_all_workloads():
    all_paths = sorted(glob.glob(os.path.join(WORKLOAD_DIR, "*.csv")))
    datasets = []

    for path in all_paths:
        df = pd.read_csv(path).sort_values("Layer #")
        x = df["Layer #"].values
        y = df["Activations(KB)"].values
        name = os.path.splitext(os.path.basename(path))[0].replace("_stats", "").title()
        datasets.append((x, y, name))
    
    return datasets

def main():
    datasets = load_all_workloads()

    if not datasets:
        print("No CSV files found in workloads directory.")
        return

    # Combine all x and y values for the global fit
    x_all = np.concatenate([x for x, _, _ in datasets])
    y_all = np.concatenate([y for _, y, _ in datasets])

    # Fit a single polynomial to all data
    coeffs = np.polyfit(x_all, y_all, POLY_DEGREE)
    poly = np.poly1d(coeffs)
    xs_fit = np.linspace(x_all.min(), x_all.max(), 300)

    # Plotting
    plt.figure(figsize=(10, 6))

    for x, y, name in datasets:
        plt.plot(x, y, label=name)

    plt.plot(xs_fit, poly(xs_fit), 'k--', label="Trend Line")

    # X-axis as spaced integer ticks
    x_min = int(x_all.min())
    x_max = int(x_all.max())
    total_layers = x_max - x_min + 1

    # Auto-determine spacing based on total number of layers
    step = 1 if total_layers <= 20 else 2 if total_layers <= 40 else 5 if total_layers <= 80 else 10
    xticks = np.arange(x_min, x_max + 1, step)
    plt.xticks(xticks)

    # Labels & title
    plt.xlabel("Layer #")
    plt.ylabel("Activations (KB)")
    plt.title("Amount of Activation Trend Among Different Neural Network")

    # Legend with custom title
    legend = plt.legend(title="Activation Trend Among Different Neural Networks")
    plt.setp(legend.get_title(), fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()