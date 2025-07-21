import os
import numpy as np
import pandas as pd
import math
import hashlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, LogLocator

XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16

chipletSpecs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 64.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}
# Tops should be multiplied by 10e12 

chipletTypesDict = {
    "Standard":    {"Size": 16384,  "Bits/cell": 2, "TOPS": 30e12,  "Energy/MAC": 0.87e-12},
    "Shared":      {"Size": 583696, "Bits/cell": 1, "TOPS": 27e12,  "Energy/MAC": 0.30e-12},
    "Adder":       {"Size": 4096,   "Bits/cell": 1, "TOPS": 11e12,  "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536,  "Bits/cell": 2, "TOPS": 35e12,  "Energy/MAC": 0.22e-12},
    "ADC_Less":    {"Size": 163840,  "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12},
}

def customizeOU(row: int, col: int, chipletName: str):
    baseRow, baseCol = chipletSpecs[chipletName]["base"]
    epm = chipletSpecs[chipletName]["energy_per_mac"]
    rowKnob = chipletSpecs[chipletName]["rowKnob"]
    rowKnobPercent = rowKnob / 100.0  # Convert to fraction
    colKnob = chipletSpecs[chipletName]["colKnob"]
    colKnobPercent = colKnob / 100.0  # Convert to fraction

    EnergyRow = epm *  rowKnobPercent * (row / baseRow)
    EnergyCol = epm *  colKnobPercent * (col / baseCol)

    # Energy per mac is Energy Total
    EnergyTotal = EnergyRow + EnergyCol 
    tops = chipletSpecs[chipletName]["tops"] * (row / baseRow) * (col / baseCol) # Adjust TOPS

    powerDensity = EnergyTotal * tops * 1e12  # Convert to pJ (picojoules)
    # Not focusing on tops for now, it is not scaled by 10e12 also. 
    return EnergyTotal, tops, powerDensity



# Writes the OU stats to a CSV file inside OU_Data_Tables folder
def writeOUFile(maxRow: int, maxCol: int, chipletName: str, folder: str = "OU_Data_Tables"):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{chipletName}_OU_Stats.csv")
    with open(filename, "w") as file:
        file.write("Rows,Cols,Energy,TOPS,Time,Power Density\n")
        for i in range(4, maxRow + 1, 4):
            for j in range(4, maxCol + 1, 4):
                energy, tops, power_density = customizeOU(i, j, chipletName)
                time = simulateMACoperation(i, j, 80123456, chipletName)  # Simulate MAC operation time
                file.write(f"{i},{j},{energy:.5e},{tops:.5e},{time:.5e},{power_density:.5e}\n")
    print(f"Saved: {filename}")
    return filename

def readOUFileAndConstrainByPowerDensity(file_path: str, power_density_threshold: float = 8):
    df = pd.read_csv(file_path)
    filtered = df[df["Power Density"] <= power_density_threshold]
    folder = os.path.dirname(file_path)
    base = os.path.basename(file_path.replace(".csv", ""))
    new_filename = f"{base}_Filtered_PD{power_density_threshold}.csv"
    new_path= os.path.join(folder, new_filename)
    filtered.to_csv(new_path, index=False)
    print(f"Filtered data saved to: {new_path}")
    return filtered

def plotPowerDensityHeatmap(file_path: str, chipletName: str, log_scale: bool = False):
    df = pd.read_csv(file_path)
    pivot = df.pivot(index="Rows", columns="Cols", values="Power Density")

    # Clean up zeros/NaNs
    pivot_safe = pivot.replace(0, np.nan).dropna(how="all").dropna(axis=1, how="all")

    # vmin = pivot_safe.min().min()
    # vmax = pivot_safe.max().max()
    vmin = 1e-4 # 0.0001 W
    vmax = 40 # 40 W
    if log_scale and (vmin <= 0 or vmax <= 0):
        raise ValueError("Log scale requires all values to be positive and non-zero.")

    plt.figure(figsize=(12, 8))
    plt.title(f"{chipletName} Power Density Heatmap")

    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        heatmap = plt.imshow(pivot_safe, origin="lower", cmap="turbo", aspect="auto", norm=norm)
    else:
        heatmap = plt.imshow(pivot_safe, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax)

    # Colorbar formatting
    cbar = plt.colorbar(heatmap)
    # Force tick placement
    # Force ticks only at powers of 10, but manually add 30W
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 30]
    cbar.set_ticks(ticks)

    # Force readable tick labels
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{x:.1f}W" if x >= 1 else f"{x*1000:.1f}mW")
    )

    # Axis ticks
    if chipletName == "Shared":
        # After computing pivot_safe
        col_labels = pivot_safe.columns
        row_labels = pivot_safe.index

        # Define tick frequency (e.g., every 8th label)
        x_ticks = np.arange(0, len(col_labels), 7)
        y_ticks = np.arange(0, len(row_labels), 7)

        plt.xticks(ticks=x_ticks, labels=col_labels[x_ticks], rotation=45)
        plt.yticks(ticks=y_ticks, labels=row_labels[y_ticks])
        annotations = generate_annotation_coords(start=28, step=64, max_dim=764)  # Adjust step for Shared chiplet
    elif chipletName == "Accumulator":
        # After computing pivot_safe
        col_labels = pivot_safe.columns
        row_labels = pivot_safe.index

        # Define tick frequency (e.g., every 8th label)
        x_ticks = np.arange(0, len(col_labels), 2)
        y_ticks = np.arange(0, len(row_labels), 2)

        plt.xticks(ticks=x_ticks, labels=col_labels[x_ticks], rotation=45)
        plt.yticks(ticks=y_ticks, labels=row_labels[y_ticks])
        annotations = generate_annotation_coords(start=12, step=24, max_dim=256)  # Adjust step for Accumulator chiplet
    else:
        plt.xticks(ticks=np.arange(len(pivot_safe.columns)), labels=pivot_safe.columns, rotation=45)
        plt.yticks(ticks=np.arange(len(pivot_safe.index)), labels=pivot_safe.index)
        # Annotate specific cells
        if chipletName == "Adder":
            annotations = generate_annotation_coords(start=4, step=8, max_dim=64)
        else:
            annotations = generate_annotation_coords()


    plt.xlabel("Cols")
    plt.ylabel("Rows")

    for row, col in annotations:
        try:
            value = pivot_safe.loc[row, col]
            if value > 0.3:
                plt.text(pivot_safe.columns.get_loc(col), pivot_safe.index.get_loc(row),
                    f"{value:.1f}W", color='black', ha='center', va='center', fontsize=8, fontweight='bold')
            elif value <= 0.3 and value > 0:
                value *= 1000  # Convert to mW for better visibility
                plt.text(pivot_safe.columns.get_loc(col), pivot_safe.index.get_loc(row),
                    f"{value:.1f}mW", color='black', ha='center', va='center', fontsize=8, fontweight='bold')
            else:
                plt.text(pivot_safe.columns.get_loc(col), pivot_safe.index.get_loc(row),
                    "N/A", color='black', ha='center', va='center', fontsize=8, fontweight='bold')
        except KeyError:
            continue  # In case the exact row/col is missing due to cl

    plt.tight_layout()

    plt.savefig(f"OUHeatmaps/{chipletName}_OU_Power_Density.png", dpi=300)

    plt.show()



def generate_annotation_coords(start=8, step=12, max_dim=128):
    """
    Generates 24 (row, col) coordinate pairs for annotation,
    spaced evenly within the range of a 4-128 OU heatmap grid.

    Returns:
        List of (row, col) tuples
    """
    coords = []
    for r in range(start, max_dim + 1, step):
        for c in range(start, max_dim + 1, step):
            coords.append((r, c))
    return coords


def simulateMACoperation(row: int, col: int, numMACs: int, chipletName: str):
    baseRow, baseCol = chipletSpecs[chipletName]["base"]
    TOPS = chipletSpecs[chipletName]["tops"] * (row / baseRow) * (col / baseCol) # Adjust TOPS
    TOPS *= 1e12  # Convert to TOPS (trillions of operations per second)
    time = numMACs / TOPS  # Time in seconds
    return time
    

def get_approx_foldable_factors(n, min_row, max_row, max_col, step=16):
    """
    Finds (row, col) pairs such that:
    - row ≥ min_row and ≤ max_row
    - col ≤ max_col
    - row and col are multiples of `step`
    - row * col ≥ n
    """
    factors = []

    # Step 1: Adjust min_row to next multiple of `step`
    if min_row % step != 0:
        min_row = ((min_row // step) + 1) * step

    # Step 2: Try rows from min_row to max_row in steps of `step`
    for row in range(min_row, max_row + 1, step):
        raw_col = math.ceil(n / row)

        # Step 3: Round up col to next multiple of `step`
        if raw_col % step != 0:
            col = ((raw_col // step) + 1) * step
        else:
            col = raw_col

        # Step 4: Check if it fits within constraints
        if col <= max_col:
            factors.append((row, col))

    return factors

def select_lowest_row_config(factor_list):
    """
    Selects the config with the smallest row value.
    If multiple have the same row, returns the one with the smallest column.
    """
    if not factor_list:
        return None
    return min(factor_list, key=lambda x: (x[0], x[1]))

# Pareto-based rank seletion function
def rank_based_selection(configs):
    if not configs:
        print("No valid OU configurations found for ranking.")
        return None  # gracefully handle empty input
    
    # Rank by latency (lower is better)
    latency_sorted = sorted(configs, key=lambda x: x["latency"])
    for i, cfg in enumerate(latency_sorted):
        cfg["latency_rank"] = i

    # Rank by energy (lower is better)
    energy_sorted = sorted(configs, key=lambda x: x["energy"])
    for i, cfg in enumerate(energy_sorted):
        cfg["energy_rank"] = i

    # Combine ranks (lower sum is better balance)
    for cfg in configs:
        cfg["total_rank"] = cfg["latency_rank"] + cfg["energy_rank"]

    # Return config with lowest total rank
    best_config = min(configs, key=lambda x: x["total_rank"])
    return best_config

def computeCrossbarMetrics(chipletName: str, workloadStatsCSV: str):
    df = pd.read_csv(workloadStatsCSV)

    results = []
    layer = 0
    for _, row in df.iterrows():
        layer += 1
        weightsKB = row["Weights(KB)"]
        macs = row["MACs"]
        weightSparsity = row["Weight_Sparsity(0-1)"]
        activationSparsity = row["Activation_Sparsity(0-1)"]
        activationsKB = row["Activations(KB)"]

        # later will change it to accept parameters for row and col
        baseOURow = chipletSpecs[chipletName]["base"][0]
        baseOUCol = chipletSpecs[chipletName]["base"][1]

        # For trying trial and error 
        ouRow = 100
        ouCol = 100

        crossbarsReq = math.ceil(weightsKB * 1024 * 8 / (chipletTypesDict[chipletName]["Size"] * chipletTypesDict[chipletName]["Bits/cell"]))
        minRequiredCrossbars = math.ceil(math.sqrt(chipletTypesDict[chipletName]["Size"]* (1 - weightSparsity)))
        MACSperCrossbar = math.ceil(macs / crossbarsReq)
        activationsPerCrossbar = (activationsKB * 1024 * 8 * (1 - activationSparsity)) / crossbarsReq

        # Step 1, get required row for accounting for activation sparsity
        rowReq = math.ceil(baseOURow * (1 - activationSparsity))
        
        # Step 2, find required OU dimensions
        adjustedOUDimensionReq = math.ceil(baseOUCol * baseOURow * (1 - weightSparsity))
        # Step 3, get array of foldable factors to select from
        factors = get_approx_foldable_factors(adjustedOUDimensionReq, rowReq, baseOURow, baseOUCol)
        # Step 4, select config with lowest rows
        minCrossbarDimensions = select_lowest_row_config(factors)

        colReq = math.ceil(adjustedOUDimensionReq / rowReq)

        step = 4
        colLimit = 32 # FOR IR LIMITATION, only for Standard 
        possibleOUConfigs = []
        ## Now we want to select optimial OU such that latency and energy is minimized
        with open("workload_OU_config_results.txt", "w") as f:
            for ou_row in range(step, minCrossbarDimensions[0]+1, step):
                for ou_col in range(step, colLimit+1, step):
                    OUrequired = math.ceil(minCrossbarDimensions[0] * minCrossbarDimensions[1]) / (ou_row * ou_col)
                    latency  = (((4*ou_row) + ou_col) * OUrequired)
                    energy = ((chipletSpecs[chipletName]["rowKnob"] * ou_row) + (chipletSpecs[chipletName]["colKnob"] * ou_col)) * OUrequired

                    epm, tops, power_density = customizeOU(ou_row, ou_col, chipletName)

                    # Check if power density is within acceptable limits
                    if power_density > 8.0:
                        continue  # Skip this config entirely, do not want to include configs that exceed power density


                    # print(f"OU Row: {ou_row}, OU Col: {ou_col}, Latency: {latency}, Energy: {energy}, Power Density: {power_density:.2f} W, TOPS: {tops:.2e}, EPM: {epm:.2e}")
                    f.write(f"OU Row: {ou_row}, OU Col: {ou_col}, Latency: {latency}, Energy: {energy}, Power Density: {power_density:.2f} W, TOPS: {tops:.2e}, EPM: {epm:.2e}\n")
                    possibleOUConfigs.append({
                        "ou_row": ou_row,
                        "ou_col": ou_col,
                        "latency": latency,
                        "energy": energy,
                        "power_density": power_density
                    })
                    
        best_config = rank_based_selection(possibleOUConfigs)
        OU_cycles = math.ceil(( baseOUCol * baseOURow * (1 - weightSparsity) * (1 - activationSparsity)) / (ouRow * ouCol))
        cyclyes_latency = ouCol * math.log2(ouRow) * OU_cycles

        # not needed for now, but can be used later
        energy = crossbarsReq * math.log2(ouRow) * ouRow * ouCol * OU_cycles

        results.append({
            "layer": layer,
            "Weight_Sparsity": weightSparsity,
            "crossbars_required": crossbarsReq ,
            "min_required_crossbars": f"{minRequiredCrossbars} x {minRequiredCrossbars}",
            "MACs_per_crossbar": MACSperCrossbar,
            "activations_per_crossbar": activationsPerCrossbar,
            "Minimum_row_Req": rowReq,
            "Minimum OU Dimension Required": adjustedOUDimensionReq,
            "cycles": OU_cycles,
            "Latency": cyclyes_latency,
            "ou_row": ouRow,
            "ou_col": ouCol,
            "chiplet_name": chipletName,
            "Factors": factors,
            "Minimum Crossbar Dimensions": f"{rowReq} x {colReq}",
            "Minimum Crossbar Dimensions Rounded": minCrossbarDimensions,
            "Rank Based Pareto Config": best_config,

        })

    return results



def computeCrossbarMetricsSweepOnOneLayer(chipletName: str, workloadStatsCSV: str, layer_index: int = 0):
    df = pd.read_csv(workloadStatsCSV)

    if layer_index >= len(df):
        raise IndexError(f"Layer index {layer_index} is out of range. CSV has {len(df)} rows.")

    row = df.iloc[layer_index]

    weightsKB = row["Weights(KB)"]
    macs = row["MACs"]
    weightSparsity = row["Weight_Sparsity(0-1)"]
    activationSparsity = row["Activation_Sparsity(0-1)"]
    activationsKB = row["Activations(KB)"]


    configs = []

    crossbarsReq = math.ceil(weightsKB * 1024 * 8 / (chipletTypesDict[chipletName]["Size"] * chipletTypesDict[chipletName]["Bits/cell"]))
    minRequiredCrossbars = math.ceil(math.sqrt(chipletTypesDict[chipletName]["Size"]* (1 - weightSparsity)))
    MACSperCrossbar = math.ceil(macs / crossbarsReq)
    activationsPerCrossbar = (activationsKB * 1024 * 8 * (1 - activationSparsity)) / crossbarsReq

    layer_stats = {
        "layer": layer_index + 1,
        "crossbars_required": crossbarsReq,
        "min_required_crossbars": f"{minRequiredCrossbars} x {minRequiredCrossbars}",
        "MACs_per_crossbar": MACSperCrossbar,
        "activations_per_crossbar": activationsPerCrossbar,
    }

    baseOURow = chipletSpecs[chipletName]["base"][0]
    baseOUCol = chipletSpecs[chipletName]["base"][1]
    for row in range(4, baseOURow + 1, 4):
        for col in range(4, baseOUCol + 1, 4):
            ouRow = row
            ouCol = col 
            
            OU_cycles = math.ceil((baseOUCol * baseOURow * (1 - weightSparsity) * (1 - activationSparsity)) / (ouRow * ouCol))
            cycles_latency = ouCol * math.log2(ouRow) * OU_cycles

            # Energy formula from Odin
            energy = crossbarsReq * math.log2(ouRow) * ouRow * ouCol * OU_cycles

            # Use the tops / energy to get edp of the OU for the given layer
            energy, tops, power_density = customizeOU(ouRow, ouCol, chipletName)

            configs.append({
                "Latency": cycles_latency,
                "Cycles": OU_cycles,
                "ou_row": ouRow,
                "ou_col": ouCol,
                "Energy_per_Mac": energy,
                "TOPS": tops,
                "Power Density": power_density,
            })

    return configs, layer_stats

def sweepFixedOUConfigsAcrossAllLayers(chipletName: str, workloadStatsCSV: str, power_density_threshold: float = 8.0, step = 4):
    df = pd.read_csv(workloadStatsCSV)
    num_layers = len(df)

    baseRow = chipletSpecs[chipletName]["base"][0]
    baseCol = chipletSpecs[chipletName]["base"][1]

    results = []

    for ouRow in range(step, baseRow + 1, step):
        for ouCol in range(step, baseCol + 1, step):
            total_latency = 0
            valid_layers = 0
            skip = False

            for i in range(num_layers):
                row = df.iloc[i]
                weightsKB = row["Weights(KB)"]
                macs = row["MACs"]
                weightSparsity = row["Weight_Sparsity(0-1)"]
                activationSparsity = row["Activation_Sparsity(0-1)"]

                #### WORK HERE 
                # rowReq = math.ceil(baseRow * (1- activationSparsity))

                # Estimate cycles and latency for this OU config
                OU_cycles = math.ceil((baseRow * baseCol * (1 - weightSparsity) * (1 - activationSparsity)) / (ouRow * ouCol))
                if chipletName == "Shared":
                    cycles_latency = ouCol * math.log2(ouRow) * OU_cycles
                ## NEED TO VERIFY, BUT SHARED DOES NOT HAVE ADC EACH BITLINE SO WE SCALE BY COLS
                else:
                    cycles_latency = math.log2(ouRow) * OU_cycles

                # Energy and power density
                energy, tops, power_density = customizeOU(ouRow, ouCol, chipletName)

                if power_density > power_density_threshold:
                    skip = True
                    break  # Skip this config entirely

                total_latency += cycles_latency
                valid_layers += 1

            if not skip and valid_layers == num_layers:
                results.append({
                    "ou_row": ouRow,
                    "ou_col": ouCol,
                    "Latency": total_latency,
                    "Energy_per_Mac": energy,
                    "TOPS": tops,
                    "Power Density": power_density,
                })

    return results

# Saves the Configs for a given layer to a CSV file
def saveConfigsToCSV(configs, layer_stats):
    # Define the output file path
    output_folder = "OU_Filtered_CSVs"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"Layer{layer_stats['layer']}_Filtered_OU.csv")

    # Save to CSV
    df_filtered = pd.DataFrame(configs)
    df_filtered = df_filtered[["ou_row", "ou_col", "Latency", "Power Density", "TOPS", "Energy_per_Mac"]]
    df_filtered.to_csv(output_file, index=False)

    print(f"\n✅ Filtered OU configs saved to: {output_file}")
    return output_file

# Configs should be filtered by power density before calling this function
def saveWorkloadConfigsToCSV(configs, chipletName, step):
    output_folder = f"OU_Workload_Filtered_CSVs_step={step}"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{chipletName}_Workload_Filtered_OU_step={step}.csv")

    # Save to CSV
    df_filtered = pd.DataFrame(configs)
    df_filtered = df_filtered[["ou_row", "ou_col", "Latency", "Power Density", "TOPS", "Energy_per_Mac"]]
    df_filtered.to_csv(output_file, index=False)

    print(f"\n✅ Filtered OU configs saved to: {output_file}")
    return output_file

## for mapping colors to specific configurations
def get_color_for_config(config_str):
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    hash_val = int(hashlib.md5(config_str.encode()).hexdigest(), 16)
    return colors[hash_val % len(colors)]

def plotTopLatencyConfigs(configs, chipletName, step, top_n=10):
    # Filter and sort by lowest latency
    sorted_configs = sorted(configs, key=lambda x: x["Latency"])
    top_configs = sorted_configs[:top_n]

    # Extract values for plotting
    labels = [f"{c['ou_row']}x{c['ou_col']}" for c in top_configs]
    latencies = [c["Latency"] for c in top_configs]
    power_densities = [c["Power Density"] for c in top_configs]
    colors = [get_color_for_config(label) for label in labels]


    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, latencies, color=colors, edgecolor='black')
    plt.xlabel("OU Config (rows x cols)", fontweight='bold')
    plt.ylabel("Latency (cycles)", fontweight='bold')
    plt.title(f"{chipletName} Top {top_n} by Latency (PD < 8W) on VGG16 (OU Step={step})", fontweight='bold')

    # Annotate each bar with power density
    for bar, pd in zip(bars, power_densities):
        height = bar.get_height()
        if pd > 0.3:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{pd:.1f}W", 
                 ha='center', va='bottom', fontsize=9, color='black')
        else:
            pd_mw = pd * 1000
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{pd_mw:.1f}mW", 
                 ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout()
    plt.savefig(f"OU_Workload_Filtered_CSVs_step={step}/{chipletName}_Top_Latency_Configs.png", dpi=300)
    plt.show()

def compare_reciprocal_configs(csv_path):
    df = pd.read_csv(csv_path)

    # Create a dictionary for quick lookup of reciprocal configs
    config_map = {
        (int(row["ou_row"]), int(row["ou_col"])): row for _, row in df.iterrows()
    }

    comparisons = []

    checked_pairs = set()

    for (row, col), data in config_map.items():
        reciprocal_key = (col, row)
        
        if (row, col) in checked_pairs or (reciprocal_key not in config_map) or row == col:
            continue  # Skip if already compared or reciprocal doesn't exist or if square
        
        reciprocal = config_map[reciprocal_key]

        latency_a = int(data["Latency"])
        latency_b = int(reciprocal["Latency"])
        pd_a = float(data["Power Density"])
        pd_b = float(reciprocal["Power Density"])

        comparisons.append({
            "OU_A": f"{row}x{col}",
            "OU_B": f"{col}x{row}",
            "Latency_A": latency_a,
            "Latency_B": latency_b,
            "Latency_Ratio (A/B)": round(latency_a / latency_b, 3),
            "PD_A": round(pd_a, 3),
            "PD_B": round(pd_b, 3),
            "PD_Ratio (A/B)": round(pd_a / pd_b, 3)
        })

        checked_pairs.add((row, col))
        checked_pairs.add(reciprocal_key)

    result_df = pd.DataFrame(comparisons)
    return result_df

if __name__ == "__main__":
    #### For getting the power density heatmap constrained by 8 Watts for Standard ####
    # filtered_standard = readOUFileAndConstrainByPowerDensity("OU_Data_Tables/Standard_OU_Stats.csv")
    # plotPowerDensityHeatmap("OU_Data_Tables/Standard_OU_Stats_Filtered_PD8.csv", "Standard", log_scale=True)

    workload_csv = "workloads/vgg16_stats.csv"
    chiplets = ["Standard", "Shared", "Adder", "Accumulator", "ADC_Less"]
    chiplets = ["Shared", "Accumulator"]

    res = computeCrossbarMetrics(chipletName="Standard", workloadStatsCSV=workload_csv)
    for r in res:
        print(f"Layer: {r['layer']}")
        print(f"Weight Sparsity: {r['Weight_Sparsity']}")
        print(f"Minimum Row Requirement: {r['Minimum_row_Req']}")
        print(f"Crossbars Required: {r['crossbars_required']}")
        print(f"Minimum OU Dimension Required: {r['Minimum OU Dimension Required']}")
        print(f"Factors: {r['Factors']}")
        print(f"Minimum Crossbar Dimensions: {r['Minimum Crossbar Dimensions']}")
        print(f"Minimum Crossbar Dimensions Rounded: {r['Minimum Crossbar Dimensions Rounded']}")
        print(f"Rank Based Pareto Config: {r['Rank Based Pareto Config']}")
        print("-" * 40)

    for r in res:
        #print(f"Layer {r['layer']}: Minimum Crossbar Dimensions: {r['Minimum Crossbar Dimensions']}")
        # print(f"Layer {r['layer']}: Optimal OU Config: {r['Rank Based Pareto Config']}")
        print(f"Layer {r['layer']}: Minimal Xbar Dimension: {r['Minimum Crossbar Dimensions']} Optimal OU Config: {r['Rank Based Pareto Config']['ou_row']}x{r['Rank Based Pareto Config']['ou_col']}, Latency: {r['Rank Based Pareto Config']['latency']}, Energy: {r['Rank Based Pareto Config']['energy']}, Power Density: {r['Rank Based Pareto Config']['power_density']:.2f} W")

    # configs, layerStats = computeCrossbarMetricsSweepOnOneLayer("Standard", workload_csv, layer_index = 2)
    # # Filter for configs under power density threshold
    # filtered_configs = [r for r in configs if r["Power Density"] < 8]
    # saveConfigsToCSV(filtered_configs, layerStats)
    # plotTopLatencyConfigs(filtered_configs, top_n=15)

    ### FOR GETTING TOP 15 CONFIGS FOR EACH CHIPLET IN PLOTS AND CSVs ###
    # step = 64
    # for chiplet in chiplets:
    #     configs = sweepFixedOUConfigsAcrossAllLayers(chiplet, workload_csv, power_density_threshold=8.0, step=step)
    #     saveWorkloadConfigsToCSV(configs, chiplet, step=step)
    #     plotTopLatencyConfigs(configs, chiplet, step=step, top_n=15)

    # file_to_compare = "OU_Workload_Filtered_CSVs_step=32/Standard_Workload_Filtered_OU_step=32.csv"
    # print(compare_reciprocal_configs(file_to_compare))

    # Energy, tops , power_density = customizeOU(96, 128, "Standard")
    print(customizeOU(96, 128, "Standard"))  # Example usage
    print(customizeOU(64, 128, "Standard"))  # Example usage
