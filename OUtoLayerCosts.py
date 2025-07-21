import mapperV3
import math
import pandas as pd
import os

XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16
MAX_CHIPLET_POWER = 8.0  # Watts

chiplet_specs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 64.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}
### 128 x 128 ->   == (row = 0.87x 87.5% = 0.76125 ; col = 0.125 x 12.5% = 0.015625) -> 0.776875
### 128 x 64 ->    == (row = 0.87 x 87.5% = 0.76125 : col = 0.12 x 6.25% = 0.0075) -> 0.76875
### 128 x 32 ->    == (row = 0.87 x 87.5% = 0.76125 : col = 0.12 x 3.125% = 0.00375) -> 0.76500
### 128 x 16 ->    == (row = 0.87 x 87.5% = 0.76125 : col = 0.12 x 1.5625% = 0.001875) -> 0.763125
### 128 x 8 ->     == (row = 0.87 x 87.5% = 0.76125 : col = 0.12 x 0.78125% = 0.0009375) -> 0.7621875
### 128 x 4 ->     == (row = 0.87 x 87.5% = 0.76125 : col = 0.12 x 0.390625% = 0.00046875) -> 0.76171875

### 64 x 128 ->    == (row = 0.87 x 43.75% = 0.38125 : col = 0.12 x 12.5% = 0.015) -> 0.39625
### 32 x 128 ->    == (row = 0.87 x 21.875% = 0.19125 : col = 0.12 x 12.5% = 0.015) -> 0.20625
         
### 4 x 4 -> == (row = .22 x 0.00859375 = 0.001890625 : col = .22 x 0.00796875 = 0.001753125) -> 0.00364375
### 128 x 128 -> == (row = .22 x 0.875 = 0.1925 : col = .22 x 0.125 = 0.0275) -> 0.2200

# Formula
# E_rows = Total_Energy/Mac * rowKnob * (rows/base_rows)
# E_cols = Total_Energy/Mac * colKnob * (cols/base_cols)
# E_total = E_rows + E_cols 

# -----------------------------------------------------------------------------
# Chip specs & capacity helper 
# -----------------------------------------------------------------------------
chipletTypesDict = {
    "Standard":    {"Size": 16384,  "Bits/cell": 2, "TOPS": 30e12,  "Energy/MAC": 0.87e-12},
    "Shared":      {"Size": 583696, "Bits/cell": 1, "TOPS": 275e12,  "Energy/MAC": 0.30e-12},
    "Adder":       {"Size": 4096,   "Bits/cell": 1, "TOPS": 11e12,  "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536,  "Bits/cell": 2, "TOPS": 35e12,  "Energy/MAC": 0.22e-12},
    "ADC_Less":    {"Size": 163840,  "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12},
}

def RecalculateChipletSpecs(rows: int, cols: int, chipletType: str):
    """
    Recalculate the chiplet specs based on the number of rows and columns.
    """
    spec = chiplet_specs[chipletType]
    base_rows, base_cols = spec["base"]
    # Constraint check
    if rows > base_rows or cols > base_cols:
        raise ValueError(f"Requested OU size ({rows}, {cols}) exceeds base size ({base_rows}, {base_cols}) for {chipletType}")

    rs = rows / base_rows
    cs = cols / base_cols

    rowE = spec["rowKnob"]*rs + spec["colKnob"]*cs 
    e_per_mac = spec["energy_per_mac"] * (rowE/100)

    # Adjusted TOPS
    tops = spec["tops"] * rs * cs
    print(f"Chiplet {chipletType} with {rows} rows and {cols} cols: {tops} TOPS, {e_per_mac} J/MAC")
    return rows, cols, tops, e_per_mac

def scheduler(csv_path, chip_distribution, ouRows, ouCols):
    df = pd.read_csv(csv_path)
    df["Adjusted_Weights_bits"] = df["Weights(KB)"]*(1-df["Weight_Sparsity(0-1)"])*1024*8

    # build chip inventory
    inv = []
    types = list(chipletTypesDict.keys())
    for ct, cnt in zip(types, chip_distribution):
        for i in range(cnt):
            inv.append({"id":f"{ct}_{i}", "type":ct,
                        "capacity_left":mapperV3.get_chip_capacity_bits(ct)})
    layers = []

    # For each layer, allocate bits to chips and compute metrics to append to layers (results)
    for _, row in df.iterrows():
        layer_id     = int(row["Layer #"])
        rem_bits     = row["Adjusted_Weights_bits"]
        total_macs   = row["MACs"]
        allocs       = []
        total_bits   = rem_bits
        for chip in inv:
            if rem_bits <= 0: break
            if chip["capacity_left"] <= 0: continue
            alloc = min(rem_bits, chip["capacity_left"])

            AS           = row["Activation_Sparsity(0-1)"]
            weight_nonzero_bits = alloc

            # how many bits per crossbar
            cap = chipletTypesDict[chip["type"]]["Size"] * chipletTypesDict[chip["type"]]["Bits/cell"]

            # number of crossbars you really need to hold those non‑zero bits
            xbars_req = math.ceil(weight_nonzero_bits / cap)

            # if you spread the non‑zeros evenly across them:
            per_xbar_nonzeros = weight_nonzero_bits / xbars_req

            # fraction of each xbar that’s empty
            xbar_sparsity = (cap - per_xbar_nonzeros) / cap

            r, c, tops, epm = RecalculateChipletSpecs(ouRows, ouCols, chip["type"])

            frac        = alloc / total_bits
            macs_assigned = total_macs * frac
            util        = alloc / chip["capacity_left"]

            chip["capacity_left"] -= alloc
            rem_bits    -= alloc

            allocs.append({
                "chip_id": chip["id"],
                "chip_type": chip["type"],
                "allocated_bits": int(alloc),
                "MACs_assigned": int(macs_assigned),
                "Chiplets_reqd": math.ceil(xbars_req/(TILES_PER_CHIPLET*XBARS_PER_TILE)),
                "Crossbars_used": xbars_req,
                "Crossbar_sparsity": xbar_sparsity,
                "weight sparsity":row["Weight_Sparsity(0-1)"],
                "Activation Sparsity": AS,
                "ou_row": r,
                "ou_col": c,
                "optimized_tops": tops,
                "optimized_energy_per_mac": epm,
                "chiplet_resource_taken": util*100
            })

        if rem_bits > 0:
            raise RuntimeError(f"Layer {layer_id} not fully allocated: {rem_bits:.0f} bits remain")

        t, e, p, edp, maxp = mapperV3.compute_layer_time_energy(allocs, total_macs)
        layers.append({
            "layer": layer_id,
            "allocations": allocs,
            "time_s": t,
            "energy_J": e,
            "avg_power_W": p,
            "edp": edp,
            "max_chiplet_power_W": maxp
        })

    return layers

# To save the results of a LAYER to a CSV file, only shows that OU size and layer results
def save_layer_results_csv(chiplet_name: str,
                           rows: int,
                           cols: int,
                           layer_results: list,
                           out_dir: str = "HomoOULayerComputeResults") -> str:
    """
    Dump per-layer compute metrics to a CSV file.

    Parameters
    ----------
    chiplet_name : str
        Human-readable chiplet type (e.g. "Shared").
    rows, cols : int
        OU size used when you called `scheduler`.
    layer_results : list[dict]
        The list returned by `scheduler`; each dict must contain
        'layer', 'time_s', 'energy_J', and 'max_chiplet_power_W'.
    out_dir : str, optional
        Folder to hold all experiment CSVs; created if missing.

    Returns
    -------
    str
        Full path to the CSV so the caller can log or open it later.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Thin dataframe for exactly the fields we want
    df_layers = pd.DataFrame([
        {
            "layer"               : lr["layer"],
            "time_s"              : lr["time_s"],
            "energy_J"            : lr["energy_J"],
        }
        for lr in layer_results
    ])

    fname  = f"{chiplet_name}_{rows}x{cols}.csv"
    fpath  = os.path.join(out_dir, fname)
    df_layers.to_csv(fpath, index=False)
    print(f"✓ Layer metrics written to {fpath}")
    return fpath

# CSV file appending function for multiple OU sizes of same chiplet type
def append_layer_results_csv(chiplet_name: str,
                             rows: int,
                             cols: int,
                             layer_results: list,
                             out_dir: str = "HomoOULayerComputeResults") -> str:
    """
    Appends per-layer compute metrics to a shared CSV file,
    grouped by OU size. Adds a summary row at the end of each block.

    Parameters
    ----------
    chiplet_name : str
        e.g. "Shared"
    rows, cols : int
        Current OU size
    layer_results : list[dict]
        Result from scheduler()
    out_dir : str
        Folder to write into (created if missing)

    Returns
    -------
    str : path to the file written
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{chiplet_name}_OU_Sweep.csv"
    fpath = os.path.join(out_dir, fname)

    # Compute block-wide summary
    latency = max(lr["time_s"] for lr in layer_results)
    energy  = sum(lr["energy_J"] for lr in layer_results)
    # EDP = Energy Delay Product
    edp     = latency * energy
    
    # S
    # scale = chipletTypesDict[chiplet_name]["Size"] / (rows * cols)
    # edp *= scale  # scale EDP by shared's size for normalization

    # Write block to CSV string
    lines = []
    lines.append(f"OU Size: {rows}x{cols}\n")
    lines.append("layer,time_s,energy_J\n")
    for lr in layer_results:
        lines.append(f"{lr['layer']},{lr['time_s']:.6e},{lr['energy_J']:.6e}\n")
    lines.append(f"Total,,,\n")  # spacer
    lines.append(f"Latency (max),{latency:.6e},\n")
    lines.append(f"Energy (sum),,{energy:.6e},\n")
    lines.append(f"EDP, , ,{edp:.6e}\n")
    lines.append("\n")  # blank line between groups

    # Append to file
    with open(fpath, "a") as f:
        f.writelines(lines)

    print(f"✓ Appended results for {rows}x{cols} to {fpath}")
    return fpath

def print_detailed_results(layer_results: list):
    """
    Print per-layer compute metrics to the console.

    Parameters
    ----------
    layer_results : list[dict]
        The list returned by `scheduler`; each dict must contain
        'layer', 'time_s', 'energy_J', and 'max_chiplet_power_W'.
    """
    # per-layer details
    for lr in layer_results:
        print(f"\nLayer {lr['layer']}:")
        for a in lr["allocations"]:
            print(" ", a)
        print(f"Layer: {lr['layer']}, Time: {lr['time_s']:.3e}s, Energy: {lr['energy_J']:.3e}J, "
              f"EDP: {lr['edp']:.3e}, MaxP: {lr['max_chiplet_power_W']:.3e}W, ")

    # global summary for computation costs
    compute_energy = sum(l["energy_J"] for l in layer_results)
    compute_latency  = max(l["time_s"]   for l in layer_results)
    compute_workload_edp = compute_energy * compute_latency
    print("\nWorkload summary:")
    print(f"  Final Compute latency: {compute_latency:.3e} s")
    print(f"  Final Compute energy : {compute_energy:.3e} J")
    print(f"  Final Compute EDP    : {compute_workload_edp:.3e} J·s")

    # check violations
    bad = [l["layer"] for l in layer_results if l["max_chiplet_power_W"] > MAX_CHIPLET_POWER]
    if bad:
        print(f"\nWarning: layers {bad} exceed the {MAX_CHIPLET_POWER}W peak‑power cap.")


# ------------------------------------------------------------
# Sweep all multiples-of-4 OU configurations for one chiplet
# ------------------------------------------------------------
def sweep_ou_sizes(chiplet_type: str,
                   chip_dist: list[int],
                   workload_csv: str,
                   step: int = 4):
    """
    Iterate through (row, col) = (4…base_r, 4…base_c) in `step` increments,
    run the scheduler, and append results to one CSV.

    Parameters
    ----------
    chiplet_type : str   e.g. "Standard", "Shared"
    chip_dist    : list  distribution vector for scheduler()
    workload_csv : str   path to the workload stats file
    step         : int   row/col increment (default 4)
    """
    base_r, base_c = chiplet_specs[chiplet_type]["base"]
    for rows in range(step, base_r + 1, step):
        for cols in range(step, base_c + 1, step):
            try:
                results = scheduler(workload_csv, chip_dist, rows, cols)
                append_layer_results_csv(chiplet_type, rows, cols, results)
            except Exception as e:
                # Skip illegal or infeasible configurations
                print(f"✗ {chiplet_type} {rows}x{cols} skipped: {e}")


def get_different_OU_stats(csv_path):
    results = []
    rows = cols = None
    latency = energy = edp = None

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("OU Size:"):
                size_str = line.split("OU Size:")[1].strip()
                rows, cols = map(int, size_str.split("x"))

            elif line.startswith("Latency"):
                latency = float(line.split(",")[1])

            elif line.startswith("Energy"):
                energy = float(line.split(",")[2])

            elif line.startswith("EDP"):
                edp = float(line.split(",")[-1])
                results.append((rows, cols, latency, energy, edp))

    return results

def print_summary_csv_format(file_path):
    summary = get_different_OU_stats(file_path)
    print("Rows,Cols,Latency,Energy,EDP")  # header
    for row, col, lat, en, edp in summary:
        print(f"{row},{col},{lat:.6e},{en:.6e},{edp:.6e}")

if __name__ == "__main__":
    workload_csv = "workloads/vgg16_stats.csv"
    df = pd.read_csv(workload_csv)
    workload = [
        { "layer": int(row["Layer #"]), "activations_kb": float(row["Activations(KB)"]) }
        for _, row in df.iterrows()
    ]
    chip_dist    = [0, 0, 0, 0, 120]# hetOU
    results = scheduler(workload_csv, chip_dist, 4, 4)
    print_detailed_results(results)
    ##### Writes and APPENDS to file so only run once
    sweep_ou_sizes("ADC_Less", chip_dist, workload_csv, step=4)

    #print_summary_csv_format("HomoOULayerComputeResults/ADC_Less_OU_Sweep.csv")

