import pandas as pd
import pprint
import math

# -----------------------------------------------------------------------------
# Import chiplet specs from pim_scaling.py
# -----------------------------------------------------------------------------
# Chiplet specs with TOPS and energy_per_mac
chiplet_specs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 94.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}

# -----------------------------------------------------------------------------
# Chip specs & capacity helper 
# -----------------------------------------------------------------------------
chipletTypesDict = {
    "Standard": {
        "Size": 16384,
        "Bits/cell": 2,
        "TOPS": 30e12,
        "Energy/MAC": 0.87e-12
    }, # Storage is 6MB
    "Shared": {
        "Size": 583696,
        "Bits/cell": 1,
        "TOPS": 27e12,
        "Energy/MAC": 0.30e-12
    }, # Storage: 107MB
    "Adder": {
        "Size": 4096,
        "Bits/cell": 1,
        "TOPS": 11e12,
        "Energy/MAC": 0.18e-12
    }, # Storage: 0.75MB
    "Accumulator": {
        "Size": 65536,
        "Bits/cell": 2,
        "TOPS": 35e12,
        "Energy/MAC": 0.22e-12
    }, # Storage: 24MB
    "ADC_Less": {
        "Size": 16384,
        "Bits/cell": 1,
        "TOPS": 3.8e12,
        "Energy/MAC": 0.27e-12,
        "non_mac": 6e5,
        "non_mac_energy": 0.6e-11
    } # Storage: 48MB
}

def get_chip_capacity_bits(chip_type, tiles=16, xbars=96):
    """
    Returns total storage capacity of one chip (in bits),
    computed as: Size  * Bits/cell * #crossbars * #tiles.
    """
    info = chipletTypesDict[chip_type]
    return info["Size"] * info["Bits/cell"] * xbars * tiles

# -----------------------------------------------------------------------------
# getOUSize implementation for optimal crossbar dimensions
# -----------------------------------------------------------------------------
def getOUSize(xbar_sparsity, num_xbars, chiplet_type, weight_bits):
    """
    Determines optimal crossbar dimensions (row/col size) that minimizes EDP and peak power
    while meeting activation sparsity requirements.
    
    Args:
        xbar_sparsity: Effective crossbar sparsity (IS + WS) / num_xbars
        num_xbars: Number of crossbars required for the layer
        chiplet_type: Type of chiplet ("Standard", "Shared", etc.)
        weight_bits: Total weight bits required for the layer
        
    Returns:
        (optimal_row_size, optimal_col_size, scaled_tops, scaled_energy_per_mac)
    """
    # Get chiplet specs from pim_scaling module
    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy_per_mac = spec["energy_per_mac"]
    
    # Scaling factors to search through
    scales = [1.0, 0.5, 0.25, 0.125]
    
    # Track best configuration
    best_config = None
    best_weighted_metric = float('inf')  # Lower is better
    
    # Weight factors for multi-objective optimization (can be adjusted)
    edp_weight = 0.7  # Higher priority on EDP
    power_weight = 0.3  # Lower priority on power
    
    for r_scale in scales:
        for c_scale in scales:
            # Calculate scaled dimensions
            r = int(base_r * r_scale)
            c = int(base_c * c_scale)
            
            # Skip if total cells are less than needed based on sparsity
            required_cells = (1 - xbar_sparsity) * base_r * base_c
            if r * c < required_cells:
                continue
            
            # Calculate energy scaling based on row and column scaling
            rowE = spec["rowKnob"] * r_scale
            colE = spec["colKnob"] * c_scale
            otherE = 100 - spec["rowKnob"] - spec["colKnob"]
            energy_ratio = (rowE + colE + otherE) / 100
            scaled_energy = base_energy_per_mac * energy_ratio
            
            # Calculate TOPS scaling
            scaled_tops = base_tops * (r_scale * c_scale)
            tops_ops = scaled_tops * 1e12  # Convert to ops/s
            
            # Calculate peak power and EDP
            peak_power = scaled_energy * tops_ops  # Watts
            edp = scaled_energy / tops_ops if tops_ops > 0 else float('inf')  # J*s per MAC²
            
            # Combined weighted metric - normalize by base values for fair comparison
            weighted_metric = (edp_weight * edp / (base_energy_per_mac / (base_tops * 1e12))) + \
                             (power_weight * peak_power / (base_energy_per_mac * base_tops * 1e12))
            
            # Update best configuration if better
            if weighted_metric < best_weighted_metric:
                best_weighted_metric = weighted_metric
                best_config = (r, c, scaled_tops, scaled_energy)
    
    if best_config is None:
        # Fallback to base configuration if no valid configuration found
        return base_r, base_c, base_tops, base_energy_per_mac
    
    return best_config

# -----------------------------------------------------------------------------
# New helper: per‑layer time/energy/power from your allocation
# -----------------------------------------------------------------------------
def compute_layer_time_energy(allocation_list, total_macs):
    """
    allocation_list: [ {chip_type, allocated_bits, ...}, ... ]
    total_macs: from CSV's 'MACs' column
    Returns (time_s, energy_J, avg_power_W)
    """
    total_bits = sum(a["allocated_bits"] for a in allocation_list)
    per_chip_times = []
    per_chip_energies = []
    per_chip_edp = []

    for a in allocation_list:
        frac = a["allocated_bits"] / total_bits
        macs_i = total_macs * frac
        
        # Use optimized TOPS and Energy/MAC values if available in allocation
        if "optimized_tops" in a and "optimized_energy_per_mac" in a:
            tops_i = a["optimized_tops"] * 1e12  # Convert to ops/sec
            energy_per_mac_i = a["optimized_energy_per_mac"]
        else:
            # Use default values from chiplet specs
            spec = chipletTypesDict[a["chip_type"]]
            tops_i = spec["TOPS"]
            energy_per_mac_i = spec["Energy/MAC"]

        t_i = macs_i / tops_i
        e_i = macs_i * energy_per_mac_i
        edp_i = macs_i**2 * energy_per_mac_i/(tops_i)

        per_chip_times.append(t_i)
        per_chip_energies.append(e_i)
        per_chip_edp.append(edp_i)

    # layer finishes when the slowest chiplet finishes
    layer_time = max(per_chip_times)
    # energy sums across chips
    layer_energy = sum(per_chip_energies)
    # average power = E / T
    layer_power = layer_energy / layer_time
    # EDP
    layer_edp = sum(per_chip_edp)
    return layer_time, layer_energy, layer_power, layer_edp

# -----------------------------------------------------------------------------
# Scheduler function
# -----------------------------------------------------------------------------
def scheduler(csv_path, chip_distribution):
    """
    Reads workload CSV and allocates each layer's adjusted weights across chips.

    Args:
      - csv_path: path to your CSV file with columns
          Layer, Weights (KB), MACS, Weight_Sparsity(0-1), Activation_Sparsity(0-1), Activations (KB)
      - chip_distribution: list of ints [n_standard, n_shared, n_adder, n_accumulator, n_adc_less]

    Returns:
      A list of dicts, one per layer, each containing:
        - layer: layer number
        - allocations: list of {chip_id, chip_type, allocated_bits}
    """
    # Build inventory of chip instances
    chip_types = list(chipletTypesDict.keys())
    chip_inventory = []
    for chip_type, count in zip(chip_types, chip_distribution):
        for idx in range(count):
            chip_inventory.append({
                "id":      f"{chip_type}_{idx}",
                "type":    chip_type,
                "capacity_left": get_chip_capacity_bits(chip_type)
            })

    # Load workload and compute adjusted weights in bits
    df = pd.read_csv(csv_path)

    df["Adjusted_Weights_KB"]  = df["Weights(KB)"] * (1-df["Weight_Sparsity(0-1)"])
    # KB → bits: *1024 (per your spec) * 8 for bytes to bits
    df["Adjusted_Weights_bits"] = df["Adjusted_Weights_KB"] * 1024 * 8
    
    print(df.head())
    # Allocate each layer across the chips
    layer_allocations = []
    used_chip_order = []
    for _, row in df.iterrows():
        layer_id      = int(row["Layer #"])
        remaining_bits = row["Adjusted_Weights_bits"]
        allocations   = []

        for chip in chip_inventory:
            if remaining_bits <= 0:
                break
            if chip["capacity_left"] <= 0:
                continue

            # allocate as much as we can on this chip
            alloc = min(remaining_bits, chip["capacity_left"])
            
            # check how much percentage of the chiplet this layer is taking up. 
            # get the weight requirement per layer using the df in bits
            weightReq = row["Weights(KB)"] * 1024 * 8 
            
            # find the requied number of crossbars (n) required per layer 
            specName = chip["type"]
            spec = chipletTypesDict[specName]
            specSize = spec["Size"] * spec["Bits/cell"]
            XbarReqDecimal = weightReq / specSize
            XbarReqCeil = math.ceil(XbarReqDecimal)
            
            # get the inherent sparsity (IS) by using formula: (ceil(n) - n)/ceil(n)
            inherentSparsityMapped = (XbarReqCeil - XbarReqDecimal) / XbarReqCeil
            # get the weight sparsity (WS) amount by the df for that layer. 
            weightSparsity_per_Xbar = row["Weight_Sparsity(0-1)"] / XbarReqCeil
            # print(weightSparsity_per_Xbar)
            # get effective crossbar (Xbar) percentage using formula: IS + (WS)/(ceil(n))
            XbarSparsity = (inherentSparsityMapped + weightSparsity_per_Xbar)
            # Get optimal crossbar dimensions and performance metrics
            ActivationSparsity = row["Activation_Sparsity(0-1)"]
            optimal_ou_row, optimal_ou_col, optimal_tops, optimal_energy_per_mac = getOUSize(
                XbarSparsity, 
                XbarReqCeil, 
                specName, 
                weightReq
            )

            if remaining_bits > chip["capacity_left"]:
                util = 1
            else:
                util = remaining_bits/chip["capacity_left"]
            
            chip["capacity_left"] -= alloc
            remaining_bits -= alloc

            allocations.append({
                "chip_id": chip["id"],
                "chip_type": chip["type"],
                "allocated_bits": int(alloc),
                "utilization": util,
                "Crossbar_used": XbarReqCeil,
                "Crossbar Sparsity": XbarSparsity,
                "Activation Sparsity": ActivationSparsity,
                "optimal_ou_row": optimal_ou_row,
                "optimal_ou_col": optimal_ou_col,
                "optimized_tops": optimal_tops,
                "optimized_energy_per_mac": optimal_energy_per_mac

            })
            used_chip_order.append(chip["id"])

        if remaining_bits > 0:
            raise RuntimeError(f"Layer {layer_id} could not be fully allocated: {remaining_bits:.0f} bits remain")

        t, e, p, edp = compute_layer_time_energy(allocations, row["MACs"])

        layer_allocations.append({
            "layer": layer_id,
            "allocations": allocations,
            "time_s": t,
            "energy_J": e,
            "avg_power_W": p,
            "edp": edp
        })

    return layer_allocations, chip_inventory, used_chip_order

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Path to your workload CSV
    workload_csv = "workloads/vgg16_stats.csv"

    # Example: [10 Standard, 0 Shared, 0 Adder, 0 Accumulator, 1 ADC_Less]
    chip_dist = [24, 28, 0, 18, 12]

    allocations, inventory, used_order = scheduler(workload_csv, chip_dist)

    # Pretty‑print the results
    #print("\n=== Layer Allocations ===")
    #pprint.pprint(allocations, width=120)

    # # Summary: count chips used per type
    # # A chip is "used" if capacity_left < original_capacity
    # orig_caps = {chip["id"] : get_chip_capacity_bits(chip["type"]) for chip in inventory}
    # # used_summary = {}
    # # for chip in inventory:
    # #     if chip["capacity_left"] < orig_caps[chip["id"]]:
    # #         used_summary[chip["type"]] = used_summary.get(chip["type"], 0) + 1
    
    # # print("\n=== Chips Used Per Type ===")
    # # for ctype, cnt in used_summary.items():
    # #     print(f" {ctype}: {cnt}")

    # # Last chip used and its fill percentage
    # last_chip_id = used_order[-1]
    # last_chip = next(c for c in inventory if c["id"] == last_chip_id)
    # orig_capacity = orig_caps[last_chip_id]
    # used_bits = orig_capacity - last_chip["capacity_left"]
    # pct_full = (used_bits / orig_capacity) * 100

    # print(f"\nLast chip used: {last_chip_id} ({last_chip['type']})")
    # print(f"  -> Filled: {used_bits:.0f} bits / {orig_capacity} bits = {pct_full:.2f}%")

    for lr in allocations:
        print(f"\nLayer {lr['layer']}:")
        pprint.pprint(lr["allocations"], width=80)
        print(f"  → Time: {lr['time_s']:.3e} s,  Energy: {lr['energy_J']:.3e} J,  Power: {lr['avg_power_W']:.3e} W, EDP: {lr['edp']:.3e} J•s")