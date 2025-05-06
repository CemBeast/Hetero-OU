import pandas as pd
import pprint
import math
import numpy as np

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

# Constants for chiplet calculations
XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16
MAX_CHIPLET_POWER = 8.0  # 8 Watts maximum per chiplet

def get_chip_capacity_bits(chip_type, tiles=16, xbars=96):
    """
    Returns total storage capacity of one chip (in bits),
    computed as: Size  * Bits/cell * #crossbars * #tiles.
    """
    info = chipletTypesDict[chip_type]
    return info["Size"] * info["Bits/cell"] * xbars * tiles

def calculate_chiplets_needed(weight_bits, chiplet_type):
    """
    Calculate the number of chiplets needed based on weight bits and chiplet capacity.
    
    Args:
        weight_bits: Total weight bits required for the layer
        chiplet_type: Type of chiplet ("Standard", "Shared", etc.)
        
    Returns:
        Number of chiplets needed (ceiling value)
    """
    # Get capacity per crossbar
    info = chipletTypesDict[chiplet_type]
    xbar_capacity = info["Size"] * info["Bits/cell"]
    
    # Calculate crossbars needed
    xbars_needed = math.ceil(weight_bits / xbar_capacity)
    
    # Calculate chiplets needed (each chiplet has TILES_PER_CHIPLET * XBARS_PER_TILE crossbars)
    chiplets_needed = math.ceil(xbars_needed / (TILES_PER_CHIPLET * XBARS_PER_TILE))
    
    return max(1, chiplets_needed)  # At least one chiplet is needed

# -----------------------------------------------------------------------------
# getOUSize implementation for optimal crossbar dimensions with power constraint
# -----------------------------------------------------------------------------
def getOUSize(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, total_macs=None):
    """
    Determines optimal crossbar dimensions (row/col size) that minimizes EDP and peak power
    while meeting activation sparsity requirements and power constraints using pymoo for multi-objective optimization.
    
    Args:
        xbar_sparsity: Effective crossbar sparsity (IS + WS) / num_xbars
        num_xbars: Number of crossbars required for the layer
        chiplet_type: Type of chiplet ("Standard", "Shared", etc.)
        weight_bits: Total weight bits required for the layer
        activation_sparsity: Minimum activation sparsity that determines the minimum row requirement
        total_macs: Total MACs for this layer (needed for power calculation)
        
    Returns:
        (optimal_row_size, optimal_col_size, scaled_tops, scaled_energy_per_mac)
    """
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError:
        print("WARNING: pymoo not installed. Falling back to manual search.")
        return _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, total_macs)
    
    # Get chiplet specs from pim_scaling module
    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy_per_mac = spec["energy_per_mac"]
    
    # Calculate minimum row requirement based on activation sparsity
    min_rows = int(base_r * activation_sparsity)
    
    # Calculate number of chiplets needed
    num_chiplets = calculate_chiplets_needed(weight_bits, chiplet_type)
    
    # Define scaling factors for the search space
    scales = np.array([0.0675, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0])
    
    # Define the multi-objective optimization problem
    class CrossbarOptProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=2,               # two variables: row_scale and col_scale
                n_obj=2,               # two objectives: minimize EDP and minimize peak power
                n_constr=1,            # three constraints:required cells, and power limit
                xl=np.array([0, 0]),   # lower bounds for scales (will be mapped to discrete values)
                xu=np.array([6, 6])    # upper bounds for scales (maps to indices of scales array)
            )
         
        def _evaluate(self, x, out, *args, **kwargs):
            # Map continuous decision variables to discrete scale factors
            x_mapped = np.round(x).astype(int)
            
            # Clip to valid indices
            x_mapped = np.clip(x_mapped, 0, len(scales)-1)
            
            # Map indices to actual scale factors
            row_scales = scales[x_mapped[:, 0]]
            col_scales = scales[x_mapped[:, 1]]
            
            # Calculate actual dimensions
            rows = (base_r * row_scales).astype(int)
            cols = (base_c * col_scales).astype(int)
            
            # Calculate energy and TOPS scaling for each configuration
            f1 = np.zeros(len(x))  # EDP
            f2 = np.zeros(len(x))  # Peak power
            g1 = np.zeros(len(x))  # Row constraint: rows >= min_rows
            # g2 = np.zeros(len(x))  # Cell constraint: rows*cols >= required_cells
            
            required_cells = (1 - xbar_sparsity) * base_r * base_c # total cells needed
            
            for i in range(len(x)):
                r_scale = row_scales[i]
                c_scale = col_scales[i]
                r = rows[i]
                c = cols[i]
                
                # Calculate energy scaling
                rowE = spec["rowKnob"] * r_scale
                colE = spec["colKnob"] * c_scale
                otherE = 100 - spec["rowKnob"] - spec["colKnob"]
                energy_ratio = (rowE + colE + otherE) / 100
                scaled_energy = base_energy_per_mac * energy_ratio
                
                # Calculate TOPS scaling
                scaled_tops = base_tops * (r_scale * c_scale)
                tops_ops = scaled_tops * 1e12  # Convert to ops/s
                
                # Calculate objectives
                
                edp = scaled_energy / tops_ops if tops_ops > 0 else 1e10
                peak_power = scaled_energy * tops_ops

                
                # Calculate per-chiplet power if total_macs is provided
                chiplet_power = float('inf')
                if total_macs is not None:
                    # Assuming equal distribution of MACs across chiplets
                    macs_per_chiplet = total_macs / num_chiplets
                    time_s = macs_per_chiplet / tops_ops
                    energy_j = macs_per_chiplet * scaled_energy
                    edp = energy_j * time_s

                
                # Normalize objectives
                f1[i] = edp / (base_energy_per_mac / (base_tops * 1e12))
                f2[i] = peak_power / (base_energy_per_mac * base_tops * 1e12)
                
                # Constraints (g <= 0 means constraint is satisfied)
                g1[i] = peak_power - MAX_CHIPLET_POWER                       # Power constraint: chiplet_power must be <= MAX_CHIPLET_POWER
                # g2[i] = required_cells - (r * c)  # Ensure r*c >= required_cells
      
                
            out["F"] = np.column_stack([f1, f2])
            out["G"] = np.column_stack([g1])
    
    # Create and solve the problem
    problem = CrossbarOptProblem()
    
    algorithm = NSGA2(
        pop_size=50,
        n_offsprings=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", 150)
    
    try:
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=False,
            verbose=False
        )
        
        # If optimization successful, get the best compromise solution
        if res.X is not None and len(res.X) > 0:
            # We prefer power reduction over EDP, so apply a weighted sum to choose the best solution
            F = res.F
            weights = np.array([0.3, 0.7])  # 30% weight to EDP, 70% to power
            weighted_F = np.sum(F * weights, axis=1)
            best_idx = np.argmin(weighted_F)
            
            # Map to discrete scale factors
            x_best = np.round(res.X[best_idx]).astype(int)
            x_best = np.clip(x_best, 0, len(scales)-1)
            row_scale = scales[x_best[0]]
            col_scale = scales[x_best[1]]
            
            # Calculate final values
            r = int(base_r * row_scale)
            c = int(base_c * col_scale)
            
            # Calculate energy scaling
            rowE = spec["rowKnob"] * row_scale
            colE = spec["colKnob"] * col_scale
            otherE = 100 - spec["rowKnob"] - spec["colKnob"]
            energy_ratio = (rowE + colE + otherE) / 100
            scaled_energy = base_energy_per_mac * energy_ratio
            
            # Calculate TOPS scaling
            scaled_tops = base_tops * (row_scale * col_scale)
            
            # Check chiplet power
            if total_macs is not None:
                macs_per_chiplet = total_macs / num_chiplets
                time_s = macs_per_chiplet / (scaled_tops * 1e12)
                energy_j = macs_per_chiplet * scaled_energy
                chiplet_power = energy_j / time_s
                print(f"Selected configuration: r={r}, c={c}, chiplet_power={chiplet_power:.2f}W (limit: {MAX_CHIPLET_POWER}W)")
            
            return r, c, scaled_tops, scaled_energy
    except Exception as e:
        print(f"Pymoo optimization failed: {e}. Falling back to manual search.")
        
    # Fallback to manual search if pymoo fails
    return _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, total_macs)

def _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, total_macs=None):
    """
    Fallback manual search method for optimal crossbar dimensions when pymoo is not available.
    Uses the same parameters and returns the same values as getOUSize.
    Includes power constraint.
    """
    # Get chiplet specs from pim_scaling module
    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy_per_mac = spec["energy_per_mac"]
    
    # Calculate minimum row requirement based on activation sparsity
    min_rows = int(base_r * activation_sparsity)
    
    # Calculate number of chiplets needed
    num_chiplets = calculate_chiplets_needed(weight_bits, chiplet_type)
    
    # Scaling factors to search through - smaller increments for more precise search
    scales = [1.0, 0.9, 0.75, 0.5, 0.25, 0.125, 0.0675]
    
    # Track best configuration
    best_config = None
    best_weighted_metric = float('inf')  # Lower is better
    
    # Weight factors for multi-objective optimization
    edp_weight = 0.3    # Lower priority on EDP
    power_weight = 0.7  # Higher priority on power reduction
    
    for r_scale in scales:
        for c_scale in scales:
            # Calculate scaled dimensions
            r = int(base_r * r_scale)
            c = int(base_c * c_scale)
            
            # Skip if row size is less than minimum required by activation sparsity
            if r < min_rows:
                continue
                
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
            
            # Check power constraint if total_macs is provided
            if total_macs is not None:
                # Calculate per-chiplet power
                macs_per_chiplet = total_macs / num_chiplets
                time_s = macs_per_chiplet / tops_ops
                energy_j = macs_per_chiplet * scaled_energy
                chiplet_power = energy_j / time_s if time_s > 0 else float('inf')
                
                # Skip if power constraint is violated
                if chiplet_power > MAX_CHIPLET_POWER:
                    continue
            
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
        # If no valid configuration with power constraint, relax power constraint and try again
        print(f"Warning: No configuration found that meets the power constraint of {MAX_CHIPLET_POWER}W. Finding best configuration without power constraint.")
        
        best_weighted_metric = float('inf')
        for r_scale in scales:
            for c_scale in scales:
                r = int(base_r * r_scale)
                c = int(base_c * c_scale)
                
                if r < min_rows:
                    continue
                    
                required_cells = (1 - xbar_sparsity) * base_r * base_c
                if r * c < required_cells:
                    continue
                
                rowE = spec["rowKnob"] * r_scale
                colE = spec["colKnob"] * c_scale
                otherE = 100 - spec["rowKnob"] - spec["colKnob"]
                energy_ratio = (rowE + colE + otherE) / 100
                scaled_energy = base_energy_per_mac * energy_ratio
                
                scaled_tops = base_tops * (r_scale * c_scale)
                tops_ops = scaled_tops * 1e12
                
                peak_power = scaled_energy * tops_ops
                edp = scaled_energy / tops_ops if tops_ops > 0 else float('inf')
                
                weighted_metric = (edp_weight * edp / (base_energy_per_mac / (base_tops * 1e12))) + \
                                 (power_weight * peak_power / (base_energy_per_mac * base_tops * 1e12))
                
                if weighted_metric < best_weighted_metric:
                    best_weighted_metric = weighted_metric
                    best_config = (r, c, scaled_tops, scaled_energy)
                    
                    # Calculate and report estimated chiplet power
                    if total_macs is not None:
                        macs_per_chiplet = total_macs / num_chiplets
                        time_s = macs_per_chiplet / tops_ops
                        energy_j = macs_per_chiplet * scaled_energy
                        chiplet_power = energy_j / time_s if time_s > 0 else float('inf')
                        print(f"Selected configuration (power constraint relaxed): r={r}, c={c}, chiplet_power={chiplet_power:.2f}W (limit was: {MAX_CHIPLET_POWER}W)")
    
    if best_config is None:
        # Fallback to base configuration if no valid configuration found
        print(f"Warning: No valid configuration found. Using base configuration.")
        return base_r, base_c, base_tops, base_energy_per_mac
    
    return best_config

# -----------------------------------------------------------------------------
# New helper: per‑layer time/energy/power from your allocation
# -----------------------------------------------------------------------------
def compute_layer_time_energy(allocation_list, total_macs):
    """
    allocation_list: [ {chip_type, allocated_bits, ...}, ... ]
    total_macs: from CSV's 'MACs' column
    Returns (time_s, energy_J, avg_power_W, edp)
    """
    total_bits = sum(a["allocated_bits"] for a in allocation_list)
    per_chip_times = []
    per_chip_energies = []
    per_chip_edp = []
    per_chip_powers = []

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
        power_i = e_i / t_i if t_i > 0 else 0

        per_chip_times.append(t_i)
        per_chip_energies.append(e_i)
        per_chip_edp.append(edp_i)
        per_chip_powers.append(power_i)

    # layer finishes when the slowest chiplet finishes
    layer_time = max(per_chip_times)
    # energy sums across chips
    layer_energy = sum(per_chip_energies)
    # average power = E / T
    layer_power = layer_energy / layer_time
    # EDP
    layer_edp = sum(per_chip_edp)
    
    # Calculate per-chiplet power - for verification with constraint
    max_chip_power = max(per_chip_powers) if per_chip_powers else 0
    
    return layer_time, layer_energy, layer_power, layer_edp, max_chip_power

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
        total_macs = row["MACs"]
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
            weightSparsity = row["Weight_Sparsity(0-1)"] / XbarReqCeil
            
            # get effective crossbar (Xbar) percentage using formula: (IS + WS)/(ceil(n))
            XbarSparsity = (inherentSparsityMapped + weightSparsity)

            # Get activation sparsity from the input data
            ActivationSparsity = row["Activation_Sparsity(0-1)"]
            
            # Get optimal crossbar dimensions and performance metrics
            # Pass total_macs to enable power constraint enforcement
            optimal_ou_row, optimal_ou_col, optimal_tops, optimal_energy_per_mac = getOUSize(
                XbarSparsity, 
                XbarReqCeil, 
                specName, 
                weightReq,
                ActivationSparsity,
                total_macs  # Pass total_macs for power calculation
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

        # Include max_chip_power in the return values to verify power constraint compliance
        t, e, p, edp, max_chip_power = compute_layer_time_energy(allocations, row["MACs"])

        layer_allocations.append({
            "layer": layer_id,
            "allocations": allocations,
            "time_s": t,
            "energy_J": e,
            "avg_power_W": p,
            "edp": edp,
            "max_chiplet_power_W": max_chip_power
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

    # Print layer details with power constraint verification
    for lr in allocations:
        print(f"\nLayer {lr['layer']}:")
        print(f"  → Time: {lr['time_s']:.3e} s,  Energy: {lr['energy_J']:.3e} J")
        print(f"  → Power: {lr['avg_power_W']:.3e} W, Max Chiplet Power: {lr['max_chiplet_power_W']:.3e} W")
        
        # Check if power constraint is met
        power_status = "✓" if lr['max_chiplet_power_W'] <= MAX_CHIPLET_POWER else "❌"
        print(f"  → Power constraint ({MAX_CHIPLET_POWER}W): {power_status}")
        
        # Count chiplets used in this layer
        chip_types_used = {}
        for alloc in lr["allocations"]:
            chip_types_used[alloc["chip_type"]] = chip_types_used.get(alloc["chip_type"], 0) + 1
        
        print(f"  → Chiplets used: {sum(chip_types_used.values())} ({', '.join([f'{v} {k}' for k, v in chip_types_used.items()])})")
        print(f"  → EDP: {lr['edp']:.3e} J·s")
        
        # Print the first few allocations details
        for i, alloc in enumerate(lr["allocations"][:2]):
            print(f"    - {alloc['chip_type']} (OU: {alloc['optimal_ou_row']}×{alloc['optimal_ou_col']}, "
                  f"TOPS: {alloc['optimized_tops']:.2f}, Energy/MAC: {alloc['optimized_energy_per_mac']:.2e} J)")
        
        if len(lr["allocations"]) > 2:
            print(f"    - ... and {len(lr['allocations'])-2} more allocations")