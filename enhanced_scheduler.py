import pandas as pd
import numpy as np
import os
import math
from communication_model import CommunicationModel

# -----------------------------------------------------------------------------
# Chiplet specs with TOPS (in Tera‑ops/s) and energy_per_mac (in J)
# -----------------------------------------------------------------------------
chiplet_specs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 64.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}

# -----------------------------------------------------------------------------
# Chip specs & capacity helper 
# -----------------------------------------------------------------------------
chipletTypesDict = {
    "Standard":    {"Size": 16384,  "Bits/cell": 2, "TOPS": 30e12,  "Energy/MAC": 0.87e-12},
    "Shared":      {"Size": 583696, "Bits/cell": 1, "TOPS": 6.75e12,  "Energy/MAC": 0.30e-12},
    "Adder":       {"Size": 4096,   "Bits/cell": 1, "TOPS": 11e12,  "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536,  "Bits/cell": 2, "TOPS": 35e12,  "Energy/MAC": 0.22e-12},
    "ADC_Less":    {"Size": 163840,  "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12},
}
def enhanced_scheduler(csv_path, chip_distribution, include_communication=True, output_dir="communication_output"):
    """
    Enhanced scheduler that includes the original functionality plus communication pattern analysis.
    
    Args:
        csv_path: Path to the workload CSV
        chip_distribution: List of chiplet counts by type
        include_communication: Whether to include communication analysis
        output_dir: Directory to save communication pattern CSVs
    
    Returns:
        Dictionary with results including both computation and communication metrics
    """
    # Run the original scheduler to get layer allocations
    df = pd.read_csv(csv_path)
    df["Adjusted_Weights_bits"] = df["Weights(KB)"]*(1-df["Weight_Sparsity(0-1)"])*1024*8

    # Build chip inventory (same as original scheduler)
    inv = []
    types = list(chipletTypesDict.keys())
    for ct, cnt in zip(types, chip_distribution):
        for i in range(cnt):
            inv.append({"id":f"{ct}_{i}", "type":ct,
                        "capacity_left":get_chip_capacity_bits(ct)})

    layers = []
    for _, row in df.iterrows():
        layer_id     = int(row["Layer #"])
        rem_bits     = row["Adjusted_Weights_bits"]
        total_macs   = row["MACs"]
        allocs       = []
        total_bits   = rem_bits
        
        # Track activations for communication modeling
        activation_bits = row["Activations(KB)"] * 1024 * 8
        
        for chip in inv:
            if rem_bits <= 0: break
            if chip["capacity_left"] <= 0: continue

            alloc = min(rem_bits, chip["capacity_left"])
            AS = row["Activation_Sparsity(0-1)"]
            weight_nonzero_bits = alloc

            # Calculate crossbar requirements same as original
            cap = chipletTypesDict[chip["type"]]["Size"] * chipletTypesDict[chip["type"]]["Bits/cell"]
            xbars_req = math.ceil(weight_nonzero_bits / cap)
            per_xbar_nonzeros = weight_nonzero_bits / xbars_req
            xbar_sparsity = (cap - per_xbar_nonzeros) / cap

            r, c, tops, epm = getOUSize(
                    xbar_sparsity,
                    xbars_req,
                    chip["type"],
                    weight_nonzero_bits,
                    row["Activation_Sparsity(0-1)"]
                )
            frac = alloc / total_bits
            macs_assigned = total_macs * frac
            util = alloc / chip["capacity_left"]

            chip["capacity_left"] -= alloc
            rem_bits -= alloc

            # Enhanced allocation with activation information for communication modeling
            allocs.append({
                "chip_id": chip["id"],
                "chip_type": chip["type"],
                "allocated_bits": int(alloc),
                "MACs_assigned": int(macs_assigned),
                "Chiplets_reqd": math.ceil(xbars_req/(TILES_PER_CHIPLET*XBARS_PER_TILE)),
                "Crossbars_used": xbars_req,
                "Crossbar_sparsity": xbar_sparsity,
                "weight sparsity": row["Weight_Sparsity(0-1)"],
                "Activation Sparsity": AS,
                "Activation_bits": int(activation_bits * frac),  # Store activation bits for this allocation
                "optimal_ou_row": r,
                "optimal_ou_col": c,
                "optimized_tops": tops,
                "optimized_energy_per_mac": epm,
                "chiplet_resource_taken": util*100,
                "layer_percentage": frac  # Store percentage of layer on this chiplet
            })

        if rem_bits > 0:
            raise RuntimeError(f"Layer {layer_id} not fully allocated: {rem_bits:.0f} bits remain")

        t, e, p, edp, maxp = compute_layer_time_energy(allocs, total_macs)
        layers.append({
            "layer": layer_id,
            "allocations": allocs,
            "time_s": t,
            "energy_J": e,
            "avg_power_W": p,
            "edp": edp,
            "max_chiplet_power_W": maxp,
            "total_macs": total_macs,
            "total_activation_bits": int(activation_bits)
        })

    # Calculate computation metrics (same as original)
    total_energy = sum(l["energy_J"] for l in layers)
    total_time = sum(l["time_s"] for l in layers)
    workload_edp = total_energy * total_time
    
    # Check violations (same as original)
    power_violations = [l["layer"] for l in layers if l["max_chiplet_power_W"] > MAX_CHIPLET_POWER]
    
    results = {
        "layers": layers,
        "computation": {
            "total_energy_J": total_energy,
            "total_latency_s": total_time,
            "workload_edp": workload_edp,
            "power_violations": power_violations
        }
    }
    
    # Add communication analysis if requested
    if include_communication:
        comm_model = CommunicationModel(chip_distribution, csv_path)
        comm_model.analyze_layer_allocations(layers)
        
        # Generate NoI and NoC traffic pattern CSVs
        os.makedirs(output_dir, exist_ok=True)
        noi_df = comm_model.generate_noi_csv(os.path.join(output_dir, "NoI_traffic.csv"))
        noc_dfs = comm_model.generate_noc_csvs(output_dir)
        
        # Get communication summary
        comm_summary = comm_model.get_communication_summary()
        
        # Add communication results
        results["communication"] = {
            "summary": comm_summary,
            "noi_df": noi_df,
            "noc_dfs": noc_dfs
        }
    
    return results

def print_enhanced_results(results):
    """
    Print enhanced results including both computation and communication metrics.
    
    Args:
        results: Dictionary with results from enhanced_scheduler
    """
    print("\n==== Computation Results ====")
    comp = results["computation"]
    print(f"  Final energy : {comp['total_energy_J']:.3e} J")
    print(f"  Final latency: {comp['total_latency_s']:.3e} s")
    print(f"  Final EDP    : {comp['workload_edp']:.3e} J·s")
    
    if comp["power_violations"]:
        print(f"\nWarning: layers {comp['power_violations']} exceed the {MAX_CHIPLET_POWER}W peak‑power cap.")
    
    if "communication" in results:
        print("\n==== Communication Results ====")
        comm = results["communication"]["summary"]
        print(f"  Total NoI packets: {comm['total_noi_packets']:.3e}")
        print(f"  Total NoC packets: {comm['total_noc_packets']:.3e}")
        print(f"  Avg NoI packets/link: {comm['avg_noi_packets_per_link']:.3e}")
        print(f"  Avg NoC packets/link: {comm['avg_noc_packets_per_link']:.3e}")
        print(f"  Total NoI traffic: {comm['noi_traffic_bits']/8/1024:.3e} KB")
        print(f"  Total NoC traffic: {comm['noc_traffic_bits']/8/1024:.3e} KB")
        
        # Communication bottleneck warnings
        if comm['avg_noi_packets_per_link'] > 1000:
            print("\nWarning: High average NoI packet count per link. Consider optimizing chiplet allocation.")
        if comm['avg_noc_packets_per_link'] > 500:
            print("\nWarning: High average NoC packet count per link. Consider optimizing intra-chiplet data flow.")

if __name__ == "__main__":
    workload_csv = "workloads/vgg16_stats.csv"
    chip_dist = [32, 8, 4, 16, 0]  # Example distribution
    
    # Run enhanced scheduler with communication modeling
    results = enhanced_scheduler(workload_csv, chip_dist)
    
    # Print results
    print_enhanced_results(results)
    
    print("\nCommunication analysis files generated in 'communication_output' directory:")
    print("  - NoI_traffic.csv: Inter-chiplet communication traffic pattern")
    print("  - NoC_traffic_*.csv: Intra-chiplet communication traffic patterns per chiplet")