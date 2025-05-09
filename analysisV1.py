import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from communication_model import CommunicationModel
from advanced_communication_analysis import CommunicationAnalyzer
from enhanced_scheduler import enhanced_scheduler, print_enhanced_results

def run_complete_analysis(workload_csv, chip_distribution, output_dir="output"):
    """
    Run a complete analysis including scheduling, communication modeling, and visualization.
    
    Args:
        workload_csv: Path to the workload CSV.
        chip_distribution: List of chiplet counts by type.
        output_dir: Directory for output files.
        
    Returns:
        Dictionary with full analysis results.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    comm_dir = os.path.join(output_dir, "communication")
    os.makedirs(comm_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Run enhanced scheduler
    print("Running enhanced scheduler...")
    results = enhanced_scheduler(workload_csv, chip_distribution, output_dir=comm_dir)
    
    # Extract communication data
    noi_df = results["communication"]["noi_df"]
    noc_dfs = results["communication"]["noc_dfs"]
    
    # Convert DataFrames to numpy arrays for analysis
    chiplet_ids = list(noi_df.index)
    noi_traffic = noi_df.values
    
    noc_traffic = {}
    for chiplet_id, df in noc_dfs.items():
        noc_traffic[chiplet_id] = df.values
    
    # Define chiplet types (example dictionary, adjust as needed)
    chipletTypesDict = {
        "Standard": 0,
        "Shared": 1,
        "Adder": 2,
        "Accumulator": 3,
        "ADC_Less": 4
    }
    
    # Create chiplet mapping
    chiplet_mapping = {}
    chiplet_types = list(chipletTypesDict.keys())
    current_index = 0
    
    for chiplet_type, count in zip(chiplet_types, chip_distribution):
        for i in range(count):
            chiplet_id = f"{chiplet_type}_{i}"
            chiplet_mapping[chiplet_id] = {
                "global_index": current_index,
                "type": chiplet_type,
                "type_index": i,
                "allocated_layers": []
            }
            current_index += 1
    
    # Update chiplet mapping with allocated layers
    for layer in results["layers"]:
        layer_id = layer["layer"]
        for alloc in layer["allocations"]:
            chiplet_id = alloc["chip_id"]
            if chiplet_id in chiplet_mapping:
                chiplet_mapping[chiplet_id]["allocated_layers"].append({
                    "layer_id": layer_id,
                    "allocation_percentage": alloc["layer_percentage"],
                    "macs_assigned": alloc["MACs_assigned"]
                })
    
    # Initialize communication analyzer
    analyzer = CommunicationAnalyzer(noi_traffic, noc_traffic, chiplet_mapping)
    
    # Run various analyses
    print("Analyzing communication patterns...")
    hotspots = analyzer.identify_hotspots()
    traffic_stats = analyzer.calculate_traffic_distribution()
    comm_cost = analyzer.calculate_communication_cost()
    layer_comm = analyzer.analyze_layer_communication(results["layers"])
    optimization = analyzer.optimize_chiplet_mapping(results["layers"])
    
    # Generate visualizations
    print("Generating visualizations...")
    noi_viz_path = os.path.join(viz_dir, "noi_traffic.png")
    analyzer.visualize_noi_traffic(noi_viz_path)
    
    noc_viz_dir = os.path.join(viz_dir, "noc")
    os.makedirs(noc_viz_dir, exist_ok=True)
    analyzer.visualize_all_noc_traffic(noc_viz_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary_path = os.path.join(output_dir, "communication_summary.csv")
    
    # Communication summary by chiplet type
    chiplet_type_summary = {}
    for chiplet_id, info in chiplet_mapping.items():
        chiplet_type = info["type"]
        if chiplet_type not in chiplet_type_summary:
            chiplet_type_summary[chiplet_type] = {
                "count": 0,
                "total_noc_traffic": 0,
                "total_outgoing_noi_traffic": 0,
                "total_incoming_noi_traffic": 0
            }
        
        chiplet_type_summary[chiplet_type]["count"] += 1
        
        # Add NoC traffic if available
        if chiplet_id in noc_traffic:
            chiplet_type_summary[chiplet_type]["total_noc_traffic"] += np.sum(noc_traffic[chiplet_id])
        
        # Add NoI traffic
        idx = info["global_index"]
        chiplet_type_summary[chiplet_type]["total_outgoing_noi_traffic"] += np.sum(noi_traffic[idx, :])
        chiplet_type_summary[chiplet_type]["total_incoming_noi_traffic"] += np.sum(noi_traffic[:, idx])
    
    # Create summary DataFrame
    summary_data = []
    for chiplet_type, stats in chiplet_type_summary.items():
        summary_data.append({
            "Chiplet Type": chiplet_type,
            "Count": stats["count"],
            "Total NoC Traffic (packets)": stats["total_noc_traffic"],
            "Avg NoC Traffic per Chiplet": stats["total_noc_traffic"] / stats["count"] if stats["count"] > 0 else 0,
            "Total Outgoing NoI Traffic (packets)": stats["total_outgoing_noi_traffic"],
            "Total Incoming NoI Traffic (packets)": stats["total_incoming_noi_traffic"],
            "Avg NoI Traffic per Chiplet": (stats["total_outgoing_noi_traffic"] + stats["total_incoming_noi_traffic"]) / stats["count"] if stats["count"] > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    
    # Create layer communication CSV
    layer_comm_path = os.path.join(output_dir, "layer_communication.csv")
    layer_comm.to_csv(layer_comm_path, index=False)
    
    # Compile all analysis results
    analysis_results = {
        "scheduler_results": results,
        "communication_analysis": {
            "hotspots": hotspots,
            "traffic_stats": traffic_stats,
            "communication_cost": comm_cost,
            "layer_communication": layer_comm,
            "optimization_suggestions": optimization
        },
        "output_files": {
            "noi_traffic_csv": os.path.join(comm_dir, "NoI_traffic.csv"),
            "noc_traffic_dir": comm_dir,
            "noi_visualization": noi_viz_path,
            "noc_visualizations_dir": noc_viz_dir,
            "summary_report": summary_path,
            "layer_communication_report": layer_comm_path
        }
    }
    
    # Print key results
    print_enhanced_results(results)
    
    print("\n==== Communication Analysis Summary ====")
    print(f"NoI communication cost: {comm_cost['noi_cost']:.3e}")
    print(f"NoC communication cost: {comm_cost['noc_cost']:.3e}")
    print(f"Total communication cost: {comm_cost['total_cost']:.3e}")
    
    print("\nCommunication hotspots:")
    if hotspots["noi"]:
        print(f"  Found {len(hotspots['noi'])} NoI hotspots")
        for i, hotspot in enumerate(hotspots["noi"][:3], 1):  # Show top 3
            print(f"  {i}. {hotspot['source']} → {hotspot['destination']}: {hotspot['packets']:.1f} packets")
    else:
        print("  No NoI hotspots identified")
    
    print("\nOptimization suggestions:")
    for i, suggestion in enumerate(optimization["optimization_suggestions"][:3], 1):  # Show top 3
        print(f"  {i}. Layers {suggestion['layer_pair']}: {suggestion['suggested_action']}")
        print(f"     Potential savings: {suggestion['potential_savings_bits']/8/1024:.2f} KB")
    
    print(f"\nOutput files saved to {output_dir}")
    
    return analysis_results

if __name__ == "__main__":
    workload_csv = "workloads/vgg16_stats.csv"
    
    # Define a mixed chiplet distribution (adjust according to your needs)
    chip_dist = [8, 4, 2, 6, 0]  # [Standard, Shared, Adder, Accumulator, ADC_Less]
    
    # Run complete analysis
    analysis_results = run_complete_analysis(workload_csv, chip_dist)