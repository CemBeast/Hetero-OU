import pandas as pd
import numpy as np
import os
import math

# Existing constants - I'll add communication-specific ones
XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16
MAX_CHIPLET_POWER = 8.0  # Watts

# Communication-specific constants
NOC_BUS_WIDTH = 16  # bits
NOI_BUS_WIDTH = 32  # bits

class CommunicationModel:
    def __init__(self, chip_distribution, workload_csv):
        """
        Initialize the communication model with chiplet distribution and workload.
        
        Args:
            chip_distribution: List of chiplet counts by type
            workload_csv: Path to the workload CSV file
        """
        self.chip_distribution = chip_distribution
        self.workload_df = pd.read_csv(workload_csv)
        self.chiplet_types = list(chipletTypesDict.keys())
        self.chiplet_count = sum(chip_distribution)
        self.chiplet_mapping = self._create_chiplet_mapping()
        
        # Initialize traffic matrices
        self.noi_traffic = np.zeros((self.chiplet_count, self.chiplet_count))
        self.noc_traffic = {chiplet_id: np.zeros((TILES_PER_CHIPLET, TILES_PER_CHIPLET)) 
                           for chiplet_id in self.chiplet_mapping.keys()}
        
    def _create_chiplet_mapping(self):
        """Create a mapping of chiplet IDs to their type and index"""
        mapping = {}
        current_index = 0
        
        for type_idx, (chiplet_type, count) in enumerate(zip(self.chiplet_types, self.chip_distribution)):
            for i in range(count):
                chiplet_id = f"{chiplet_type}_{i}"
                mapping[chiplet_id] = {
                    "global_index": current_index,
                    "type": chiplet_type,
                    "type_index": i,
                    "allocated_layers": []
                }
                current_index += 1
                
        return mapping
    
    def analyze_layer_allocations(self, layer_allocations):
        """
        Analyze layer allocations to determine communication patterns.
        
        Args:
            layer_allocations: List of layer allocation details from the scheduler
        """
        # Reset traffic matrices for a new analysis
        self.noi_traffic.fill(0)
        for chiplet_id in self.noc_traffic:
            self.noc_traffic[chiplet_id].fill(0)
            
        # Update chiplet mapping with allocated layers
        for layer in layer_allocations:
            layer_id = layer["layer"]
            for alloc in layer["allocations"]:
                chiplet_id = alloc["chip_id"]
                if chiplet_id in self.chiplet_mapping:
                    self.chiplet_mapping[chiplet_id]["allocated_layers"].append({
                        "layer_id": layer_id,
                        "allocation_percentage": alloc["allocated_bits"] / sum(a["allocated_bits"] for a in layer["allocations"]),
                        "macs_assigned": alloc["MACs_assigned"]
                    })
        
        # Calculate communication traffic based on layer dependencies
        self._calculate_traffic_patterns()
        
    def _calculate_traffic_patterns(self):
        """Calculate both NoI and NoC traffic patterns based on layer allocations"""
        # Process each layer sequentially (assuming layer dependencies follow layer numbers)
        for layer_idx in range(len(self.workload_df)):
            current_layer = layer_idx + 1  # Assuming layer IDs start from 1
            
            # Skip if this is the first layer (no dependencies)
            if layer_idx == 0:
                continue
                
            # Get activations size for this layer (in KB)
            try:
                activation_size = self.workload_df.loc[self.workload_df["Layer #"] == current_layer, "Activations(KB)"].values[0]
                activation_bits = activation_size * 1024 * 8  # Convert KB to bits
            except (IndexError, KeyError):
                # Layer not found in workload, skip
                continue
                
            # Find chiplets that have the current layer allocated
            current_chiplets = []
            for chiplet_id, info in self.chiplet_mapping.items():
                for layer_alloc in info["allocated_layers"]:
                    if layer_alloc["layer_id"] == current_layer:
                        current_chiplets.append((chiplet_id, layer_alloc["allocation_percentage"]))
            
            # Find chiplets that have the previous layer allocated
            prev_layer = current_layer - 1
            prev_chiplets = []
            for chiplet_id, info in self.chiplet_mapping.items():
                for layer_alloc in info["allocated_layers"]:
                    if layer_alloc["layer_id"] == prev_layer:
                        prev_chiplets.append((chiplet_id, layer_alloc["allocation_percentage"]))
            
            # Calculate inter-chiplet (NoI) traffic
            for prev_chiplet, prev_percentage in prev_chiplets:
                for curr_chiplet, curr_percentage in current_chiplets:
                    # Skip if it's the same chiplet (handled by NoC)
                    if prev_chiplet == curr_chiplet:
                        continue
                        
                    # Calculate traffic volume based on allocation percentages
                    traffic_volume = activation_bits * prev_percentage * curr_percentage
                    
                    # Convert to packets based on NoI bus width
                    packets = math.ceil(traffic_volume / NOI_BUS_WIDTH)
                    
                    # Update NoI traffic matrix
                    src_idx = self.chiplet_mapping[prev_chiplet]["global_index"]
                    dst_idx = self.chiplet_mapping[curr_chiplet]["global_index"]
                    self.noi_traffic[src_idx, dst_idx] += packets
            
            # Calculate intra-chiplet (NoC) traffic for chiplets that have both layers
            for chiplet_id, info in self.chiplet_mapping.items():
                # Find if this chiplet has both current and previous layer
                has_prev = any(layer_alloc["layer_id"] == prev_layer for layer_alloc in info["allocated_layers"])
                has_curr = any(layer_alloc["layer_id"] == current_layer for layer_alloc in info["allocated_layers"])
                
                if has_prev and has_curr:
                    # This chiplet handles both layers, so there's internal NoC traffic
                    # For simplicity, distribute traffic evenly across tiles (can be refined later)
                    prev_percentage = sum(layer_alloc["allocation_percentage"] 
                                         for layer_alloc in info["allocated_layers"] 
                                         if layer_alloc["layer_id"] == prev_layer)
                    curr_percentage = sum(layer_alloc["allocation_percentage"]
                                         for layer_alloc in info["allocated_layers"]
                                         if layer_alloc["layer_id"] == current_layer)
                    
                    traffic_volume = activation_bits * prev_percentage * curr_percentage
                    packets = math.ceil(traffic_volume / NOC_BUS_WIDTH)
                    
                    # Distribute traffic across tiles (simplified model)
                    # Later this can be refined with actual tile mapping information
                    for src_tile in range(TILES_PER_CHIPLET):
                        for dst_tile in range(TILES_PER_CHIPLET):
                            if src_tile != dst_tile:  # Skip self-communication
                                # Assign a fraction of the traffic between tiles
                                # This simplified model distributes traffic evenly
                                self.noc_traffic[chiplet_id][src_tile, dst_tile] += packets / (TILES_PER_CHIPLET * (TILES_PER_CHIPLET - 1))
    
    def generate_noi_csv(self, output_path):
        """
        Generate the NoI traffic pattern CSV.
        
        Args:
            output_path: Path to save the NoI CSV file
        """
        # Create a DataFrame for the NoI traffic matrix
        chiplet_ids = [chiplet_id for chiplet_id in sorted(self.chiplet_mapping.keys(), 
                                                          key=lambda x: self.chiplet_mapping[x]["global_index"])]
        
        noi_df = pd.DataFrame(self.noi_traffic, 
                             index=chiplet_ids,
                             columns=chiplet_ids)
        
        # Save to CSV
        noi_df.to_csv(output_path)
        print(f"NoI traffic pattern saved to {output_path}")
        
        return noi_df
    
    def generate_noc_csvs(self, output_dir):
        """
        Generate NoC traffic pattern CSVs, one per chiplet.
        
        Args:
            output_dir: Directory to save the NoC CSV files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        noc_dfs = {}
        
        # Generate a CSV for each chiplet
        for chiplet_id, traffic_matrix in self.noc_traffic.items():
            # Only generate CSVs for chiplets that have allocated layers
            if not self.chiplet_mapping[chiplet_id]["allocated_layers"]:
                continue
                
            # Create tile labels
            tile_labels = [f"Tile_{i}" for i in range(TILES_PER_CHIPLET)]
            
            # Create DataFrame
            noc_df = pd.DataFrame(traffic_matrix,
                                 index=tile_labels,
                                 columns=tile_labels)
            
            # Save to CSV
            output_path = os.path.join(output_dir, f"NoC_traffic_{chiplet_id}.csv")
            noc_df.to_csv(output_path)
            print(f"NoC traffic pattern for {chiplet_id} saved to {output_path}")
            
            noc_dfs[chiplet_id] = noc_df
            
        return noc_dfs
    
    def get_communication_summary(self):
        """Get a summary of the communication patterns"""
        total_noi_packets = np.sum(self.noi_traffic)
        total_noc_packets = sum(np.sum(matrix) for matrix in self.noc_traffic.values())
        
        # Calculate average packets per link
        noi_links = self.chiplet_count * (self.chiplet_count - 1)  # Excluding self-links
        noc_links = sum(1 for chiplet_id in self.noc_traffic 
                      if len(self.chiplet_mapping[chiplet_id]["allocated_layers"]) > 0) * TILES_PER_CHIPLET * (TILES_PER_CHIPLET - 1)
        
        avg_noi_packets = total_noi_packets / noi_links if noi_links > 0 else 0
        avg_noc_packets = total_noc_packets / noc_links if noc_links > 0 else 0
        
        return {
            "total_noi_packets": total_noi_packets,
            "total_noc_packets": total_noc_packets,
            "avg_noi_packets_per_link": avg_noi_packets,
            "avg_noc_packets_per_link": avg_noc_packets,
            "noi_traffic_bits": total_noi_packets * NOI_BUS_WIDTH,
            "noc_traffic_bits": total_noc_packets * NOC_BUS_WIDTH
        }

# Extend the existing scheduler to include communication modeling
def extended_scheduler(csv_path, chip_distribution, output_dir="communication_output"):
    """
    Extended scheduler that includes communication pattern analysis.
    
    Args:
        csv_path: Path to the workload CSV
        chip_distribution: List of chiplet counts by type
        output_dir: Directory to save communication pattern CSVs
    
    Returns:
        Tuple of (layer_results, comm_summary)
    """
    # Run the original scheduler to get layer allocations
    layer_results = scheduler(csv_path, chip_distribution)
    
    # Initialize communication model
    comm_model = CommunicationModel(chip_distribution, csv_path)
    
    # Analyze layer allocations for communication patterns
    comm_model.analyze_layer_allocations(layer_results)
    
    # Generate NoI and NoC traffic pattern CSVs
    os.makedirs(output_dir, exist_ok=True)
    noi_df = comm_model.generate_noi_csv(os.path.join(output_dir, "NoI_traffic.csv"))
    noc_dfs = comm_model.generate_noc_csvs(output_dir)
    
    # Get communication summary
    comm_summary = comm_model.get_communication_summary()
    
    return layer_results, comm_summary, noi_df, noc_dfs

# Usage example
if __name__ == "__main__":
    workload_csv = "workloads/vgg16_stats.csv"
    chip_dist = [1000, 1, 0, 40, 0, 0]  # hetOU
    
    # Run extended scheduler with communication modeling
    results, comm_summary, noi_df, noc_dfs = extended_scheduler(workload_csv, chip_dist)
    
    # Display original scheduler results
    print("\n==== Original Scheduler Results ====")
    total_energy = sum(l["energy_J"] for l in results)
    max_latency = sum(l["time_s"] for l in results)
    workload_edp = total_energy * max_latency
    print(f"  Final energy : {total_energy:.3e} J")
    print(f"  Final latency: {max_latency:.3e} s")
    print(f"  Final EDP    : {workload_edp:.3e} J·s")
    
    # Display communication summary
    print("\n==== Communication Summary ====")
    for key, value in comm_summary.items():
        print(f"  {key}: {value:.3e}")
        
    # Check for potential communication bottlenecks
    if comm_summary["avg_noi_packets_per_link"] > 1000:
        print("\nWarning: High average NoI packet count per link. Consider optimizing chiplet allocation.")
    if comm_summary["avg_noc_packets_per_link"] > 500:
        print("\nWarning: High average NoC packet count per link. Consider optimizing intra-chiplet data flow.")