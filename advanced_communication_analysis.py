import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import seaborn as sns
from collections import defaultdict

class CommunicationAnalyzer:
    """Advanced analysis tools for chiplet communication patterns"""
    
    def __init__(self, noi_traffic, noc_traffic, chiplet_mapping):
        """
        Initialize with communication traffic data
        
        Args:
            noi_traffic: NoI traffic matrix
            noc_traffic: Dictionary of NoC traffic matrices by chiplet ID
            chiplet_mapping: Mapping of chiplet IDs to metadata
        """
        self.noi_traffic = noi_traffic
        self.noc_traffic = noc_traffic
        self.chiplet_mapping = chiplet_mapping
        
    def identify_hotspots(self, threshold_percentile=95):
        """
        Identify communication hotspots in both NoI and NoC
        
        Args:
            threshold_percentile: Percentile above which links are considered hotspots
            
        Returns:
            Dictionary with hotspot information
        """
        results = {"noi": [], "noc": {}}
        
        # Find NoI hotspots
        noi_flat = self.noi_traffic.flatten()
        noi_threshold = np.percentile(noi_flat[noi_flat > 0], threshold_percentile)
        
        for i in range(self.noi_traffic.shape[0]):
            for j in range(self.noi_traffic.shape[1]):
                if self.noi_traffic[i, j] > noi_threshold:
                    # Get chiplet IDs from indices
                    src_id = None
                    dst_id = None
                    for chiplet_id, info in self.chiplet_mapping.items():
                        if info["global_index"] == i:
                            src_id = chiplet_id
                        if info["global_index"] == j:
                            dst_id = chiplet_id
                    
                    if src_id and dst_id:
                        results["noi"].append({
                            "source": src_id,
                            "destination": dst_id,
                            "packets": self.noi_traffic[i, j],
                            "source_type": self.chiplet_mapping[src_id]["type"],
                            "destination_type": self.chiplet_mapping[dst_id]["type"]
                        })
        
        # Find NoC hotspots for each chiplet
        for chiplet_id, traffic in self.noc_traffic.items():
            if traffic.size == 0:
                continue
                
            noc_flat = traffic.flatten()
            if len(noc_flat[noc_flat > 0]) == 0:
                continue
                
            noc_threshold = np.percentile(noc_flat[noc_flat > 0], threshold_percentile)
            results["noc"][chiplet_id] = []
            
            for i in range(traffic.shape[0]):
                for j in range(traffic.shape[1]):
                    if traffic[i, j] > noc_threshold:
                        results["noc"][chiplet_id].append({
                            "source_tile": i,
                            "destination_tile": j,
                            "packets": traffic[i, j]
                        })
        
        return results
    
    def calculate_traffic_distribution(self):
        """
        Calculate traffic distribution statistics
        
        Returns:
            Dictionary with distribution statistics
        """
        results = {
            "noi": {
                "max": np.max(self.noi_traffic),
                "mean": np.mean(self.noi_traffic[self.noi_traffic > 0]) if np.any(self.noi_traffic > 0) else 0,
                "median": np.median(self.noi_traffic[self.noi_traffic > 0]) if np.any(self.noi_traffic > 0) else 0,
                "std": np.std(self.noi_traffic[self.noi_traffic > 0]) if np.any(self.noi_traffic > 0) else 0,
                "total_traffic": np.sum(self.noi_traffic),
                "active_links": np.sum(self.noi_traffic > 0),
                "traffic_per_link_type": defaultdict(float)
            },
            "noc": {}
        }
        
        # Calculate traffic per link type for NoI
        for i in range(self.noi_traffic.shape[0]):
            for j in range(self.noi_traffic.shape[1]):
                if self.noi_traffic[i, j] > 0:
                    # Get chiplet types from indices
                    src_type = None
                    dst_type = None
                    for chiplet_id, info in self.chiplet_mapping.items():
                        if info["global_index"] == i:
                            src_type = info["type"]
                        if info["global_index"] == j:
                            dst_type = info["type"]
                    
                    if src_type and dst_type:
                        link_type = f"{src_type}-to-{dst_type}"
                        results["noi"]["traffic_per_link_type"][link_type] += self.noi_traffic[i, j]
        
        # Calculate traffic statistics for each chiplet's NoC
        for chiplet_id, traffic in self.noc_traffic.items():
            if traffic.size == 0 or not np.any(traffic > 0):
                results["noc"][chiplet_id] = {
                    "max": 0,
                    "mean": 0,
                    "median": 0,
                    "std": 0,
                    "total_traffic": 0,
                    "active_links": 0
                }
                continue
                
            results["noc"][chiplet_id] = {
                "max": np.max(traffic),
                "mean": np.mean(traffic[traffic > 0]),
                "median": np.median(traffic[traffic > 0]),
                "std": np.std(traffic[traffic > 0]),
                "total_traffic": np.sum(traffic),
                "active_links": np.sum(traffic > 0)
            }
        
        return results
    
    def visualize_noi_traffic(self, output_path="noi_traffic_visualization.png"):
        """
        Create a visualization of NoI traffic patterns
        
        Args:
            output_path: Path to save the visualization
        """
        # Create a directed graph for the NoI traffic
        G = nx.DiGraph()
        
        # Chiplet IDs ordered by global index
        chiplet_ids = [None] * len(self.chiplet_mapping)
        for chiplet_id, info in self.chiplet_mapping.items():
            chiplet_ids[info["global_index"]] = chiplet_id
        
        # Add nodes to the graph
        for chiplet_id in chiplet_ids:
            if chiplet_id:  # Skip None entries if any
                G.add_node(chiplet_id, type=self.chiplet_mapping[chiplet_id]["type"])
        
        # Add edges with traffic weights
        max_traffic = np.max(self.noi_traffic) if np.any(self.noi_traffic > 0) else 1
        for i in range(self.noi_traffic.shape[0]):
            for j in range(self.noi_traffic.shape[1]):
                if self.noi_traffic[i, j] > 0 and i < len(chiplet_ids) and j < len(chiplet_ids):
                    src_id = chiplet_ids[i]
                    dst_id = chiplet_ids[j]
                    if src_id and dst_id:  # Skip None entries if any
                        G.add_edge(src_id, dst_id, weight=self.noi_traffic[i, j],
                                  relative_weight=self.noi_traffic[i, j]/max_traffic)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(G, k=0.15, iterations=50)
        
        # Group nodes by type for coloring
        chiplet_types = set(info["type"] for info in self.chiplet_mapping.values())
        color_map = plt.cm.get_cmap('tab10', len(chiplet_types))
        type_to_color = {t: color_map(i) for i, t in enumerate(chiplet_types)}
        node_colors = [type_to_color[self.chiplet_mapping[node]["type"]] for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=250, node_color=node_colors, alpha=0.8)
        
        # Draw edges with width based on traffic volume
        edges = G.edges()
        weights = [G[u][v]['relative_weight'] * 5 for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, 
                            edge_color='grey', arrows=True, arrowsize=15)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Create legend for chiplet types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=type_to_color[t], markersize=10, label=t)
                          for t in chiplet_types]
        plt.legend(handles=legend_elements, title="Chiplet Types")
        
        # Set title and axis properties
        plt.title('NoI Traffic Pattern Between Chiplets', fontsize=16)
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"NoI traffic visualization saved to {output_path}")
    
    def visualize_noc_traffic(self, chiplet_id, output_path=None):
        """
        Create a visualization of NoC traffic patterns for a specific chiplet
        
        Args:
            chiplet_id: ID of the chiplet to visualize
            output_path: Path to save the visualization
        """
        if chiplet_id not in self.noc_traffic:
            print(f"No NoC traffic data for chiplet {chiplet_id}")
            return
            
        traffic = self.noc_traffic[chiplet_id]
        if not np.any(traffic > 0):
            print(f"No traffic data for chiplet {chiplet_id}")
            return
            
        # If no output path provided, create one
        if output_path is None:
            output_path = f"noc_traffic_{chiplet_id}.png"
            
        # Create a heatmap of the NoC traffic
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        ax = sns.heatmap(traffic, cmap="YlOrRd", annot=True, fmt=".1f", 
                        xticklabels=[f"Tile {i}" for i in range(traffic.shape[1])],
                        yticklabels=[f"Tile {i}" for i in range(traffic.shape[0])])
        
        # Set labels and title
        plt.xlabel("Destination Tile")
        plt.ylabel("Source Tile")
        plt.title(f"NoC Traffic Pattern for Chiplet {chiplet_id} ({self.chiplet_mapping[chiplet_id]['type']})")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"NoC traffic visualization for {chiplet_id} saved to {output_path}")
    
    def visualize_all_noc_traffic(self, output_dir="noc_visualizations"):
        """
        Create visualizations for all chiplets with NoC traffic
        
        Args:
            output_dir: Directory to save the visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for chiplet_id in self.noc_traffic:
            output_path = os.path.join(output_dir, f"noc_traffic_{chiplet_id}.png")
            self.visualize_noc_traffic(chiplet_id, output_path)
    
    def calculate_communication_cost(self, noi_cost_per_packet=2.0, noc_cost_per_packet=1.0):
        """
        Calculate communication cost based on traffic patterns and per-packet costs
        
        Args:
            noi_cost_per_packet: Energy cost per NoI packet
            noc_cost_per_packet: Energy cost per NoC packet
            
        Returns:
            Dictionary with communication cost information
        """
        noi_total_cost = np.sum(self.noi_traffic) * noi_cost_per_packet
        
        noc_costs = {}
        noc_total_cost = 0
        for chiplet_id, traffic in self.noc_traffic.items():
            chiplet_cost = np.sum(traffic) * noc_cost_per_packet
            noc_costs[chiplet_id] = chiplet_cost
            noc_total_cost += chiplet_cost
            
        return {
            "noi_cost": noi_total_cost,
            "noc_cost": noc_total_cost,
            "total_cost": noi_total_cost + noc_total_cost,
            "noc_costs_by_chiplet": noc_costs
        }
    
    def analyze_layer_communication(self, layer_allocations):
        """
        Analyze communication by layer to identify communication-heavy layers
        
        Args:
            layer_allocations: List of layer allocations from scheduler
            
        Returns:
            DataFrame with layer communication metrics
        """
        layer_metrics = []
        
        # Create a mapping of layers to chiplets
        layer_to_chiplets = defaultdict(list)
        for chiplet_id, info in self.chiplet_mapping.items():
            for layer_alloc in info.get("allocated_layers", []):
                layer_id = layer_alloc["layer_id"]
                layer_to_chiplets[layer_id].append({
                    "chiplet_id": chiplet_id,
                    "percentage": layer_alloc.get("allocation_percentage", 0)
                })
        
        # Analyze communication for each layer
        for layer_idx, layer in enumerate(layer_allocations):
            layer_id = layer["layer"]
            
            # Skip first layer (no dependencies)
            if layer_idx == 0:
                continue
                
            prev_layer_id = layer_allocations[layer_idx-1]["layer"]
            
            # Get chiplets for current and previous layers
            curr_chiplets = layer_to_chiplets.get(layer_id, [])
            prev_chiplets = layer_to_chiplets.get(prev_layer_id, [])
            
            # Count cross-chiplet links
            cross_chiplet_links = 0
            for prev_alloc in prev_chiplets:
                for curr_alloc in curr_chiplets:
                    if prev_alloc["chiplet_id"] != curr_alloc["chiplet_id"]:
                        cross_chiplet_links += 1
            
            # Calculate activation bits that need to be communicated
            activation_bits = layer.get("total_activation_bits", 0)
            
            # Calculate percentage of activations that cross chiplet boundaries
            cross_chiplet_percentage = 0
            if cross_chiplet_links > 0:
                total_links = len(prev_chiplets) * len(curr_chiplets)
                cross_chiplet_percentage = cross_chiplet_links / total_links if total_links > 0 else 0
            
            estimated_noi_traffic = activation_bits * cross_chiplet_percentage
            
            layer_metrics.append({
                "layer_id": layer_id,
                "prev_layer_id": prev_layer_id,
                "activation_bits": activation_bits,
                "cross_chiplet_links": cross_chiplet_links,
                "cross_chiplet_percentage": cross_chiplet_percentage * 100,  # to percentage
                "estimated_noi_traffic_bits": estimated_noi_traffic,
                "estimated_noi_packets": math.ceil(estimated_noi_traffic / 32)  # 32-bit NoI bus width
            })
        
        return pd.DataFrame(layer_metrics)
    
    def optimize_chiplet_mapping(self, layer_allocations, iterations=1000):
        """
        Suggest an optimized chiplet mapping to minimize inter-chiplet communication
        
        Args:
            layer_allocations: List of layer allocations from scheduler
            iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # This is a placeholder for a more sophisticated optimization
        # In a real implementation, this would use techniques like simulated annealing
        # or genetic algorithms to find an optimal mapping
        
        # For now, we'll identify the layers with heaviest communication
        layer_comm = self.analyze_layer_communication(layer_allocations)
        heaviest_layers = layer_comm.sort_values("estimated_noi_traffic_bits", ascending=False).head(5)
        
        suggestions = []
        for _, row in heaviest_layers.iterrows():
            suggestions.append({
                "layer_pair": (row["prev_layer_id"], row["layer_id"]),
                "suggested_action": "Consider mapping these layers to the same chiplet types",
                "potential_savings_bits": row["estimated_noi_traffic_bits"]
            })
        
        return {
            "heaviest_communication_layers": heaviest_layers,
            "optimization_suggestions": suggestions
        }