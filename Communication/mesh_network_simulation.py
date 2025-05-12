import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json

class MeshNetworkSimulator:
    def __init__(self, config_file=None):
        """Initialize the simulator with default or provided configuration."""
        # Default configuration
        self.config = {
            "subgraph_size": 4,  # 4x4 mesh
            "intra_subgraph_freq_ghz": 2.0,  # GHz
            "inter_subgraph_freq_ghz": 1.15,  # GHz
            "intra_subgraph_energy_pj_bit": 10.0,  # pJ/bit
            "inter_subgraph_energy_pj_bit": 50.0,  # pJ/bit
            "packet_size_bytes": 32,  # bytes
            "packets_per_node": 100000,  # number of packets each node sends
            "buffer_size": 4,  # packets
            "gateway_node": (1, 2),  # 1_node_6 in 0-indexed coordinates (row 1, col 2)
            "routing": "xy"  # x-y routing
        }
        
        # Override with provided configuration if available
        if config_file:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        # Derived parameters
        self.subgraph_dim = self.config["subgraph_size"]
        self.total_nodes = self.subgraph_dim * self.subgraph_dim
        self.packet_size_bits = self.config["packet_size_bytes"] * 8
        self.intra_cycle_time = 1 / self.config["intra_subgraph_freq_ghz"] * 1e-9  # seconds
        self.inter_cycle_time = 1 / self.config["inter_subgraph_freq_ghz"] * 1e-9  # seconds
        
        # Initialize network structure
        self.initialize_network()
    
    def initialize_network(self):
        """Initialize the network structures."""
        # Create node structures for both subgraphs
        self.subgraph1 = {}
        self.subgraph2 = {}
        
        for i in range(self.subgraph_dim):
            for j in range(self.subgraph_dim):
                # Node ID format: (row, col)
                node_id = (i, j)
                
                # Create nodes in subgraph 1
                self.subgraph1[node_id] = {
                    'buffer': deque(maxlen=self.config["buffer_size"]),
                    'packets_to_send': self.config["packets_per_node"],
                    'packets_sent': 0,
                    'packets_received': 0,
                    'energy_consumed': 0.0
                }
                
                # Create nodes in subgraph 2
                self.subgraph2[node_id] = {
                    'buffer': deque(maxlen=self.config["buffer_size"]),
                    'packets_to_send': 0,  # Subgraph 2 nodes don't initiate sending
                    'packets_sent': 0,
                    'packets_received': 0,
                    'energy_consumed': 0.0
                }
        
        # Initialize gateway node (the single connection between subgraphs)
        self.gateway_node = self.config["gateway_node"]
        
        # Create tracking variables
        self.current_time = 0.0
        self.total_energy = 0.0
        self.network_state_history = []
    
    def run_simulation(self):
        """Run the full simulation until all packets are delivered."""
        total_packets = self.total_nodes * self.config["packets_per_node"]
        delivered_packets = 0
        
        print(f"Starting simulation with {total_packets} total packets to deliver")
        
        # Continue until all packets are delivered
        while delivered_packets < total_packets:
            # Process one time step
            packets_this_step = self.simulate_time_step()
            delivered_packets += packets_this_step
            
            # Record state periodically (e.g., every 1000 time steps)
            if len(self.network_state_history) % 1000 == 0:
                print(f"Time: {self.current_time:.6f}s, Delivered: {delivered_packets}/{total_packets} packets")
        
        print(f"Simulation complete! Total time: {self.current_time:.6f}s, Total energy: {self.total_energy/1e12:.6f}J")
        return {
            'total_time_seconds': self.current_time,
            'total_energy_joules': self.total_energy / 1e12,  # Convert from pJ to J
            'energy_per_packet_nanojoules': (self.total_energy / total_packets) / 1e3,  # Convert from pJ to nJ
            'throughput_packets_per_second': total_packets / self.current_time
        }
    
    def simulate_time_step(self):
        """Simulate one time step in the network."""
        packets_delivered = 0
        
        # Phase 1: Route packets within subgraph 1 toward gateway
        self.route_within_subgraph(self.subgraph1, is_source=True)
        
        # Phase 2: Transfer packets across gateway
        packets_transferred = self.transfer_across_gateway()
        
        # Phase 3: Route packets within subgraph 2 to destination
        packets_delivered = self.route_within_subgraph(self.subgraph2, is_source=False)
        
        # Advance time
        self.current_time += self.intra_cycle_time
        
        # Record network state
        self.network_state_history.append({
            'time': self.current_time,
            'packets_delivered': packets_delivered,
            'total_energy': self.total_energy
        })
        
        return packets_delivered
    
    def route_within_subgraph(self, subgraph, is_source=True):
        """Route packets within a subgraph following X-Y routing."""
        packets_delivered = 0
        
        # Create a copy of the buffer states to simulate simultaneous packet movement
        buffer_state = {node_id: list(node['buffer']) for node_id, node in subgraph.items()}
        
        # For source subgraph, inject new packets if nodes have packets to send
        if is_source:
            for node_id, node in subgraph.items():
                if node['packets_to_send'] > 0 and len(node['buffer']) < self.config["buffer_size"]:
                    # Create new packet and add to buffer
                    destination = node_id  # Same coordinates in the other subgraph
                    new_packet = {
                        'source': node_id,
                        'destination': destination,
                        'current_location': node_id,
                        'next_hop': self.determine_next_hop(node_id, self.gateway_node)
                    }
                    node['buffer'].append(new_packet)
                    node['packets_to_send'] -= 1
                    node['packets_sent'] += 1
                    
                    # Account for energy to create packet
                    energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                    node['energy_consumed'] += energy_consumed
                    self.total_energy += energy_consumed
        
        # Process all nodes and move packets according to routing
        for node_id, node in subgraph.items():
            # Process only if the buffer has packets
            if node['buffer']:
                packet = node['buffer'].popleft()
            
            # Check if at hub already
            if node_id == hub_node:
                # Keep packet in hub buffer for inter-subgraph transfer
                node['buffer'].append(packet)
            else:
                # Get next hop toward hub
                next_hop = packet['next_hop']
                
                # Check if next hop buffer has space
                if len(subgraph[next_hop]['buffer']) < (
                    subgraph[next_hop]['buffer'].maxlen
                ):
                    packet['current_location'] = next_hop
                    packet['next_hop'] = self.determine_next_hop(next_hop, hub_node)
                    
                    # Move packet to next node's buffer
                    subgraph[next_hop]['buffer'].append(packet)
                    
                    # Account for energy to transmit packet
                    energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                    node['energy_consumed'] += energy_consumed
                    self.total_energy += energy_consumed
                else:
                    # Buffer full, keep packet in current buffer
                    node['buffer'].append(packet)popleft()  # Get the first packet in buffer
            
            if is_source:
                # In source subgraph, route toward gateway
                if node_id == self.gateway_node:
                    # If at gateway, wait for inter-subgraph transfer
                    node['buffer'].append(packet)
                else:
                    # Move to next hop
                    next_hop = packet['next_hop']
                    
                    # Check if next hop buffer has space
                    if len(subgraph[next_hop]['buffer']) < self.config["buffer_size"]:
                        packet['current_location'] = next_hop
                        if next_hop == self.gateway_node:
                            # Reached gateway, next hop is to cross to subgraph 2
                            packet['next_hop'] = self.gateway_node
                        else:
                            # Update next hop
                            packet['next_hop'] = self.determine_next_hop(next_hop, self.gateway_node)
                        
                        # Move packet to next node's buffer
                        subgraph[next_hop]['buffer'].append(packet)
                        
                        # Account for energy to transmit packet
                        energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                        node['energy_consumed'] += energy_consumed
                        self.total_energy += energy_consumed
                    else:
                        # Buffer full, keep packet in current buffer
                        node['buffer'].append(packet)
            else:
                # In destination subgraph, route toward final destination
                destination = packet['destination']
                
                if node_id == destination:
                    # Packet reached its destination
                    node['packets_received'] += 1
                    packets_delivered += 1
                else:
                    # Move to next hop toward destination
                    next_hop = packet['next_hop']
                    
                    # Check if next hop buffer has space
                    if len(subgraph[next_hop]['buffer']) < self.config["buffer_size"]:
                        packet['current_location'] = next_hop
                        packet['next_hop'] = self.determine_next_hop(next_hop, destination)
                        
                        # Move packet to next node's buffer
                        subgraph[next_hop]['buffer'].append(packet)
                        
                        # Account for energy to transmit packet
                        energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                        node['energy_consumed'] += energy_consumed
                        self.total_energy += energy_consumed
                    else:
                        # Buffer full, keep packet in current buffer
                        node['buffer'].append(packet)
        
        return packets_delivered
    
    def transfer_across_gateway(self):
        """Transfer packets from subgraph 1 gateway to subgraph 2 gateway."""
        packets_transferred = 0
        
        # Get gateway node in subgraph 1
        gateway1 = self.subgraph1[self.gateway_node]
        
        # Process packets at gateway
        if gateway1['buffer']:
            packet = gateway1['buffer'].popleft()
            
            # Transfer packet to gateway in subgraph 2
            destination = packet['destination']
            next_hop = self.determine_next_hop(self.gateway_node, destination)
            
            packet['current_location'] = self.gateway_node  # Now in subgraph 2
            packet['next_hop'] = next_hop
            
            # Add to subgraph 2 gateway buffer
            self.subgraph2[self.gateway_node]['buffer'].append(packet)
            
            # Account for energy for inter-subgraph transfer (higher energy cost)
            energy_consumed = self.packet_size_bits * self.config["inter_subgraph_energy_pj_bit"]
            gateway1['energy_consumed'] += energy_consumed
            self.total_energy += energy_consumed
            
            packets_transferred += 1
            
            # Add time delay for inter-subgraph transfer
            self.current_time += self.inter_cycle_time - self.intra_cycle_time
        
        return packets_transferred
    
    def determine_next_hop(self, current, target):
        """Determine next hop using X-Y routing."""
        current_row, current_col = current
        target_row, target_col = target
        
        # X-Y routing: first move horizontally, then vertically
        if current_col != target_col:
            # Move horizontally
            next_col = current_col + 1 if current_col < target_col else current_col - 1
            return (current_row, next_col)
        elif current_row != target_row:
            # Move vertically
            next_row = current_row + 1 if current_row < target_row else current_row - 1
            return (next_row, current_col)
        else:
            # Already at destination
            return current
    
    def visualize_results(self):
        """Visualize simulation results."""
        # Extract data for plotting
        times = [state['time'] for state in self.network_state_history]
        packets_delivered = [state['packets_delivered'] for state in self.network_state_history]
        energy_consumed = [state['total_energy'] / 1e12 for state in self.network_state_history]  # Convert to Joules
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot packets delivered over time
        ax1.plot(times, np.cumsum(packets_delivered))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cumulative Packets Delivered')
        ax1.set_title('Packet Delivery Progress')
        ax1.grid(True)
        
        # Plot energy consumption over time
        ax2.plot(times, energy_consumed)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Total Energy Consumed (J)')
        ax2.set_title('Energy Consumption')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('mesh_network_simulation_results.png')
        plt.show()
        
        # Create heatmap of energy consumption per node
        plt.figure(figsize=(12, 5))
        
        # Subgraph 1 energy heatmap
        plt.subplot(1, 2, 1)
        energy_matrix1 = np.zeros((self.subgraph_dim, self.subgraph_dim))
        for (i, j), node in self.subgraph1.items():
            energy_matrix1[i, j] = node['energy_consumed'] / 1e9  # Convert to nJ
        
        plt.imshow(energy_matrix1, cmap='hot')
        plt.colorbar(label='Energy (nJ)')
        plt.title('Subgraph 1 Energy Consumption')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # Subgraph 2 energy heatmap
        plt.subplot(1, 2, 2)
        energy_matrix2 = np.zeros((self.subgraph_dim, self.subgraph_dim))
        for (i, j), node in self.subgraph2.items():
            energy_matrix2[i, j] = node['energy_consumed'] / 1e9  # Convert to nJ
        
        plt.imshow(energy_matrix2, cmap='hot')
        plt.colorbar(label='Energy (nJ)')
        plt.title('Subgraph 2 Energy Consumption')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        plt.tight_layout()
        plt.savefig('mesh_network_energy_heatmap.png')
        plt.show()

class ConcentratedMeshSimulator:
    def __init__(self, config_file=None):
        """Initialize the hub-based concentrated mesh simulator."""
        # Default configuration
        self.config = {
            "subgraph_size": 4,  # 4x4 mesh
            "intra_subgraph_freq_ghz": 2.0,  # GHz
            "inter_subgraph_freq_ghz": 1.15,  # GHz
            "intra_subgraph_energy_pj_bit": 10.0,  # pJ/bit
            "inter_subgraph_energy_pj_bit": 50.0,  # pJ/bit
            "packet_size_bytes": 32,  # bytes
            "packets_per_node": 100000,  # number of packets each node sends
            "buffer_size": 4,  # packets
            "hub_node_subgraph1": (1, 2),  # Hub in subgraph 1 (0-indexed)
            "hub_node_subgraph2": (1, 2),  # Hub in subgraph 2 (0-indexed)
            "communication_pattern": "broadcast"  # "broadcast", "gather", or "point-to-point"
        }
        
        # Override with provided configuration if available
        if config_file:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        # Derived parameters
        self.subgraph_dim = self.config["subgraph_size"]
        self.total_nodes = self.subgraph_dim * self.subgraph_dim
        self.packet_size_bits = self.config["packet_size_bytes"] * 8
        self.intra_cycle_time = 1 / self.config["intra_subgraph_freq_ghz"] * 1e-9  # seconds
        self.inter_cycle_time = 1 / self.config["inter_subgraph_freq_ghz"] * 1e-9  # seconds
        
        # Initialize network structure
        self.initialize_network()
    
    def initialize_network(self):
        """Initialize the network structures."""
        # Create node structures for both subgraphs
        self.subgraph1 = {}
        self.subgraph2 = {}
        
        for i in range(self.subgraph_dim):
            for j in range(self.subgraph_dim):
                # Node ID format: (row, col)
                node_id = (i, j)
                
                # Create nodes in subgraph 1
                self.subgraph1[node_id] = {
                    'buffer': deque(maxlen=self.config["buffer_size"]),
                    'packets_to_send': self.config["packets_per_node"],
                    'packets_sent': 0,
                    'packets_received': 0,
                    'energy_consumed': 0.0
                }
                
                # Create nodes in subgraph 2
                self.subgraph2[node_id] = {
                    'buffer': deque(maxlen=self.config["buffer_size"]),
                    'packets_to_send': 0,  # Subgraph 2 nodes don't initiate sending
                    'packets_sent': 0,
                    'packets_received': 0,
                    'energy_consumed': 0.0
                }
        
        # Hub nodes have larger buffers for broadcast/gather
        self.hub_node1 = self.config["hub_node_subgraph1"]
        self.hub_node2 = self.config["hub_node_subgraph2"]
        
        # Increase buffer size for hub nodes
        self.subgraph1[self.hub_node1]['buffer'] = deque(maxlen=self.config["buffer_size"] * 4)
        self.subgraph2[self.hub_node2]['buffer'] = deque(maxlen=self.config["buffer_size"] * 4)
        
        # Create tracking variables
        self.current_time = 0.0
        self.total_energy = 0.0
        self.network_state_history = []
    
    def run_simulation(self):
        """Run the full simulation until all packets are delivered."""
        total_packets = self.total_nodes * self.config["packets_per_node"]
        delivered_packets = 0
        
        print(f"Starting hub-based simulation with {total_packets} total packets to deliver")
        print(f"Communication pattern: {self.config['communication_pattern']}")
        
        # Continue until all packets are delivered
        while delivered_packets < total_packets:
            # Process one time step
            packets_this_step = self.simulate_time_step()
            delivered_packets += packets_this_step
            
            # Record state periodically
            if len(self.network_state_history) % 1000 == 0:
                print(f"Time: {self.current_time:.6f}s, Delivered: {delivered_packets}/{total_packets} packets")
        
        print(f"Simulation complete! Total time: {self.current_time:.6f}s, Total energy: {self.total_energy/1e12:.6f}J")
        return {
            'total_time_seconds': self.current_time,
            'total_energy_joules': self.total_energy / 1e12,  # Convert from pJ to J
            'energy_per_packet_nanojoules': (self.total_energy / total_packets) / 1e3,  # Convert from pJ to nJ
            'throughput_packets_per_second': total_packets / self.current_time
        }
    
    def simulate_time_step(self):
        """Simulate one time step in the hub-based network."""
        packets_delivered = 0
        
        # Different phases based on communication pattern
        if self.config["communication_pattern"] == "gather":
            # Phase 1: Gather packets from source nodes to hub node in subgraph 1
            self.gather_to_hub(self.subgraph1, self.hub_node1)
            
            # Phase 2: Transfer packets between hub nodes
            packets_transferred = self.transfer_between_hubs()
            
            # Phase 3: Deliver packets from hub to destination nodes in subgraph 2
            packets_delivered = self.distribute_from_hub(self.subgraph2, self.hub_node2)
            
        elif self.config["communication_pattern"] == "broadcast":
            # Phase 1: Gather packets from source nodes to hub node in subgraph 1
            self.gather_to_hub(self.subgraph1, self.hub_node1)
            
            # Phase 2: Transfer packets between hub nodes
            packets_transferred = self.transfer_between_hubs()
            
            # Phase 3: Broadcast packets from hub to all nodes in subgraph 2
            packets_delivered = self.broadcast_from_hub(self.subgraph2, self.hub_node2)
            
        else:  # point-to-point with hub concentration
            # Phase 1: Route packets within subgraph 1 toward hub
            self.route_to_hub(self.subgraph1, self.hub_node1)
            
            # Phase 2: Transfer packets between hubs
            packets_transferred = self.transfer_between_hubs()
            
            # Phase 3: Route packets from hub to destinations in subgraph 2
            packets_delivered = self.route_from_hub(self.subgraph2, self.hub_node2)
        
        # Advance time
        self.current_time += self.intra_cycle_time
        
        # Record network state
        self.network_state_history.append({
            'time': self.current_time,
            'packets_delivered': packets_delivered,
            'total_energy': self.total_energy
        })
        
        return packets_delivered
    
    def gather_to_hub(self, subgraph, hub_node):
        """Gather packets from all nodes to the hub node."""
        # Inject new packets at source nodes
        for node_id, node in subgraph.items():
            if node_id != hub_node and node['packets_to_send'] > 0 and len(node['buffer']) < self.config["buffer_size"]:
                # Create new packet and add to buffer
                destination = node_id  # Same coordinates in the other subgraph
                new_packet = {
                    'source': node_id,
                    'destination': destination,  # Destination in other subgraph
                    'current_location': node_id,
                    'next_hop': self.determine_next_hop(node_id, hub_node)
                }
                node['buffer'].append(new_packet)
                node['packets_to_send'] -= 1
                node['packets_sent'] += 1
                
                # Account for energy to create packet
                energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                node['energy_consumed'] += energy_consumed
                self.total_energy += energy_consumed
        
        # Route packets toward hub
        for node_id, node in subgraph.items():
            if node_id != hub_node and node['buffer']:
                packet = node['buffer'].popleft()
                
                # Get next hop toward hub
                next_hop = packet['next_hop']
                
                # Check if next hop buffer has space
                if len(subgraph[next_hop]['buffer']) < (
                    subgraph[next_hop]['buffer'].maxlen
                ):
                    packet['current_location'] = next_hop
                    packet['next_hop'] = self.determine_next_hop(next_hop, hub_node)
                    
                    # Move packet to next node's buffer
                    subgraph[next_hop]['buffer'].append(packet)
                    
                    # Account for energy to transmit packet
                    energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                    node['energy_consumed'] += energy_consumed
                    self.total_energy += energy_consumed
                else:
                    # Buffer full, keep packet in current buffer
                    node['buffer'].append(packet)
    
    def transfer_between_hubs(self):
        """Transfer packets from hub in subgraph 1 to hub in subgraph 2."""
        packets_transferred = 0
        
        # Get hub nodes in both subgraphs
        hub1 = self.subgraph1[self.hub_node1]
        hub2 = self.subgraph2[self.hub_node2]
        
        # Process packets at hub1
        if hub1['buffer'] and len(hub2['buffer']) < hub2['buffer'].maxlen:
            packet = hub1['buffer'].popleft()
            
            # Transfer packet to hub in subgraph 2
            packet['current_location'] = self.hub_node2  # Now in subgraph 2
            
            # Keep original destination in packet
            if self.config["communication_pattern"] == "broadcast":
                # For broadcast, no specific next hop needed
                packet['next_hop'] = None
            else:
                # For other patterns, set next hop toward final destination
                packet['next_hop'] = self.determine_next_hop(
                    self.hub_node2, packet['destination']
                )
            
            # Add to hub2 buffer
            hub2['buffer'].append(packet)
            
            # Account for energy for inter-subgraph transfer (higher energy cost)
            energy_consumed = self.packet_size_bits * self.config["inter_subgraph_energy_pj_bit"]
            hub1['energy_consumed'] += energy_consumed
            self.total_energy += energy_consumed
            
            packets_transferred += 1
            
            # Add time delay for inter-subgraph transfer
            self.current_time += self.inter_cycle_time - self.intra_cycle_time
        
        return packets_transferred
    
    def broadcast_from_hub(self, subgraph, hub_node):
        """Broadcast packets from hub to all nodes in the subgraph."""
        packets_delivered = 0
        hub = subgraph[hub_node]
        
        # Process packets at hub
        if hub['buffer']:
            packet = hub['buffer'].popleft()
            
            # Broadcast to all nodes (one node per time step to simulate sequential broadcast)
            for node_id, node in subgraph.items():
                if node_id != hub_node:
                    # Create a copy of the packet for each destination node
                    broadcast_packet = packet.copy()
                    broadcast_packet['destination'] = node_id
                    
                    # Mark as received immediately (simplified broadcast model)
                    node['packets_received'] += 1
                    packets_delivered += 1
                    
                    # Account for energy for broadcasting
                    energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                    hub['energy_consumed'] += energy_consumed
                    self.total_energy += energy_consumed
        
        return packets_delivered
    
    def distribute_from_hub(self, subgraph, hub_node):
        """Distribute packets from hub to their specific destinations."""
        packets_delivered = 0
        hub = subgraph[hub_node]
        
        # Process packets at hub
        if hub['buffer']:
            packet = hub['buffer'].popleft()
            destination = packet['destination']
            
            # Direct delivery from hub to destination (simplified model)
            if destination in subgraph:
                subgraph[destination]['packets_received'] += 1
                packets_delivered += 1
                
                # Account for energy for direct delivery
                energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                hub['energy_consumed'] += energy_consumed
                self.total_energy += energy_consumed
        
        return packets_delivered
    
    def route_to_hub(self, subgraph, hub_node):
        """Route packets toward hub using X-Y routing."""
        # Similar to gather_to_hub but with X-Y routing
        # Inject new packets at source nodes
        for node_id, node in subgraph.items():
            if node['packets_to_send'] > 0 and len(node['buffer']) < self.config["buffer_size"]:
                # Create new packet and add to buffer
                destination = node_id  # Same coordinates in the other subgraph
                new_packet = {
                    'source': node_id,
                    'destination': destination,
                    'current_location': node_id,
                    'next_hop': self.determine_next_hop(node_id, hub_node)
                }
                node['buffer'].append(new_packet)
                node['packets_to_send'] -= 1
                node['packets_sent'] += 1
                
                # Account for energy to create packet
                energy_consumed = self.packet_size_bits * self.config["intra_subgraph_energy_pj_bit"]
                node['energy_consumed'] += energy_consumed
                self.total_energy += energy_consumed
        
        # Route packets using X-Y routing
        for node_id, node in subgraph.items():
            if node['buffer']:
                packet = node['buffer'].