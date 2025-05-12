#!/bin/bash

# This script runs simulation to compare mesh network vs. hub-based approaches

echo "Running Network-on-Chip comparison simulation..."

# Run with default parameters first
python mesh_network_simulation.py --compare

# Try different communication patterns
echo "Running with gather communication pattern..."
python mesh_network_simulation.py --compare --comm_pattern gather

echo "Running with point-to-point communication pattern..."
python mesh_network_simulation.py --compare --comm_pattern point-to-point

# Example of changing parameters with custom config files
echo "Creating custom config files..."
cat > custom_mesh_config.json << EOL
{
    "subgraph_size": 4,
    "intra_subgraph_freq_ghz": 2.0,
    "inter_subgraph_freq_ghz": 1.15,
    "intra_subgraph_energy_pj_bit": 10.0,
    "inter_subgraph_energy_pj_bit": 50.0,
    "packet_size_bytes": 32,
    "packets_per_node": 10000,
    "buffer_size": 8,
    "gateway_node": [1, 2],
    "routing": "xy"
}
EOL

cat > custom_hub_config.json << EOL
{
    "subgraph_size": 4,
    "intra_subgraph_freq_ghz": 2.0,
    "inter_subgraph_freq_ghz": 1.15,
    "intra_subgraph_energy_pj_bit": 10.0,
    "inter_subgraph_energy_pj_bit": 50.0,
    "packet_size_bytes": 32,
    "packets_per_node": 10000,
    "buffer_size": 8,
    "hub_node_subgraph1": [1, 2],
    "hub_node_subgraph2": [1, 2],
    "communication_pattern": "broadcast"
}
EOL

echo "Running with custom configurations (larger buffer, fewer packets)..."
python mesh_network_simulation.py --compare --mesh_config custom_mesh_config.json --hub_config custom_hub_config.json

echo "Simulation complete!"