import networkx as nx
import matplotlib.pyplot as plt
import random

FREQUENCY_HZ = 1e9  # 1 GHz

def create_mesh_noc_with_entry(size_x=2, size_y=2, chiplet_coords=(0, 0)):
    """
    Creates a mesh-style NoC topology for a chiplet with clear node naming.
    Names will follow: chipletX_Y_nodeI_J
    Also returns the (0,0) node as the NoI entry point.
    """
    i_chiplet, j_chiplet = chiplet_coords
    prefix = f"C{i_chiplet}_{j_chiplet}_N"
    G = nx.grid_2d_graph(size_x, size_y)
    mapping = {(i, j): f"{prefix}{i}_{j}" for i in range(size_x) for j in range(size_y)}
    G = nx.relabel_nodes(G, mapping)

    pos = {f"{prefix}{i}_{j}": (j, -i) for i in range(size_x) for j in range(size_y)}
    entry_node = f"{prefix}0_0"  # top-left node
    return G, pos, entry_node

def create_chiplet_mesh_system(
    mesh_dim=(2, 2),
    chiplet_noc_dim=(2, 2)
):
    """
    Creates a customizable mesh of chiplet subgraphs with proper naming and NoI entry nodes.
    mesh_dim: (rows, cols) for overall chiplet mesh
    chiplet_noc_dim: (rows, cols) for each chiplet NoC
    """
    mesh_rows, mesh_cols = mesh_dim
    noc_rows, noc_cols = chiplet_noc_dim

    full_system = nx.Graph()
    layout_pos = {}
    tile_entries = {}

    for i in range(mesh_rows):
        for j in range(mesh_cols):
            subgraph, sub_pos, entry_node = create_mesh_noc_with_entry(
                noc_rows, noc_cols, chiplet_coords=(i, j)
            )
            full_system = nx.compose(full_system, subgraph)

            offset_x, offset_y = j * (noc_cols + 2), -i * (noc_rows + 2)
            for node, (x, y) in sub_pos.items():
                layout_pos[node] = (x + offset_x, y + offset_y)

            tile_entries[(i, j)] = entry_node

    # Add NoI links between entry nodes
    for i in range(mesh_rows):
        for j in range(mesh_cols):
            curr_entry = tile_entries[(i, j)]

            if j + 1 < mesh_cols:
                right_entry = tile_entries[(i, j + 1)]
                full_system.add_edge(curr_entry, right_entry)

            if i + 1 < mesh_rows:
                bottom_entry = tile_entries[(i + 1, j)]
                full_system.add_edge(curr_entry, bottom_entry)

    # Draw the full system
    # plt.figure(figsize=(8, 8))
    # nx.draw(full_system, layout_pos, with_labels=True, node_size=600, node_color='lightyellow', font_size=7)
    # plt.title(f"{mesh_rows}x{mesh_cols} Chiplet Mesh (each {noc_rows}x{noc_cols} NoC, NoI via top-left node)")
    # plt.show()

    return full_system, tile_entries


def simulate_packet_transfer_with_chiplets(G, src, dst, params):
    """
    Simulates packet transfer across a multi-chiplet graph.
    Uses e_intra for NoC (within chiplet) and e_cross for NoI (between chiplets).
    """
    path = nx.shortest_path(G, src, dst)
    num_hops = len(path) - 1
    packet_bits = params["packet_size_bytes"] * 8

    total_latency_ns = 0
    total_energy = 0.0
    intra_energy = 0.0
    inter_energy = 0.0
    intra_hops = 0
    inter_hops = 0

    # Identify chiplet ID from a node name like "chiplet0_1_node2_1"
    def get_chiplet_id(node):
        return "_".join(node.split("_")[:2])  # e.g., "chiplet0_1"

    prev_chiplet = get_chiplet_id(path[0])

    for i in range(1, len(path)):
        curr_node = path[i]
        curr_chiplet = get_chiplet_id(curr_node)

        is_cross_chip = (curr_chiplet != prev_chiplet)
        if is_cross_chip:
            latency = params["noi_hops_latency_ns"]
            energy = packet_bits * params["e_cross"]
            inter_energy += energy
            inter_hops += 1
        else:
            latency = params["noc_hops_latency_ns"]
            energy = packet_bits * params["e_intra"]
            intra_energy += energy
            intra_hops += 1

        # Update totals
        total_latency_ns += latency
        total_energy += energy
        prev_chiplet = curr_chiplet

    cycles_intra = int(params["noc_hops_latency_ns"] * intra_hops * params["freq_intra_hz"] * 1e-9)
    cycles_inter = int(params["noi_hops_latency_ns"] * inter_hops * params["freq_inter_hz"] * 1e-9)
    total_cycles = cycles_intra + cycles_inter

    edp = total_energy * total_latency_ns
    
    time_per_packet_intra = ( packet_bits * intra_hops / params["noc_bandwidth_bits_per_sec"])
    time_per_packet_inter = (packet_bits * inter_hops / params["noi_bandwidth_bits_per_sec"])
    time_per_packet_s = time_per_packet_intra + time_per_packet_inter

    #print(f"Path: {path}")
    print(f"Total Hops: {num_hops}")
    print(f"Latency: {total_latency_ns} ns ({total_cycles} cycles)")
    print(f"Energy: {total_energy:.2e} J")
    print(f"EDP: {edp:.2e} J·ns")
    print(f"Intra hops: {intra_hops}")
    print(f"Inter hops: {inter_hops}")
    if intra_hops > 0:
        avg_intra_energy_per_hop = intra_energy / intra_hops
        print(f"Avg energy per NoC hop: {avg_intra_energy_per_hop:.2e} J")
    else:
        print("Avg energy per NoC hop: N/A (0 intra hops)")

    if inter_hops > 0:
        avg_inter_energy_per_hop = inter_energy / inter_hops
        print(f"Avg energy per NoI hop: {avg_inter_energy_per_hop:.2e} J")
    else:
        print("Avg energy per NoI hop: N/A (0 inter hops)")
    print(f"Intra energy: {intra_energy:.2e} J")
    print(f"Inter energy: {inter_energy:.2e} J")
    print(f"NoC energy usage:  {(intra_energy / total_energy) * 100:.2f}%")
    print(f"NoI energy usage:  {(inter_energy / total_energy) * 100:.2f}%")
    print(f"Time per packet (based on bandwidth): {time_per_packet_s:.2e} s")

    return {
        "path": path,
        "hops": num_hops,
        "cycles": total_cycles,
        "latency_ns": total_latency_ns,
        "energy_joules": total_energy,
        "inter_hops": inter_hops,
        "intra_hops": intra_hops,
        "avg_intra_energy_per_hop": avg_intra_energy_per_hop if intra_hops > 0 else None,
        "avg_inter_energy_per_hop": avg_inter_energy_per_hop if inter_hops > 0 else None,
        "intra_energy": intra_energy,
        "inter_energy": inter_energy,
        "edp": edp,
        "time_per_packet_s": time_per_packet_s
    }

def compare_noc_vs_noi(noc_metrics, noi_metrics):
    print("\n🟦 NoC Communication Results")
    print(f"Latency: {noc_metrics['latency_ns']} ns")
    print(f"Energy:  {noc_metrics['energy_joules']:.2e} J")
    print(f"EDP:     {noc_metrics['edp']:.2e} J·ns")

    print("\n🟨 NoI Communication Results")
    print(f"Latency: {noi_metrics['latency_ns']} ns")
    print(f"Energy:  {noi_metrics['energy_joules']:.2e} J")
    print(f"EDP:     {noi_metrics['edp']:.2e} J·ns")

    print("\n📊 Relative Comparison (NoI vs NoC)")
    print(f"Latency Ratio: {noi_metrics['latency_ns'] / noc_metrics['latency_ns']:.2f}×")
    print(f"Energy Ratio:  {noi_metrics['energy_joules'] / noc_metrics['energy_joules']:.2f}×")
    print(f"EDP Ratio:     {noi_metrics['edp'] / noc_metrics['edp']:.2f}×")

def run_random_noc_simulations(params, num_runs=10, max_chiplet_dim=10, max_mesh_dim=10):
    for i in range(num_runs):
        mesh_x = random.randint(1, max_mesh_dim)
        mesh_y = random.randint(1, max_mesh_dim)
        chiplet_x = random.randint(1, max_chiplet_dim)
        chiplet_y = random.randint(1, max_chiplet_dim)

        chiplet_graph_mesh, _ = create_chiplet_mesh_system(
            mesh_dim=(mesh_x, mesh_y),
            chiplet_noc_dim=(chiplet_x, chiplet_y)
        )

        src_chip_x = random.randint(0, mesh_x - 1)
        src_chip_y = random.randint(0, mesh_y - 1)
        dst_chip_x = random.randint(0, mesh_x - 1)
        dst_chip_y = random.randint(0, mesh_y - 1)

        src_node_x = random.randint(0, chiplet_x - 1)
        src_node_y = random.randint(0, chiplet_y - 1)
        dst_node_x = random.randint(0, chiplet_x - 1)
        dst_node_y = random.randint(0, chiplet_y - 1)

        src = f"C{src_chip_x}_{src_chip_y}_N{src_node_x}_{src_node_y}"
        dst = f"C{dst_chip_x}_{dst_chip_y}_N{dst_node_x}_{dst_node_y}"

        print(f"\n▶️ Run {i + 1}")
        print(f"Mesh Dimensions: {mesh_x} x {mesh_y}")
        print(f"Chiplet Dimensions: {chiplet_x} x {chiplet_y}")
        print(f"Source: {src}")
        print(f"Destination: {dst}")

        try:
            simulate_packet_transfer_with_chiplets(
                chiplet_graph_mesh,
                src,
                dst,
                params
            )
        except Exception as e:
            print(f"❌ Simulation failed: {e}")


def buildMeshGraph(rows, cols):
    G = nx.Graph()
            
    for i in range(rows):
        for j in range(cols):
            G.add_node((i,j))
            if i + 1 < rows:
                G.add_edge((i, j), (i + 1, j))
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))

    return G

def buildKiteGraph(rows, cols):
    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            if i + 2 < rows:
                G.add_edge((i, j), (i + 2, j))
            if j + 2 < cols:
                G.add_edge((i, j), (i, j + 2))

    for i in range(rows):
        for j in range(cols):
            node = (i,j)
            current_degree = G.degree(node)
            if current_degree == 4:
                continue
            # Define neighbor directions to try (right, down, left, up)
            neighbors = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
            for nbr in neighbors:
                if nbr in G.nodes and G.degree(nbr) < 4 and not G.has_edge(node, nbr):
                    G.add_edge(node, nbr)
                    current_degree += 1
                    if current_degree == 4:
                        break

    return G

def buildHexaMeshGraph(rows, cols):
    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            even_row = (i % 2 == 0)

            # Right neighbor
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))
            # Bottom-right neighbor
            if i + 1 < rows and (j + (0 if even_row else 1)) < cols:
                G.add_edge((i, j), (i + 1, j + (0 if even_row else 1)))
            # Bottom-left neighbor
            if i + 1 < rows and (j - (1 if even_row else 0)) >= 0:
                G.add_edge((i, j), (i + 1, j - (1 if even_row else 0)))

    return G

def simulate_packet_transfer_graph(G, src, dst, params):
    """
    Simulates packet transfer over a single-level graph topology (e.g. mesh, kite, hexamesh).
    Assumes every hop is an interposer-level (NoI) hop.
    """
    path = nx.shortest_path(G, source=src, target=dst)
    num_hops = len(path) - 1
    packet_bits = params["packet_size_bytes"] * 8

    latency_per_hop_ns = params["noi_hops_latency_ns"]
    energy_per_bit = params["e_cross"]
    bandwidth_bps = params["noi_bandwidth_bits_per_sec"]
    frequency = params["freq_inter_hz"]

    total_latency_ns = num_hops * latency_per_hop_ns
    total_energy = num_hops * packet_bits * energy_per_bit
    total_cycles = int(total_latency_ns * frequency * 1e-9)
    edp = total_energy * total_latency_ns

    time_per_packet_s = (packet_bits * num_hops) / bandwidth_bps

    print(f"Path: {path}")
    print(f"Total Hops: {num_hops}")
    print(f"Latency: {total_latency_ns} ns ({total_cycles} cycles)")
    print(f"Energy: {total_energy:.2e} J")
    print(f"EDP: {edp:.2e} J·ns")
    print(f"Time per packet (based on bandwidth): {time_per_packet_s:.2e} s")

    return {
        "path": path,
        "hops": num_hops,
        "latency_ns": total_latency_ns,
        "energy_joules": total_energy,
        "cycles": total_cycles,
        "edp": edp,
        "time_per_packet_s": time_per_packet_s
    }

# ------------ DFS -------------
def dfs_all_paths(G, start, goal, path=None, visited=None):
    if path is None:
        path = [start]
    if visited is None:
        visited = set()

    visited.add(start)

    if start == goal:
        return [path]

    paths = []
    for neighbor in G.neighbors(start):
        if neighbor not in visited:
            new_paths = dfs_all_paths(G, neighbor, goal, path + [neighbor], visited.copy())
            paths.extend(new_paths)

    return paths

def dfs_shortest_path(G, start, goal):
    all_paths = dfs_all_paths(G, start, goal)
    if not all_paths:
        return None
    return min(all_paths, key=len)

if __name__ == "__main__":
    params = {
    "noc_hops_latency_ns": 1,
    "noi_hops_latency_ns": 0.8694,  # 1.449 mm × 0.6 ns/mm
    "packet_size_bytes": 64,
    "freq_intra_hz": 2e9,
    "freq_inter_hz": 1.15e9,
    "e_intra": 10e-12,
    "e_cross": 50e-12,
    "noc_bandwidth_bits_per_sec": 32e9,
    "noi_bandwidth_bits_per_sec": 36.8e9
}
    # For ideal 50/50 split NoC/NoI we want the ratio of intra hops to inter hops
    # to be the same as the ratio of e_cross / e_intra


    # # NoC
    # print("NoC Congiguration")
    # src="C0_0_N0_0"
    # dst="C2_2_N24_24"
    # G, tile_centers_mesh = create_chiplet_mesh_system(mesh_dim=(3, 3),chiplet_noc_dim=(25, 25))
    # noc_results = simulate_packet_transfer_with_chiplets(G, src, dst, params)

    # print("\nNoI Congiguration")
    # # NoI
    # src="C0_0_N0_0"
    # dst="C24_24_N2_2"
    # G, tile_centers_mesh = create_chiplet_mesh_system(mesh_dim=(25, 25),chiplet_noc_dim=(3, 3))
    # noi_results = simulate_packet_transfer_with_chiplets(G, src, dst, params)

    # compare_noc_vs_noi(noc_results, noi_results)

    # run_random_noc_simulations(params)


    G = buildMeshGraph(4, 4)  # or buildKiteGraph / buildHexaMeshGraph
    src = (0, 0)
    dst = (3, 3)
    simulate_packet_transfer_graph(G, src, dst, params)