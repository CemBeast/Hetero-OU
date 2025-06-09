import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mapperV3
from collections import defaultdict

MAX_CHIPLET_POWER = 8.0  # Watts

def simulate_packet_transfer_graph(G, src, dst, params):
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

def build_floret_graph(rows, cols, lam, psi, workload):
    assert lam * psi == rows * cols
    pos = {(i, j): (j, -i) for i in range(rows) for j in range(cols)}
    all_nodes = [(i, j) for i in range(rows) for j in range(cols)]
    arr_list = [all_nodes[i * psi: (i + 1) * psi] for i in range(lam)]
    arr = np.array(arr_list, dtype=object)

    G = nx.Graph()
    node_labels = {}

    layer_names = list(workload.keys())
    layer_sizes = list(workload.values())

    layer_idx = 0
    for i in range(lam):
        j = 0
        while j < psi and layer_idx < len(layer_names):
            layer = layer_names[layer_idx]
            size = layer_sizes[layer_idx]
            if j + size > psi:
                break
            for k in range(size):
                node = tuple(arr[i][j + k])
                node_labels[node] = f"{node}\n{layer}"
            j += size
            layer_idx += 1

    # Connect all nodes in each SFC
    for i in range(lam):
        for j in range(psi - 1):
            G.add_edge(tuple(arr[i][j]), tuple(arr[i][j + 1]))
    # Connect head and tail of that SFC
    for i in range(lam):
        head, tail = tuple(arr[i][0]), tuple(arr[i][-1])
        G.add_edge(head, tail)
    # Connect tail of one SFC to head of the next
    for i in range(lam - 1):
        G.add_edge(tuple(arr[i][-1]), tuple(arr[i + 1][0]))
    G.add_edge(tuple(arr[-1][-1]), tuple(arr[0][0]))

    return G, arr, pos, node_labels

def display_floret_graph(DG, pos, label_map, lam, psi, show=True):
    plt.figure(figsize=(9, 9))
    colors = plt.get_cmap("tab20")

    # Group nodes into lam SFCs, each with psi chiplets
    all_nodes = list(DG.nodes)
    sfc_groups = [all_nodes[i * psi:(i + 1) * psi] for i in range(lam)]

    for i, sfc in enumerate(sfc_groups):
        nx.draw_networkx_nodes(DG, pos, nodelist=sfc, node_color=[colors(i % 20)], node_size=500)
        edges = [(sfc[j], sfc[j + 1]) for j in range(psi - 1)]
        nx.draw_networkx_edges(DG, pos, edgelist=edges, edge_color=[colors(i % 20)], arrows=True)
        # Head-to-tail links
        head, tail = sfc[0], sfc[-1]
        nx.draw_networkx_edges(DG, pos, edgelist=[(head, tail), (tail, head)], edge_color=[colors(i % 20)], width=2)

    # Dashed links between SFCs
    inter_edges = [(sfc_groups[i][-1], sfc_groups[i + 1][0]) for i in range(lam - 1)]
    inter_edges.append((sfc_groups[-1][-1], sfc_groups[0][0]))
    nx.draw_networkx_edges(DG, pos, edgelist=inter_edges, edge_color='black', style='dashed', arrows=True)

    # Labels
    nx.draw_networkx_labels(DG, pos, labels=label_map, font_size=6, font_color="black")
    nx.draw_networkx_edges(DG, pos, edgelist=DG.edges(), edge_color="gray", style="solid", alpha=0.3)

    plt.title("Floret SFCs with Layer Labels")
    plt.axis("off")
    if show:
        plt.show()

def build_floret_graph_from_results(results):
    chiplet_to_layers = defaultdict(list)
    
    # Step 1: Map each chiplet to all layers that used it
    for lr in results:
        layer_id = lr["layer"]
        for alloc in lr["allocations"]:
            chip_id = alloc["chip_id"]
            chiplet_to_layers[chip_id].append(layer_id)

    chiplet_ids = list(chiplet_to_layers.keys())
    total_chiplets = len(chiplet_ids)
    rows, cols = mapperV3.choose_dims(total_chiplets)
    lam = rows
    psi = cols

    # Step 2: Assign chiplets to (lam x psi) grid
    pos = {(i, j): (j, -i) for i in range(rows) for j in range(cols)}
    all_nodes = [(i, j) for i in range(rows) for j in range(cols)]
    arr_list = [all_nodes[i * psi: (i + 1) * psi] for i in range(lam)]

    arr = [row for row in arr_list]
    arr = [list(map(tuple, row)) for row in arr]

    G = nx.Graph()
    node_labels = {}

    # Step 3: Map chiplet names to grid positions
    mapping = {}
    chiplet_name_map = {}
    i = 0
    for x in range(rows):
        for y in range(cols):
            if i < total_chiplets:
                chiplet = chiplet_ids[i]
                mapping[(x, y)] = chiplet
                chiplet_name_map[chiplet] = pos[(x, y)]
                i += 1

    # Step 4: Build floret links and label nodes for each SFC
    for i in range(lam):
        for j in range(psi - 1):
            G.add_edge(mapping[(i, j)], mapping[(i, j + 1)])
        # Head-tail loop
        G.add_edge(mapping[(i, 0)], mapping[(i, psi - 1)])
        # G.add_edge(mapping[(i, psi - 1)], mapping[(i, 0)])

    # Connect Tail of one SFC to the head of the next
    for i in range(lam - 1):
        G.add_edge(mapping[(i, psi - 1)], mapping[(i + 1, 0)])

    G.add_edge(mapping[(lam - 1, psi - 1)], mapping[(0, 0)])

    # Step 5: Format labels
    label_map = {}
    for chiplet, layers in chiplet_to_layers.items():
        label_map[chiplet] = f"{chiplet}\nL{','.join(map(str, layers))}"

    return G, chiplet_name_map, label_map, lam, psi

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
    workload = {
        "L1": 3, "L2": 2, "L3": 5, "L4": 5, "L5": 5, "L6": 5, "L7": 5, "L8": 5, "L9": 5, 
        "L10": 5, "L11": 5, "L12": 5, "L13": 5, "L14": 5, "L15": 5, "L16": 5, "L17": 5,
        "L18": 5, "L19": 5, "L20": 10
    }

    rows, cols = 2, 7
    lam = 2
    psi = 7

    # DG, arr, pos, node_labels = build_floret_graph(rows, cols, lam, psi, workload)
    # display_floret_graph(DG, arr, pos, node_labels, lam, psi)

    workload_csv = "workloads/vgg16_stats.csv"
    df = pd.read_csv(workload_csv)
    workload = [
        { "layer": int(row["Layer #"]), "activations_kb": float(row["Activations(KB)"]) }
        for _, row in df.iterrows()
    ]
    chip_dist    = [100, 0, 0, 0, 0]# hetOU
    results      = mapperV3.scheduler(workload_csv, chip_dist)
    

    G, pos, label_map, lam, psi = build_floret_graph_from_results(results)
    display_floret_graph(G, pos, label_map, lam, psi)

    logs = mapperV3.simulate_activations_between_layers(workload, results, G, params)
    # Initialize totals
    total_hops = 0
    total_latency_ns = 0
    total_cycles = 0
    total_energy = 0.0
    total_edp = 0.0
    for log in logs:
        print(f"L{log['src_layer']} → L{log['dst_layer']}, {log['src_chiplet']} → {log['dst_chiplet']}, {log['activations_kb']} KBs,{log['hops']} hops, {log['energy_joules']:.2e} J")
        total_hops       += log["hops"]
        total_latency_ns += log["latency_ns"]
        total_cycles     += log["cycles"]
        total_energy     += log["energy_joules"]
        total_edp        += log["edp"]

    print("\n📊 Summary for Floret:")
    print(f"Total Hops: {total_hops}")
    print(f"Latency: {total_latency_ns} ns ({total_cycles} cycles)")
    print(f"Energy: {total_energy:.2e} J")
    print(f"EDP: {total_edp:.2e} J·ns")