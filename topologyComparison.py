import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import os
import mapperV3
from collections import defaultdict

def build_floret_graph(lam: int, phi: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(lam):
        for j in range(phi):
            G.add_node((i, j))
    # Intra‐row chain + wrap
    for i in range(lam):
        for j in range(phi - 1):
            G.add_edge((i, j), (i, j + 1))
        G.add_edge((i, 0), (i, phi - 1))
    # Inter‐row chain
    for i in range(lam - 1):
        G.add_edge((i, phi - 1), ((i + 1), 0))
    # Final wrapback
    G.add_edge((lam - 1, phi - 1), (0, 0))
    return G

def build_SFC_graph(rows: int, cols:int) -> nx.Graph:
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i,j))
    for i in range(rows):
        for j in range(cols - 2):
            G.add_edge((i, j),(i, j+1))
    for i in range(rows - 1):
        if i % 2 == 0:
            G.add_edge((i,0), (i + 1, 0))
        else:
            G.add_edge((i, cols - 2), (i + 1, cols - 2))
    for i in range(rows - 1):
        G.add_edge((i, cols - 1), (i+1, cols - 1))
    G.add_edge((0, cols - 2), (0, cols - 1))
    G.add_edge((rows - 1, cols - 2), (rows - 1, cols - 1))
    return G

def buildMeshGraph(rows: int, cols: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            if i + 1 < rows:
                G.add_edge((i, j), (i + 1, j))
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))
    return G

def buildKiteGraph(rows: int, cols: int) -> nx.Graph:
    G = nx.Graph()
    # Jump‐by‐2 edges
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            if i + 2 < rows:
                G.add_edge((i, j), (i + 2, j), weight=2)
            if j + 2 < cols:
                G.add_edge((i, j), (i, j + 2), weight=2)
    # Fill up to degree 4
    for i in range(rows):
        for j in range(cols):
            node = (i, j)
            deg = G.degree(node)
            if deg == 4:
                continue
            for nbr in [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]:
                if nbr in G and G.degree(nbr) < 4 and not G.has_edge(node, nbr):
                    G.add_edge(node, nbr)
                    deg += 1
                    if deg == 4:
                        break
    return G

def buildHexaMeshGraph(rows: int, cols: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
            even_row = (i % 2 == 0)
            # Right neighbor
            if j + 1 < cols:
                G.add_edge((i, j), (i, j + 1))
            # Bottom‐right
            if i + 1 < rows:
                offset = 0 if even_row else 1
                if 0 <= j + offset < cols:
                    G.add_edge((i, j), (i + 1, j + offset), weight=2)
            # Bottom‐left
            if i + 1 < rows:
                offset = 1 if even_row else 0
                if 0 <= j - offset < cols:
                    G.add_edge((i, j), (i + 1, j - offset), weight=2)
    return G

def draw_labeled_chiplet_graph(
    G: nx.Graph,
    rows: int,
    cols: int,
    chiplet_to_layers: dict[int, list[int]],
    title: str = "",
    use_curved: bool = False,
    layout: str = "grid"
) -> None:
    """
    Attach a simple default chiplet→layer mapping (chiplet_i hosts layer i+1),
    label each node accordingly, then draw the graph.

    Each node in G is assumed to be a tuple (r, c) on a `rows × cols` grid,
    and we create exactly `rows * cols` chiplets, numbered 0..(rows*cols-1),
    so that the node at (i, j) corresponds to chiplet index = (i*cols + j).

    - G: a networkx.Graph whose nodes are (i, j).
    - rows, cols: grid dimensions (so total chiplets = rows * cols).
    - title: plot title to use.
    - use_curved: if True, calls mapperV3.draw_curved_graph; otherwise, does a straight draw.
    """
    # 1) Build a per‐(r,c) label map
    labels: dict[tuple[int,int], str] = {}
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c                 # chiplet ID
            layer_list = chiplet_to_layers.get(k, [])
            base_name   = f"chiplet_{k}"
            if layer_list:
                layer_str = ",".join(str(L) for L in layer_list)
                text      = f"{base_name}\nL{layer_str}"
            else:
                text = base_name
            labels[(r, c)] = text

    # 2) Attach these labels as a node attribute (optional—nx.draw can take a labels dict directly)
    nx.set_node_attributes(G, labels, name="label")

    # 3) Compute 2D positions for each (row, col):
    if layout == "hexa":
        pos = mapperV3.getHexPositions(rows, cols)
    elif layout == "floret":
        pos = nx.spring_layout(G)
    else:
        pos = mapperV3.get_grid_positions(rows, cols)

    # 4) Draw:
    if use_curved:
        mapperV3.draw_curved_graph(G, pos, labels)  # this makes its own figure/axes
        plt.title(title)
        plt.tight_layout()
        plt.show()

    else:
        # 4b) Straight‐edge branch: start a fresh figure before drawing
        plt.figure(figsize=(6, 6))
        nx.draw(
            G,
            pos,
            labels=labels,
            node_size=800,
            node_color="lightblue",
            font_size=8,
            font_weight="bold"
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()

def load_workload_from_csv(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    workload = []
    for _, row in df.iterrows():
        layer_1b = int(row["Layer #"])
        act_kb   = float(row["Activations(KB)"])
        workload.append({
            "layer": layer_1b - 1,    # convert 1‐based → 0‐based
            "activations_kb": act_kb
        })
    workload.sort(key=lambda x: x["layer"])
    return workload

# To create activationKBs for any size layers/chiplets *since current goal is one layer to one chip
def generate_synthetic_activations(num_layers: int) -> list[dict]:
    """
    If we do not have a CSV (or num_layers ≠ 16), we synthesize activations_kb
    for each adjacent pair (layer i → i+1).  Here we simply start with a
    “base” and halve every two layers, to roughly mimic a CNN‐like pattern.

    Output is a list of length num_layers, but only entries 0..(num_layers-2)
    matter (because we simulate transfers from layer i→i+1 for i=0..num_layers-2).
    """
    base = 32768  # or any number you like
    synthetic = []
    for i in range(num_layers):
        # “activations from layer i → i+1”:
        #   = base / (2**floor(i/2))
        # (so 0,1 use base; 2,3 use base/2; 4,5 use base/4; etc.)
        val = base / (2 ** (i // 2))
        synthetic.append({
            "layer": i,
            "activations_kb": float(val)
        })
    return synthetic

def simulate_activations_between_layers(
    workload: list[dict],
    G: nx.Graph,
    rows: int,
    cols: int,
    layer_to_chiplets: dict[int, list[int]],
    params: dict
) -> list[dict]:
    """
    Inter‐chiplet activation transfers, assuming “layer i” → node (i//cols, i%cols).
    """
    # layer→activations_kb
    layer_to_act = { entry["layer"]: entry["activations_kb"] for entry in workload }
    max_layer = max(layer_to_act.keys())

    #  We'll need to convert an integer chiplet ID into its (row,col) tuple on the grid.
    #     E.g. chiplet ID k → (r, c) where r = k // cols, c = k % cols.
    def chiplet_id_to_tuple(k: int) -> tuple[int, int]:
        return (k // cols, k % cols)

    log = []
    for i in range(max_layer + 1):
        src_layer = i
        dst_layer = i + 1
        act_kb = layer_to_act.get(src_layer, 0.0)
        packet_bits = act_kb * 8192  # bits

        # map to tuple‐nodes
        src_chiplets = layer_to_chiplets.get(src_layer, [])
        dst_chiplets = layer_to_chiplets.get(dst_layer, [])

        for sc in src_chiplets:
            for dc in dst_chiplets:
                # if same chiplet, skip inter‐chip energy
                if sc == dc:
                    continue

                # Convert sc,dc into tuple‐nodes on the grid
                sc_tup = chiplet_id_to_tuple(sc)
                dc_tup = chiplet_id_to_tuple(dc)

                if sc_tup not in G or dc_tup not in G:
                    print(f"⚠️ chiplet {sc} or {dc} not in G, skipping")
                    continue

                try:
                    path = nx.shortest_path(G, source=sc_tup, target=dc_tup, weight="weight")
                except nx.NetworkXNoPath:
                    print(f"❌ no path {sc_tup}→{dc_tup}, skip")
                    continue

                # Physical hops (each step between nodes)
                raw_hops = len(path) - 1
                # Weighted hops (true cost, e.g. jump-by-2 = 2)
                weighted_hops = sum(G[path[i]][path[i+1]].get("weight", 1) for i in range(len(path) - 1))
                latency_ns = weighted_hops * params["noi_hops_latency_ns"]
                energy_j = weighted_hops * packet_bits * params["e_cross"]
                cycles = int(latency_ns * params["freq_inter_hz"] * 1e-9)
                edp = energy_j * latency_ns
                time_s = (packet_bits * weighted_hops) / params["noi_bandwidth_bits_per_sec"]
                #print(f"Activations: {act_kb}, from layer{src_layer + 1} to layer{dst_layer + 1}")

                record = {
                    "src_layer":     src_layer,
                    "dst_layer":     dst_layer,
                    "src_chiplet":   sc,
                    "dst_chiplet":   dc,
                    "activations_kb":act_kb,
                    "hops":          raw_hops,
                    "weighted_hops": weighted_hops,
                    "latency_ns":    latency_ns,
                    "energy_joules": energy_j,
                    "cycles":        cycles,
                    "edp":           edp,
                    "time_s":        time_s,
                    "path":          path
                }
                log.append(record)
    return log

def choose_grid_dimensions(n: int) -> tuple[int, int]:
    """
    Return (rows, cols) so that rows * cols == n and |rows - cols| is minimized.
    In other words, pick the factor of n that is closest to sqrt(n).
    """
    if n <= 1:
        return (n, 1) if n == 1 else (1, 1)
    root = int(math.isqrt(n))
    for r in range(root, 0, -1):
        if n % r == 0:
            return (r, n // r)
    # (Falls back to 1 × n, but the loop above always succeeds at r=1.)
    return (1, n)

def generate_layer_to_chiplets(num_chiplets: int) -> dict[int, list[int]]:
    layer_to_chiplets = {i: [] for i in range(1, 17)}
    chiplet_id = 0
    remaining_chiplets = num_chiplets

    for layer in range(1, 17):
        layers_left = 16 - layer + 1
        chips_left = num_chiplets - chiplet_id

        # Assign fewer chiplets early on, more later
        if layer == 16:
            chips_this_layer = chips_left
        else:
            max_for_layer = chips_left - (layers_left - 1)
            max_for_layer = max(1, max_for_layer)
            # Bias toward assigning fewer chiplets in early layers
            upper = min(2, max_for_layer)
            chips_this_layer = random.randint(1, upper)

        for _ in range(chips_this_layer):
            if chiplet_id >= num_chiplets:
                break
            layer_to_chiplets[layer].append(chiplet_id)
            chiplet_id += 1

    return layer_to_chiplets

def runCustomSim(rows: int, cols: int):
    num_chiplets = rows * cols
    workload = generate_synthetic_activations(num_chiplets)
    num_layers = len(workload)
    # “Hard‑coded” mapping: to test floret
    layer_to_chiplets = {
        1:  [0],        # layer 1 lives on chiplet 0 
        2:  [0],        # layer 2 → chiplet 1
        3:  [0],     # layer 3 is split across chiplets 2 and 3
        4:  [1],        # layer 4 lives on chiplet 3
        5:  [1],        # layer 5 → chiplet 4
        6:  [1],        # layer 6 → chiplet 5
        7:  [2],        # layer 7 → chiplet 6
        8:  [3],        # layer 8 → chiplet 7
        9:  [4],        # layer 9 → chiplet 8
        10: [5],        # layer 10 → chiplet 9
        11: [6],       # layer 11 → chiplet 10
        12: [7,8,9,10],       # layer 12 → chiplet 11
        13: [11,12,13],       # layer 13 → chiplet 12
        14: [14],       # layer 14 → chiplet 13
        15: [15],       # layer 15 → chiplet 14
        16: [15]
    }
    # layer_to_chiplets = {
    #     1:  [0],        # layer 1 lives on chiplet 0 
    #     2:  [1],        # layer 2 → chiplet 1
    #     3:  [2],     # layer 3 is split across chiplets 2 and 3
    #     4:  [3],        # layer 4 lives on chiplet 3
    #     5:  [4],        # layer 5 → chiplet 4
    #     6:  [5],        # layer 6 → chiplet 5
    #     7:  [6],        # layer 7 → chiplet 6
    #     8:  [7],        # layer 8 → chiplet 7
    #     9:  [8],        # layer 9 → chiplet 8
    #     10: [9],        # layer 10 → chiplet 9
    #     11: [10],       # layer 11 → chiplet 10
    #     12: [11],       # layer 12 → chiplet 11
    #     13: [12],       # layer 13 → chiplet 12
    #     14: [13],       # layer 14 → chiplet 13
    #     15: [14],       # layer 15 → chiplet 14
    #     16: [15]
    # }
    layer_to_chiplets = generate_layer_to_chiplets(rows*cols)
    print(layer_to_chiplets)
    # 4) invert to chiplet→layers
    chiplet_to_layers = defaultdict(list)
    for layer_idx, clist in layer_to_chiplets.items():
        for c in clist:
            chiplet_to_layers[c].append(layer_idx)

    rows, cols = choose_grid_dimensions(num_layers)
    print(f"Using rows={rows}, cols={cols} (exactly {rows*cols} = {num_layers})")

    # 3) Build your topology of choice (tuple‐nodes):
    floret_G = build_floret_graph(rows, cols)
    mesh_G   = buildMeshGraph(rows, cols)
    kite_G   = buildKiteGraph(rows, cols)
    hexa_G   = buildHexaMeshGraph(rows, cols)
    SFC_G = build_SFC_graph(rows,cols)

    for name, G_top in [
        ("floret", floret_G),
        ("mesh",   mesh_G),
        ("kite",   kite_G),
        ("hexa",   hexa_G),
        ("SFC", SFC_G)
    ]:
        print(f"\n===== Simulating on {name} topology =====")
        sim_log = simulate_activations_between_layers(workload, G_top, rows, cols, layer_to_chiplets, params)

        total_hops       = sum(r["hops"]          for r in sim_log)
        total_weighted_hops = sum(r["weighted_hops"] for r in sim_log)
        total_latency_ns = sum(r["latency_ns"]    for r in sim_log)
        total_energy     = sum(r["energy_joules"] for r in sim_log)
        total_edp        = sum(r["edp"]           for r in sim_log)
        # for rec in sim_log:
        #     print(f"    L{rec['src_layer']}→L{rec['dst_layer']}, "
        #           f"C{rec['src_chiplet']}→C{rec['dst_chiplet']}, "
        #           f"hops={rec['hops']}, weighted_hops={rec['weighted_hops']} energy={rec['energy_joules']:.2e} J")

        print(f"Topology = {name}")
        print(f"  #transfers:    {len(sim_log)}")
        print(f"  total_hops:     {total_hops}")
        print(f"  total_weighted_hoops:   {total_weighted_hops}")
        print(f"  total_latency:  {total_latency_ns:.1f} ns")
        print(f"  total_energy:   {total_energy:.2e} J")
        print(f"  total_EDP:      {total_edp:.2e} J·ns")

    draw_labeled_chiplet_graph(G=floret_G, rows=rows, cols=cols, chiplet_to_layers=chiplet_to_layers, title=f"Floret",use_curved=True,layout="floret")
    # draw_labeled_chiplet_graph(G=mesh_G, rows=rows, cols=cols, chiplet_to_layers=chiplet_to_layers, title=f"Mesh",use_curved=False, layout="grid")
    # draw_labeled_chiplet_graph(G=kite_G, rows=rows, cols=cols, chiplet_to_layers=chiplet_to_layers, title=f"Kite",use_curved=True, layout="grid")
    # draw_labeled_chiplet_graph(G=hexa_G, rows=rows, cols=cols, chiplet_to_layers=chiplet_to_layers, title=f"Hexamesh",use_curved=False, layout="hexa")
    #draw_labeled_chiplet_graph(G=SFC_G, rows=rows, cols=cols, chiplet_to_layers=chiplet_to_layers, title=f"SFC",use_curved=False, layout="grid")

if __name__ == "__main__":
    params = {
        "noi_hops_latency_ns":       0.8694,
        "e_cross":                   50e-12,
        "freq_inter_hz":             1.15e9,
        "noi_bandwidth_bits_per_sec":36.8e9
    }

    csvs = [
        "workloads/resnet18_stats.csv",
        "workloads/resnet50_stats.csv",
        "workloads/resnet152_stats.csv",
        "workloads/vgg16_stats.csv",
        "workloads/vgg19_stats.csv",
        "workloads/densenet121_stats.csv",
        "workloads/mobilenetv2_stats.csv"
    ]
    
    runCustomSim(4,4)

    # for csv_path in csvs:
    #     # 1) Load CSV
    #     if not os.path.isfile(csv_path):
    #         print(f"ERROR: '{csv_path}' does not exist. Skipping.")
    #         continue
    #     workload = load_workload_from_csv(csv_path)
    #     num_layers = len(workload)
    #     print(f"Loaded {num_layers} layers from {csv_path}.")

    # # This makes the shape more square like but wastes chiplets 
    #     # 2) Shape → lam×phi (one layer per chiplet)
    #     # rows = int(np.floor(np.sqrt(num_layers)))
    #     # if rows == 0:
    #     #     rows = 1
    #     # cols = int(np.ceil(num_layers / rows))
    # # Gets the exact number of chiplets for it
    #     rows, cols = choose_grid_dimensions(num_layers)
    #     print(f"Using rows={rows}, cols={cols} (exactly {rows*cols} = {num_layers})")

    #     # 3) Build your topology of choice (tuple‐nodes):
    #     floret_G = build_floret_graph(rows, cols)
    #     mesh_G   = buildMeshGraph(rows, cols)
    #     kite_G   = buildKiteGraph(rows, cols)
    #     hexa_G   = buildHexaMeshGraph(rows, cols)
    #     for name, G_top in [
    #         ("floret", floret_G),
    #         ("mesh",   mesh_G),
    #         ("kite",   kite_G),
    #         ("hexa",   hexa_G)
    #     ]:
    #         print(f"\n===== Simulating on {name} topology =====")
    #         sim_log = simulate_activations_between_layers(workload, G_top, rows, cols, params)

    #         total_hops       = sum(r["hops"]          for r in sim_log)
    #         total_latency_ns = sum(r["latency_ns"]    for r in sim_log)
    #         total_energy     = sum(r["energy_joules"] for r in sim_log)
    #         total_edp        = sum(r["edp"]           for r in sim_log)

    #         print(f"Topology = {name}")
    #         print(f"  #transfers:    {len(sim_log)}")
    #         print(f"  total_hops:     {total_hops}")
    #         print(f"  total_latency:  {total_latency_ns:.1f} ns")
    #         print(f"  total_energy:   {total_energy:.2e} J")
    #         print(f"  total_EDP:      {total_edp:.2e} J·ns")
    #         # if sim_log:
    #         #     #print("  First 5 transfers:")
    #         #     for rec in sim_log:
    #         #         print(f"    L{rec['src_layer']}→L{rec['dst_layer']}, "
    #         #               f"{rec['src_chiplet']}→{rec['dst_chiplet']}, "
    #         #               f"hops={rec['hops']}, energy={rec['energy_joules']:.2e} J")
    #         # else:
    #         #     print("  (no inter‐chiplet transfers)")

    #     # 4) (Optional) Draw the floret with labels
    #     base = os.path.splitext(os.path.basename(csv_path))[0]    # e.g. "vgg16_stats"
    #     draw_labeled_chiplet_graph(G=floret_G, rows=rows, cols=cols, title=f"Floret for {base}",use_curved=True,layout="grid")
    #     draw_labeled_chiplet_graph(G=mesh_G, rows=rows, cols=cols, title=f"Mesh for {base}",use_curved=False, layout="grid")
    #     draw_labeled_chiplet_graph(G=kite_G, rows=rows, cols=cols, title=f"Kite for {base}",use_curved=True, layout="grid")
    #     draw_labeled_chiplet_graph(G=hexa_G, rows=rows, cols=cols, title=f"Hexamesh for {base}",use_curved=False, layout="hexa")