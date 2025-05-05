import pandas as pd
import pprint

# Unified dictionary of all chiplet types
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

def get_chip_capacity_bits(chip_type, tiles=16, xbars=96):
    """
    Returns total storage capacity of one chip (in bits),
    computed as: Size  * Bits/cell * #crossbars * #tiles.
    """
    info = chipletTypesDict[chip_type]
    return info["Size"] * info["Bits/cell"] * xbars * tiles

# -----------------------------------------------------------------------------
# 2) Scheduler function
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
    # 2.1 Build inventory of chip instances
    chip_types = list(chipletTypesDict.keys())
    chip_inventory = []
    for chip_type, count in zip(chip_types, chip_distribution):
        for idx in range(count):
            chip_inventory.append({
                "id":      f"{chip_type}_{idx}",
                "type":    chip_type,
                "capacity_left": get_chip_capacity_bits(chip_type)
            })

    # 2.2 Load workload and compute adjusted weights in bits
    df = pd.read_csv(csv_path)
    df["Adjusted_Weights_KB"]  = df["Weights(KB)"] * df["Weight_Sparsity(0-1)"]
    # KB → bits: *1024 (per your spec)
    df["Adjusted_Weights_bits"] = df["Adjusted_Weights_KB"] * 1024

    # 2.3 Allocate each layer across the chips
    layer_allocations = []
    used_chip_order = []
    for _, row in df.iterrows():
        layer_id      = int(row["Layer #"])
        remaining_bits = row["Adjusted_Weights_bits"]
        allocations   = []

        for chip in chip_inventory:
            if remaining_bits <= 0:
                break
            if chip["capacity_left"] <= 0:
                continue

            # allocate as much as we can on this chip
            alloc = min(remaining_bits, chip["capacity_left"])
            chip["capacity_left"] -= alloc
            remaining_bits          -= alloc

            allocations.append({
                "chip_id":       chip["id"],
                "chip_type":     chip["type"],
                "allocated_bits": int(alloc)
            })
            used_chip_order.append(chip["id"])

        if remaining_bits > 0:
            raise RuntimeError(f"Layer {layer_id} could not be fully allocated: {remaining_bits:.0f} bits remain")

        layer_allocations.append({
            "layer":       layer_id,
            "allocations": allocations
        })

    return layer_allocations, chip_inventory, used_chip_order

# -----------------------------------------------------------------------------
# 3) Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Path to your workload CSV
    workload_csv = "workloads/vgg16_stats.csv"

    # Example: [10 Standard, 0 Shared, 0 Adder, 0 Accumulator, 1 ADC_Less]
    chip_dist = [24, 28, 0, 18, 12]

    allocations, inventory, used_order = scheduler(workload_csv, chip_dist)

    # 3.1 Pretty‑print the results
    print("\n=== Layer Allocations ===")
    pprint.pprint(allocations, width=120)

    # 3.2 Summary: count chips used per type
    #   A chip is "used" if capacity_left < original_capacity
    orig_caps = {chip["id"] : get_chip_capacity_bits(chip["type"]) for chip in inventory}
    used_summary = {}
    for chip in inventory:
        if chip["capacity_left"] < orig_caps[chip["id"]]:
            used_summary[chip["type"]] = used_summary.get(chip["type"], 0) + 1
    
    print("\n=== Chips Used Per Type ===")
    for ctype, cnt in used_summary.items():
        print(f" {ctype}: {cnt}")

    # 3.3 Last chip used and its fill percentage
    last_chip_id = used_order[-1]
    last_chip = next(c for c in inventory if c["id"] == last_chip_id)
    orig_capacity = orig_caps[last_chip_id]
    used_bits = orig_capacity - last_chip["capacity_left"]
    pct_full = (used_bits / orig_capacity) * 100

    print(f"\nLast chip used: {last_chip_id} ({last_chip['type']})")
    print(f"  -> Filled: {used_bits:.0f} bits / {orig_capacity} bits = {pct_full:.2f}%")