import math
import itertools
import random
import csv
import matplotlib.pyplot as plt

# Define chiplet types
chiplet_types = {
    "Standard": {"Size": 16384, "Bits/cell": 2, "TOPS": 30e12, "Energy/MAC": 0.87e-12},
    "Shared": {"Size": 583696, "Bits/cell": 1, "TOPS": 27e12, "Energy/MAC": 0.30e-12},
    "Adder": {"Size": 4096, "Bits/cell": 1, "TOPS": 11e12, "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536, "Bits/cell": 2, "TOPS": 35e12, "Energy/MAC": 0.22e-12},
    "ADC_Less": {"Size": 16384, "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12}
}

def computeTimeEnergy(weights, numMACs, chipletType, chipletCount):
    size = chipletType["Size"]
    bitsPerCell = chipletType["Bits/cell"]
    tops = chipletType["TOPS"]
    energyPerMAC = chipletType["Energy/MAC"]
    storage = math.ceil((weights * (8 / bitsPerCell)) / (size * 40 * 16))
    storageScale = storage / chipletCount
    time = (numMACs / tops) * storageScale
    energy = (numMACs * energyPerMAC) * storageScale
    return time, energy

# Chiplet configuration
chiplet_config = {
    "Standard": 2, "Adder": 3, "Accumulator": 2, "Shared": 2, "ADC_Less": 2
}

# Generate MAC operations
random.seed(42)
mac_ops = [(random.randint(int(5e7), int(1.2e8)), random.randint(int(5e7), int(1.2e8))) for _ in range(10)]

# Get chiplet permutations
chiplets = sum(([k]*v for k, v in chiplet_config.items()), [])
perms = random.sample(list(itertools.permutations(chiplets, len(mac_ops))), 10)

# Compute latency/energy per permutation
results = []
for p in perms:
    T, E = 0, 0
    for i, name in enumerate(p):
        t, e = computeTimeEnergy(*mac_ops[i], chiplet_types[name], chiplet_config[name])
        T += t
        E += e
    results.append((p, T, E))

# Save results to CSV
with open("chiplet_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Ordering_Label", "Permutation", "Total_Latency", "Total_Energy"])
    for idx, (perm, T, E) in enumerate(results, 1):
        label = f"Ordering {idx}"
        writer.writerow([label, " → ".join(perm), T, E])

# Normalize for plotting
base_latency = results[0][1]
base_energy = results[0][2]
normalized_latencies = [T / base_latency for _, T, _ in results]
normalized_energies = [E / base_energy for _, _, E in results]
ordering_labels = [f"Ordering {i+1}" for i in range(len(results))]

# Plot and save as PNG
fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axs[0].bar(ordering_labels, normalized_latencies, color='dodgerblue')
axs[0].set_ylabel("Normalized Latency")
axs[0].set_title("Normalized Latency per Chiplet Ordering (Base = Ordering 1)")
axs[0].grid(True, axis='y')

axs[1].bar(ordering_labels, normalized_energies, color='seagreen')
axs[1].set_ylabel("Normalized Energy")
axs[1].set_title("Normalized Energy per Chiplet Ordering (Base = Ordering 1)")
axs[1].set_xticks(range(len(ordering_labels)))
axs[1].set_xticklabels(ordering_labels, rotation=45, ha='right')
axs[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig("chiplet_plot.png")  # ✅ Save instead of showing
plt.close()