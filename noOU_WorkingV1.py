
import math
import itertools
import random
import matplotlib.pyplot as plt

# Define chiplet types
chiplet_types = {
    "Standard": {"Size": 16384, "Bits/cell": 2, "TOPS": 30e12, "Energy/MAC": 0.87e-12},
    "Shared": {"Size": 583696, "Bits/cell": 1, "TOPS": 27e12, "Energy/MAC": 0.30e-12},
    "Adder": {"Size": 4096, "Bits/cell": 1, "TOPS": 11e12, "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536, "Bits/cell": 2, "TOPS": 35e12, "Energy/MAC": 0.22e-12},
    "ADC_Less": {"Size": 16384, "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12}
}

# Compute function
def computeTimeEnergy(weights, numMACs, chipletType, chipletCount):
    numTiles = 40
    numChiplets = 16
    size = chipletType["Size"]
    bitsPerCell = chipletType["Bits/cell"]
    tops = chipletType["TOPS"]
    energyPerMAC = chipletType["Energy/MAC"]

    storage = math.ceil((weights * (8 / bitsPerCell)) / (size * numTiles * numChiplets))
    storageScale = storage / chipletCount
    timeTaken = numMACs / tops
    energy = numMACs * energyPerMAC
    realTime = storageScale * timeTaken
    realEnergy = storageScale * energy
    return realTime, realEnergy

# Heterogeneous chiplet configuration
hetero_chiplet_configuration = {
    "Standard": 2,
    "Adder": 3,
    "Accumulator": 2,
    "Shared": 2,
    "ADC_Less": 2
}

# Generate 10 MAC operations
random.seed(42)
mac_operations_extended = [
    (random.randint(int(5e7), int(1.2e8)), random.randint(int(5e7), int(1.2e8))) for _ in range(10)
]

# Generate chiplet sequence and permutations
chiplet_sequence = []
for name, count in hetero_chiplet_configuration.items():
    chiplet_sequence.extend([name] * count)

valid_permutations = list(itertools.permutations(chiplet_sequence, len(mac_operations_extended)))
sampled_permutations = random.sample(valid_permutations, 10)

# Compute results
extended_results = []
for perm in sampled_permutations:
    total_time = 0
    total_energy = 0
    for i, chiplet_name in enumerate(perm):
        weights, macs = mac_operations_extended[i]
        chiplet = chiplet_types[chiplet_name]
        chiplet_count = hetero_chiplet_configuration[chiplet_name]
        time, energy = computeTimeEnergy(weights, macs, chiplet, chiplet_count)
        total_time += time
        total_energy += energy
    extended_results.append((perm, total_time, total_energy))

# Plot results
perm_labels_ext = [" → ".join(p) for p, _, _ in extended_results]
cumulative_latencies_ext = [r[1] for r in extended_results]
cumulative_energies_ext = [r[2] for r in extended_results]

fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

axs[0].bar(range(len(perm_labels_ext)), cumulative_latencies_ext, color='deepskyblue')
axs[0].set_ylabel("Total Latency (s)")
axs[0].set_title("Cumulative Latency per Chiplet Ordering")
axs[0].grid(True, axis='y')

axs[1].bar(range(len(perm_labels_ext)), cumulative_energies_ext, color='mediumseagreen')
axs[1].set_ylabel("Total Energy (J)")
axs[1].set_title("Cumulative Energy per Chiplet Ordering")
axs[1].set_xticks(range(len(perm_labels_ext)))
axs[1].set_xticklabels(perm_labels_ext, rotation=60, ha='right')
axs[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig("fig.png")