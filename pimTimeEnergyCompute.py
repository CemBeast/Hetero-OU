import csv
import math
import copy
import pandas as pd
import pyperclip  # pip install pyperclip



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

def computeTimeEnergy(weights, numMACs, chipletType, chipletCount):
    numArrays = 40 # Fixed for now 40 or 96
    numTiles = 16 # Fixed for now

    size = chipletType["Size"] # Crossbar capacity
    #print(f"Chip size: {size}")
    bitsPerCell = chipletType["Bits/cell"]
    #print(f"Bits per cell: {bitsPerCell}")
    tops = chipletType["TOPS"]
    #print(f"Chip TOPS: {tops}") # consider heterogenous OU tops 
    energyPerMAC = chipletType["Energy/MAC"]
    #print(f"Energy/MAC: {energyPerMAC}") # consider heterogenous OU energy 
    

    storage = math.ceil((weights * (8 /bitsPerCell)) / (size * numArrays * numTiles))
    #print(f"Storage: {storage}")
    # change to consider heterogenous OUs
    storageScale = storage / chipletCount # if StorageScale is above 1 then the weights are too much for the number of chips
    #print(f"Storage Scale: {storageScale}")
    timeTaken = numMACs / tops
    #print(f"Time Taken before scale: {timeTaken}")
    energy = numMACs * energyPerMAC
    #print(f"Energy before scale: {energy}")

    realTime = storageScale * timeTaken
    realEnergy = storageScale * energy
    avgPower = realEnergy/realTime
    #print(f"Real Time: {realTime}  Real Energy:{realEnergy}") # Add Avg Power
    return realTime, realEnergy, avgPower

def runWorkloadFromCSV(csvPath, chipletName, chipletCount, chipletDict):
    chipletType = chipletDict[chipletName]
    print(f"Running Workload on {chipletName} with a count of {chipletCount}")

    table_data = []
    totalTime = 0
    totalEnergy = 0
    totalPower = 0

    with open(csvPath, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            layer = row["Layer"]
            weights = int(row["Weights"])
            MACS = int(row["MACs"])

            t, e, p = computeTimeEnergy(weights, MACS, chipletType, chipletCount)
            powerConsumption = p * chipletCount # Take avg power and multiply it by chip count for power
            
            totalTime += t
            totalEnergy += e
            totalPower += powerConsumption

            table_data.append({
                "Layer": layer,
                "Time (s)": t,
                "Energy (J)": e,
                "Power Avg (W)": p,
                "Power Consumption (W)": powerConsumption
            })
    table_data.append({
        "Layer": "TOTAL",
        "Time (s)": totalTime,
        "Energy (J)": totalEnergy,
        "Power Avg (W)": totalEnergy / totalTime,
        "Power Consumption (W)": totalPower
    })
    # Create and format DataFrame
    df = pd.DataFrame(table_data)

    # Format 'Power Avg (W)' and 'Power Consumption (W)' to plain float strings with 2 decimals
    df["Power Avg (W)"] = df["Power Avg (W)"].apply(lambda x: f"{x:.3f}")
    df["Power Consumption (W)"] = df["Power Consumption (W)"].apply(lambda x: f"{x:.3f}")

    # Print the table, keeping the rest in scientific notation
    print(df.to_string(index=False, float_format="%.4e"))

    # Copy to clipboard for Excel
    text = df.to_csv(sep='\t', index=False, float_format="%.4e")
    pyperclip.copy(text)
    print("\n✅ Table copied to clipboard (tab-separated, ready for Excel)")

# The important thing is that we want to see the interactions of different OU sizes with 
# Different chips and see what could be the best to work with

# Avoid the ordering/ permutations as there can be infinite so it is challenging (for now)
# f2 dummy func input is Chip type, col x rows, output is TOPS and Energy/MAC


# Harsh will create a function that will calculate TOPS and Energy/MAC dependent 
# on the OU size. With this information, we will run comparisons of different OU sizes on
# each of the chiplet types and then compare from there
# For now we use this dummy function
# (placeholder for real logic)
def dummy(chipletType, rows, cols):
    return 0.1, 1

# Make a deep copy of the original chipletTypesDict to not modify original numbers
customChipletDict = copy.deepcopy(chipletTypesDict)

# Example row/col sizes for each chiplet type (you can adjust this) to change TOPS and Energy/MAC
crossbarDims = {
    "Standard": (64, 64),
    "Shared": (96, 96),
    "Adder": (32, 32),
    "Accumulator": (128, 128),
    "ADC_Less": (64, 64)
}

# Applys dummy() to update each entry in the copied dict
for chipName, chipType in customChipletDict.items():
    rows, cols = crossbarDims[chipName] # get dimensions from the example dictionary above
    area = rows * cols # computes size
    new_tops, new_energy = dummy(chipType, rows, cols) # Get new TOPS and Energy/Mac
    chipType["Size"] = area # REassign Values in the chip dictionary
    chipType["TOPS"] = new_tops
    chipType["Energy/MAC"] = new_energy

# You can now use customChipletDict like chipletTypesDict without affecting the original print is to see if it works
#print(customChipletDict)

#runWorkloadFromCSV(csvPath, "Accumulator", 7, customChipletDict)

# Now we wawnt to run the workload with a given set of chips 
#  [24, 28, 0, 18, 12] where ech element value corresponds to the toatal 
# number of chiplets from 𝑆𝑡𝑎𝑛𝑑𝑎𝑟𝑑 , 𝑆ℎ𝑎𝑟𝑒𝑑 , 𝐴𝑑𝑑𝑒𝑟 , and 𝐴𝑐𝑐𝑢𝑚𝑢𝑙𝑎𝑡𝑜𝑟, and 𝐴𝐷𝐶1Less types

# This assumes that we can only run one layer at a time, so if the storage is 0.4 it will
# run on just one chip and not try to see if it can fit more from the next layer. 
# Runs layer by layer for workload rather than all possible weights that fit in 1 chip
def runHeterogeneousWorkload(csvPath, chipletCounts, chipletDict):
    chipletNames = ["Standard", "Shared", "Adder", "Accumulator", "ADC_Less"]
    numTiles = 96
    numChiplets = 16  # Fixed architecture value

    # Track remaining chiplets for each type - Dictionary type for chip name and count
    chipletPools = {name: chipletCounts[i] for i, name in enumerate(chipletNames)}

    table_data = []
    totalTime = totalEnergy = totalPower = 0

    with open(csvPath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            layer = row["Layer"]
            weights = int(row["Weights"])
            macs = int(row["MACs"])
            assigned = False

            for chipName in chipletNames:
                chip = chipletDict[chipName]
                bitsPerCell = chip["Bits/cell"]
                size = chip["Size"]

                # Compute how many chiplets this layer needs
                chipsNeeded = math.ceil((weights * (8 / bitsPerCell)) / (size * numTiles * numChiplets))

                if chipletPools[chipName] >= chipsNeeded:
                    chipletPools[chipName] -= chipsNeeded  # Deduct used chips
                    t, e, p = computeTimeEnergy(weights, macs, chip, chipsNeeded)
                    power = p * chipsNeeded

                    totalTime += t
                    totalEnergy += e
                    totalPower += power

                    table_data.append({
                        "Layer": layer,
                        "Chiplet": chipName,
                        "Chiplets Used": chipsNeeded,
                        "Time (s)": t,
                        "Energy (J)": e,
                        "Power Avg (W)": p,
                        "Power Consumption (W)": power
                    })

                    assigned = True
                    break  # Stop after assigning to one chip group

            if not assigned:
                print(f"❌ Not enough chiplets to assign layer {layer}")
                table_data.append({
                    "Layer": layer,
                    "Chiplet": "UNASSIGNED",
                    "Chiplets Used": "N/A",
                    "Time (s)": 0,
                    "Energy (J)": 0,
                    "Power Avg (W)": 0,
                    "Power Consumption (W)": 0
                })

    # Append totals
    table_data.append({
        "Layer": "TOTAL",
        "Chiplet": "-",
        "Chiplets Used": "-",
        "Time (s)": totalTime,
        "Energy (J)": totalEnergy,
        "Power Avg (W)": totalEnergy / totalTime if totalTime > 0 else 0,
        "Power Consumption (W)": totalPower
    })

    # Show how many chiplets were used and how many are left
    print("\n🔧 Chiplet Usage Summary:")
    print("Chiplet Type | Initial | Used | Remaining")
    for name in chipletNames:
        initial = chipletCounts[chipletNames.index(name)]
        remaining = chipletPools[name]
        used = initial - remaining
        print(f"{name:<13} | {initial:^7} | {used:^5} | {remaining:^9}")

    df = pd.DataFrame(table_data)
    df["Power Avg (W)"] = df["Power Avg (W)"].apply(lambda x: f"{float(x):.3f}" if x != "N/A" else x)
    df["Power Consumption (W)"] = df["Power Consumption (W)"].apply(lambda x: f"{float(x):.3f}" if x != "N/A" else x)

    print(df.to_string(index=False, float_format="%.4e"))

    pyperclip.copy(df.to_csv(sep='\t', index=False, float_format="%.4e"))
    print("\n✅ Heterogeneous table copied to clipboard (tab-separated for Excel)")



chipCounts = [24, 28, 0, 18, 12]  # Standard, Shared, Adder, Accumulator, ADC_Less
runHeterogeneousWorkload("workload.csv", chipCounts, chipletTypesDict)

def runHeterogeneousWorkload2(csvPath, chipletCounts, chipletDict):
    chipletNames = ["Standard", "Shared", "Adder", "Accumulator", "ADC_Less"]
    numTiles = 96
    numChiplets = 16  # Fixed architecture value

    chipletPools = {name: chipletCounts[i] for i, name in enumerate(chipletNames)}
    table_data = []
    totalTime = totalEnergy = totalPower = 0

    with open(csvPath, 'r') as file:
        reader = list(csv.DictReader(file))

    i = 0
    while i < len(reader):
        layer_group = []     # layers to assign together
        total_storage_bits = 0
        total_weights = 0
        total_macs = 0

        # Try to group layers until one doesn't fit
        for j in range(i, len(reader)):
            layer = reader[j]["Layer"]
            weights = int(reader[j]["Weights"])
            macs = int(reader[j]["MACs"])
            storage_bits = weights * 8

            total_storage_bits += storage_bits
            total_weights += weights
            total_macs += macs
            layer_group.append((j, layer, weights, macs))

            # Check if this group can fit in any chip type
            fits_any = False
            for chipName in chipletNames:
                chip = chipletDict[chipName]
                bitsPerCell = chip["Bits/cell"]
                chipStorage = chip["Size"] * bitsPerCell * numTiles * numChiplets
                chipsNeeded = math.ceil(total_storage_bits / chipStorage)
                if chipletPools[chipName] >= chipsNeeded:
                    fits_any = True
                    break

            if not fits_any:
                if len(layer_group) == 1:
                # Let large single layers go through, even if they exceed one chiplet
                    break
                else:
                    layer_group.pop()
                    total_storage_bits -= storage_bits
                    total_weights -= weights
                    total_macs -= macs
                    break

        # Now assign the group to the best chiplet type
        assigned = False
        for chipName in chipletNames:
            chip = chipletDict[chipName]
            bitsPerCell = chip["Bits/cell"]
            chipStorage = chip["Size"] * bitsPerCell * numTiles * numChiplets
            chipsNeeded = math.ceil(total_storage_bits / chipStorage)

            if chipletPools[chipName] >= chipsNeeded:
                chipletPools[chipName] -= chipsNeeded
                if chipsNeeded == 0:
                    print(f"⚠️ Skipping empty layer group starting at row {i}")
                    i += 1  # prevent infinite loop
                    break
                t, e, p = computeTimeEnergy(total_weights, total_macs, chip, chipsNeeded)
                power = p * chipsNeeded

                totalTime += t
                totalEnergy += e
                totalPower += power

                for (_, layer, _, _) in layer_group:
                    table_data.append({
                        "Layer": layer,
                        "Chiplet": chipName,
                        "Chiplets Used": chipsNeeded,
                        "Time (s)": t,
                        "Energy (J)": e,
                        "Power Avg (W)": p,
                        "Power Consumption (W)": power
                    })

                assigned = True
                break

        # If not assigned, mark all in group as unassigned
        if not assigned:
            for (_, layer, _, _) in layer_group:
                print(f"❌ Not enough chiplets to assign layer {layer}")
                table_data.append({
                    "Layer": layer,
                    "Chiplet": "UNASSIGNED",
                    "Chiplets Used": "N/A",
                    "Time (s)": 0,
                    "Energy (J)": 0,
                    "Power Avg (W)": 0,
                    "Power Consumption (W)": 0
                })

        i += len(layer_group)

    # Totals row
    table_data.append({
        "Layer": "TOTAL",
        "Chiplet": "-",
        "Chiplets Used": "-",
        "Time (s)": totalTime,
        "Energy (J)": totalEnergy,
        "Power Avg (W)": totalEnergy / totalTime if totalTime > 0 else 0,
        "Power Consumption (W)": totalPower
    })

    # Usage summary
    print("\n🔧 Chiplet Usage Summary:")
    print("Chiplet Type | Initial | Used | Remaining")
    for name in chipletNames:
        initial = chipletCounts[chipletNames.index(name)]
        remaining = chipletPools[name]
        used = initial - remaining
        print(f"{name:<13} | {initial:^7} | {used:^5} | {remaining:^9}")

    df = pd.DataFrame(table_data)
    df["Power Avg (W)"] = df["Power Avg (W)"].apply(lambda x: f"{float(x):.3f}" if x != "N/A" else x)
    df["Power Consumption (W)"] = df["Power Consumption (W)"].apply(lambda x: f"{float(x):.3f}" if x != "N/A" else x)

    print(df.to_string(index=False, float_format="%.4e"))
    pyperclip.copy(df.to_csv(sep='\t', index=False, float_format="%.4e"))
    print("\n✅ Heterogeneous table copied to clipboard (tab-separated for Excel)")

# chipCounts = [24, 28, 0, 18, 12]  # Standard, Shared, Adder, Accumulator, ADC_Less
# runHeterogeneousWorkload2("workload.csv", chipCounts, chipletTypesDict)


def runHeterogeneousWorkloadFromKB(csvPath, chipletCounts, chipletDict):
    chipletNames = ["Standard", "Shared", "Adder", "Accumulator", "ADC_Less"]
    numTiles = 96
    numChiplets = 16  # Fixed architecture value

    chipletPools = {name: chipletCounts[i] for i, name in enumerate(chipletNames)}
    table_data = []
    totalTime = totalEnergy = totalPower = 0

    with open(csvPath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            layer = row["Layer"]
            weights_kb = float(row["Weights(KB)"])
            macs = int(row["MACs"])

            # Convert KB (assuming float64 → 8 bytes) to bit count
            weights_bits = int(weights_kb * 1024)  # 1 KB = 1024 bits (since 8B already handled)

            assigned = False
            for chipName in chipletNames:
                chip = chipletDict[chipName]
                bitsPerCell = chip["Bits/cell"]
                size = chip["Size"]

                chipsNeeded = math.ceil((weights_bits * (1 / bitsPerCell)) / (size * numTiles * numChiplets))

                if chipletPools[chipName] >= chipsNeeded:
                    chipletPools[chipName] -= chipsNeeded
                    t, e, p = computeTimeEnergy(weights_bits, macs, chip, chipsNeeded)
                    power = p * chipsNeeded

                    totalTime += t
                    totalEnergy += e
                    totalPower += power

                    table_data.append({
                        "Layer": layer,
                        "Chiplet": chipName,
                        "Chiplets Used": chipsNeeded,
                        "Time (s)": t,
                        "Energy (J)": e,
                        "Power Avg (W)": p,
                        "Power Consumption (W)": power
                    })

                    assigned = True
                    break

            if not assigned:
                print(f"❌ Not enough chiplets to assign layer {layer}")
                table_data.append({
                    "Layer": layer,
                    "Chiplet": "UNASSIGNED",
                    "Chiplets Used": "N/A",
                    "Time (s)": 0,
                    "Energy (J)": 0,
                    "Power Avg (W)": 0,
                    "Power Consumption (W)": 0
                })

    table_data.append({
        "Layer": "TOTAL",
        "Chiplet": "-",
        "Chiplets Used": "-",
        "Time (s)": totalTime,
        "Energy (J)": totalEnergy,
        "Power Avg (W)": totalEnergy / totalTime if totalTime > 0 else 0,
        "Power Consumption (W)": totalPower
    })

    print("\n🔧 Chiplet Usage Summary:")
    print("Chiplet Type | Initial | Used | Remaining")
    for name in chipletNames:
        initial = chipletCounts[chipletNames.index(name)]
        remaining = chipletPools[name]
        used = initial - remaining
        print(f"{name:<13} | {initial:^7} | {used:^5} | {remaining:^9}")

    df = pd.DataFrame(table_data)
    df["Power Avg (W)"] = df["Power Avg (W)"].apply(lambda x: f"{float(x):.3f}" if x != "N/A" else x)
    df["Power Consumption (W)"] = df["Power Consumption (W)"].apply(lambda x: f"{float(x):.3f}" if x != "N/A" else x)

    print(df.to_string(index=False, float_format="%.4e"))

    pyperclip.copy(df.to_csv(sep='\t', index=False, float_format="%.4e"))
    print("\n✅ Heterogeneous table copied to clipboard (tab-separated for Excel)")



# Todo make it run with workloadds file format
csvPath = "workloads/vgg16_stats.csv"
chipCounts = [24, 28, 0, 18, 12]  # Standard, Shared, Adder, Accumulator, ADC_Less
runHeterogeneousWorkloadFromKB(csvPath, chipCounts, chipletTypesDict)