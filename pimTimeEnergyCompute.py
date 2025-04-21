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
    },
    "Shared": {
        "Size": 583696,
        "Bits/cell": 1,
        "TOPS": 27e12,
        "Energy/MAC": 0.30e-12
    },
    "Adder": {
        "Size": 4096,
        "Bits/cell": 1,
        "TOPS": 11e12,
        "Energy/MAC": 0.18e-12
    },
    "Accumulator": {
        "Size": 65536,
        "Bits/cell": 2,
        "TOPS": 35e12,
        "Energy/MAC": 0.22e-12
    },
    "ADC_Less": {
        "Size": 16384,
        "Bits/cell": 1,
        "TOPS": 3.8e12,
        "Energy/MAC": 0.27e-12,
        "non_mac": 6e5,
        "non_mac_energy": 0.6e-11
    }
}

def computeTimeEnergy(weights, numMACs, chipletType, chipletCount):
    numTiles = 40 # Fixed for now
    numChiplets = 16 # Fixed for now

    size = chipletType["Size"] # Crossbar capacity
    #print(f"Chip size: {size}")
    bitsPerCell = chipletType["Bits/cell"]
    #print(f"Bits per cell: {bitsPerCell}")
    tops = chipletType["TOPS"]
    #print(f"Chip TOPS: {tops}") # consider heterogenous OU tops 
    energyPerMAC = chipletType["Energy/MAC"]
    #print(f"Energy/MAC: {energyPerMAC}") # consider heterogenous OU energy 
    

    storage = math.ceil((weights * (8 /bitsPerCell)) / (size * numTiles * numChiplets))
    #print(f"Storage: {storage}")
    # change to consider heterogenous OUs
    storageScale = storage / chipletCount
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



csvPath = "workload.csv"
runWorkloadFromCSV(csvPath, "Accumulator", 7, chipletTypesDict)
# runWorkloadFromCSV(csvPath, "Adder", 10)
# runWorkloadFromCSV(csvPath, "ADC_Less", 10)
# runWorkloadFromCSV(csvPath, "Shared", 10)



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
print(customChipletDict)

runWorkloadFromCSV(csvPath, "Accumulator", 7, customChipletDict)