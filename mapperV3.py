import pandas as pd
import math
import numpy as np

# -----------------------------------------------------------------------------
# Chiplet specs with TOPS (in Tera‑ops/s) and energy_per_mac (in J)
# -----------------------------------------------------------------------------
chiplet_specs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 94.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}

# -----------------------------------------------------------------------------
# Chip specs & capacity helper 
# -----------------------------------------------------------------------------
chipletTypesDict = {
    "Standard":    {"Size": 16384,  "Bits/cell": 2, "TOPS": 30e12,  "Energy/MAC": 0.87e-12},
    "Shared":      {"Size": 583696, "Bits/cell": 1, "TOPS": 27e12,  "Energy/MAC": 0.30e-12},
    "Adder":       {"Size": 4096,   "Bits/cell": 1, "TOPS": 11e12,  "Energy/MAC": 0.18e-12},
    "Accumulator": {"Size": 65536,  "Bits/cell": 2, "TOPS": 35e12,  "Energy/MAC": 0.22e-12},
    "ADC_Less":    {"Size": 16384,  "Bits/cell": 1, "TOPS": 3.8e12, "Energy/MAC": 0.27e-12},
}

XBARS_PER_TILE = 96
TILES_PER_CHIPLET = 16
MAX_CHIPLET_POWER = 8.0  # Watts

def get_chip_capacity_bits(chip_type, tiles=TILES_PER_CHIPLET, xbars=XBARS_PER_TILE):
    info = chipletTypesDict[chip_type]
    return info["Size"] * info["Bits/cell"] * xbars * tiles

def calculate_chiplets_needed(weight_bits, chiplet_type):
    info = chipletTypesDict[chiplet_type]
    xbar_capacity = info["Size"] * info["Bits/cell"]
    xbars_needed = math.ceil(weight_bits / xbar_capacity)
    return max(1, math.ceil(xbars_needed / (TILES_PER_CHIPLET * XBARS_PER_TILE)))

# -----------------------------------------------------------------------------
# getOUSize: finds best crossbar dims under 8 W instantaneous peak power
# -----------------------------------------------------------------------------
def getOUSize(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, _=None):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError:
        return _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity)

    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy = spec["energy_per_mac"]
    min_rows = int(base_r * activation_sparsity)
    scales = np.array([0.0675, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0])

    class ProblemDef(Problem):
        def __init__(self):
            super().__init__(
                n_var=2, n_obj=2, n_constr=1,
                xl=np.array([0, 0]), xu=np.array([len(scales)-1, len(scales)-1])
            )
        def _evaluate(self, X, out, *args, **kwargs):
            idx = np.round(X).astype(int)
            idx = np.clip(idx, 0, len(scales)-1)
            row_scales = scales[idx[:,0]]
            col_scales = scales[idx[:,1]]

            F1 = np.zeros(len(X))
            F2 = np.zeros(len(X))
            G1 = np.zeros(len(X))

            for i in range(len(X)):
                r = int(base_r * row_scales[i])
                c = int(base_c * col_scales[i])
                if r < min_rows:
                    # invalid: too few rows
                    F1[i] = F2[i] = 1e6
                    G1[i] = 1e6
                    continue

                # scaled energy per MAC
                rowE = spec["rowKnob"] * row_scales[i]
                colE = spec["colKnob"] * col_scales[i]
                other = 100 - spec["rowKnob"] - spec["colKnob"]
                e_per_mac = base_energy * ((rowE + colE + other) / 100)

                # instantaneous TOPS → ops/s
                tops_ops = base_tops * row_scales[i] * col_scales[i] * 1e12

                # instantaneous peak power
                peak_power = e_per_mac * tops_ops

                # normalized EDP and power objectives
                edp = e_per_mac / tops_ops
                F1[i] = edp / (base_energy / (base_tops * 1e12))
                F2[i] = peak_power / (base_energy * base_tops * 1e12)

                # constraint: ≤ 8 W
                G1[i] = peak_power - MAX_CHIPLET_POWER

            out["F"] = np.column_stack([F1, F2])
            out["G"] = G1.reshape(-1,1)

    algorithm = NSGA2(
        pop_size=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 150)
    res = minimize(ProblemDef(), algorithm, termination, seed=1, verbose=False)

    # choose best by 0.3*EDP + 0.7*Power
    weights = np.array([0.3, 0.7])
    best_idx = np.argmin((res.F * weights).sum(axis=1))
    i0, i1 = np.round(res.X[best_idx]).astype(int)
    i0, i1 = np.clip([i0, i1], 0, len(scales)-1)

    r = int(base_r * scales[i0])
    c = int(base_c * scales[i1])
    rowE = spec["rowKnob"] * scales[i0]
    colE = spec["colKnob"] * scales[i1]
    e_per_mac = base_energy * ((rowE + colE + (100-spec["rowKnob"]-spec["colKnob"])) / 100)
    scaled_tops = base_tops * scales[i0] * scales[i1]

    return r, c, scaled_tops, e_per_mac

def _getOUSize_manual(xbar_sparsity, num_xbars, chiplet_type, weight_bits, activation_sparsity, _=None):
    spec = chiplet_specs[chiplet_type]
    base_r, base_c = spec["base"]
    base_tops = spec["tops"]
    base_energy = spec["energy_per_mac"]
    min_rows = int(base_r * activation_sparsity)
    scales = [1.0, 0.9, 0.75, 0.5, 0.25, 0.125, 0.0675]
    best = None
    best_metric = float('inf')

    for rs in scales:
        for cs in scales:
            r, c = int(base_r*rs), int(base_c*cs)
            if r < min_rows:
                continue

            rowE = spec["rowKnob"]*rs + spec["colKnob"]*cs + (100-spec["rowKnob"]-spec["colKnob"])
            e_per_mac = base_energy * (rowE/100)
            tops_ops = base_tops * rs * cs * 1e12
            peak_power = e_per_mac * tops_ops
            if peak_power > MAX_CHIPLET_POWER:
                continue

            edp = e_per_mac / tops_ops
            norm_edp   = edp   / (base_energy/(base_tops*1e12))
            norm_pow   = peak_power/(base_energy*base_tops*1e12)
            metric = 0.3*norm_edp + 0.7*norm_pow

            if metric < best_metric:
                best_metric = metric
                best = (r, c, base_tops*rs*cs, e_per_mac)

    return best or (base_r, base_c, spec["tops"], spec["energy_per_mac"])

# -----------------------------------------------------------------------------
# Layer compute helper
# -----------------------------------------------------------------------------
def compute_layer_time_energy(allocation_list, total_macs):
    total_bits = sum(a["allocated_bits"] for a in allocation_list)
    times, energies = [], []
    for a in allocation_list:
        frac = a["allocated_bits"] / total_bits
        macs_i = total_macs * frac
        tops_i = a["optimized_tops"] * 1e12
        epm   = a["optimized_energy_per_mac"]
        t_i = macs_i / tops_i
        e_i = macs_i * epm
        times.append(t_i)
        energies.append(e_i)
    layer_time   = max(times)
    layer_energy = sum(energies)
    layer_power  = layer_energy / layer_time if layer_time>0 else 0
    layer_edp    = layer_energy * layer_time
    max_cpwr     = max((e/t if t>0 else 0) for e,t in zip(energies, times))
    return layer_time, layer_energy, layer_power, layer_edp, max_cpwr

# -----------------------------------------------------------------------------
# Scheduler + detailed per-layer & global summary
# -----------------------------------------------------------------------------
def scheduler(csv_path, chip_distribution):
    df = pd.read_csv(csv_path)
    df["Adjusted_Weights_bits"] = df["Weights(KB)"]*(1-df["Weight_Sparsity(0-1)"])*1024*8

    # build chip inventory
    inv = []
    types = list(chipletTypesDict.keys())
    for ct, cnt in zip(types, chip_distribution):
        for i in range(cnt):
            inv.append({"id":f"{ct}_{i}", "type":ct,
                        "capacity_left":get_chip_capacity_bits(ct)})

    layers = []
    for _, row in df.iterrows():
        layer_id     = int(row["Layer #"])
        rem_bits     = row["Adjusted_Weights_bits"]
        total_macs   = row["MACs"]
        allocs       = []
        total_bits   = rem_bits
        for chip in inv:
            if rem_bits <= 0: break
            if chip["capacity_left"] <= 0: continue


            alloc = min(rem_bits, chip["capacity_left"])
            # weightReq    = row["Weights(KB)"]*1024*8
            # cap          = chipletTypesDict[chip["type"]]["Size"]*chipletTypesDict[chip["type"]]["Bits/cell"]
            # XbarReqCeil  = math.ceil(weightReq / cap)
            # inherentS    = (XbarReqCeil - weightReq/cap)/XbarReqCeil
            # WS           = row["Weight_Sparsity(0-1)"]/XbarReqCeil
            # XbarSparsity = inherentS + WS
            AS           = row["Activation_Sparsity(0-1)"]
            weight_nonzero_bits = alloc

            # how many bits per crossbar
            cap = chipletTypesDict[chip["type"]]["Size"] * chipletTypesDict[chip["type"]]["Bits/cell"]

            # number of crossbars you really need to hold those non‑zero bits
            xbars_req = math.ceil(weight_nonzero_bits / cap)

            # if you spread the non‑zeros evenly across them:
            per_xbar_nonzeros = weight_nonzero_bits / xbars_req

            # fraction of each xbar that’s empty
            xbar_sparsity = (cap - per_xbar_nonzeros) / cap

            r, c, tops, epm = getOUSize(
                    xbar_sparsity,
                    xbars_req,
                    chip["type"],
                    weight_nonzero_bits,
                    row["Activation_Sparsity(0-1)"]
                )
            frac        = alloc / total_bits
            macs_assigned = total_macs * frac
            util        = alloc / chip["capacity_left"]

            chip["capacity_left"] -= alloc
            rem_bits    -= alloc

            allocs.append({
                "chip_id": chip["id"],
                "chip_type": chip["type"],
                "allocated_bits": int(alloc),
                "MACs_assigned": int(macs_assigned),
                "Chiplets_reqd": math.ceil(xbars_req/(TILES_PER_CHIPLET*XBARS_PER_TILE)),
                "Crossbars_used": xbars_req,
                "Crossbar_sparsity": xbar_sparsity,
                "weight sparsity":row["Weight_Sparsity(0-1)"],
                "Activation Sparsity": AS,
                "optimal_ou_row": r,
                "optimal_ou_col": c,
                "optimized_tops": tops,
                "optimized_energy_per_mac": epm,
                "chiplet_resource_taken": util*100
            })

        if rem_bits > 0:
            raise RuntimeError(f"Layer {layer_id} not fully allocated: {rem_bits:.0f} bits remain")

        t, e, p, edp, maxp = compute_layer_time_energy(allocs, total_macs)
        layers.append({
            "layer": layer_id,
            "allocations": allocs,
            "time_s": t,
            "energy_J": e,
            "avg_power_W": p,
            "edp": edp,
            "max_chiplet_power_W": maxp
        })

    return layers

if __name__ == "__main__":
    workload_csv = "workloads/vgg16_stats.csv"
    chip_dist    = [0, 0, 0, 38, 12]
    results      = scheduler(workload_csv, chip_dist)

    # per-layer details
    for lr in results:
        print(f"\nLayer {lr['layer']}:")
        for a in lr["allocations"]:
            print(" ", a)
        print(f"  → Time: {lr['time_s']:.3e}s, Energy: {lr['energy_J']:.3e}J, "
              f"Power: {lr['avg_power_W']:.3e}W, MaxP: {lr['max_chiplet_power_W']:.3e}W, EDP: {lr['edp']:.3e}")

    # global summary
    total_energy = sum(l["energy_J"] for l in results)
    max_latency  = max(l["time_s"]   for l in results)
    workload_edp = total_energy * max_latency
    print("\nWorkload summary:")
    print(f"  Final energy : {total_energy:.3e} J")
    print(f"  Final latency: {max_latency:.3e} s")
    print(f"  Final EDP    : {workload_edp:.3e} J·s")

    # check violations
    bad = [l["layer"] for l in results if l["max_chiplet_power_W"] > MAX_CHIPLET_POWER]
    if bad:
        print(f"\nWarning: layers {bad} exceed the {MAX_CHIPLET_POWER}W peak‑power cap.")
