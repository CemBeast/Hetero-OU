import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# point this at your workloads folder
DATA_DIR = "./workloads"

def process_csv(file_path):
    # --- load data ---
    df = pd.read_csv(file_path)
    
    # --- parse workload name ---
    base = os.path.basename(file_path)                
    name = os.path.splitext(base)[0]                  
    workload = name.replace("_stats", "").title()     
    
    # --- compute stats ---
    act = df["Activations(KB)"]
    sp  = df["Activation_Sparsity(0-1)"]
    act_mean, act_std = act.mean(), act.std()
    sp_mean,  sp_std  = sp.mean(),  sp.std()
    
    # --- print stats ---
    print(f"\n=== {workload} ===")
    print(f"Activations (KB):      mean = {act_mean:.2f}, std = {act_std:.2f}")
    print(f"Activation Sparsity:   mean = {sp_mean:.4f}, std = {sp_std:.4f}")
    
    # --- plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = df["Layer #"]
    
    # Activations plot
    ax1.plot(x, act)
    ax1.set_xlabel("Layer #")
    ax1.set_ylabel("Activations (KB)")
    ax1.set_title(f"{workload}: Activations per Layer")
    ax1.text(
        0.5, -0.2,
        f"mean = {act_mean:.2f}\nstd  = {act_std:.2f}",
        transform=ax1.transAxes,
        ha='center', va='top'
    )
    
    # Sparsity plot as percent
    ax2.plot(x, sp * 100)
    ax2.set_xlabel("Layer #")
    ax2.set_ylabel("Activation Sparsity (%)")
    ax2.set_title(f"{workload}: Activation Sparsity per Layer")
    ax2.text(
        0.5, -0.2,
        f"mean = {sp_mean * 100:.2f}%\nstd  = {sp_std * 100:.2f}%",
        transform=ax2.transAxes,
        ha='center', va='top'
    )
    
    plt.tight_layout()
    
    # --- save to folder ---
    output_dir = "Workload_Activation_Stats"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{workload}.png"))
    
    plt.show()

def main():
    pattern = os.path.join(DATA_DIR, "*.csv")
    for csv_file in sorted(glob.glob(pattern)):
        process_csv(csv_file)

if __name__ == "__main__":
    main()