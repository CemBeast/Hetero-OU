import math
import matplotlib.pyplot as plt

# 1) Chiplet specs (with TOPS and energy_per_mac)
chiplet_specs = {
    "Standard":    {"base": (128, 128), "rowKnob": 87.5, "colKnob": 12.5, "tops": 30.0, "energy_per_mac": 0.87e-12},
    "Shared":      {"base": (764, 764), "rowKnob": 68.0, "colKnob": 1.3,  "tops": 27.0, "energy_per_mac": 0.30e-12},
    "Adder":       {"base": (64,  64),  "rowKnob": 81.0, "colKnob": 0.4,  "tops": 11.0, "energy_per_mac": 0.18e-12},
    "Accumulator": {"base": (256, 256),"rowKnob": 55.0, "colKnob": 51.0, "tops": 35.0, "energy_per_mac": 0.22e-12},
    "ADC_Less":    {"base": (128, 128),"rowKnob": 94.4, "colKnob": 20.0, "tops": 3.8,  "energy_per_mac": 0.27e-12},
}

# 2) Scale factors (powers of two)
scales = [1.0, 0.5, 0.25, 0.125]
def plot_scaling_trends(chiplet_specs, scales):
    # Loop over each chiplet
    for name, spec in chiplet_specs.items():
        base_r, base_c = spec["base"]
        labels, energies, tops_scaled, powers, edp_mac2, weighted_edp = [], [], [], [], [], []

        # Generate metrics for each (row_scale, col_scale)
        for r_scale in scales:
            for c_scale in scales:
                r = int(base_r * r_scale)
                c = int(base_c * c_scale)
                # Energy per MAC scaling
                rowE = spec["rowKnob"] * r_scale
                colE = spec["colKnob"] * c_scale
                otherE = 100 - spec["rowKnob"] - spec["colKnob"]
                energy_ratio = (rowE + colE + otherE) / 100
                e_scaled = spec["energy_per_mac"] * energy_ratio

                # TOPS scaling and convert to ops/s
                tops_tops = spec["tops"] * (r_scale * c_scale)
                tops_ops = tops_tops * 1e12  # tera → ops/s

                # Peak power and EDP/MAC^2
                p_scaled = e_scaled * tops_ops            # Watts
                edp_scaled = e_scaled / tops_ops if tops_ops > 0 else 0  # J*s per MAC^2

                # Weighted by total cells (surrogate for storage)
                cells = r * c
                w_edp = edp_scaled * cells

                labels.append(f"{r}×{c}")
                energies.append(e_scaled)
                tops_scaled.append(tops_tops)
                powers.append(p_scaled)
                edp_mac2.append(edp_scaled)
                weighted_edp.append(w_edp)

        # Plot all 5 metrics as subplots
        fig, axes = plt.subplots(5, 1, figsize=(8, 16), sharex=True)
        fig.suptitle(f"{name}: Scaling Trends (with Weighted EDP)", y=0.95)

        axes[0].plot(labels, energies, marker='o')
        axes[0].set_ylabel('Energy per MAC (J)')

        axes[1].plot(labels, tops_scaled, marker='o')
        axes[1].set_ylabel('Scaled TOPS')

        axes[2].plot(labels, powers, marker='o')
        axes[2].set_ylabel('Peak Power (W)')

        axes[3].plot(labels, edp_mac2, marker='o')
        axes[3].set_ylabel('EDP/MAC² (J·s/MAC²)')

        axes[4].plot(labels, weighted_edp, marker='o')
        axes[4].set_ylabel('EDP/MAC² × Cells (J·s per MAC²·cell)')
        axes[4].set_xlabel('Crossbar Size (Rows×Columns)')

        for ax in axes:
            ax.grid(True, linestyle='--', linewidth=0.5)
        plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

plot_scaling_trends(chiplet_specs, scales)