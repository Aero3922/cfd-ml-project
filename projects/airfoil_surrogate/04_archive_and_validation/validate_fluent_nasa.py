import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Configuration
PROJECT_ROOT = r"D:\cfd-ml-project"
NASA_FORCE_FILE = os.path.join(PROJECT_ROOT, "data", "CLCD_Ladson_expdata.dat")
CFL3D_FORCES_FILE = os.path.join(PROJECT_ROOT, "data", "n0012clcd_cfl3d_sst.dat")
FLUENT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Airfoil_sweep")

TARGET_AOAS = [0, 10, 15]

# Data Loaders & Filters
def parse_nasa_forces(filepath, target_zone="80 grit"):
    """Parse experimental Ladson lift and drag data."""
    alpha, cl, cd = [], [], []
    in_zone = False
    if not os.path.exists(filepath):
        print(f"Warning: Ladson force file not found at {filepath}")
        return np.array(alpha), np.array(cl), np.array(cd)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("zone"):
                in_zone = (target_zone.lower() in line.lower())
                continue
            if in_zone and line and not line.startswith("variables"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        alpha.append(float(parts[0]))
                        cl.append(float(parts[1]))
                        cd.append(float(parts[2]))
                    except ValueError:
                        pass
    return np.array(alpha), np.array(cl), np.array(cd)


def load_cfl3d_forces(filepath):
    """Load the NASA numerical benchmark anchor points"""
    alpha, cl, cd = [], [], []
    if not os.path.exists(filepath):
        print(f"Warning: CFL3D force file not found at {filepath}")
        return np.array(alpha), np.array(cl), np.array(cd)
        
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith("variables") and not line.startswith("#") and line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        alpha.append(float(parts[0]))
                        cl.append(float(parts[1]))
                        cd.append(float(parts[2]))
                    except ValueError:
                        pass
    return np.array(alpha), np.array(cl), np.array(cd)


def extract_fluent_forces(data_dir):
    """Scrape final converged Cl/Cd values from local Fluent out files"""
    fluent_results = []
    history_files = glob.glob(os.path.join(data_dir, "**", "history_*.out"), recursive=True)
    
    for hfile in history_files:
        case_id = os.path.basename(hfile).replace("history_", "").replace(".out", "")
        try:
            if "minus" in case_id:
                aoa = -float(case_id.replace("minus", "").replace("deg", ""))
            else:
                aoa = float(case_id.replace("deg", ""))
        except ValueError:
            continue
            
        try:
            with open(hfile, 'r') as f:
                lines = f.readlines()
            
            # Scan upwards to grab the last converged iteration
            for line in reversed(lines):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        cl = float(parts[1])
                        cd = float(parts[2])
                        fluent_results.append((aoa, cl, cd))
                        break
                    except ValueError:
                        continue 
        except Exception as e:
            print(f"Failed to read {hfile}: {e}")
            
    fluent_results.sort(key=lambda x: x[0])
    return np.array([x[0] for x in fluent_results]), np.array([x[1] for x in fluent_results]), np.array([x[2] for x in fluent_results])

def filter_by_target_aoa(alpha, cl, cd, targets):
    """Finds exactly the closest single data point for each target AoA"""
    filtered_alpha, filtered_cl, filtered_cd = [], [], []
    for t in targets:
        if len(alpha) == 0: 
            continue
        # Find the index of the single closest Angle of Attack
        idx = np.argmin(np.abs(alpha - t))
        filtered_alpha.append(alpha[idx])
        filtered_cl.append(cl[idx])
        filtered_cd.append(cd[idx])
    return np.array(filtered_alpha), np.array(filtered_cl), np.array(filtered_cd)

# Main Execution & Plotting
# 1. Load Data
raw_nasa_alpha, raw_nasa_cl, raw_nasa_cd = parse_nasa_forces(NASA_FORCE_FILE, "80 grit")
cfl3d_alpha, cfl3d_cl, cfl3d_cd = load_cfl3d_forces(CFL3D_FORCES_FILE)
raw_fluent_alpha, raw_fluent_cl, raw_fluent_cd = extract_fluent_forces(FLUENT_DATA_DIR)

# 2. Filter Data to only exactly matching 0, 10, 15 degrees
nasa_alpha, nasa_cl, nasa_cd = filter_by_target_aoa(raw_nasa_alpha, raw_nasa_cl, raw_nasa_cd, TARGET_AOAS)
fluent_alpha, fluent_cl, fluent_cd = filter_by_target_aoa(raw_fluent_alpha, raw_fluent_cl, raw_fluent_cd, TARGET_AOAS)

# 3. Setup Plot (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# AXIS 0: Lift Curve
axs[0].scatter(nasa_alpha, nasa_cl, facecolors='none', edgecolors='k', s=100, label='NASA Exp (Fixed Trans)')
if len(cfl3d_alpha) > 0:
    axs[0].scatter(cfl3d_alpha, cfl3d_cl, color='red', marker='*', s=200, zorder=6, label='NASA CFL3D (Benchmark)')
axs[0].scatter(fluent_alpha, fluent_cl, color='blue', marker='s', s=80, label='Fluent SST k-$\omega$ (Current)')

axs[0].set_xlabel('Angle of Attack ($\degree$)', fontsize=12)
axs[0].set_ylabel('Lift Coefficient ($C_l$)', fontsize=12)
axs[0].set_title('Lift Anchor Points (0°, 10°, 15°)', fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(fontsize=10)
axs[0].set_xlim([-2, 18])


# AXIS 1: Drag Polar (Cl vs Cd)
axs[1].scatter(nasa_cl, nasa_cd, facecolors='none', edgecolors='k', s=100, label='NASA Exp (Fixed Trans)')
if len(cfl3d_cd) > 0:
    axs[1].scatter(cfl3d_cl, cfl3d_cd, color='red', marker='*', s=200, zorder=6, label='NASA CFL3D (Benchmark)')
axs[1].scatter(fluent_cl, fluent_cd, color='blue', marker='s', s=80, label='Fluent SST k-$\omega$ (Current)')

axs[1].set_xlabel('Lift Coefficient ($C_l$)', fontsize=12)
axs[1].set_ylabel('Drag Coefficient ($C_d$)', fontsize=12)
axs[1].set_title('Drag Polar Anchor Points (0°, 10°, 15°)', fontsize=14)
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(fontsize=10)
# Adjust limits dynamically to fit the maximum Cd nicely
axs[1].set_ylim([0, max(max(cfl3d_cd)*1.2 if len(cfl3d_cd)>0 else 0, 0.03)])

# Final Polish and Save
plt.tight_layout()
output_path = os.path.join(PROJECT_ROOT, "reports", "macro_forces_validation_anchor_points.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)

print(f"Plot successfully saved to: {output_path}")