import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
PROJECT_ROOT = r"D:\cfd-ml-project"
NASA_CP_FILE = os.path.join(PROJECT_ROOT,"data", "CP_Ladson.dat")
NASA_FORCE_FILE = os.path.join(PROJECT_ROOT, "data", "CLCD_Ladson_expdata.dat")
FLUENT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Airfoil_sweep")

# Parse NASA wind tunnel data
def parse_nasa_cp(filepath, target_zone):
    x, cp = [], []
    in_zone = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("zone"):
                in_zone = (target_zone.lower() in line.lower())
                continue
            if in_zone and line and not line.isalpha():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x.append(float(parts[0]))
                        cp.append(float(parts[1]))
                    except ValueError:
                        pass
    return np.array(x), np.array(cp)

def parse_nasa_forces(filepath, target_zone="80 grit"):
    alpha, cl, cd = [], [], []
    in_zone = False
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

# parse local fluent batch results
def extract_fluent_forces(data_dir):
    """Scrape final Cl/Cd values from the out files."""
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
            
            # scan upwards to grab the last converged iteration
            for line in reversed(lines):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        cl = float(parts[1])
                        cd = float(parts[2])
                        fluent_results.append((aoa, cl, cd))
                        break # found the final values, move to next file
                    except ValueError:
                        continue 
        except Exception as e:
            print(f"Failed to read {hfile}: {e}")
            
    fluent_results.sort(key=lambda x: x[0])
    return np.array([x[0] for x in fluent_results]), np.array([x[1] for x in fluent_results]), np.array([x[2] for x in fluent_results])

def load_fluent_cp(filepath):
    """Load space-delimited surface pressure exports."""
    try:
        # handle fluent's weird spacing
        df = pd.read_csv(filepath, sep=r'\s+', on_bad_lines='skip')
        df.columns = [str(c).strip() for c in df.columns]
        
        # fallback if headers are shifted
        if not any('x-coordinate' in c for c in df.columns):
            df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, on_bad_lines='skip')
            df.columns = [str(c).strip() for c in df.columns]
            
        x_col = [c for c in df.columns if 'x-coordinate' in c][0]
        cp_col = [c for c in df.columns if 'pressure-coefficient' in c][0]
        
        # clean up any text artifacts
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[cp_col] = pd.to_numeric(df[cp_col], errors='coerce')
        df = df.dropna(subset=[x_col, cp_col])
        
        # sort along the chord so plots don't zigzag
        df = df.sort_values(by=x_col)
        return df[x_col].values, df[cp_col].values
    except Exception as e:
        print(f"Parse error for {filepath}: {e}")
        return np.array([]), np.array([])

# main plotting execution
nasa_cp_x, nasa_cp_val = parse_nasa_cp(NASA_CP_FILE, "Re=6 million, alpha=10.0254")
nasa_alpha, nasa_cl, nasa_cd = parse_nasa_forces(NASA_FORCE_FILE, "80 grit")
fluent_alpha, fluent_cl, fluent_cd = extract_fluent_forces(FLUENT_DATA_DIR)

fluent_cp_up_file = os.path.join(FLUENT_DATA_DIR, "10deg", "cp_upper_10deg.csv")
fluent_cp_low_file = os.path.join(FLUENT_DATA_DIR, "10deg", "cp_lower_10deg.csv")

x_up, cp_up = load_fluent_cp(fluent_cp_up_file)
x_low, cp_low = load_fluent_cp(fluent_cp_low_file)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Cp curve (10 deg)
if len(x_up) > 0:
    axs[0].scatter(nasa_cp_x, nasa_cp_val, facecolors='none', edgecolors='k', label='NASA Exp (10.02°)', zorder=5)
    axs[0].plot(x_up, cp_up, 'b-', linewidth=2, label='Fluent Upper (10°)')
    axs[0].plot(x_low, cp_low, 'r-', linewidth=2, label='Fluent Lower (10°)')
    axs[0].set_xlabel('x/c')
    axs[0].set_ylabel('Pressure Coefficient ($C_p$)')
    axs[0].set_title('Cp Distribution at 10° AoA')
    axs[0].invert_yaxis()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

# Lift curve
axs[1].scatter(nasa_alpha, nasa_cl, facecolors='none', edgecolors='k', label='NASA Exp (Fixed Trans)')
axs[1].plot(fluent_alpha, fluent_cl, 'b-o', label='Fluent SST k-$\omega$')
axs[1].set_xlabel('Angle of Attack ($\degree$)')
axs[1].set_ylabel('Lift Coefficient ($C_l$)')
axs[1].set_title('Lift Curve')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend()
axs[1].set_xlim([-5, 20])

# Drag polar
axs[2].scatter(nasa_cd, nasa_cl, facecolors='none', edgecolors='k', label='NASA Exp (Fixed Trans)')
axs[2].plot(fluent_cd, fluent_cl, 'r-o', label='Fluent SST k-$\omega$')
axs[2].set_xlabel('Drag Coefficient ($C_d$)')
axs[2].set_ylabel('Lift Coefficient ($C_l$)')
axs[2].set_title('Drag Polar')
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend()
axs[2].set_xlim([0, 0.05])

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "reports", "fluent_vs_nasa_validation.png"), dpi=300)
print("Plot successfully saved to reports/fluent_vs_nasa_validation.png")