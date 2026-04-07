import os
import time
import argparse

# Hide tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# Configuration
PROJECT_ROOT = r"D:\cfd-ml-project"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "demo_outputs")
CFD_TIME_FILE = os.path.join(PROJECT_ROOT, "airfoil_cfd_run_time.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_avg_cfd_time(filepath):
    """Dynamically parses the CFD batch execution report for the average time."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Average CFD Time/Case" in line:
                    # Extracts the number after the colon
                    val_str = line.split(":")[-1].strip()
                    return float(val_str)
    except Exception as e:
        print(f"Warning: Could not read {filepath} ({e}). Defaulting to 794.79")
        return 794.79
        
    print("Warning: 'Average CFD Time/Case' not found. Defaulting to 794.79")
    return 794.79


def main(aoa):
    print(f"Initiating Aerodynamic Inference for AoA = {aoa} degree")
    
    # Dynamically read the CFD benchmark time
    avg_cfd_time_sec = get_avg_cfd_time(CFD_TIME_FILE)
    
    # Load Models & Scalers
    print("Loading models and scalers")
    
    # Physics-Informed CNN (Cp Curve Prediction)
    pinn_cnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "pinn_cnn_model_structured.keras"))
    cnn_in_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_input_scaler_structured.pkl"))
    
    # Direct MLP (Used ONLY for Drag (Cd) prediction, as PINN handles Cl)
    cl_cd_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cl_cd_direct_model_structured.keras"))
    cl_in_scaler = joblib.load(os.path.join(MODELS_DIR, "cl_cd_input_scaler_structured.pkl"))
    cl_out_scaler = joblib.load(os.path.join(MODELS_DIR, "cl_cd_output_scaler_structured.pkl"))
    
    # Load exact CFD spatial grid for accurate plotting and integration
    data = np.load(DATA_PATH)
    x_grid = data["x_grid"]
    dx_array = x_grid[1:] - x_grid[:-1]
    
    # Preprocess Inputs
    aoa_array = np.array([[aoa]])
    aoa_cnn_scaled = cnn_in_scaler.transform(aoa_array)
    aoa_cl_scaled = cl_in_scaler.transform(aoa_array)
    
    print("Running ML Surrogate Inference")
    start_time = time.time()
    
    # 1. Physics-Informed Spatial Cp Prediction
    cp_pred = pinn_cnn_model.predict(aoa_cnn_scaled, verbose=0)
    cp_upper = cp_pred[0, :, 0]
    cp_lower = cp_pred[0, :, 1]
    
    # 2. MATHEMATICAL INTEGRATION: Calculate Lift (Cl) from PINN-CNN curves
    dcp = cp_lower - cp_upper
    mids = (dcp[:-1] + dcp[1:]) / 2.0
    cn_pred = np.sum(mids * dx_array)
    cl_pred = cn_pred * np.cos(np.radians(aoa))
    
    # 3. Direct MLP: Calculate Drag (Cd) ONLY
    forces_scaled = cl_cd_model.predict(aoa_cl_scaled, verbose=0)
    forces_pred = cl_out_scaler.inverse_transform(forces_scaled)
    cd_pred = forces_pred[0][1] # Index 1 is Cd
    
    end_time = time.time()
    
    # Calculate performance metrics
    inference_time = end_time - start_time
    inference_time = max(inference_time, 0.001) # Prevent division by zero if cached
    speedup = avg_cfd_time_sec / inference_time
    
    print(f"\nINFERENCE RESULTS")
    print(f"Predicted Lift Coefficient (Cl): {cl_pred:.4f} (Integrated from PINN-CNN)")
    print(f"Predicted Drag Coefficient (Cd): {cd_pred:.4f} (Via Direct MLP)")
    print(f"Inference Time: {inference_time:.4f} seconds ({speedup:,.0f}x faster than CFD)")
    
    # Export Artifacts
    # A) Export CSV
    df = pd.DataFrame({
        "x/c": x_grid,
        "Cp_Upper": cp_upper,
        "Cp_Lower": cp_lower
    })
    csv_path = os.path.join(OUTPUT_DIR, f"cp_prediction_aoa_{aoa}.csv")
    df.to_csv(csv_path, index=False)
    
    # B) Generate Engineering Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, cp_upper, color="blue", linewidth=2, label="PINN-CNN Upper Surface")
    plt.plot(x_grid, cp_lower, color="orange", linewidth=2, linestyle="--", label="PINN-CNN Lower Surface")
    plt.gca().invert_yaxis() # Standard aerodynamic convention
    plt.xlabel("x/c", fontsize=12)
    plt.ylabel("Pressure Coefficient ($C_p$)", fontsize=12)
    plt.title(f"Physics-Informed CNN Surrogate Prediction (AoA = {aoa}$^\circ$)\n$C_l$: {cl_pred:.4f} | $C_d$: {cd_pred:.4f}", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"cp_plot_aoa_{aoa}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # C) Generate Output Summary Report
    summary_text = f"""
           SURROGATE INFERENCE SUMMARY

Target Angle of Attack: {aoa} degrees

Calculated Global Forces:
- Lift Coefficient (Cl): {cl_pred:.4f} [Calculated via spatial integration of PINN-CNN]
- Drag Coefficient (Cd): {cd_pred:.4f} [Calculated via Direct MLP regression]

Performance Metrics:
- Average ANSYS Fluent RANS Time: {avg_cfd_time_sec:.2f} seconds
- ML Surrogate Inference Time   : {inference_time:.4f} seconds
- Computational Speedup         : ~{speedup:,.0f}x 

Expected Extrapolation Constraints (AoA > 12 deg):
- Lift (Cl) RMSE vs CFD: ~0.046 (via PINN-CNN integration)
- Surface Pressure (Cp) RMSE vs CFD: ~0.725
- Utilized Architectures: Physics-Informed 1D-CNN (for Cp, Cl), Direct Dense MLP (for Cd)

Artifacts successfully generated in {OUTPUT_DIR}:
- Engineering Plot: {os.path.basename(plot_path)}
- Spatial Data CSV: {os.path.basename(csv_path)}
"""
    
    summary_path = os.path.join(OUTPUT_DIR, f"summary_aoa_{aoa}.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\nInference complete. Artifacts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerodynamic ML Surrogate Inference Tool")
    parser.add_argument("--aoa", type=float, default=4.0, help="Target Angle of Attack in degrees (e.g., 16.5)")
    args = parser.parse_args()
    
    main(args.aoa)