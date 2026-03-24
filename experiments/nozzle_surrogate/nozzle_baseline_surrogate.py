import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import time

# configuration
PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "nozzle_dataset.npz")

# Load & Prepare data
print("Loading Nozzle Dataset")
data = np.load(DATA_PATH)
X_npr = data['npr']
Y_thrust = data['thrust']
Y_mach = data['mach']
x_grid = data['x_grid']

# Split into Training (80%) and Testing (20%)
X_train_raw, X_test_raw, Y_t_train_raw, Y_t_test_raw, Y_m_train, Y_m_test = train_test_split(
    X_npr, Y_thrust, Y_mach, test_size=0.2, random_state=42
)

# Initialize Scalers
scaler_x = MinMaxScaler()
scaler_thrust = MinMaxScaler()

# Fit and transform ONLY on the training data to prevent data leakage
X_train = scaler_x.fit_transform(X_train_raw)
Y_t_train = scaler_thrust.fit_transform(Y_t_train_raw)

# Transform the test data
X_test = scaler_x.transform(X_test_raw)
Y_t_test = scaler_thrust.transform(Y_t_test_raw)

# Mach numbers are already well-bounded (0 to ~3.5), so leave them unscaled 
print(f"Dataset split complete. Training samples: {len(X_train)} | Testing samples: {len(X_test)}")


# Build & Train Model A: Thrust Predictor (MLP)
print("\nTraining Model A: Thrust Predictor (MLP)")
model_A = models.Sequential([
    layers.InputLayer(input_shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])

model_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')

# Train the Thrust model
history_A = model_A.fit(X_train, Y_t_train, validation_data=(X_test, Y_t_test), 
                        epochs=400, batch_size=8, verbose=0)
print("Model A Training Complete.")

# Build & Train Model B: Mach Curve Predictor (1D CNN)
print("\nTraining Model B: Mach Curve Predictor (1D CNN)")
model_B = models.Sequential([
    layers.InputLayer(input_shape=(1,)),
    
    # Expand the single NPR value into spatial features (400 points to match new grid)
    layers.Dense(400 * 16, activation='relu'), 
    layers.Reshape((400, 16)), # Reshape into a 1D sequence with 16 channels
    
    # Apply 1D Convolutions to map the physical shockwave shapes
    layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'),
    layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu'),
    layers.Conv1D(filters=1, kernel_size=5, padding='same', activation='linear'),
    
    layers.Flatten() # Flatten back to a 400-point 1D array
])

model_B.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')

# Train the Mach model
history_B = model_B.fit(X_train, Y_m_train, validation_data=(X_test, Y_m_test), 
                        epochs=600, batch_size=8, verbose=0)
print("Model B Training Complete.")

# Evaluation & metrics
print("SURROGATE PERFORMANCE")

# Evaluate Model A: Thrust
pred_thrust_scaled = model_A.predict(X_test, verbose=0)
pred_thrust = scaler_thrust.inverse_transform(pred_thrust_scaled)
actual_thrust = scaler_thrust.inverse_transform(Y_t_test)

thrust_rmse = np.sqrt(mean_squared_error(actual_thrust, pred_thrust))
thrust_mape = np.mean(np.abs((actual_thrust - pred_thrust) / actual_thrust)) * 100

print("Model A: Thrust predictor")
print(f"  Thrust RMSE:       {thrust_rmse:.2f} Newtons")
print(f"  Thrust Error (%):  {thrust_mape:.2f} %")

# Evaluate Model B: Mach Curve
predictions_mach = model_B.predict(X_test, verbose=0)
mach_rmse = np.sqrt(mean_squared_error(Y_m_test, predictions_mach))

print("\nModel B: Mach distribution (1D CNN)")
print(f"  Mach Curve RMSE:   {mach_rmse:.4f}")

# Timing and Speedup vs Fluent
Average_CFD_time_per_case = 566.95 # seconds 

start_time = time.time()
_ = model_B.predict(X_test, verbose=0)  
end_time = time.time()

time_per_sample = (end_time - start_time) / len(X_test)
speedup_factor = Average_CFD_time_per_case / time_per_sample

print("\nCOMPUTATIONAL SPEEDUP")
print(f"  Fluent Time:       {Average_CFD_time_per_case:.2f} seconds/case")
print(f"  ML Inference Time: {time_per_sample:.5f} seconds/case")
print(f"  Speedup Factor:    {speedup_factor:,.0f}x faster than CFD")

# Plot Results
plt.figure(figsize=(14, 5))

# Plot 1: Thrust Validation
plt.subplot(1, 2, 1)
test_npr_unscaled = scaler_x.inverse_transform(X_test)
plt.scatter(test_npr_unscaled, actual_thrust, color='blue', label='ANSYS CFD (Ground Truth)')
plt.scatter(test_npr_unscaled, pred_thrust, color='red', marker='x', label='ML Prediction')
plt.title("Thrust vs NPR Validation")
plt.xlabel("Nozzle Pressure Ratio (NPR)")
plt.ylabel("Thrust (N)")
plt.legend()
plt.grid(True, alpha=0.5)

# Plot 2: Mach Curve Validation
plt.subplot(1, 2, 2)
# Pick a random test case to plot (e.g., the first one in the randomized test set)
sample_idx = 0 
plt.plot(x_grid, Y_m_test[sample_idx], 'b-', linewidth=2, label='ANSYS CFD')
plt.plot(x_grid, predictions_mach[sample_idx], 'r--', linewidth=2, label='ML Prediction')
plt.title(f"Mach Distribution (Test Case NPR: {test_npr_unscaled[sample_idx][0]:.1f})")
plt.xlabel("Axial Distance (m)")
plt.ylabel("Mach Number")
plt.axvline(x=0.0, color='k', linestyle=':', label='Throat (x=0.0 m)')
plt.xlim([-0.5, 0.5])
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
plt.savefig(os.path.join(PROJECT_ROOT, "reports", "CD_nozzle_baseline_surrogate_results.png"), dpi=300, bbox_inches='tight')
plt.show()