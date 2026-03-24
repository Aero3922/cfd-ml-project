import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load & Prepare data
PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "nozzle_dataset.npz")

print("Loading Nozzle Dataset")
data = np.load(DATA_PATH)
X_npr = data['npr']
Y_thrust = data['thrust']
Y_mach = data['mach']
x_grid = data['x_grid']

# Split first, then scale to prevent data leakage
X_train_raw, X_test_raw, Y_t_train_raw, Y_t_test_raw, Y_m_train, Y_m_test = train_test_split(
    X_npr, Y_thrust, Y_mach, test_size=0.2, random_state=42
)

scaler_x = MinMaxScaler()
scaler_thrust = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train_raw)
Y_t_train = scaler_thrust.fit_transform(Y_t_train_raw)

X_test = scaler_x.transform(X_test_raw)
Y_t_test = scaler_thrust.transform(Y_t_test_raw)

print(f"Dataset Split - Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

#  Build Multi-Task Physcis-Aware Model (PINN)
print("\nBuilding Physics-Aware Multi-Output Model")
input_layer = layers.Input(shape=(1,), name='npr_input')

# Shared "Latent Physics" Core
shared = layers.Dense(128, activation='relu')(input_layer)
shared = layers.Dense(256, activation='relu')(shared)

# Branch 1: Macroscopic Physics (Thrust Constraint)
thrust_branch = layers.Dense(64, activation='relu')(shared)
thrust_output = layers.Dense(1, activation='linear', name='thrust_out')(thrust_branch)

# Branch 2: Microscopic Physics (Mach Curve)
mach_branch = layers.Dense(400 * 16, activation='relu')(shared)
mach_branch = layers.Reshape((400, 16))(mach_branch)
mach_branch = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(mach_branch)
mach_branch = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation='relu')(mach_branch)
mach_output = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation='linear')(mach_branch)
mach_output = layers.Flatten(name='mach_out')(mach_output)

model_pinn = models.Model(inputs=input_layer, outputs=[thrust_output, mach_output])

# Apply physics Loss weights (The Lambda Parameter)
# apply a lambda weight of 2.0 to the thrust branch to enforce the physical constraint
losses = {'thrust_out': 'mse', 'mach_out': 'mse'}
loss_weights = {'thrust_out': 2.0, 'mach_out': 1.0}

model_pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
                   loss=losses, loss_weights=loss_weights)

# Train the surrogate
print("Training Physics-Aware Model")
start_train = time.time()
history = model_pinn.fit(
    X_train, 
    {'thrust_out': Y_t_train, 'mach_out': Y_m_train},
    validation_data=(X_test, {'thrust_out': Y_t_test, 'mach_out': Y_m_test}),
    epochs=600, batch_size=8, verbose=0
)
train_time = time.time() - start_train
print(f"Training Complete in {train_time:.1f} seconds.")

# Save final model
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "nozzle_pinn_model.keras")

model_pinn.save(model_path)
print(f"Final Model saved to: {model_path}")

# Evaluation & metrics
print("PHYSICS-AWARE SURROGATE PERFORMANCE")

predictions = model_pinn.predict(X_test, verbose=0)
pred_thrust_scaled = predictions[0]
pred_mach = predictions[1]

# Unscale Thrust
pred_thrust = scaler_thrust.inverse_transform(pred_thrust_scaled)
actual_thrust = scaler_thrust.inverse_transform(Y_t_test)

thrust_rmse = np.sqrt(mean_squared_error(actual_thrust, pred_thrust))
thrust_mape = np.mean(np.abs((actual_thrust - pred_thrust) / actual_thrust)) * 100
mach_rmse = np.sqrt(mean_squared_error(Y_m_test, pred_mach))

print(f"  Thrust RMSE:       {thrust_rmse:.2f} Newtons")
print(f"  Thrust Error (%):  {thrust_mape:.3f} %")
print(f"  Mach Curve RMSE:   {mach_rmse:.4f}")

# Plot results
plt.figure(figsize=(14, 5))

# Plot 1: Loss Curves (Log Scale)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Total Loss', color='black', linewidth=2)
plt.plot(history.history['thrust_out_loss'], label='Thrust Loss (Physics Constraint)', color='red', linestyle='--')
plt.plot(history.history['mach_out_loss'], label='Mach Loss (Data Matching)', color='blue', linestyle='--')
plt.title("PINN Training Convergence")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.5)

# Plot 2: Mach Curve Comparison
plt.subplot(1, 2, 2)
sample_idx = 0  # Change this index to view different test set NPRs
test_npr_unscaled = scaler_x.inverse_transform(X_test)

plt.plot(x_grid, Y_m_test[sample_idx], 'k-', linewidth=3, label='ANSYS CFD (Ground Truth)')
plt.plot(x_grid, pred_mach[sample_idx], 'g--', linewidth=2, label='Physics-Aware PINN')

plt.title(f"Mach Distribution with Constraint (NPR: {test_npr_unscaled[sample_idx][0]:.1f})")
plt.xlabel("Axial Distance (m)")
plt.ylabel("Mach Number")
plt.axvline(x=0.0, color='gray', linestyle=':', label='Throat (x=0.0 m)')
plt.xlim([-0.5, 0.5])
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
plt.savefig(os.path.join(PROJECT_ROOT, "reports", "CD_nozzle_pinn_surrogate_results.png"), dpi=300, bbox_inches='tight')
plt.show()