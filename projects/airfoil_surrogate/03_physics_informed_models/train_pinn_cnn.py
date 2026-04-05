import os
import random
import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Flagship Model: Physics-Informed 1D-CNN (PINN-CNN)
# Objective: Combine the spatial awareness of a 1D-CNN with a custom Navier-Stokes 
# integration constraint to achieve high-fidelity extrapolation for both local 
# pressure distributions and global aerodynamic forces.

# setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

print("Loading CFD baseline data for PINN-CNN training...")
data = np.load(DATA_PATH)
X_raw = data["X"]
y_cp = data["y_cp"]
y_cl = data["y_cl"].reshape(-1, 1)
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
NUM_POINTS = len(x_grid) 

# CRITICAL FIX: Airfoils use non-uniform meshes (cosine clustering). 
# We must calculate the exact varying dx for every single panel.
dx_array = x_grid[1:] - x_grid[:-1]
dx_tensor = tf.constant(dx_array, dtype=tf.float32)
dx_tensor = tf.reshape(dx_tensor, (1, -1)) # Shape (1, 149) for broadcasting

# Structured split (Extrapolation)
aoa_flat = aoa.flatten()
train_mask = aoa_flat <= 8
val_mask   = (aoa_flat > 8) & (aoa_flat <= 12)
test_mask  = aoa_flat > 12

aoa_train, aoa_val, aoa_test = aoa[train_mask], aoa[val_mask], aoa[test_mask]
cp_train, cp_val, cp_test = y_cp[train_mask], y_cp[val_mask], y_cp[test_mask]
cl_train, cl_val, cl_test = y_cl[train_mask], y_cl[val_mask], y_cl[test_mask]

# Scale inputs 
input_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_input_scaler_structured.pkl"))
aoa_train_scaled = input_scaler.transform(aoa_train)
aoa_val_scaled   = input_scaler.transform(aoa_val)
aoa_test_scaled  = input_scaler.transform(aoa_test)

AOA_MEAN = tf.constant(input_scaler.mean_[0], dtype=tf.float32)
AOA_STD = tf.constant(input_scaler.scale_[0], dtype=tf.float32)

# Data formatting for CNN
def reshape_to_channels(cp_flat):
    upper = cp_flat[:, :NUM_POINTS]
    lower = cp_flat[:, NUM_POINTS:]
    return np.stack((upper, lower), axis=-1)

cp_train_cnn = reshape_to_channels(cp_train)
cp_val_cnn   = reshape_to_channels(cp_val)
cp_test_cnn  = reshape_to_channels(cp_test)

BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices((
    aoa_train_scaled.astype(np.float32),
    cp_train_cnn.astype(np.float32),
    cl_train.astype(np.float32)
)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    aoa_val_scaled.astype(np.float32),
    cp_val_cnn.astype(np.float32),
    cl_val.astype(np.float32)
)).batch(BATCH_SIZE)

# PINN-CNN architecture
print("\nInitializing Physics-Informed 1D-CNN")
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(1,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(NUM_POINTS * 32, activation="relu"),
    keras.layers.Reshape((NUM_POINTS, 32)),
    keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    keras.layers.Conv1D(2, kernel_size=3, padding="same", activation="linear")
])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
lambda_physics_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)

# ADAPTIVE PHYSICS-INFORMED TRAINING LOOP
@tf.function
def train_step(x_batch, y_cp_batch, y_cl_batch):
    with tf.GradientTape() as tape:
        cp_pred = model(x_batch, training=True)
        data_loss = tf.reduce_mean(tf.square(y_cp_batch - cp_pred))

        cp_upper = cp_pred[:, :, 0]
        cp_lower = cp_pred[:, :, 1]
        
        # CORRECT NON-UNIFORM TRAPEZOIDAL INTEGRATION
        dcp = cp_lower - cp_upper
        mids = (dcp[:, :-1] + dcp[:, 1:]) / 2.0
        cn_pred = tf.reduce_sum(mids * dx_tensor, axis=1, keepdims=True)

        aoa_real = (x_batch * AOA_STD) + AOA_MEAN
        aoa_rad = aoa_real * (np.pi / 180.0)
        cl_pred = cn_pred * tf.cos(aoa_rad)

        phys_loss = tf.reduce_mean(tf.square(y_cl_batch - cl_pred))
        total_loss = data_loss + (lambda_physics_var * phys_loss)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return data_loss, phys_loss, total_loss

@tf.function
def val_step(x_batch, y_cp_batch, y_cl_batch):
    cp_pred = model(x_batch, training=False)
    data_loss = tf.reduce_mean(tf.square(y_cp_batch - cp_pred))

    cp_upper = cp_pred[:, :, 0]
    cp_lower = cp_pred[:, :, 1]
    
    # CORRECT NON-UNIFORM TRAPEZOIDAL INTEGRATION
    dcp = cp_lower - cp_upper
    mids = (dcp[:, :-1] + dcp[:, 1:]) / 2.0
    cn_pred = tf.reduce_sum(mids * dx_tensor, axis=1, keepdims=True)

    aoa_real = (x_batch * AOA_STD) + AOA_MEAN
    aoa_rad = aoa_real * (np.pi / 180.0)
    cl_pred = cn_pred * tf.cos(aoa_rad)

    phys_loss = tf.reduce_mean(tf.square(y_cl_batch - cl_pred))
    total_loss = data_loss + (lambda_physics_var * phys_loss)

    return data_loss, phys_loss, total_loss

# --- 6. EXECUTION WITH ANNEALING SCHEDULE ---
EPOCHS = 350
history = {"train_total_loss": [], "val_total_loss": [], "val_phys_loss": []}

RAMP_START_EPOCH = 50   
RAMP_END_EPOCH = 250    
LAMBDA_MAX = 1.5        

print("Starting PINN Optimization with True Non-Uniform Integration")

for epoch in range(EPOCHS):
    if epoch < RAMP_START_EPOCH:
        current_lambda = 0.0
    elif epoch >= RAMP_END_EPOCH:
        current_lambda = LAMBDA_MAX
    else:
        progress = (epoch - RAMP_START_EPOCH) / (RAMP_END_EPOCH - RAMP_START_EPOCH)
        current_lambda = progress * LAMBDA_MAX
        
    lambda_physics_var.assign(current_lambda)

    t_total = keras.metrics.Mean()
    v_total = keras.metrics.Mean()
    v_phys = keras.metrics.Mean()

    for xb, ycp, ycl in train_dataset:
        _, _, tl = train_step(xb, ycp, ycl)
        t_total.update_state(tl)

    for xb, ycp, ycl in val_dataset:
        _, pl, tl = val_step(xb, ycp, ycl)
        v_total.update_state(tl)
        v_phys.update_state(pl)

    history["train_total_loss"].append(t_total.result().numpy())
    history["val_total_loss"].append(v_total.result().numpy())
    history["val_phys_loss"].append(v_phys.result().numpy())

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Lambda: {current_lambda:.2f} | Train Loss: {t_total.result():.4f} | Val Loss: {v_total.result():.4f} | Phys Error: {v_phys.result():.6f}")

# --- 7. EXTRAPOLATION EVALUATION ---
print("\nEvaluating PINN-CNN on High-Alpha Extrapolation Set (>12 deg)...")
pred_test = model.predict(aoa_test_scaled, verbose=0)

cp_upper = pred_test[:, :, 0]
cp_lower = pred_test[:, :, 1]

cn = np.trapz(cp_lower - cp_upper, x=x_grid, axis=1)
cl_pred_test = cn * np.cos(np.radians(aoa_test.flatten()))

rmse_cl_test = np.sqrt(np.mean((cl_pred_test - cl_test.flatten())**2))
print(f"PINN-CNN TEST RMSE (Cl): {rmse_cl_test:.5f}")

test_flat_pred = pred_test.reshape(pred_test.shape[0], -1)
test_flat_true = cp_test_cnn.reshape(cp_test_cnn.shape[0], -1)
rmse_cp_test = np.sqrt(mean_squared_error(test_flat_true, test_flat_pred))
print(f"PINN-CNN TEST RMSE (Cp): {rmse_cp_test:.5f}")

model.save(os.path.join(MODELS_DIR, "pinn_cnn_model_structured.keras"))

# plot
plt.figure(figsize=(10, 6))
plt.plot(history["train_total_loss"], label="Train Total Loss", color='blue', linewidth=2)
plt.plot(history["val_total_loss"], label="Val Total Loss", color='orange', linewidth=2)
plt.plot(history["val_phys_loss"], label="Physics Penalty ($C_l$ Error)", color='green', linestyle=':', linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("PINN-CNN Convergence")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "pinn_cnn_training_convergence.png"), dpi=300)
plt.show()

sample_idx = -1  
aoa_sample = aoa_test[sample_idx][0]

plt.figure(figsize=(10,6))
plt.plot(x_grid, cp_test_cnn[sample_idx, :, 0], color='black', linewidth=2, label="CFD Upper (Truth)")
plt.plot(x_grid, cp_test_cnn[sample_idx, :, 1], color='gray', linewidth=2, linestyle='-.', label="CFD Lower (Truth)")
plt.plot(x_grid, cp_upper[sample_idx, :], color='blue', linewidth=2, linestyle='--', label="PINN-CNN Upper")
plt.plot(x_grid, cp_lower[sample_idx, :], color='orange', linewidth=2, linestyle='--', label="PINN-CNN Lower")

plt.gca().invert_yaxis()
plt.xlabel("Chord Fraction (x/c)")
plt.ylabel("Pressure Coefficient ($C_p$)")
plt.title(f"PINN-CNN Extrapolation (AoA = {aoa_sample:.1f}$^\circ$)")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "pinn_cnn_cp_prediction.png"), dpi=300)
plt.show()