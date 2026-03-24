import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Configuration & Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Load Data & Scalers
print("Loading dataset and scalers")
data = np.load(DATA_PATH)
X_raw = data["X"]
y_cp = data["y_cp"]
y_cl = data["y_cl"].reshape(-1, 1)
x_grid = data["x_grid"]

aoa = X_raw[:, 0].reshape(-1, 1)
NUM_POINTS = y_cp.shape[1] // 2  # 100 points per surface
dx = tf.constant(x_grid[1] - x_grid[0], dtype=tf.float32)

# Load the exact scalers from cnn model
input_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_input_scaler.pkl"))
output_scaler = joblib.load(os.path.join(MODELS_DIR, "cnn_output_scaler.pkl"))

# Extract scaler constants for TensorFlow differential math (Inverse Scaling in the loop)
out_mean = tf.constant(output_scaler.mean_, dtype=tf.float32)
out_scale = tf.constant(output_scaler.scale_, dtype=tf.float32)

# Preprocess Data
aoa_train, aoa_val, cp_train, cp_val, cl_train, cl_val = train_test_split(
    aoa, y_cp, y_cl, test_size=0.2, random_state=SEED
)

aoa_train_scaled = input_scaler.transform(aoa_train)
aoa_val_scaled = input_scaler.transform(aoa_val)

cp_train_scaled = output_scaler.transform(cp_train)
cp_val_scaled = output_scaler.transform(cp_val)

# Reshape into CNN spatial channels (N, 100, 2)
def reshape_to_channels(cp_flat):
    upper = cp_flat[:, :NUM_POINTS]
    lower = cp_flat[:, NUM_POINTS:]
    return np.stack((upper, lower), axis=-1)

cp_train_cnn = reshape_to_channels(cp_train_scaled)
cp_val_cnn = reshape_to_channels(cp_val_scaled)

# Convert to TensorFlow datasets for the custom training loop
train_dataset = tf.data.Dataset.from_tensor_slices((
    aoa_train_scaled.astype(np.float32), 
    cp_train_cnn.astype(np.float32), 
    cl_train.astype(np.float32)
)).batch(4)

# Build 1D-CNN Architecture
print("\nInitializing Physics-Informed 1D-CNN")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_POINTS * 32, activation="relu"),
    tf.keras.layers.Reshape((NUM_POINTS, 32)),
    tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(2, kernel_size=3, padding="same", activation="linear")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Custom Physics-Informed Training Loop
# Weighting factor: How much should the network care about physics vs data?
LAMBDA_PHYSICS = 0.5 

@tf.function
def train_step(x_batch, y_cp_batch, y_cl_batch):
    with tf.GradientTape() as tape:
        # 1. Forward Pass (Predict Scaled Cp)
        cp_pred_scaled = model(x_batch, training=True)
        
        # 2. Data Loss (MSE of the curves)
        data_loss = tf.reduce_mean(tf.square(y_cp_batch - cp_pred_scaled))
        
        # 3. Physics Constraint (Trapezoidal Integration)
        # Flatten the (N, 100, 2) prediction to (N, 200) to apply the inverse scaler math
        cp_pred_flat_scaled = tf.concat([cp_pred_scaled[:, :, 0], cp_pred_scaled[:, :, 1]], axis=1)
        cp_pred_real = (cp_pred_flat_scaled * out_scale) + out_mean
        
        # Unpack back to upper and lower
        cp_upper_real = cp_pred_real[:, :NUM_POINTS]
        cp_lower_real = cp_pred_real[:, NUM_POINTS:]
        
        # Integrate: sum( (f(x) + f(x+1))/2 ) * dx
        dcp = cp_lower_real - cp_upper_real
        mids = (dcp[:, :-1] + dcp[:, 1:]) / 2.0
        cl_pred_integrated = tf.reduce_sum(mids, axis=1, keepdims=True) * dx
        
        # Calculate Physics Loss (MSE between integrated Cl and true CFD Cl)
        # Change y_cl_batch to float32 to match cl_pred_integrated types
        y_cl_batch = tf.cast(y_cl_batch, tf.float32)
        phys_loss = tf.reduce_mean(tf.square(y_cl_batch - cl_pred_integrated))
        
        # 4. Total Loss Calculation
        total_loss = data_loss + (LAMBDA_PHYSICS * phys_loss)
        
    # Backpropagation
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return data_loss, phys_loss, total_loss

# Execution
EPOCHS = 350
history = {"data_loss": [], "phys_loss": [], "total_loss": []}

print("Starting PINN Training")
for epoch in range(EPOCHS):
    epoch_data_loss = tf.keras.metrics.Mean()
    epoch_phys_loss = tf.keras.metrics.Mean()
    epoch_total_loss = tf.keras.metrics.Mean()
    
    for x_batch, y_cp_batch, y_cl_batch in train_dataset:
        d_loss, p_loss, t_loss = train_step(x_batch, y_cp_batch, y_cl_batch)
        epoch_data_loss.update_state(d_loss)
        epoch_phys_loss.update_state(p_loss)
        epoch_total_loss.update_state(t_loss)
        
    history["data_loss"].append(epoch_data_loss.result().numpy())
    history["phys_loss"].append(epoch_phys_loss.result().numpy())
    history["total_loss"].append(epoch_total_loss.result().numpy())
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Data Loss: {epoch_data_loss.result():.4f} | Phys Loss (Cl): {epoch_phys_loss.result():.4f}")

# Save the flagship model
model.save(os.path.join(MODELS_DIR, "pinn_cnn_model.keras"))
print("Physics-Informed CNN saved successfully")

# Plotting Convergence
plt.figure(figsize=(10, 6))
plt.plot(history["data_loss"], label="Data Loss (Cp Curve Shape)", color="blue")
plt.plot(history["phys_loss"], label="Physics Loss (Lift Integration)", color="red", linestyle="--")
plt.plot(history["total_loss"], label="Total Combined Loss", color="black", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Log Scale)")
plt.title("Physics-Informed CNN: Training Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(REPORTS_DIR, "Pinn_cnn_Training_convergence.png"), dpi=300)
plt.show()