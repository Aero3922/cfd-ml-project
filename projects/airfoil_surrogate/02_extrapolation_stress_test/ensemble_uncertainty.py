import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Extrapolation Stress Test: Epistemic Uncertainty
# Objective: Deploy a Deep Ensemble to evaluate if standard ML architectures 
# exhibit high uncertainty when extrapolating to unseen high Angles of Attack.

# Setup
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("Loading aerodynamic dataset")
data = np.load(DATA_PATH)
X_raw = data["X"]
y_cl = data["y_cl"].reshape(-1, 1)
y_cd = data["y_cd"].reshape(-1, 1)

aoa = X_raw[:, 0].reshape(-1, 1)
y_forces = np.hstack((y_cl, y_cd))
print(f"Loaded {len(aoa)} cases. Target shape: {y_forces.shape}")

# Structured split (Extrapolation)
aoa_flat = aoa.flatten()
train_mask = aoa_flat <= 8
val_mask   = (aoa_flat > 8) & (aoa_flat <= 12)
test_mask  = aoa_flat > 12

X_train, X_val, X_test = aoa[train_mask], aoa[val_mask], aoa[test_mask]
y_train, y_val, y_test = y_forces[train_mask], y_forces[val_mask], y_forces[test_mask]

# Scale inputs and outputs
aoa_scaler = StandardScaler()
X_train_scaled = aoa_scaler.fit_transform(X_train)
X_val_scaled   = aoa_scaler.transform(X_val)
X_test_scaled  = aoa_scaler.transform(X_test)

force_scaler = StandardScaler()
y_train_scaled = force_scaler.fit_transform(y_train)
y_val_scaled   = force_scaler.transform(y_val)
y_test_scaled  = force_scaler.transform(y_test)

# Train ensemble
n_models = 3
ensemble_models = []

print(f"\nTraining Deep Ensemble ({n_models} models)")
for i in range(n_models):
    tf.random.set_seed(SEED + i)
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="linear")
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.fit(X_train_scaled, y_train_scaled, epochs=250, batch_size=4, verbose=0)
    
    ensemble_models.append(model)
    print(f"Ensemble member {i+1}/{n_models} trained")

# Evaluation & Uncertainty calculation (on test set)
raw_preds = [m.predict(X_test_scaled, verbose=0) for m in ensemble_models]
real_preds = np.array([force_scaler.inverse_transform(p) for p in raw_preds])

# Calculate Mean and Standard Deviation (Uncertainty) across ensemble members
mean_preds = np.mean(real_preds, axis=0)
std_preds = np.std(real_preds, axis=0)

cl_max = np.max(np.abs(y_test[:, 0]))
cd_max = np.max(np.abs(y_test[:, 1]))

cl_errors = np.abs(mean_preds[:, 0] - y_test[:, 0]) / cl_max * 100
cd_errors = np.abs(mean_preds[:, 1] - y_test[:, 1]) / cd_max * 100

print("\nExtrapolation Uncertainty Report (> 12 deg)")
for i in range(len(X_test)):
    print(f"AoA: {X_test[i, 0]:5.1f} deg | Cl Truth: {y_test[i, 0]:.4f} | Cl Mean Pred: {mean_preds[i, 0]:.4f} | Uncertainty (\u00B1): {std_preds[i, 0]:.4f}")

print(f"\nExtrapolation Average Full-Scale Error (Cl): {np.mean(cl_errors):.2f}%")
print(f"Extrapolation Average Full-Scale Error (Cd): {np.mean(cd_errors):.2f}%")