import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import keras
import joblib

# Setup & Load
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = np.load(r"D:\cfd-ml-project\data\datasets\airfoil_dataset.npz")
X = data["X"]
y_cp = data["y_cp"]
x_grid = data["x_grid"]

print(f"Loaded {len(X)} cases. Target shape: {y_cp.shape}")

# Extract AoA and reshape
aoa = X[:, 0].reshape(-1, 1)

# Scaling
input_scaler = StandardScaler()
aoa_scaled = input_scaler.fit_transform(aoa)

output_scaler = StandardScaler()
y_cp_scaled = output_scaler.fit_transform(y_cp)

os.makedirs(r"D:\cfd-ml-project\models", exist_ok=True)
joblib.dump(input_scaler, r"D:\cfd-ml-project\models\aoa_scaler.pkl")
joblib.dump(output_scaler, r"D:\cfd-ml-project\models\cp_scaler.pkl")


# Split by case to prevent data leakage
X_train, X_val, y_train, y_val = train_test_split(
    aoa_scaled, y_cp_scaled, test_size=0.2, random_state=SEED
)

early_stop = keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)

# MODEL A: Vector Output (AoA -> 200 point Cp curve)
print("\n Training Model A (Vector Output)")
model_A = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(200) 
])

model_A.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

history_A = model_A.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=4,
    callbacks=[early_stop],
    verbose=0 
)

model_A.save(r"D:\cfd-ml-project\models\model_A_full_curve.keras")
print("Model A trained and saved.")

# MODEL B: Pointwise Output ([AoA, x/c, surface_flag] ---> 1 Cp point)

print("\n Training Model B (Pointwise Output)")

def make_point_dataset(X_cases, y_cases, x_grid):
    """Flattens 2D arrays into pointwise [AoA, x/c, surface_flag] rows."""
    X_pts, y_pts = [], []
    for i in range(len(X_cases)):
        aoa = X_cases[i, 0]
        
        # Upper surface (flag = 1)
        for j, x in enumerate(x_grid):
            X_pts.append([aoa, x, 1.0])
            y_pts.append(y_cases[i, j])
            
        # Lower surface (flag = 0)
        for j, x in enumerate(x_grid):
            X_pts.append([aoa, x, 0.0])
            y_pts.append(y_cases[i, 100 + j])
            
    return np.array(X_pts), np.array(y_pts)

X_train_B, y_train_B = make_point_dataset(X_train, y_train, x_grid)
X_val_B, y_val_B = make_point_dataset(X_val, y_val, x_grid)

model_B = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1)
])

model_B.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

history_B = model_B.fit(
    X_train_B, y_train_B,
    validation_data=(X_val_B, y_val_B),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

model_B.save(r"D:\cfd-ml-project\models\model_B_pointwise.keras")
print("Model B trained and saved.")

# COMPARISON: Which architecture is better?
# Predict on validation sets
pred_A = model_A.predict(X_val, verbose=0)
pred_B_flat = model_B.predict(X_val_B, verbose=0)

# Reshape Model B's flat predictions back into (cases, 200 points) to compare fairly
pred_B = pred_B_flat.reshape(len(X_val), 200)

# Calculate global RMSE for the scaled validation data
rmse_A = np.sqrt(mean_squared_error(y_val, pred_A))
rmse_B = np.sqrt(mean_squared_error(y_val, pred_B))

print("\n--- Final Results ---")
print(f"Model A (Vector Output) RMSE:    {rmse_A:.5f}")
print(f"Model B (Pointwise Output) RMSE: {rmse_B:.5f}")

if rmse_A < rmse_B:
    print("\n[Conclusion] Model A is more accurate. Saving as primary surrogate.")
else:
    print("\n[Conclusion] Model B is more accurate. Saving as primary surrogate.")