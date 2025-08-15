import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pinn_model import PINN
from utils_pinn import inverse_transform_outputs
import config

# === Load model + scalers ===
model = PINN(input_dim=47, output_dim=len(config.TARGET_COLUMNS))  # 47 = example input dim
model.compile(optimizer='adam')
model.load_weights("models/PINN/pinn_model_weights.h5")

with open("models/PINN/input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)
with open("models/PINN/output_scaler.pkl", "rb") as f:
    output_scaler = pickle.load(f)

# === Load your new data ===
input_df = pd.read_csv("data/processed/predict_input.csv")  # Replace with actual path
physics_keys = ['t_rise_est', 't_fall_est', 'f_ring_est', 'overshoot_norm_1', 'Vbus']

physics_inputs = {k: input_df[k].values.astype("float32") for k in physics_keys}
X_raw = input_df.drop(columns=config.TARGET_COLUMNS + config.DROP_COLUMNS + physics_keys, errors='ignore')
X_scaled = input_scaler.transform(X_raw).astype("float32")

# === Predict ===
pred_scaled = model({"features": X_scaled, **physics_inputs}, training=False).numpy()
preds = inverse_transform_outputs(pred_scaled, output_scaler)

# === Save or view ===
output_df = pd.DataFrame(preds, columns=config.TARGET_COLUMNS)
output_df.to_csv("data/processed/pinn_predictions.csv", index=False)
print("✅ Predictions saved to: data/processed/pinn_predictions.csv")
