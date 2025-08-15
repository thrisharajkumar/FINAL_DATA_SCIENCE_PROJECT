import os
import tensorflow as tf
import pickle
from pinn_model import PINN
from utils_pinn import load_data_and_prepare
import config

# Load and preprocess data
X_train, X_val, X_test, y_train, y_val, y_test, input_scaler, output_scaler, physics_inputs = load_data_and_prepare()

model = PINN(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# Data format function
def format_batch(X, y):
    return ({'features': X, **physics_inputs}, y)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(format_batch).batch(128)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(format_batch).batch(128)

# Train model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=50)

# Save outputs
os.makedirs(os.path.join(config.MODELS_DIR, "PINN"), exist_ok=True)
model.save_weights(os.path.join(config.MODELS_DIR, "PINN", "pinn_model_weights.h5"))

with open(os.path.join(config.MODELS_DIR, "PINN", "input_scaler.pkl"), "wb") as f:
    pickle.dump(input_scaler, f)
with open(os.path.join(config.MODELS_DIR, "PINN", "output_scaler.pkl"), "wb") as f:
    pickle.dump(output_scaler, f)
