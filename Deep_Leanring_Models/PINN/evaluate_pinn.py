import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pinn_model import PINN
from utils_pinn import load_data_and_prepare, inverse_transform_outputs
import config

# Load preprocessed data
X_train, X_val, X_test, y_train, y_val, y_test, input_scaler, output_scaler, physics_inputs = load_data_and_prepare()

# Reload model
model = PINN(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
model.compile(optimizer='adam')
model.load_weights(os.path.join(config.MODELS_DIR, "PINN", "pinn_model_weights.h5"))

# Predictions
preds = model({"features": X_test, **physics_inputs}, training=False).numpy()
y_test_orig = inverse_transform_outputs(y_test, output_scaler)
preds_orig = inverse_transform_outputs(preds, output_scaler)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
r2 = r2_score(y_test_orig, preds_orig)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
