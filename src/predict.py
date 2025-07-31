# src/train_final_model.py

import os
import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    TRAIN_FILE, BEST_PARAMS_PATH,
    INPUT_SCALER_PATH, OUTPUT_SCALER_PATH, FINAL_MODEL_PATH,
    TARGET_COLUMNS
)
from src.model_builder import build_model
from src.preprocessing import preprocess_from_file

def train_final_model():
    print("🔄 Loading and preprocessing training data...")
    X, y, input_scaler, output_scaler, feature_names = preprocess_from_file(TRAIN_FILE)

    print("📦 Loading best hyperparameters...")
    best_params = joblib.load(BEST_PARAMS_PATH)

    print("🏗️  Building final model...")
    model = build_model(
        input_shape=X.shape[1],
        output_shape=len(TARGET_COLUMNS),
        hidden_layers=best_params["n_layers"],
        units=best_params["n_units"],
        activation=best_params["activation"],
        dropout_rate=best_params["dropout"],
        optimizer=best_params["optimizer"],
        learning_rate=best_params["learning_rate"]
    )

    print("🚀 Training final model on full training data...")
    model.fit(
        X, y,
        batch_size=best_params["batch_size"],
        epochs=100,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    print(f"💾 Saving model to {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)
    print("✅ Final model, scalers, and features saved successfully.")

if __name__ == "__main__":
    train_final_model()
