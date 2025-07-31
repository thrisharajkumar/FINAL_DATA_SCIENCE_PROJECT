import os
import joblib
import json
from tensorflow.keras.callbacks import EarlyStopping
from src.config import (
    TRAIN_FILE, BEST_PARAMS_PATH,
    FINAL_MODEL_PATH
)
from src.model_builder import build_model
from src.preprocessing import preprocess_and_split_train

def train_final_model():
    print("🔄 Preprocessing training data (excluding holdout)...")
    X_train, y_train, X_val, y_val, _, _, feature_names = preprocess_and_split_train(TRAIN_FILE)

    os.makedirs("models", exist_ok=True)
    with open("models/input_features.json", "w") as f:
        json.dump(feature_names, f)

    print("📦 Loading best hyperparameters...")
    best_params = joblib.load(BEST_PARAMS_PATH)

    print("🏗️ Building model...")
    model = build_model(
        input_shape=X_train.shape[1],
        output_shape=y_train.shape[1],
        hidden_layers=best_params["n_layers"],
        units=best_params["n_units"],
        activation=best_params["activation"],
        dropout_rate=best_params["dropout"],
        optimizer=best_params["optimizer"],
        learning_rate=best_params["learning_rate"]
    )

    print("🚀 Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_params["batch_size"],
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    print(f"💾 Saving model to {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)

if __name__ == "__main__":
    train_final_model()
