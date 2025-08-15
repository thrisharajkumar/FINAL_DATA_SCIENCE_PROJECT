import os, joblib, pickle
from tensorflow.keras.callbacks import EarlyStopping

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config import TRAIN_FOR_MODEL_FILE, FINAL_MODEL_PATH_ANN, BEST_PARAMS_PATH_ANN, HISTORY_PATH_ANN
from src.ANN.model_builder import build_model
from src.ANN.preprocessing import preprocess_and_split_train

def train_final_model():
    print("🔄 Preprocessing training data...")
    X_train, y_train, X_val, y_val, _, _, _ = preprocess_and_split_train(TRAIN_FOR_MODEL_FILE)

    print("📦 Loading best Optuna hyperparameters...")
    best_params = joblib.load(BEST_PARAMS_PATH_ANN)

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

    print("🚀 Training final model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_params["batch_size"],
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    print(f"💾 Saving model to {FINAL_MODEL_PATH_ANN}")
    model.save(FINAL_MODEL_PATH_ANN)

    print(f"📈 Saving training history to {HISTORY_PATH_ANN}")
    os.makedirs(os.path.dirname(HISTORY_PATH_ANN), exist_ok=True)
    with open(HISTORY_PATH_ANN, "wb") as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    train_final_model()
