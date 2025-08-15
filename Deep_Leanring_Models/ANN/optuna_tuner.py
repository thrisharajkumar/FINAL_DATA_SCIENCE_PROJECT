import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import optuna
import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

from src.ANN.model_builder import build_model
from src.ANN.preprocessing import preprocess_from_file
from config import TRAIN_5_MOSFET_FILE, BEST_PARAMS_PATH_ANN, NUM_OUTPUTS

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_units = trial.suggest_int("n_units", 64, 512)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    # Get processed features and targets
    X, y, _, _, _ = preprocess_from_file(TRAIN_5_MOSFET_FILE)

    # FAST MODE: use only 10% for tuning
    sample_size = int(0.1 * len(X))
    X_sample, y_sample = X[:sample_size], y[:sample_size]

    model = build_model(
        input_shape=X.shape[1],
        output_shape=NUM_OUTPUTS,
        hidden_layers=n_layers,
        units=n_units,
        activation=activation,
        dropout_rate=dropout,
        optimizer=optimizer,
        learning_rate=learning_rate
    )

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=100,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )

    y_pred = model.predict(X_sample, verbose=0)
    mean_r2 = r2_score(y, y_pred, multioutput='uniform_average')
    return 1.0 - mean_r2  # Optuna minimizes

def main():
    print("🚀 Starting Optuna Tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("✅ Best trial:", study.best_trial.params)

    joblib.dump(study.best_trial.params, BEST_PARAMS_PATH_ANN)

if __name__ == "__main__":
    main()
