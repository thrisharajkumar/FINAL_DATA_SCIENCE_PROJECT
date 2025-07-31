# src/optuna_tuner.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import joblib
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import preprocess_from_file
from src.model_builder import build_model
from src.config import TRAIN_FILE, NUM_OUTPUTS, BEST_PARAMS_PATH

def objective(trial):
    # Hyperparameter suggestions
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_units = trial.suggest_int("n_units", 64, 512)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    # Preprocess with scaling + top features
    X, y, _, _, _ = preprocess_from_file(TRAIN_FILE, num_features=50)
    sample_size = int(0.2 * len(X))
    X_sample, y_sample = X[:sample_size], y[:sample_size]

    # Build model
    model = build_model(
        input_shape=X_sample.shape[1],
        output_shape=NUM_OUTPUTS,
        hidden_layers=n_layers,
        units=n_units,
        activation=activation,
        dropout_rate=dropout,
        optimizer=optimizer,
        learning_rate=learning_rate
    )

    # Train with EarlyStopping
    model.fit(
        X_sample, y_sample,
        validation_split=0.2,
        epochs=50,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # Evaluate with mean R²
    y_pred = model.predict(X_sample, verbose=0)
    mean_r2 = r2_score(y_sample, y_pred, multioutput='uniform_average')
    return 1.0 - mean_r2  # Minimize 1 - R²

def run_optimization(n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best Trial:")
    print(study.best_trial.params)
    joblib.dump(study.best_trial.params, BEST_PARAMS_PATH)

if __name__ == "__main__":
    run_optimization(n_trials=30)
