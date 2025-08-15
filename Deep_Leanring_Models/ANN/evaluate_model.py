import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model

from config import (
    TRAIN_FOR_MODEL_FILE, MERGED_TEST_WITH_FEATURES_FILE,
    TRAIN_5_MOSFET_FILE, TEST_1_MOSFET_FILE,
    DROP_COLUMNS, TARGET_COLUMNS,
    INPUT_SCALER_PATH_ANN, OUTPUT_SCALER_PATH_ANN, FINAL_MODEL_PATH_ANN
)

def load_and_preprocess(path, input_scaler, output_scaler):
    df = pd.read_csv(path)
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMNS]
    X_scaled = input_scaler.transform(X)
    y_scaled = output_scaler.transform(y)
    return X_scaled, y_scaled, y.values  # return original y for inverse check

def evaluate_split(name, X, y_true_scaled, y_true_original, model, output_scaler):
    y_pred_scaled = model.predict(X)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred, multioutput='raw_values'))
    r2 = r2_score(y_true_original, y_pred, multioutput='raw_values')
    print(f"\n📊 Evaluation on {name} Set:")
    for i, col in enumerate(TARGET_COLUMNS):
        print(f"{col:<30} | RMSE: {rmse[i]:.4f} | R²: {r2[i]:.4f}")
    return pd.DataFrame({"Target": TARGET_COLUMNS, "RMSE": rmse, "R2": r2, "Dataset": name})

if __name__ == "__main__":
    print("🔍 Loading model and scalers...")
    model = load_model(FINAL_MODEL_PATH_ANN)
    input_scaler = joblib.load(INPUT_SCALER_PATH_ANN)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH_ANN)

    print("📦 Loading datasets...")
    df_full = pd.read_csv(TRAIN_FOR_MODEL_FILE)
    df_full = df_full.dropna()
    X_full = df_full.drop(columns=DROP_COLUMNS)
    y_full = df_full[TARGET_COLUMNS].values

    # Pre-split into 70-15-15
    total = len(df_full)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    X_train = input_scaler.transform(X_full[:train_end])
    y_train = output_scaler.transform(y_full[:train_end])
    X_val = input_scaler.transform(X_full[train_end:val_end])
    y_val = output_scaler.transform(y_full[train_end:val_end])
    X_test = input_scaler.transform(X_full[val_end:])
    y_test = output_scaler.transform(y_full[val_end:])

    results = []
    results.append(evaluate_split("Train", X_train, y_train, y_full[:train_end], model, output_scaler))
    results.append(evaluate_split("Validation", X_val, y_val, y_full[train_end:val_end], model, output_scaler))
    results.append(evaluate_split("Internal Test", X_test, y_test, y_full[val_end:], model, output_scaler))

    # Holdout
    X_holdout, y_holdout_scaled, y_holdout = load_and_preprocess(MERGED_TEST_WITH_FEATURES_FILE, input_scaler, output_scaler)
    results.append(evaluate_split("Holdout (Merged Test)", X_holdout, y_holdout_scaled, y_holdout, model, output_scaler))

    # Seen 5 MOSFETs
    X_5, y_5_scaled, y_5 = load_and_preprocess(TRAIN_5_MOSFET_FILE, input_scaler, output_scaler)
    results.append(evaluate_split("5-MOSFET Seen", X_5, y_5_scaled, y_5, model, output_scaler))

    # Unseen 1 MOSFET
    X_unseen, y_unseen_scaled, y_unseen = load_and_preprocess(TEST_1_MOSFET_FILE, input_scaler, output_scaler)
    results.append(evaluate_split("1-MOSFET Unseen", X_unseen, y_unseen_scaled, y_unseen, model, output_scaler))

    # Combine all results
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv("evaluation_results_ann.csv", index=False)
    print("\n✅ Evaluation complete. Results saved to 'evaluation_results_ann.csv'")
