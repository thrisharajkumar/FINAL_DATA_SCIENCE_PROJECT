# src/predict_test_full_features.py

import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from src.config import (
    TEST_FILE, FINAL_MODEL_PATH,
    INPUT_SCALER_PATH, OUTPUT_SCALER_PATH,
    TARGET_COLUMNS
)

def main():
    print("📂 Loading test data...")
    df = pd.read_csv(TEST_FILE)

    print("🧹 Dropping DeviceID and target columns...")
    input_columns = df.drop(columns=['DeviceID'] + TARGET_COLUMNS).columns
    X_test = df[input_columns]

    print("📦 Loading model and scalers...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)
    model = load_model(FINAL_MODEL_PATH)

    print("🔄 Scaling test inputs...")
    X_scaled = input_scaler.transform(X_test)

    print("🔮 Making predictions...")
    y_pred_scaled = model.predict(X_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)

    print("💾 Saving predictions to CSV...")
    predictions_df = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)
    output_path = os.path.join("data", "processed", "predictions.csv")
    predictions_df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
