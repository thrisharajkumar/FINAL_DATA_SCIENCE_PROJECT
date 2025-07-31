# src/predict_single_input.py

import json
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.config import (
    FINAL_MODEL_PATH, INPUT_SCALER_PATH, OUTPUT_SCALER_PATH,
    FEATURES_JSON_PATH, TARGET_COLUMNS
)

def predict_single(input_csv_path):
    print("📂 Loading input data...")
    df = pd.read_csv(input_csv_path)

    print("📋 Loading selected features...")
    with open(FEATURES_JSON_PATH, "r") as f:
        selected_features = json.load(f)

    # ⚠️ Filter the DataFrame to only use selected features
    try:
        X = df[selected_features]
    except KeyError as e:
        missing = list(set(selected_features) - set(df.columns))
        print("❌ Missing columns in input:", missing)
        return

    print("📦 Loading scalers and model...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)
    model = load_model(FINAL_MODEL_PATH)

    print("🔄 Scaling input...")
    X_scaled = input_scaler.transform(X)

    print("🔮 Predicting EMI outputs...")
    predictions_scaled = model.predict(X_scaled)
    predictions = output_scaler.inverse_transform(predictions_scaled)

    results_df = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    print("✅ Prediction complete. Sample output:\n")
    print(results_df.head())

    # Save predictions
    output_path = "data/predictions/single_prediction_output.csv"
    results_df.to_csv(output_path, index=False)
    print(f"📁 Saved predictions to: {output_path}")


if __name__ == "__main__":
    predict_single("data/raw/prediction_input_selected_47.csv")
