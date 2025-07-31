import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

from src.config import (
    TRAIN_FILE, INPUT_SCALER_PATH, OUTPUT_SCALER_PATH,
    FINAL_MODEL_PATH, TARGET_COLUMNS, DROP_COLUMNS
)

def main():
    print("🔄 Loading full training data...")
    df = pd.read_csv(TRAIN_FILE).dropna()
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMNS]

    print("🧪 Splitting into 80/20 train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("📦 Loading scalers and model...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)
    model = load_model(FINAL_MODEL_PATH)

    print("🔄 Scaling input and output...")
    X_test_scaled = input_scaler.transform(X_test)
    y_test_scaled = output_scaler.transform(y_test)

    print("🔮 Predicting on held-out training split...")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    y_true = output_scaler.inverse_transform(y_test_scaled)

    print("📊 Computing R² scores and saving plots...")
    r2_scores = {}
    for i, col in enumerate(TARGET_COLUMNS):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        r2_scores[col] = r2

        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, i], label="Actual", linestyle="--")
        plt.plot(y_pred[:, i], label="Predicted", alpha=0.8)
        plt.title(f"{col} (R²: {r2:.4f})")
        plt.xlabel("Sample")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join("data", "processed", f"{col}_train_eval_plot.png")
        plt.savefig(plot_path)
        plt.close()

    print("\n📈 R² Scores on Training Split (20% held-out):")
    for col, score in r2_scores.items():
        print(f"{col}: {score:.4f}")

    # Save merged CSV
    comparison_df = pd.DataFrame()
    for i, col in enumerate(TARGET_COLUMNS):
        comparison_df[f"Actual_{col}"] = y_true[:, i]
        comparison_df[f"Predicted_{col}"] = y_pred[:, i]

    output_csv = os.path.join("data", "processed", "train_split_predictions_vs_actuals.csv")
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n📝 Prediction vs Actual CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()
