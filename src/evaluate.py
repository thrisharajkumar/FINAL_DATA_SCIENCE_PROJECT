import os
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
from src.config import FINAL_MODEL_PATH, TRAIN_FILE, UNSEEN_HOLDOUT_FILE, OUTPUT_SCALER_PATH, TARGET_COLUMNS
from src.preprocessing import preprocess_and_split_train, preprocess_holdout_test

def evaluate():
    model = load_model(FINAL_MODEL_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)

    print("\n📊 Evaluating on internal test split...")
    _, _, _, _, X_test, y_test, _ = preprocess_and_split_train(TRAIN_FILE)
    y_pred_internal = model.predict(X_test)
    y_true_internal = output_scaler.inverse_transform(y_test)
    y_pred_internal = output_scaler.inverse_transform(y_pred_internal)

    print("\n🔍 Internal Test R² Scores:")
    for i, col in enumerate(TARGET_COLUMNS):
        r2 = r2_score(y_true_internal[:, i], y_pred_internal[:, i])
        print(f"{col}: R² = {r2:.4f}")

    print("\n🧪 Evaluating on 10% holdout set (last 50K rows)...")
    X_holdout, y_holdout_scaled, y_holdout_true = preprocess_holdout_test(UNSEEN_HOLDOUT_FILE)
    y_pred_holdout = model.predict(X_holdout)
    y_pred_holdout = output_scaler.inverse_transform(y_pred_holdout)

    print("\n🔍 Holdout R² Scores:")
    for i, col in enumerate(TARGET_COLUMNS):
        r2 = r2_score(y_holdout_true[:, i], y_pred_holdout[:, i])
        print(f"{col}: R² = {r2:.4f}")

if __name__ == "__main__":
    evaluate()
