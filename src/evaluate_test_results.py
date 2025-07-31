# src/evaluate_test_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from src.config import TEST_FILE, TARGET_COLUMNS

def main():
    print("📊 Loading actual test data and predictions...")
    df_actual = pd.read_csv(TEST_FILE)
    df_pred = pd.read_csv(os.path.join("data", "processed", "predictions.csv"))

    # Rename predicted columns
    df_pred.columns = [f"{col}_pred" for col in TARGET_COLUMNS]

    # Merge
    df_merged = pd.concat([df_actual[TARGET_COLUMNS], df_pred], axis=1)

    # Drop rows with NaNs
    df_merged = df_merged.dropna()

    r2_scores = {}

    for col in TARGET_COLUMNS:
        y_true = df_merged[col]
        y_pred = df_merged[f"{col}_pred"]

        r2 = r2_score(y_true, y_pred)
        r2_scores[col] = r2

        # Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(y_true.values, label='Actual', linestyle='--')
        plt.plot(y_pred.values, label='Predicted', alpha=0.8)
        plt.title(f"{col} (R²: {r2:.4f})")
        plt.xlabel("Sample")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join("data", "processed", f"{col}_test_plot.png")
        plt.savefig(plot_path)
        plt.close()

    print("\n📈 R² Scores on Cleaned Test Set:")
    for k, v in r2_scores.items():
        print(f"{k}: {v:.4f}")



if __name__ == "__main__":
    main()
