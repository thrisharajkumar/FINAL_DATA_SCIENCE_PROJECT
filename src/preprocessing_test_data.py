import pandas as pd
import json
import os
from src.config import TEST_FILE, FEATURES_JSON_PATH

# Output path (optional: you can overwrite or save elsewhere)
OUTPUT_PATH = os.path.join(os.path.dirname(TEST_FILE), "filtered_test_data.csv")

def preprocess_test_data():
    # Load test data
    df = pd.read_csv(TEST_FILE)
    print(f"✅ Loaded test data: {df.shape}")

    # Load selected features
    with open(FEATURES_JSON_PATH, 'r') as f:
        selected_features = json.load(f)

    # Ensure all selected features are present
    missing = set(selected_features) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required features in test data: {missing}")

    # Filter only selected features
    df_filtered = df[selected_features]
    df_filtered.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved filtered test data to: {OUTPUT_PATH}")
    print(f"🎯 Shape of filtered test data: {df_filtered.shape}")

if __name__ == "__main__":
    preprocess_test_data()
