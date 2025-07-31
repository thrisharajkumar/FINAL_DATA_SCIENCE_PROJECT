# scripts/split_train_and_holdout.py

import pandas as pd
import os

INPUT_FILE = "data/raw/merged_train_with_features.csv"
TRAIN_OUT = "data/processed/train_for_model.csv"
HOLDOUT_OUT = "data/processed/unseen_holdout.csv"

def split_holdout():
    print(f"🔍 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    holdout_size = int(0.10 * len(df))
    train_df = df.iloc[:-holdout_size].reset_index(drop=True)
    holdout_df = df.iloc[-holdout_size:].reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    holdout_df.to_csv(HOLDOUT_OUT, index=False)

    print(f"Saved train data: {TRAIN_OUT} ({len(train_df)} rows)")
    print(f"Saved unseen holdout: {HOLDOUT_OUT} ({len(holdout_df)} rows)")

if __name__ == "__main__":
    split_holdout()
