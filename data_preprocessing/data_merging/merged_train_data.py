import pandas as pd
import glob
import os
import sys

# Add root to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import MERGED_DATA_DIR, MERGED_TRAIN_ALL_FILE

# === Step 1: Get all merged MOSFET CSVs ===
all_files = sorted(glob.glob(os.path.join(MERGED_DATA_DIR, "merged_*.csv")))

# === Step 2: Concatenate all files into one training DataFrame ===
train_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# === Step 3: Save full training dataset ===
os.makedirs(os.path.dirname(MERGED_TRAIN_ALL_FILE), exist_ok=True)
train_df.to_csv(MERGED_TRAIN_ALL_FILE, index=False)
print(f"Full training dataset saved to: {MERGED_TRAIN_ALL_FILE}")
