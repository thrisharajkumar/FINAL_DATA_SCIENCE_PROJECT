import pandas as pd
import glob
import os

# Step 1: Get all file paths
all_files = sorted(glob.glob("merged_data/merged_*.csv"))

# Step 2: Specify MOSFETs to include in training
train_ids = ["C2M0025120D", "C2M0040120D", "C2M0080120D", 
             "C2M0280120D", "C2M1000170D"]

# Step 3: Classify files into train/test based on MOSFET name
train_files = [f for f in all_files if any(mid in f for mid in train_ids)]
test_files = [f for f in all_files if f not in train_files]

# Step 4: Merge training files
train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

# Step 5: Use first remaining file as test (you can change this logic)
# "C2M0160120D"
test_df = pd.read_csv(test_files[0]) if test_files else None

# Step 6: Save outputs
train_df.to_csv("merged_train.csv", index=False)
if test_df is not None:
    test_df.to_csv("merged_test.csv", index=False)
    print(f"Merged training data saved to: merged_train.csv")
    print(f"Test data saved to: merged_test.csv (from {os.path.basename(test_files[0])})")
else:
    print("No test files found — only training file saved.")
