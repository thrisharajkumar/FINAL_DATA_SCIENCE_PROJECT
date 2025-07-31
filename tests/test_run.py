


import pandas as pd


X_unseen_df = pd.read_csv("data/raw/merged_test_with_features.csv").dropna()
print("📋 Columns in Unseen Test File:")
print(X_unseen_df.columns.tolist())
