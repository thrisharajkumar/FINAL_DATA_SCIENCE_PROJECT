import pandas as pd
from sklearn.model_selection import train_test_split
import config
import os

"""
Saved 90% training data to: 
C:\Users\pc\Desktop\PROJECT_THESIS_Thrisha_Rajkumar\data\processed\train_for_model.csv 
(483114 rows)
Saved 10% holdout test data to: 
C:\Users\pc\Desktop\PROJECT_THESIS_Thrisha_Rajkumar\data\processed\merged_test_with_features.csv 
(53680 rows)
"""

def split_holdout(df, test_size=0.10, seed=42):
    """
    Split merged_train_with_features.csv into 90% train and 10% holdout.
    """
    train_df, holdout_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=seed)
    return train_df, holdout_df

def main():
    input_path = config.MERGED_TRAIN_WITH_FEATURES_FILE
    output_train_path = config.TRAIN_FOR_MODEL_FILE
    output_test_path = config.MERGED_TEST_WITH_FEATURES_FILE

    # Load full data
    df = pd.read_csv(input_path)
    print(f"Loaded {df.shape[0]} rows from {input_path}")

    # Split
    train_df, holdout_df = split_holdout(df)

    # Save 90% training data
    train_df.to_csv(output_train_path, index=False)
    print(f"Saved 90% training data to: {output_train_path} ({train_df.shape[0]} rows)")

    # Save 10% holdout data
    holdout_df.to_csv(output_test_path, index=False)
    print(f"Saved 10% holdout test data to: {output_test_path} ({holdout_df.shape[0]} rows)")

if __name__ == "__main__":
    main()
